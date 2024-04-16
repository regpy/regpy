from regpy import vecsps
from regpy.operators import Operator
import multiprocessing as mp 
from warnings import warn
from weakref import WeakValueDictionary
from regpy.util import classlogger
from enum import Enum
import os
import time

class ExitCode(Enum):
    SUCCESS=1
    ERROR=2
    TIMEOUT=3


class OperatorAsWorker(mp.Process):
    r""" 
    Process that represents an operator and can be used to do operator
    evaluations in parallel. 
    
    Parameters
    ----------
    name : string
        name of the process
    conn : mp.connection.Connection
        connection object to receive commands and 
        send the results back to master
    F : operators.Operator
        the regpy operator
    """
    log = classlogger
    def __init__(self, name, conn,F):
        super(OperatorAsWorker, self).__init__()
        self.F = F
        """the operator"""
        self.name = name
        """name of the process"""
        self.conn = conn
        """connection to master"""

    def run(self):
        """Starts the process. While running the process may receive the commands:
        'eval_nodiff': evaluates the operator with differentiate=False
        'eval_diff': evaluates the operator with differentiate=True
        'deriv': returns linearize
        'eval_nodiff': returns adjoint
        'break': ends process

        Raises:
            TypeError: Error is raised if unknown command is received
        """
        terminate=False
        while not terminate:
            res=None
            exit_code=ExitCode.ERROR
            try:    
                command = self.conn.recv()
                self.log.debug(self.name+ ' executing '+command[0])
                if command[0] ==  'eval_nodiff':
                    res=self.F(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'eval_diff':
                    res, self.deriv = self.F.linearize(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'deriv':
                    res = self.deriv(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'adjoint':
                    if self.F.linear:
                        res = self.F.adjoint(command[1])
                    else:
                        res = self.deriv.adjoint(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'break':
                    terminate=True
                else:
                    raise TypeError(self.name+': unknown command ',command[0])
            except TypeError:
                exit_code=ExitCode.ERROR
                res=TypeError(f"Error in subprocess: {self.name}: unknown command",command[0])
            except:
                exit_code=ExitCode.ERROR
                res=RuntimeError(f"Error in subprocess: An error occured during the computation of {command[0]}")
            if(not terminate):
                self.conn.send([exit_code,res])
        return 0
            

def check_running(conns,conn_m):
    parent_id=os.getppid()
    terminated=False
    while(os.getppid()==parent_id and not terminated):
        if(conn_m.poll(10)):
            terminated=True
    if(not terminated):
        time.sleep(10)
        for conn in conns:
            if(conn.poll()):
                conn.recv()
            conn.send(['break'])
        print("Closed remaining background processes.")


class ParallelInterface:
    MAX_SUBPROCESSES=128
    parallel_instances=[WeakValueDictionary()]
    _min_id_inst=0
    _id_manager=0

    def total_subprocess_count():
        tot_sum=0
        for p_inst in ParallelInterface.parallel_instances:
            tot_sum+=sum([instance.subprocess_count for instance in p_inst.values() if instance.running])
        return tot_sum

    def warn_subprocess_count():
        sp_count=ParallelInterface.total_subprocess_count()
        if(sp_count> ParallelInterface.MAX_SUBPROCESSES):
            warn(f"Warning: There are already {sp_count} subprocesses running.",stacklevel=2)

    def terminate_managed_instances(manager_id):
        for i in range(manager_id,len(ParallelInterface.parallel_instances)):
            for instance in ParallelInterface.parallel_instances[i].values():
                instance.terminate_all()


    def terminate_all_instances():
        ParallelInterface.terminate_all_managed_instances(0)

    def add_manager():
        ParallelInterface.parallel_instances.append(WeakValueDictionary())
        ParallelInterface._id_manager+=1
        return ParallelInterface._id_manager


    def __init__(self,conns,subprocess_count,end_command="break"):
        self.conns=conns
        self.subprocess_count=subprocess_count
        self.end_command=end_command
        ParallelInterface.parallel_instances[ParallelInterface._id_manager][ParallelInterface._min_id_inst]=self
        ParallelInterface._min_id_inst+=1
        self.running=True
        ParallelInterface.warn_subprocess_count()
        conn_m, conn_w = mp.Pipe()
        self.conn_watcher=conn_m
        process = mp.Process(target=check_running, args=(conns,conn_w))
        process.start()

    def terminate_all(self):
        if(self.running):
            for conn in self.conns:
                if(conn.poll()):
                    rec_d=conn.recv()
                conn.send([self.end_command])
            self.subprocess_count=0
            self.conn_watcher.send(['break'])
            self.running=False

    def handle_errors(self,rec_d):
        if(rec_d[0]==ExitCode.ERROR):
            self.terminate_all()
            raise rec_d[1]
        elif(rec_d[0]==ExitCode.TIMEOUT):
            self.terminate_all()
            raise TimeoutError("Subprocess timed out!")

    def compute_all(self,command,args_same=[],args_specific=[]):
        if(not self.running):
            raise RuntimeError(f"Computation of {command} is impossible, because process {self} was already terminated.")
        same_info=[command]+args_same
        for i,conn in enumerate(self.conns):
            conn.send(same_info+[arg[i] for arg in args_specific])
        rec_data=[conn.recv() for conn in self.conns]
        for rec_d in rec_data:
            self.handle_errors(rec_d)
        return (rec_d[1] for rec_d in rec_data)

    def __del__(self):
        self.terminate_all()



class ParallelVectorOfOperators(Operator,ParallelInterface):
    """Vector of operators in which all components are evaluated in parallel. 
    The functionality is identical to the sequential analog VectorOfOperators: For

        T_i : X -> Y_i

    we define

        T := VectorOfOperators(T_i) : X -> DirectSum(Y_i)

    by `T(x)_i := T_i(x)`. 
    
    Parameters
    ----------
    *ops : tuple of Operator
    codomain : vecsps.VectorSpace or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, ops,  domain=None, codomain=None):
        assert all([isinstance(op, Operator) for op in ops])
        assert ops

        if domain is None:
            self.domain = ops[0].domain
        else:
            self.domain = domain
        assert all(op.domain == self.domain for op in ops)

        if codomain is None:
            codomain = vecsps.DirectSum
        if isinstance(codomain, vecsps.VectorSpace):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in ops))
        else:
            raise TypeError('codomain={} is neither a VectorSpace nor callable'.format(codomain))
        assert all(op.codomain == c for op, c in zip(ops, codomain))

        conns = []
        it = 0
        for op in ops:
            conn_m, conn_w = mp.Pipe()
            conns.append(conn_m)
            G = OperatorAsWorker(type(op).__name__+' as worker '+str(it),conn_w,op)
            G.start()
            it += 1
        Operator.__init__(self,domain=self.domain, codomain=codomain, linear=all(op.linear for op in ops))
        ParallelInterface.__init__(self,conns,len(conns))
        

    def _eval(self, x, differentiate=False):
        if differentiate:
            return self.codomain.join(*self.compute_all('eval_diff',[x]))
        else:
            return self.codomain.join(*self.compute_all('eval_nodiff',[x]))

    def _derivative(self, x):
        return self.codomain.join(*self.compute_all('deriv',[x]))

    def _adjoint(self, y):
        assert self.running
        elms = self.codomain.split(y)
        return sum(self.compute_all('adjoint',args_specific=[elms]))

class ParallelExecutionManager:

    def __init__(self):
        pass

    def __enter__(self):
        ParallelInterface.warn_subprocess_count()
        self.manager_id=ParallelInterface.add_manager()
        return self

    def __exit__(self,type, value, traceback):
        ParallelInterface.terminate_managed_instances(self.manager_id)







