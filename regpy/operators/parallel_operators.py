from regpy import util, vecsps
from regpy.operators import Operator
import multiprocessing as mp 
from warnings import warn
from regpy.util import classlogger

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
        while not terminate:#TODO improve error handling
            command = self.conn.recv()
            self.log.debug(self.name+ ' executing '+command[0])
            if command[0] ==  'eval_nodiff':
                res=self.F(command[1])
                self.conn.send(res)
            elif command[0] == 'eval_diff':
                res, self.deriv = self.F.linearize(command[1])
                self.conn.send(res)
            elif command[0] == 'deriv':
                res = self.deriv(command[1])
                self.conn.send(res)
            elif command[0] == 'adjoint':
                if self.F.linear:
                    res = self.F.adjoint(command[1])
                else:
                    res = self.deriv.adjoint(command[1])
                self.conn.send(res)
            elif command[0] == 'break':
                terminate=True
            else:
                raise TypeError(self.name+': unknown command ',command[0])
        print("Finished process")
        return 0

class ParallelInterface:
    def __init__(self,conns,subprocess_count,parallel_manager=None,end_command="break"):
        self.conns=conns
        self.subprocess_count=subprocess_count
        self.end_command=end_command
        self.parallel_manager=parallel_manager
        if(self.parallel_manager!=None):
            self.pid=self.parallel_manager.append(self)
        self.running=True

    def terminate_all(self,call_manager=False):
        for conn in self.conns:
            conn.send([self.end_command])
        self.subprocess_count=0
        self.running=False
        if(self.parallel_manager!=None and call_manager):
            self.parallel_manager.terminated(self.pid)

    def __del__(self):
        if(self.running):
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

    def __init__(self, ops,  domain=None, codomain=None,parallel_manager=None):
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
        ParallelInterface.__init__(self,conns,len(conns),parallel_manager)

    def _eval(self, x, differentiate=False):
        assert self.running
        if differentiate:
            for conn_m in self.conns:
                conn_m.send(['eval_diff',x])
        else:
            for conn_m in self.conns:
                conn_m.send(['eval_nodiff',x])   
        aux = self.codomain.join(*(conn_m.recv() for conn_m in self.conns))
        return aux

    def _derivative(self, x):
        assert self.running
        for conn_m in self.conns:
                conn_m.send(['deriv',x])   
        return self.codomain.join(*(conn_m.recv() for conn_m in self.conns))

    def _adjoint(self, y):
        assert self.running
        elms = self.codomain.split(y)
        result = self.domain.zeros()    
        for conn_m, elm in zip(self.conns, elms):
            conn_m.send(['adjoint',elm])
            result += conn_m.recv()
        return result

class ParallelExecutionManager:
    MAX_SUBPROCESSES=1#128
    total_subprocesses=0

    def __init__(self):
        self._min_id=0
        self.managed_ops={}
        self._managed_processes=0

    def check_subprocess_count():
        if(ParallelExecutionManager.total_subprocesses> ParallelExecutionManager.MAX_SUBPROCESSES):
            warn(f"Warning: There are already {ParallelExecutionManager.total_subprocesses} subprocesses running.")

    def __enter__(self):
        ParallelExecutionManager.check_subprocess_count()
        return self

    def __exit__(self,type, value, traceback):
        self.terminate_all()

    @property
    def managed_processes(self):
        return self._managed_processes
    
    @managed_processes.setter
    def managed_processes(self,new_ammount):
        print(new_ammount)
        ParallelExecutionManager.total_subprocesses+=new_ammount-self._managed_processes
        ParallelExecutionManager.check_subprocess_count()
        self._managed_processes=new_ammount

    def append(self,parallel_op):
        assert isinstance(parallel_op,ParallelInterface)
        id=self._min_id
        self._min_id+=1
        self.managed_ops.update({id:(parallel_op,parallel_op.subprocess_count)})
        self.managed_processes+=parallel_op.subprocess_count
        return id

    def terminated(self,id):
        _,subprocess_count=self.managed_ops.pop()
        self.managed_processes-=subprocess_count

    def terminate_all(self):
        for id in self.managed_ops.keys():
            self.managed_ops[id][0].terminate_all()
        self.managed_ops.clear()
        self.managed_processes=0
        self._min_id=0






