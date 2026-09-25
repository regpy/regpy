from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from types import NotImplementedType
    from numpy.typing import ArrayLike, DTypeLike
    import numpy as np
    from regpy.vecsps.base import ReadOnlyVector, Scalar, Vector
    from multiprocessing.connection import Connection
    from types import TracebackType
from enum import Enum
import multiprocessing as mp 
import os
import time
import logging
from warnings import warn
from weakref import WeakValueDictionary

from regpy.util import ClassLogger, Errors
from regpy.vecsps import VectorSpaceBase
from regpy.vecsps import DirectSum as DirectSumVS

from .base import Operator

__all__ = [
    "OperatorAsWorker", "ParallelExecutionManager", "ParallelVectorOfOperators", "DistributedVectorOfOperators"
]

class ExitCode(Enum):
    SUCCESS=1
    ERROR=2
    TIMEOUT=3

class OperatorAsWorker(mp.Process):
    r"""Process that represents an operator and can be used to do operator
    evaluations in parallel. 
    
    Parameters
    ----------
    name: str
        name of the process
    conn: Connection
        connection object to receive commands and 
        send the results back to master
    F: Operator
        the regpy operator
    """
    log = ClassLogger()
    def __init__(self, name: str, conn: Connection,F: Operator) -> None:
        super(OperatorAsWorker, self).__init__()
        self.F = F
        """the operator"""
        self.name = name
        """name of the process"""
        self.conn = conn
        """connection to master"""
        self._codomain_buffer = None
        """Reusable output buffer for evaluations and derivatives."""
        self._domain_buffer = None
        """Reusable output buffer for adjoint evaluations."""

    def _get_codomain_buffer(self) -> Vector:
        if self._codomain_buffer is None:
            self._codomain_buffer = self.F.codomain.empty()
        return self._codomain_buffer

    def _get_domain_buffer(self) -> Vector:
        if self._domain_buffer is None:
            self._domain_buffer = self.F.domain.empty()
        return self._domain_buffer

    def run(self) -> None:
        r"""Starts the process. While running the process may receive the commands:
        'eval_nodiff': evaluates the operator with differentiate=False
        'eval_diff': evaluates the operator with differentiate=True
        'deriv': evaluates the derivative
        'adjoint': evaluates the adjoint
        'ieval_nodiff': evaluates into a reusable worker-side codomain buffer
        'ieval_diff': linearizes into a reusable worker-side codomain buffer
        'ideriv': evaluates the derivative into that codomain buffer
        'iadjoint': evaluates the adjoint into a reusable worker-side domain buffer
        'break': ends process
        
        Raises
        ------
        TypeError
            Error is raised if unknown command is received
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
                elif command[0] == 'ieval_nodiff':
                    res = self.F(command[1], out=self._get_codomain_buffer())
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'eval_diff':
                    res, self.deriv = self.F.linearize(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'ieval_diff':
                    res, self.deriv = self.F.linearize(
                        command[1], out=self._get_codomain_buffer()
                    )
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'deriv':
                    if self.F.linear:
                        res = self.F(command[1])
                    else:
                        res = self.deriv(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'ideriv':
                    if self.F.linear:
                        res = self.F(command[1], out=self._get_codomain_buffer())
                    else:
                        res = self.deriv(
                            command[1], out=self._get_codomain_buffer()
                        )
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'adjoint':
                    if self.F.linear:
                        res = self.F.adjoint(command[1])
                    else:
                        res = self.deriv.adjoint(command[1])
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'iadjoint':
                    if self.F.linear:
                        res = self.F.adjoint(
                            command[1], out=self._get_domain_buffer()
                        )
                    else:
                        res = self.deriv.adjoint(
                            command[1], out=self._get_domain_buffer()
                        )
                    exit_code=ExitCode.SUCCESS
                elif command[0] == 'break':
                    terminate=True
                else:
                    raise TypeError(self.name+': unknown command ',command[0])
            except TypeError:
                exit_code=ExitCode.ERROR
                res=TypeError(f"Error in subprocess: {self.name}: unknown command",command[0])
            except BaseException as error:
                exit_code=ExitCode.ERROR
                res=RuntimeError(f"{error} in subprocess {self.name}: An error occurred during the computation of {command[0]}")
            if(not terminate):
                self.conn.send([exit_code,res])
        return 0
            

def check_running(conns: list[Connection],conn_m: Connection) -> None:
    r"""Function that runs in separate watcher process and checks if main process is alive.
    Terminates subprocesses after 10 seconds if main process is killed.
    
    Parameters
    ----------
    conns: list[Connection]
        connections to subprocesses of main process
    conn_m: Connection
        Conn m.
    """
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
        logging.log(logging.INFO,"Closed remaining background processes.")


class ParallelInterface:
    r"""Interface for parallel processing 
    
    Parameters
    ----------
    conns: list[Connection]
        List of connections used to send commands to worker processes
        and receive results.
    end_command: str [default: 'break']
        Command that terminates sub processes. 
    """


    MAX_SUBPROCESSES=128
    """maximal number of subprocesses until warning is raised"""
    parallel_instances=[WeakValueDictionary()]
    """list of dictionaries containing weak references to subprocesses. Used internally for terminating subprocesses."""
    _min_id_inst=0
    _id_manager=0

    def total_subprocess_count() -> int:
        r"""Calculates the total number of running processes.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        tot_sum=0
        for p_inst in ParallelInterface.parallel_instances:
            tot_sum+=sum([instance.subprocess_count for instance in p_inst.values() if instance.running])
        return tot_sum

    def warn_subprocess_count() -> None:
        r"""Produces a warning if the total number of running processes is higher than
        MAX_SUBPROCESSES.
        """
        sp_count=ParallelInterface.total_subprocess_count()
        if(sp_count> ParallelInterface.MAX_SUBPROCESSES):
            warn(f"Warning: There are already {sp_count} subprocesses running.",stacklevel=2)

    def terminate_managed_instances(manager_id: int) -> None:
        r"""Terminate all instances of ParallelInterface associated with manager_id or a higher id
        
        Parameters
        ----------
        manager_id: int
            id of ParallelExecutionManager
        """
        for i in range(manager_id,len(ParallelInterface.parallel_instances)):
            for instance in ParallelInterface.parallel_instances[i].values():
                instance.terminate_all()
        if(manager_id>0):
            ParallelInterface._id_manager=manager_id-1
            ParallelInterface.parallel_instances=ParallelInterface.parallel_instances[:manager_id]
        else:
            ParallelInterface._id_manager=0
            ParallelInterface.parallel_instances=[WeakValueDictionary()]

    def terminate_all_instances() -> None:
        r"""Terminates all instances of ParallelInterface.
        """
        ParallelInterface.terminate_all_managed_instances(0)


    def add_manager() -> None:
        r"""Adds a new manager section and returns the corresponding manager id.
        """
        ParallelInterface.parallel_instances.append(WeakValueDictionary())
        ParallelInterface._id_manager+=1
        return ParallelInterface._id_manager


    def __init__(self,conns: list[Connection],end_command: str="break") -> None:
        self.conns=conns
        """Connection to subprocesses"""
        self.subprocess_count=len(conns)
        """Number of subprocesses"""
        self.end_command=end_command
        """Command used to end sub processes"""
        #Add current instance to weak dictionary at current manager id
        ParallelInterface.parallel_instances[ParallelInterface._id_manager][ParallelInterface._min_id_inst]=self
        ParallelInterface._min_id_inst+=1
        self.running=True
        """Flag which indicates if subprocesses of this object are still running"""
        ParallelInterface.warn_subprocess_count()
        #Setup watcher process
        conn_m, conn_w = mp.Pipe()
        self.conn_watcher=conn_m
        process = mp.Process(target=check_running, args=(conns,conn_w))
        process.start()

    def terminate_all(self) -> None:
        r"""Terminate all running subprocesses.
        """
        if(self.running):
            for conn in self.conns:
                if(conn.poll()):
                    rec_d=conn.recv()
                conn.send([self.end_command])
            self.subprocess_count=0
            self.conn_watcher.send(['break'])
            self.running=False

    def handle_errors(self,rec_d: list[Any]) -> None:
        r"""Handles received errors by displaying massage and stopping subprocesses.
        
        Parameters
        ----------
        rec_d: list[Any]
            list where the first entry is an ExitCode that indicates whether an error occurred
            in the subprocess
        """
        if(rec_d[0]==ExitCode.ERROR):
            self.terminate_all()
            raise rec_d[1]
        elif(rec_d[0]==ExitCode.TIMEOUT):
            self.terminate_all()
            raise TimeoutError("Subprocess timed out!")

    def compute_all(self,command: str,arg_same: np.ndarray | None=None,args_specific: list[Any]=[]) -> list[Any]:
        r"""Sends command and argument to all subprocesses and returns results.
        
        Parameters
        ----------
        command: str
            command describing task for subprocess
        arg_same: np.ndarray | None [default: None]
            argument send to all subprocesses.
        args_specific: list[Any] [default: []]
            List of arguments where args_specific[j] is send to subprocess j.
        
        
        Returns
        -------
        list[Any]
            Result described above.
        """
        if(not self.running):
            raise RuntimeError(f"Computation of {command} is impossible, because process {self} was already terminated.")
        same_info=[command]
        if(arg_same is not None):
            same_info.append(arg_same)
            for conn in self.conns:
                conn.send(same_info)
        elif(len(args_specific)==len(self.conns)):
            for i,conn in enumerate(self.conns):
                conn.send(same_info+[args_specific[i]])
        else:
            raise ValueError("Invalid number of arguments for parallel operators!")
        rec_data=[conn.recv() for conn in self.conns]
        for rec_d in rec_data:
            self.handle_errors(rec_d)
        return (rec_d[1] for rec_d in rec_data)

    def __del__(self) -> None:
        r"""Terminates all subprocesses when object is garbage collected.
        """
        self.terminate_all()



class ParallelVectorOfOperators(Operator,ParallelInterface):
    r"""Vector of operators in which all components are evaluated in parallel. 
    The functionality is identical to the sequential analog `VectorOfOperators`: For
    
        T_i : X -> Y_i
    
    we define
    
        T := VectorOfOperators(T_i) : X -> DirectSum(Y_i)
    
    by `T(x)_i := T_i(x)`. 
    
    Parameters
    ----------
    ops: Operator
        Ops.
    domain: VectorSpaceBase | None [default: None]
        Domain vector space.
    codomain: VectorSpaceBase | None [default: None]
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.DirectSum.
    """

    def __init__(self, ops: Operator,  domain: VectorSpaceBase | None=None, codomain: VectorSpaceBase | None=None) -> None:
        if not isinstance(ops,(list,tuple)) or len(ops) == 0:
            raise ValueError(Errors.value_error('At least one operator must be given in the ParallelVectorOperator'))
        if any([not isinstance(op, Operator) for op in ops]):
            raise TypeError(Errors.type_error('All arguments must be instances of regpy.operators.Operator in the ParallelVectorOperator. The given arguments are of types: {}'.format([type(op) for op in ops])))

        if domain is None:
            self.domain = ops[0].domain
        else:
            self.domain = domain
        if any(op.domain != self.domain for op in ops):
            raise ValueError(Errors.value_error('All operators must have the same domain in a ParallelVectorOfOperators.'))

        if codomain is None:
            codomain = DirectSumVS
        if isinstance(codomain, VectorSpaceBase):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in ops))
        else:
            raise TypeError(Errors.type_error('codomain={} is neither a VectorSpaceBase nor callable'.format(codomain),self))
        if any(op.codomain != c for op, c in zip(ops, codomain)):
            raise ValueError(Errors.value_error('All operators must have the same codomain as the corresponding summand of the codomain.',self))

        conns = []
        it = 0
        for op in ops:
            conn_m, conn_w = mp.Pipe()
            conns.append(conn_m)
            G = OperatorAsWorker(type(op).__name__+' as worker '+str(it),conn_w,op)
            G.start()
            it += 1
        Operator.__init__(self,domain=self.domain, codomain=codomain, linear=all(op.linear for op in ops))
        ParallelInterface.__init__(self,conns)
        

    def _eval(self, x: ReadOnlyVector, differentiate: bool=False) -> Vector:
        if differentiate:
            return self.codomain.join(*self.compute_all('eval_diff',x))
        else:
            return self.codomain.join(*self.compute_all('eval_nodiff',x))

    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        command = 'ieval_diff' if differentiate else 'ieval_nodiff'
        results = self.compute_all(command, x)
        for summand, out_i, result_i in zip(self.codomain, out, results):
            summand.copyto(out_i, result_i)
        return out

    def _derivative(self, x: ReadOnlyVector) -> Vector:
        return self.codomain.join(*self.compute_all('deriv',x))

    def _iderivative(self, x: ReadOnlyVector, out: Vector) -> Vector:
        results = self.compute_all('ideriv', x)
        for summand, out_i, result_i in zip(self.codomain, out, results):
            summand.copyto(out_i, result_i)
        return out

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        return sum(self.compute_all('adjoint',args_specific=y))

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        results = iter(self.compute_all('iadjoint', args_specific=y))
        first = next(results)
        self.domain.copyto(out, first)
        for result in results:
            out += result
        return out
    
class DistributedVectorOfOperators(Operator,ParallelInterface):
    r"""Vector of operators in which all components are evaluated in parallel and the input
    is assumed to be from direct sum of spaces that is then distributed to the operators that
    need it
    
    .. math::
        T_i : X_{i_1}\times X_{i_2}... -> Y_i
    
    we define
    
        T := VectorOfOperators(T_i) : DirectSum(X_{i_j}) -> DirectSum(Y_i)
    
    by `T(x)_i := T_i(x_i1,x_i2,...)`. 
    
    Parameters
    ----------
    ops: Operator
        Ops.
    domain: VectorSpaceBase
        The domain of the operator. It should usually be a direct sum of vector spaces
    distribution_mat: list[list[int]]
        The matrix that indicates which parts of the arguments are passed to which operator. If the entry M_i,j is True the
        j-th component of the argument is passed to the i-th operator.
    codomain: VectorSpaceBase | None [default: None]
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.DirectSum.
    """

    def __init__(self, ops: Operator,  domain: VectorSpaceBase,distribution_mat: list[list[int]], codomain: VectorSpaceBase | None=None) -> None:
        if not isinstance(ops,(list,tuple)) or len(ops) == 0:
            raise ValueError(Errors.value_error('At least one operator must be given in the DistributedVectorOfOperators'))
        if any([not isinstance(op, Operator) for op in ops]):
            raise TypeError(Errors.type_error('All arguments must be instances of regpy.operators.Operator in the DistributedVectorOfOperators. The given arguments are of types: {}'.format([type(op) for op in ops])))

        if not isinstance(domain,DirectSumVS):
            raise TypeError(Errors.not_instance(domain,DirectSumVS, add_info="DistributedVectorOfOperators: Domain needs to be a DirectSum vector space!"))

        if codomain is None:
            codomain = DirectSumVS
        if isinstance(codomain, VectorSpaceBase):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in ops))
        else:
            raise TypeError(Errors.type_error('DistributedVectorOfOperators: codomain={} is neither a VectorSpaceBase nor callable'.format(codomain)))
        if any(op.codomain != c for op, c in zip(ops, codomain)):
            raise ValueError(Errors.value_error(f'DistributedVectorOfOperators: The summands of the codomain need to match with the codomain of the operators!'))
        self.distribution_mat=distribution_mat
        self.distribution_lists=[[j for j in range(distribution_mat.shape[1]) if distribution_mat[i,j]] for i in range(distribution_mat.shape[0])]
        for op,indices in zip(ops,self.distribution_lists):
            if len(indices) == 1 and op.domain != domain[indices[0]]:
                raise ValueError(Errors.value_error(f"The domain of index {indices[0]} that should be used by the operator {op} does not match! "+"\n\t "+f"domain = {domain[indices[0]]} "+"\n\t "+f"op.domain = {op.domain}."))
            elif not isinstance(op.domain,DirectSumVS) or any((d != domain[indices[j]] for j,d in enumerate(op.domain.summands))):
                raise ValueError(Errors.value_error(f"""
            Either the domain of the operator is not a DirectSum but should get multiple inputs as defined by 
            indices = {indices} or one of the domains referred to by the indices does not match the operators 
            domain summands! 
                domain.summands[indices] = {tuple(domain[ind] for ind in indices)} 
                op.domain = {op.domain.summands}."""))
        conns = []
        it = 0
        for op in ops:
            conn_m, conn_w = mp.Pipe()
            conns.append(conn_m)
            G = OperatorAsWorker(type(op).__name__+' as worker '+str(it),conn_w,op)
            G.start()
            it += 1
        self.ops=ops
        Operator.__init__(self,domain=domain, codomain=codomain, linear=all(op.linear for op in ops))
        ParallelInterface.__init__(self,conns)
    
    def _distribute(self,x: ReadOnlyVector) -> list[ReadOnlyVector]:
        x_ops=[]
        for indices,op in zip(self.distribution_lists,self.ops):
            if(len(indices)==1):
                x_ops.append(x[indices[0]])
            else:
                x_ops.append(op.domain.join(*[x[j] for j in indices]))
        return x_ops
    
    def _collect(self,x_res: Vector) -> Vector:
        elms=self.domain.zeros()
        for i,indices in enumerate(self.distribution_lists):
            if len(indices) == 1:
                elms[indices[0]] += x_res[i]
            else:
                for x_j,j in zip(x_res[i],indices):
                    elms[j] += x_j
        return elms

    def _icollect(self, x_res: Vector, out: Vector) -> Vector:
        touched = [False] * len(self.domain)
        for result_i, indices in zip(x_res, self.distribution_lists):
            if len(indices) == 1:
                contributions = ((result_i, indices[0]),)
            else:
                contributions = zip(result_i, indices)
            for contribution, j in contributions:
                if touched[j]:
                    out[j] += contribution
                else:
                    self.domain[j].copyto(out[j], contribution)
                    touched[j] = True

        # Components that are not used by any worker must still be overwritten.
        for j, was_touched in enumerate(touched):
            if not was_touched:
                self.domain[j].fill(out[j], 0)
        return out

    def _eval(self, x: ReadOnlyVector, differentiate: bool=False) -> Vector:
        x_ops=self._distribute(x)
        if differentiate:
            return self.codomain.join(*self.compute_all('eval_diff',args_specific=x_ops))
        else:
            return self.codomain.join(*self.compute_all('eval_nodiff',args_specific=x_ops))

    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        x_ops = self._distribute(x)
        command = 'ieval_diff' if differentiate else 'ieval_nodiff'
        results = self.compute_all(command, args_specific=x_ops)
        for summand, out_i, result_i in zip(self.codomain, out, results):
            summand.copyto(out_i, result_i)
        return out

    def _derivative(self, x: ReadOnlyVector) -> Vector:
        x_ops=self._distribute(x)
        return self.codomain.join(*self.compute_all('deriv',args_specific=x_ops))

    def _iderivative(self, x: ReadOnlyVector, out: Vector) -> Vector:
        x_ops = self._distribute(x)
        results = self.compute_all('ideriv', args_specific=x_ops)
        for summand, out_i, result_i in zip(self.codomain, out, results):
            summand.copyto(out_i, result_i)
        return out

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        return self._collect(list(self.compute_all('adjoint',args_specific=y)))

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        results = self.compute_all('iadjoint', args_specific=y)
        return self._icollect(results, out)

class ParallelExecutionManager:
    r"""Context manager used to manage objects that use the ParallelInterface.
    Before such objects are created a ParallelExecutionManager should be entered using
    `with ParallelExecutionManager()` to guarantee that subprocesses are closed correctly.
    """

    def __init__(self) -> None:
        pass

    def __enter__(self) -> Operator | NotImplementedType:
        r"""Gets manager id from ParallelInterface.
        
        
        Returns
        -------
        Operator | NotImplementedType
            Result described above.
        """
        ParallelInterface.warn_subprocess_count()
        self.manager_id=ParallelInterface.add_manager()
        return self

    def __exit__(self,type: DTypeLike, value: Scalar | ArrayLike, traceback: TracebackType | None) -> None:
        r"""Terminates all managed instances of ParallelInterface.
        
        Parameters
        ----------
        type: DTypeLike
            Type.
        value: Scalar | ArrayLike
            Value.
        traceback: TracebackType | None
            Traceback.
        """
        ParallelInterface.terminate_managed_instances(self.manager_id)

