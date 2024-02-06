from regpy import util, vecsps
from regpy.operators import Operator
import multiprocessing as mp 
from regpy.util import classlogger

class OperatorAsWorker(mp.Process):
    log = classlogger
    def __init__(self, name, conn,F):
        super(OperatorAsWorker, self).__init__()
        self.F = F
        self.name = name
        self.conn = conn

    def run(self):
        while True:
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
                break
            else:
                raise TypeError(self.name+': unknown command ',command[0])

class ParallelVectorOfOperators(Operator):
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

        self.conn = []
        it = 0
        for op in ops:
            conn_m, conn_w = mp.Pipe()
            self.conn.append(conn_m)
            G = OperatorAsWorker(type(op).__name__+' as worker '+str(it),conn_w,op)
            G.start()
            it += 1
        super().__init__(domain=self.domain, codomain=codomain, linear=all(op.linear for op in ops))

 #   def __del__(self):
 #       for conn_m in self.conn:
 #           conn_m.send('break')

    def _eval(self, x, differentiate=False):
        if differentiate:
            for conn_m in self.conn:
                conn_m.send(['eval_diff',x])
        else:
            for conn_m in self.conn:
                conn_m.send(['eval_nodiff',x])   
        #results = []
        #for conn_m in self.conn:
        #    res = conn_m.recv()
        #    self.log.info('received result of size ' + str(res.shape))
        #    results.append(res)
        #self.log.info('now joining results. type:'+str(type(results)))
        #return self.codomain.join(*tuple(results))
        aux = self.codomain.join(*(conn_m.recv() for conn_m in self.conn))
        return aux

    def _derivative(self, x):
        for conn_m in self.conn:
                conn_m.send(['deriv',x])   
        return self.codomain.join(*(conn_m.recv() for conn_m in self.conn))

    def _adjoint(self, y):
        elms = self.codomain.split(y)
        result = self.domain.zeros()    
        for conn_m, elm in zip(self.conn, elms):
            conn_m.send(['adjoint',elm])
            result += conn_m.recv()
        return result

#   @util.memoized_property
#    def __repr__(self):
#        return util.make_repr(self, *self.ops)
#
#    def __getitem__(self, item):
#        return self.ops[item]
#
#    def __iter__(self):
#        return iter(self.ops)
