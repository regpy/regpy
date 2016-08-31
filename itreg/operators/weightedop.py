from . import LinearOperator

__all__ = ['WeightedOp']


class WeightedOp(LinearOperator):  
    def __init__(self, op, weight):
        self.domx = op.domx
        self.domy = op.domy
        self.log = op.log
        self.op = op
        self.weight = weight
        
    def __call__(self, x):
        return self.weight * self.op(x)

    def adjoint(self, x):
        return self.op.adjoint(self.weight * x)