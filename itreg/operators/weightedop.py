from . import LinearOperator

__all__ = ['WeightedOp']


class WeightedOp(LinearOperator):  
    """The Weighted operator.
    
    Constructs an operator with weighted adjoint and derivative.
    

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    weight : array
        The weight.
    """
    
    
    def __init__(self, op, weight):
        """Initialization of parameters"""
        
        self.domx = op.domx
        self.domy = op.domy
        self.log = op.log
        self.op = op
        self.weight = weight

        
    def __call__(self, x):
        return self.weight * self.op(x)

    def adjoint(self, x):
        return self.op.adjoint(self.weight*x)