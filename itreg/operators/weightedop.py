from . import LinearOperator

__all__ = ['WeightedOp']


class WeightedOp(LinearOperator):  
    """Weight the given linear operator in a certain way.
    
    This operator is used by the inner solver ``sqp.py``, which is used by the
    solver ``irnm_kl.py``.

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
        """Weight the call function."""
        
        return self.weight * self.op(x)

    def adjoint(self, x):
        """Weight the adjoint function."""
        
        return self.op.adjoint(self.weight*x)