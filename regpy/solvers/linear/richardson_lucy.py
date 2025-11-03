from ..general import Solver 

__all__ = ["RichardsonLucy"]

class RichardsonLucy(Solver):
    r"""The Richardson-Lucy Algorithm

    Minimizes :math:`-g\ln Tf, f\geq 0, Tf>0` 

    Parameters
    ----------
    op : regpy.operators.Operator
        The linear forward operator.
    data : array-like
        The non-negative data
    x_init : array_like, optional
        The initial guess "f". Must be in setting.op.domain. (Default: None)
    sigma : float , optional
        Non-negative parameter to shift the data and operator result to avoid division by zero or appearance of negative numbers. (Default: 0)
    """
    def __init__(self,op,data,x_init=None,sigma=0):
        super().__init__()
        self.op=op
        """The forward operator."""
        self.data=data
        """The measured data."""
        if(x_init==None):
            x_init=self.op.domain.ones()
        self.sigma = sigma
        """The shift."""
        self.x=x_init
        self.y=self.op(self.x)
        self.adj_ones=self.op.adjoint(self.op.codomain.ones())
        assert (self.adj_ones>0).all()


    def _next(self):
        multiplier=self.op.adjoint((self.data+self.sigma)/(self.y+self.sigma))
        self.x=multiplier*self.x/self.adj_ones
        self.y=self.op(self.x)



