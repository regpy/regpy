from regpy.util import Errors

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
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="RichardsonLucy requires the operator to be linear!"))
        """The forward operator."""
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        self.data=data
        """The measured data."""
        if x_init is not None and x_init not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(x_init,self.op.domain,vec_name="initial guess",space_name="domain"))
        self.sigma = sigma
        """The shift."""
        self.x=self.op.domain.ones() if x_init is None else x_init
        self.y=self.op(self.x)
        self.adj_ones=self.op.adjoint(self.op.codomain.ones())
        if (self.adj_ones<=0).any():
            raise RuntimeError(Errors.runtime_error("The adjoint of constant one returns some negative values! Not possible to use RichardsonLucy!"))


    def _next(self):
        multiplier=self.op.adjoint((self.data+self.sigma)/(self.y+self.sigma))
        self.x=multiplier*self.x/self.adj_ones
        self.y=self.op(self.x)



