from typing import Callable

from matplotlib.pyplot import grid
from scipy.sparse import diags, csc_matrix 
import scipy.sparse.linalg as sla
import numpy as np
from regpy.operators.numpy import ForwardFDGradient
from regpy.vecsps import UniformGridFcts
from regpy.operators import Operator

__all__ = ['PeriodicCProblem']

class PeriodicCProblem(Operator):
    """A simple forward operator for testing purposes.
    The operator maps a periodic function c to the solution of the periodic differential equation
    .. math::
        -\Delta u - k^2(1+c) u = f        
    
    Parameters
    n: int
        The number of grid points
    k:float
        wave number
    f: Callable | np.ndarray | None
        The right-hand side function. If None, the right-hand side is set to 1.
    """ 

    def __init__(self, 
                 n: int,
                 k: float,
                 f: Callable | np.ndarray | None = None):
        self.grid = UniformGridFcts((0,1,n), periodic=True)
        super().__init__(self.grid, self.grid)
        grad = ForwardFDGradient(self.grid, boundary_condition='per')
        [Dx] = grad.matrices()
        self.neg_laplacian = Dx.T @ Dx   
        self.k = k
        if f is None:
            self.f = self.domain.ones(read_only = True)
        elif f in self.domain:
            self.f = f
        else:
            self.f = f(self.grid)

    def _eval(self, c, differentiate=False):
        """Evaluate the operator at a given function c."""
        self.Lc = (self.neg_laplacian - self.k**2*diags(1.+c)).tocsc()
        """Finite difference discretization of the differential operator."""
        if differentiate:
            self.lu = sla.splu(self.Lc) 
            self.u = self.lu.solve(self.f)
        else:        
            self.u = sla.spsolve(self.Lc,self.f)
        return self.u.copy()

    def _derivative(self, h):
        return self.k**2 * self.lu.solve(self.u*h)

    def _adjoint(self, v):
        trans_type = 'T' if np.issubdtype(self.grid.dtype, np.complexfloating) else 'T'
        out = self.lu.solve(v,trans = trans_type)
        out *= (self.k)**2 * self.u
        return out
