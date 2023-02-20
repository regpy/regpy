import numpy as np
from operators import Operator
from discrs import Discretization
class TensorBasis(Operator):
    """
    We consider a rectangular grid grid = \{(x_{1,j_1},....x_{n_j_n}): j_1=0:M_1-1, ... j_n=0:M_n-1\} 
    and a tensor basis for grid function f:grid -> dtype:
        f(x_1,... x_n) = \sum_{k_1=0}^{N_1-1} ... \sum_{k_n=0}^{N_n-1} c_{k_1,...k_n} b^1_{k_1}(x_1) .... b^n_{k_n}(x_n)
    The operator TensorBasis maps the coefficient tensor c = (c_{k_1,....k_n}) to the tensor of function values 
    (f(x))_{x in grid}
    degrees:   the tuple (N_1,...N_n) of dimensions of the coefficient vector
    grid:      an instance of the class Grid in discretizations of size M_1 x .... x M_n
    bases:     a list of matrices [B_1,..., B_n] where the M_l x N_l matrix contains the function values 
               of the basis \{b^l_0, b^l_{M_l-1}} of the l-th coordinate: 
               B_l = (b^l_{k}(x_{l,j}))_{j=0:M_l-1, k=0:N_l-1}
    """
    def __init__(self,degrees,grid,bases,dtype=float): 
        domain = Discretization(degrees,dtype=dtype)
        super().__init__(domain,grid, linear=True)
        self.ndim = len(degrees)
        assert len(bases) == self.ndim
        assert len(grid.axes) == self.ndim
        assert grid.dtype == dtype
        assert np.all(bases[n].shape[1]== degrees.shape[n] for n in range(self.ndim)) 
        assert np.all(bases[n].shape[0]== len(grid.axes[n]) for n in range(self.ndim)) 
        self.dtype = dtype 
        self.degrees = degrees
        self.grid = grid
        self.bases = bases
        
    def _eval(self, Coeff):
        if self.ndim == 1:
            result = self.bases[0] @ Coeff
            # same as result = np.einsum('i,ai->a',Coeff,self.bases[0])
        elif self.ndim == 2:
            result = np.linalg.multi_dot([self.bases[0], Coeff, self.bases[1].T]) 
            # same as result = np.einsum('ij,ai,bj->ab',Coeff,self.bases[0],self.bases[1])   
            # or result = self.bases[0] @ Coeff @ self.bases[1].T
        elif self.ndim == 3:
            result = np.einsum('ijk,ai,bj,ck->abc',Coeff,self.bases[0],self.bases[1],self.bases[2])
        else:
            Raise(NotImplementedError)
        return result

    def _adjoint(self, G):
        if self.ndim == 1:
            result = self.bases[0].H @ G
            # same as result = np.einsum('a,ai->i',G,self.bases[0])
        elif self.ndim == 2:
            result = np.linalg.multi_dot([self.bases[0].conj().T, G, self.bases[1].conj()]) 
            # same as result = np.einsum('ab,ai,bj->ij',G,np.conj(self.bases[0]),np.conj(self.bases[1]))   
            # or result = self.bases[0].T @ G @ np.conj(self.bases[1])
        elif self.ndim == 3:
            result = np.einsum('abc,ai,bj,ck->ijk',G,self.bases[0].conj(), 
                self.bases[1].conj(),self.bases[2].conj())
        else:
            Raise(NotImplementedError)
        return result

def ChebyshevBasis(degrees,grid,dtype=float):
    """ Implements a tensor basis of Chebyshev polynomials
    """ 
    bases = []  
    for l in range(len(degrees)):
        x = grid.axes[l]
        intv = (grid.axes[l][0],grid.axes[l][-1])
        Nl = degrees[l]
        Bl = np.zeros((len(x),Nl))
        Id = np.eye(Nl)
        for k in range(Nl):
            pol = np.polynomial.chebyshev.Chebyshev(Id[k,:],domain = intv)
            Bl[:,k] = pol(x)
        bases.append(Bl) 
    return TensorBasis(degrees,grid,bases,dtype)
 