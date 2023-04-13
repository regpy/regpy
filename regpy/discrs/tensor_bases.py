import numpy as np
from regpy.operators import Operator
from regpy.discrs import Discretization,Grid,UniformGrid, Prod

class TensorBasis(Operator):
    """
    Consider an evaluation domain given as eval_domain = Prod(D_1,...,D_n) with D_1,...,D_n being n Discretizations
    and a tensor in the coefficiants domain coef_domain = Prod(V_1,...,V_n) then we define an operator to map coeficiants
    by a given basis to a function f: eval_domain -> dtype:
        f(d_1,...,d_n) = \sum_{k_1=0}^{N_1-1} ... \sum_{k_n=0}^{N_n-1} c_{k_1,...k_n} b^1_{k_1}(x_1) .... b^n_{k_n}(x_n)
    So that the operator TensorBasis maps the coefficient tensor c = (c_{k_1,....k_n}) to the tensor of function values
    (f(x))_{x in eval_domain}
    eval_domain:    an instance of the class Prod in discretizations of size where each D_i has size M_i
    coef_domain:    an instance of the class Prod in discretizations of size where each V_i has size N_i
    bases:          a list of matrices [B_1,..., B_n] where thematrix B_lis of size M_l x N_l and contains the function values
                    of the basis \{b^l_0, b^l_{M_l-1}} of the l-th coordinate:
                        B_l = (b^l_{k}(x_{l,j}))_{j=0:M_l-1, k=0:N_l-1}
    """
    def __init__(self,coef_domain,eval_domain,bases,dtype=float):
        assert isinstance(coef_domain,Prod)
        assert isinstance(eval_domain,Prod)
        assert len(bases) == eval_domain.ndim
        assert coef_domain.ndim == eval_domain.ndim
        assert len(bases) <= 26
        assert coef_domain.dtype == dtype and eval_domain.dtype == dtype
        assert np.all(basis.shape[1]== eval.size for (basis,eval) in zip(bases,eval_domain))
        assert np.all(basis.shape[1]== coef.size for (basis,coef) in zip(bases,coef_domain))
        super().__init__(coef_domain,eval_domain, linear=True)
        self.dtype = dtype
        self.ndim = coef_domain.ndim
        self.bases = bases

    def _eval(self, Coef):
        ## separate 1-D and 2-D because of performance
        if self.ndim == 1 and self.domain[0].size*self.codomain[0].size <= 50000000:
            return self.bases[0] @ Coef
        elif self.ndim == 1 and (self.domain[0].size+self.domain[1].size)*(self.codomain[0].size+self.codomain[1].size) <= 4000000:
            return np.linalg.multi_dot([self.bases[0], Coef, self.bases[1].T])
        else:
            self.sumrule = "".join(chr(k) for k in range(65,65+self.ndim))+","+",".join(["".join(chr(k) for k in [97+l,65+l]) for l in range(self.ndim)])+"->"+"".join(chr(k) for k in range(97,97+self.ndim))
            self.einsum_path = np.einsum_path(self.sumrule,Coef,*self.bases, optimize='optimal')[0]
            return np.einsum(self.sumrule,Coef,*self.bases,optimize=self.einsum_path)

    def _adjoint(self, G):
        ## separate 1-D and 2-D because of performance
        if self.ndim == 1 and self.domain[0].size*self.codomain[0].size <= 50000000:
            return self.bases[0].H @ G
        elif self.ndim == 2 and (self.domain[0].size+self.domain[1].size)*(self.codomain[0].size+self.codomain[1].size) <= 4000000:
            return np.linalg.multi_dot([self.bases[0].conj().T, G, self.bases[1].conj()])
        else:
            self.sumrule = "".join(chr(k) for k in range(97,97+self.ndim))+","+",".join(["".join(chr(k) for k in [97+l,65+l]) for l in range(self.ndim)])+"->"+"".join(chr(k) for k in range(65,65+self.ndim))
            self.einsum_path = np.einsum_path(self.sumrule,G,*self.bases, optimize='optimal')[0]
            return np.einsum(self.sumrule,G,*self.bases,optimize=self.path)


def ChebyshevBasis(coef_domain,eval_domain,dtype=float):
    """ Implements a tensor basis of Chebyshev polynomials
    """
    assert isinstance(coef_domain,Prod)
    assert isinstance(eval_domain,Prod)
    assert coef_domain.ndim == eval_domain.ndim
    bases = []
    for D_i, V_i in zip(eval_domain,coef_domain):
        assert isinstance(D_i,Grid)
        x = D_i.axes[0]
        N_i=V_i.size
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.chebyshev.Chebyshev(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return TensorBasis(coef_domain,eval_domain,bases,dtype)

def LegendreBasis(coef_domain,eval_domain,dtype=float):
    """ Implements a tensor basis of Legendre polynomials
    """
    assert isinstance(coef_domain,Prod)
    assert isinstance(eval_domain,Prod)
    assert coef_domain.ndim == eval_domain.ndim
    bases = []
    for D_i, V_i in zip(eval_domain,coef_domain):
        assert isinstance(D_i,Grid)
        x = D_i.axes[0]
        N_i=V_i.size
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.legendre.Legendre(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return TensorBasis(coef_domain,eval_domain,bases,dtype)
