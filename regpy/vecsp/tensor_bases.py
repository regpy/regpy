import numpy as np
from regpy.operators import Operator
from regpy.vecsp import VectorSpace,GridFcts,UniformGridFcts, Prod
from scipy.interpolate import BSpline

class TensorBasis(Operator):
    """
    Consider an evaluation domain given as eval_domain = Prod(D_1,...,D_n) with D_1,...,D_n being n VectorSpaces
    and a tensor in the coefficiants domain coef_domain = Prod(V_1,...,V_n) then we define an operator to map coeficiants
    by a given basis to a function f: eval_domain -> dtype:
        f(d_1,...,d_n) = \sum_{k_1=0}^{N_1-1} ... \sum_{k_n=0}^{N_n-1} c_{k_1,...k_n} b^1_{k_1}(x_1) .... b^n_{k_n}(x_n)
    So that the operator TensorBasis maps the coefficient tensor c = (c_{k_1,....k_n}) to the tensor of function values
    (f(x))_{x in eval_domain}
    eval_domain:    an instance of the class Prod in vector spaces of size where each D_i has size M_i
    coef_domain:    an instance of the class Prod in vector spaces of size where each V_i has size N_i
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
        assert isinstance(D_i,GridFcts)
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
        assert isinstance(D_i,GridFcts)
        x = D_i.axes[0]
        N_i=V_i.size
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.legendre.Legendre(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return TensorBasis(coef_domain,eval_domain,bases,dtype)

def BSplineBasis(k,t,dim=1,add_points=10):
    """ Implements a B-Spline basis in an arbirtary Dimension (given by dim)
    the splines are generated via BSpline from scipy.interpolate.
    In each dimension it uses the knots given in t to generate a B-Spline Basis.
    The evalutaion domain is a refined grid determined by the point added between points
    given by add_points:
        np.linspace(t[0],t[-1],t.size*add_points)
    Note, that to do that accuratly construct Splines, we use the key extrapolate=False and extend the
    orignal knot points given in t by additionally 2k points with equidistante distance to T.
    that is:
                t[0]    ...     t[-1=n+1]
    T[0]        T[k]    ...     T[-k]       T[n+2k+1]
    In the end the spline will be zero at the boundary by contruction.
    """
    assert t.ndim == 1 and isinstance(k,int) and isinstance(dim,int) and isinstance(add_points,int)
    assert t.size > k+1
    n = t.size -k-1
    coef_domain = Prod(*[UniformGridFcts(np.arange(n)) for i in range(dim)])
    eval_domain = Prod(*[UniformGridFcts(np.linspace(t[0],t[-1],t.size*add_points)) for i in range(dim)])
    basis = np.zeros((t.size*add_points,n))
    j=0
    axis = eval_domain[0].axes[0]
    # added points to to t since BSpline only gives back data in t[k] to t[n]=t[-k] and t of size n+k+1
    #assuming t to be equidistibuted points
    diff = t[1]-t[0]
    # T has t_size + 2*k points hence T[k] = t[0] and T[-k] = t[-1] hence full interval under consideration
    T = np.linspace(-k*diff+t[0],t[-1]+k*diff,t.size+2*k)
    c = np.zeros(t.size+k+1)
    for c_i in coef_domain.factors[0].iter_basis():
        c[k:k+n] = c_i
        spl_i = BSpline(T,c,k)
        basis[:,j] = spl_i(axis)
        j += 1
    return TensorBasis(coef_domain,eval_domain,[basis for i in range(dim)])
