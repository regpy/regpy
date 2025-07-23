import numpy as np
from regpy.operators import Operator
from regpy.vecsps import VectorSpace,GridFcts,UniformGridFcts, Prod
from scipy.interpolate import BSpline

class BasisTransform(Operator):
    r"""
    Consider an evaluation domain given as Tensor product :math:`D_1\otimes \dots\otimes D_n` with :math:`D_1,\dots,D_n` being :math:`n` 
    `regpy.vecsps.VectorSpace`'s and a tensor in the coefficients domain :math:`V_1\otimes \dots\otimes V_m` then we define an 
    operator mapping coefficients to some function `f: eval_domain -> dtype`:

    .. math::
        f(d_1,...,d_n) = \sum_{k_1=0}^{N_1-1} ... \sum_{k_n=0}^{N_n-1} c_{k_1,...k_n} b^1_{k_1}(x_1) .... b^n_{k_n}(x_n).

    So that the operator BasisTransform maps the coefficient tensor :math:`c = (c_\{k_1,....k_n\})` to the tensor of function values
    :math:`(f(x))_{x in eval_domain}`

    Parameters
    ----------
    eval_domain : regpy.vecsps.Prod   
        an instance of the class `regpy.vecsps.Prod` in vector spaces of size where each D_i has size M_i
    coef_domain : regpy.vecsps.Prod   
        an instance of the class `regpy.vecsps.Prod` in vector spaces of size where each V_i has size N_i
    bases : [ np.ndarray, ... ]
        a list of matrices :math:`[B_1,..., B_n]` where the matrix :math:`B_l` of size :math:`M_l \times N_l` and contains the function values
        of the basis :math:`\{b^l_0, b^l_{M_l-1}\}` of the l-th coordinate:

        .. math::
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
        r""" `dtype` of the vector spaces."""
        self.ndim = coef_domain.ndim
        r""" dimension of the `coef_domain`. """
        self.bases = bases
        r"""List of all the bases transforms as a list of `np.ndarray`\s
        """

    def _eval(self, coef):
        ## separate 1-D and 2-D because of performance
        if self.ndim == 1 and self.domain[0].size*self.codomain[0].size <= 50000000:
            return self.bases[0] @ coef
        elif self.ndim == 2 and (self.domain[0].size+self.domain[1].size)*(self.codomain[0].size+self.codomain[1].size) <= 4000000:
            return np.linalg.multi_dot([self.bases[0], coef, self.bases[1].T])
        else:
            self.sumrule = "".join(chr(k) for k in range(65,65+self.ndim))+","+",".join(["".join(chr(k) for k in [97+l,65+l]) for l in range(self.ndim)])+"->"+"".join(chr(k) for k in range(97,97+self.ndim))
            self.einsum_path = np.einsum_path(self.sumrule,coef,*self.bases, optimize='optimal')[0]
            return np.einsum(self.sumrule,coef,*self.bases,optimize=self.einsum_path)

    def _adjoint(self, G):
        ## separate 1-D and 2-D because of performance
        if self.ndim == 1 and self.domain[0].size*self.codomain[0].size <= 50000000:
            return self.bases[0].H @ G
        elif self.ndim == 2 and (self.domain[0].size+self.domain[1].size)*(self.codomain[0].size+self.codomain[1].size) <= 4000000:
            return np.linalg.multi_dot([self.bases[0].conj().T, G, self.bases[1].conj()])
        else:
            self.sumrule = "".join(chr(k) for k in range(97,97+self.ndim))+","+",".join(["".join(chr(k) for k in [97+l,65+l]) for l in range(self.ndim)])+"->"+"".join(chr(k) for k in range(65,65+self.ndim))
            self.einsum_path = np.einsum_path(self.sumrule,G,*self.bases, optimize='optimal')[0]
            return np.einsum(self.sumrule,G,*self.bases,optimize=self.einsum_path)


def chebyshev_basis(coef_nr,eval_domain,dtype=float):
    r"""Implements a tensor basis of Chebyshev polynomials for product spaces. It requires that 
    both coef_domain and eval_domain has the same dimension.

    Parameters
    ----------
    coef_nr : scalar or tuple
        Number of Coefficients of Chebychev polynomials in the coefficients basis.  
    eval_domain : regpy.vecsps.Prod
        Tensor product of `GridFcts` instances on which the Chebyshev polynomial are evaluated. 
    dtype : np.dtype, optional
       type of the underlying spaces, by default float

    Returns
    -------
    BasisTransform 
        A bases transform from coefficients of Chebychev polynomial to their evaluation.
    """
    assert isinstance(eval_domain,Prod)
    if isinstance(coef_nr, tuple):
        assert len(coef_nr) == eval_domain.ndim 
    elif isinstance(coef_nr,int):
        coef_nr = (coef_nr,)*eval_domain.ndim
    coef_domain = Prod(*[VectorSpace(nr) for nr in coef_nr])
    bases = []
    for D_i, N_i in zip(eval_domain,coef_nr):
        assert isinstance(D_i,GridFcts)
        x = D_i.axes[0]
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.chebyshev.Chebyshev(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return BasisTransform(coef_domain,eval_domain,bases,dtype)

def legendre_basis(coef_nr,eval_domain,dtype=float):
    r"""Implements a tensor basis of Legendre polynomials for product spaces. It requires that 
    both coef_domain and eval_domain has the same dimension.

    Parameters
    ----------
    coef_domain : regpy.vecsps.Prod
        Coefficients of tensor products of Legendre polynomials.  
    eval_domain : regpy.vecsps.Prod
        Tensor product of `GridFcts` instances on which the Legendre polynomial are evaluated. 
    dtype : np.dtype, optional
       type of the underlying spaces, by default float

    Returns
    -------
    BasisTransform 
        A bases transform from coefficients of Legendre polynomial to their evaluation.
    """

    assert isinstance(eval_domain,Prod)
    if isinstance(coef_nr, tuple):
        assert len(coef_nr) == eval_domain.ndim 
    elif isinstance(coef_nr,int):
        coef_nr = (coef_nr,)*eval_domain.ndim
    coef_domain = Prod(*[VectorSpace(nr) for nr in coef_nr])
    bases = []
    for D_i, N_i in zip(eval_domain,coef_nr):
        assert isinstance(D_i,GridFcts)
        x = D_i.axes[0]
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.legendre.Legendre(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return BasisTransform(coef_domain,eval_domain,bases,dtype)

def bspline_basis(k,t,dim=1,add_points=10):
    r"""Implements a B-Spline basis in an arbitrary Dimension (given by dim)
    the splines are generated via BSpline from scipy.interpolate.
    In each dimension it uses the knots given in t to generate a B-Spline Basis.
    The evaluation domain is a refined grid determined by the point added between points
    given by add_points:

        np.linspace(t[0],t[-1],t.size*add_points)
    
    Note, that to do that accurately construct Splines, we use the key extrapolate=False and extend the
    original knot points given in t by additionally 2k points with equidistant distance to T.
    
    By this construction the splines will be zero at the boundary.

    Parameters
    ----------
    k : integer
        Smoothness degree of the used Splines.
    t : np.ndarray
        One dimensional array of knot points for evaluating the Splines.
    dim : int, optional
        Dimension of the resulting spaces, by default 1
    add_points : int, optional
        Number of points to be added between each Knot for evaluation, by default 10

    Returns
    -------
    BasisTransform
        A base transform from coefficients of Splines to evaluation on a grid constructed from a refined
        grid of the given evaluation knots. 
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
    #assuming t to be equidistant points
    diff = t[1]-t[0]
    # T has t_size + 2*k points hence T[k] = t[0] and T[-k] = t[-1] hence full interval under consideration
    T = np.linspace(-k*diff+t[0],t[-1]+k*diff,t.size+2*k)
    c = np.zeros(t.size+k+1)
    for c_i in coef_domain.factors[0].iter_basis():
        c[k:k+n] = c_i
        spl_i = BSpline(T,c,k)
        basis[:,j] = spl_i(axis)
        j += 1
    return BasisTransform(coef_domain,eval_domain,[basis for i in range(dim)])
