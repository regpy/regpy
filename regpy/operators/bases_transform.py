import numpy as np
from scipy.interpolate import BSpline

from regpy.vecsps import NumPyVectorSpace,GridFcts,UniformGridFcts, Prod
from regpy.util import Errors

from .base import Operator

__all__ = ["BasisTransform", "chebyshev_basis", "legendre_basis", "bspline_basis"]

class BasisTransform(Operator):
    r"""
    Consider an evaluation domain given as Tensor product :math:`D_1\otimes \dots\otimes D_n` with :math:`D_1,\dots,D_n` being :math:`n` 
    `regpy.vecsps.VectorSpaceBase`'s and a tensor in the coefficients domain :math:`V_1\otimes \dots\otimes V_m` then we define an 
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
    bases : list(list)
        a list of matrices :math:`[B_1,..., B_n]` where the matrix :math:`B_l` of size :math:`M_l \times N_l` and contains the function values
        of the basis :math:`\{b^l_0, b^l_{M_l-1}\}` of the l-th coordinate:

        .. math::
            B_l = (b^l_{k}(x_{l,j}))_{j=0:M_l-1, k=0:N_l-1}

    """
    def __init__(self,coef_domain,eval_domain,bases,dtype=float):
        if not isinstance(coef_domain,Prod):
            raise TypeError(Errors.not_instance(coef_domain,Prod))
        if not isinstance(eval_domain,Prod):
            raise TypeError(Errors.not_instance(eval_domain,Prod))
        if len(bases) != eval_domain.ndim or coef_domain.ndim != eval_domain.ndim:
            raise ValueError(Errors.value_error(f"The number of bases vectors need to match the domain vector dimension and codomain vector dimension. "+"\n\t" +f"len(bases) = { len(bases)}"+"\n\t"+f"eval_domain.ndim = {eval_domain.ndim} "+"\n\t"+f"coef_domain.ndim = {coef_domain.ndim}."))
        if len(bases) > 26:
            raise ValueError(Errors.value_error("The number of vector dimension cannot be large then 26!"))
        if not np.issubdtype(dtype,coef_domain.dtype) or not np.issubdtype(dtype,eval_domain.dtype):
            raise ValueError(Errors.value_error(f"The dtypes of the different domains have to be identical with the given dtype. coef.dtype = {coef_domain.dtype}, eval.dtype = {eval_domain.dtype}, and dtype = {dtype}"))
        if any(basis.shape[0] != int(eval.size) for (basis,eval) in zip(bases,eval_domain)):
            raise ValueError(Errors.value_error("The vectors in the bases need to have as shape 0 the size of the eval domain!"+"\n\t"+f"bases shapes = {tuple(basis.shape[0] for basis in bases)}"+"\n\t"+f"eval_domain sizes = {tuple(int(eval.size) for eval in eval_domain)}"))
        if any(basis.shape[1] != coef.size for (basis,coef) in zip(bases,coef_domain)):
            raise ValueError(Errors.value_error("The vectors in the bases need to have as shape 1 the size of the coef domain!"+"\n\t"+f"bases shapes = {tuple(basis.shape[1] for basis in bases)}"+"\n\t"+f"coef_domain sizes = {tuple(int(coef.size) for coef in coef_domain)}"))
        super().__init__(coef_domain,eval_domain, linear=True)
        self.dtype = dtype
        r""" `dtype` of the vector spaces."""
        self.ndim = coef_domain.ndim
        r""" dimension of the `coef_domain`. """
        self.bases = tuple(np.asarray(basis) for basis in bases)
        r"""All one-dimensional basis transforms as immutable tuple of arrays."""
        self._conjugate_bases = tuple(
            np.conjugate(basis) if np.iscomplexobj(basis) else basis
            for basis in self.bases
        )

        coef_indices = "".join(chr(k) for k in range(65, 65 + self.ndim))
        eval_indices = "".join(chr(k) for k in range(97, 97 + self.ndim))
        basis_indices = ",".join(
            chr(97 + axis) + chr(65 + axis)
            for axis in range(self.ndim)
        )
        self._eval_sumrule = f"{coef_indices},{basis_indices}->{eval_indices}"
        self._adjoint_sumrule = f"{eval_indices},{basis_indices}->{coef_indices}"
        self._eval_einsum_path = None
        self._adjoint_einsum_path = None

        # Basis arrays are read-only operator data and should not be duplicated when
        # derivatives/adjoints of this linear operator are copied.
        self._consts.update({'bases', '_conjugate_bases'})

    @staticmethod
    def _copy_if_aliased(x, out):
        """Return an independent input only when ``out`` overlaps it."""
        return x.copy() if np.shares_memory(x, out) else x

    def _get_einsum_path(self, operand, *, adjoint):
        """Compute each contraction path only once and reuse it afterwards."""
        if adjoint:
            if self._adjoint_einsum_path is None:
                self._adjoint_einsum_path = np.einsum_path(
                    self._adjoint_sumrule,
                    operand,
                    *self._conjugate_bases,
                    optimize='optimal'
                )[0]
            return self._adjoint_einsum_path

        if self._eval_einsum_path is None:
            self._eval_einsum_path = np.einsum_path(
                self._eval_sumrule,
                operand,
                *self.bases,
                optimize='optimal'
            )[0]
        return self._eval_einsum_path

    def _ieval(self, coef, out):
        """Evaluate the tensor-product basis transform directly into ``out``."""
        coef = self._copy_if_aliased(coef, out)

        # Keep the BLAS-backed special cases used by the allocating implementation.
        if self.ndim == 1 and self.domain[0].size * self.codomain[0].size <= 50000000:
            np.matmul(self.bases[0], coef, out=out)
            return out

        if (self.ndim == 2 and
                (self.domain[0].size + self.domain[1].size) *
                (self.codomain[0].size + self.codomain[1].size) <= 4000000):
            basis_left, basis_right = self.bases
            n_left, n_right = coef.shape
            m_left, m_right = out.shape

            left_first_cost = (
                m_left * n_left * n_right + m_left * n_right * m_right
            )
            right_first_cost = (
                n_left * n_right * m_right + m_left * n_left * m_right
            )
            if left_first_cost <= right_first_cost:
                tmp = basis_left @ coef
                np.matmul(tmp, basis_right.T, out=out)
            else:
                tmp = coef @ basis_right.T
                np.matmul(basis_left, tmp, out=out)
            return out

        return np.einsum(
            self._eval_sumrule,
            coef,
            *self.bases,
            out=out,
            optimize=self._get_einsum_path(coef, adjoint=False)
        )

    def _iadjoint(self, values, out):
        """Apply the real-Hilbert-space adjoint directly into ``out``."""
        values = self._copy_if_aliased(values, out)

        if self.ndim == 1 and self.domain[0].size * self.codomain[0].size <= 50000000:
            np.matmul(self._conjugate_bases[0].T, values, out=out)
            return out

        if (self.ndim == 2 and
                (self.domain[0].size + self.domain[1].size) *
                (self.codomain[0].size + self.codomain[1].size) <= 4000000):
            basis_left_h = self._conjugate_bases[0].T
            basis_right_conj = self._conjugate_bases[1]
            m_left, m_right = values.shape
            n_left, n_right = out.shape

            left_first_cost = (
                n_left * m_left * m_right + n_left * m_right * n_right
            )
            right_first_cost = (
                m_left * m_right * n_right + n_left * m_left * n_right
            )
            if left_first_cost <= right_first_cost:
                tmp = basis_left_h @ values
                np.matmul(tmp, basis_right_conj, out=out)
            else:
                tmp = values @ basis_right_conj
                np.matmul(basis_left_h, tmp, out=out)
            return out

        return np.einsum(
            self._adjoint_sumrule,
            values,
            *self._conjugate_bases,
            out=out,
            optimize=self._get_einsum_path(values, adjoint=True)
        )


def chebyshev_basis(coef_nr,eval_domain) -> BasisTransform:
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
    if not isinstance(eval_domain,Prod):
        raise TypeError(Errors.not_instance(eval_domain,Prod))
    if isinstance(coef_nr, tuple) and len(coef_nr) != eval_domain.ndim: 
        raise ValueError(Errors.value_error("When given coef_nr as a tuple it has to have the same number of entries as the domain has dimensions."))
    elif isinstance(coef_nr,int):
        coef_nr = (coef_nr,)*eval_domain.ndim
    coef_domain = Prod(*[NumPyVectorSpace(nr) for nr in coef_nr])
    bases = []
    for D_i, N_i in zip(eval_domain,coef_nr):
        if not isinstance(D_i,GridFcts):
            raise TypeError(Errors.not_instance(D_i,GridFcts,"All domains in the product for a chebyshev basis need to be GridFcts"))
        x = D_i.axes[0]
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.chebyshev.Chebyshev(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return BasisTransform(coef_domain,eval_domain,bases)

def legendre_basis(coef_nr,eval_domain) -> BasisTransform:
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

    if not isinstance(eval_domain,Prod):
        raise TypeError(Errors.not_instance(eval_domain,Prod))
    if isinstance(coef_nr, tuple) and len(coef_nr) != eval_domain.ndim: 
        raise ValueError(Errors.value_error("When given coef_nr as a tuple it has to have the same number of entries as the domain has dimensions."))
    elif isinstance(coef_nr,int):
        coef_nr = (coef_nr,)*eval_domain.ndim
    coef_domain = Prod(*[NumPyVectorSpace(nr) for nr in coef_nr])
    bases = []
    for D_i, N_i in zip(eval_domain,coef_nr):
        if not isinstance(D_i,GridFcts):
            raise TypeError(Errors.not_instance(D_i,GridFcts,"All domains in the product for a legendre basis need to be GridFcts"))
        x = D_i.axes[0]
        B_i = np.zeros((len(x),N_i))
        Id = np.eye(N_i)
        for k in range(N_i):
            pol = np.polynomial.legendre.Legendre(Id[k,:],domain = (D_i.axes[0][0],D_i.axes[0][-1]))
            B_i[:,k] = pol(x)
        bases.append(B_i)
    return BasisTransform(coef_domain,eval_domain,bases)

def bspline_basis(k,t,dim=1,add_points=10) -> BasisTransform:
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
    if not isinstance(t, np.ndarray):
        raise TypeError(Errors.type_error("t for a B-Spline basis need to be an ndarray."))
    if t.ndim != 1:
        raise ValueError(Errors.value_error("t for a B-Spline basis needs to be one dimensional"))
    if not isinstance(k,int) or not isinstance(dim,int) or not isinstance(add_points,int):
        raise TypeError(Errors.type_error("k, dim, and add_points need to be integers!"))
    if t.size < k+1:
        raise ValueError(Errors.value_error("For a B-SPline basis we need t.size >= k+1."))
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
