import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla

from regpy import util
from regpy.vecsps import NumPyVectorSpace,UniformGridFcts,GridFcts

from .base import Operator

__all__ = ["MatrixMultiplication","CholeskyInverse","SuperLUInverse","Power","Exponential","FourierTransform"]

class MatrixMultiplication(Operator):
    r"""Implements an operator that does matrix-vector multiplication with a given matrix. Domain and codomain 
    are plain one dimensional `regpy.vecsps.NumPyVectorSpace` instances by default.

    Parameters
    ----------
    matrix : array-like
        The matrix.
    inverse : Operator, array-like, 'inv', 'cholesky' or None, optional
        How to implement the inverse operator. If available, this should be given as `Operator`
        or array. If `inv`, `numpy.linalg.inv` will be used. If `cholesky` or `superLU`, a
        `CholeskyInverse` or `SuperLU` instance will be returned.
    domain : regpy.vecsps.NumPyVectorSpace, optional
        The underlying vector space. If not given a `regpy.vecsps.VectorSpaceBase` with same number of elements as
        matrix columns is used. Defaults to None.
    codomain : regpy.vecsps.NumPyVectorSpace, optional
        The underlying vector space. If not given a `regpy.vecsps.VectorSpaceBase` with same number of elements as
        matrix rows is used. Defaults to None.

    Notes
    -----
    The matrix multiplication is done by applying numpy.dot to the matrix and an element of the domain. 
    The adjoint is implemented in the same way by multiplying with the adjoint matrix.
    As long as this dot product is possible and the matrix is two-dimensional, multidimensional domains and
    codomains may also be used.
    """

    def __init__(self, matrix, inverse=None, domain=None, codomain=None,dtype=None):
        assert len(matrix.shape) == 2
        assert domain is None or isinstance(domain,NumPyVectorSpace), "Domain either none or NumPyVectorSpace given was {}".format(type(domain))
        assert codomain is None or isinstance(codomain,NumPyVectorSpace), "Codomain either none or NumPyVectorSpace given was {}".format(type(codomain))
        self.matrix = matrix
        if dtype == None:
            dtype = matrix.dtype
        super().__init__(
            domain=domain or NumPyVectorSpace(matrix.shape[1],dtype = dtype),
            codomain=codomain or NumPyVectorSpace(matrix.shape[0],dtype = dtype),
            linear=True
        )
        self._inverse = inverse

    def _eval(self, x):
        return self.matrix @ x

    def _adjoint(self, y):
        if self.codomain.is_complex:
            return np.conjugate(np.conjugate(y) @ self.matrix) 
        else:
            return y @ self.matrix

    @util.memoized_property
    def inverse(self):
        if isinstance(self._inverse, Operator):
            return self._inverse
        elif isinstance(self._inverse, np.ndarray):
            return MatrixMultiplication(self._inverse, inverse=self)
        elif isinstance(self._inverse, str):
            if self._inverse == 'inv':
                return MatrixMultiplication(np.linalg.inv(self.matrix), inverse=self)
            if self._inverse == 'cholesky':
                return CholeskyInverse(self, matrix=self.matrix)
            if self._inverse == 'superLU':
                return SuperLUInverse(self)
        raise NotImplementedError

    def __repr__(self):
        return util.make_repr(self, self.matrix)


class CholeskyInverse(Operator):
    """Implements the inverse of a linear, self-adjoint operator via Cholesky decomposition. Since
    it needs to assemble a full matrix, this should not be used for high-dimensional operators.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator to be inverted.
    matrix : array-like, optional
        If a matrix of `op` is already available, it can be passed in to avoid recomputation.
    """
    def __init__(self, op, matrix=None):
        assert op.linear, "Operator is not linear."
        assert op.domain and op.domain == op.codomain, "Domain cannot be None and has to match codomain."
        assert isinstance(op.domain,NumPyVectorSpace), "Domain has to be a NumPyVectorSpace"
        domain = op.domain
        if matrix is None:
            matrix = np.empty((domain.realsize,) * 2, dtype=float)
            for j, elm in enumerate(domain.iter_basis()):
                matrix[j, :] = domain.flatten(op(elm))
        self.factorization = cho_factor(matrix)
        """The Cholesky factorization for use with `scipy.linalg.cho_solve`"""
        super().__init__(
            domain=domain,
            codomain=domain,
            linear=True
        )
        self.op = op

    def _eval(self, x):
        return self.domain.fromflat(
            cho_solve(self.factorization, self.domain.flatten(x)))

    def _adjoint(self, x):
        return self._eval(x)

    @property
    def inverse(self):
        """Returns the original operator."""
        return self.op

    def __repr__(self):
        return util.make_repr(self, self.op)


class SuperLUInverse(Operator):
    """Implements the inverse of a MatrixMultiplication Operator given by a csc_matrix using SuperLU.

    Parameters
    ----------
        op : MatrixMultiplication
            The operator to be inverted.   
    """
    def __init__(self,op):
        assert isinstance(op,MatrixMultiplication)
        assert isinstance(op.matrix, csc_matrix)
        super().__init__(
            domain=op.codomain, 
            codomain = op.domain,
            linear=True)
        self.lu = sla.splu(op.matrix)

    def _eval(self,x):
        if np.issubdtype(self.lu.U.dtype,np.complexfloating):
            return self.lu.solve(x)
        else: 
            if np.isrealobj(x):
                return self.lu.solve(x)
            else:
                return self.lu.solve(x.real) + 1j*self.lu.solve(x.imag) 

    def _adjoint(self,x):
        return self.lu.solve(x,trans='H')

    @property
    def inverse(self):
        """Returns the original operator."""
        return self.op

    def __repr__(self):
        return util.make_repr(self, self.op)


class Power(Operator):
    r"""The operator \(x \mapsto x^n\).

    Parameters
    ----------
    power : float
        The exponent.
    domain : regpy.vecsps.NumPyVectorSpace
        The underlying vector space
    """

    def __init__(self, power, domain, integer = False):
        assert isinstance(domain,NumPyVectorSpace)
        self.integer = integer
        if integer:
            assert power>=0 and int(power)==power
            power=int(power)
            self._power_bin = "{0:b}".format(power)
        self.power = power
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if self.integer:
            res = np.ones_like(x)
            if differentiate:
                self._factor = self.power*np.ones_like(x)
                if self.power>0:
                    self._dpow_bin = "{0:b}".format(self.power-1)
                    if len(self._dpow_bin)< len(self._power_bin):
                        self._dpow_bin = '0'+self._dpow_bin
                else:
                    self._dpow_bin = "{0:b}".format(0)
            powx = x.copy()
            for k in reversed(range(len(self._power_bin))):
                if self._power_bin[k] == '1':
                    res *= powx
                if differentiate:
                    if self._dpow_bin[k] == '1':
                        self._factor *= powx
                if k>0:
                    powx *= powx
        else:
            if differentiate:
                self._factor = self.power * x**(self.power - 1)
            res = x**self.power
        return res

    def _derivative(self, x):
        return self._factor * x

    def _adjoint(self, y):
        return np.conjugate(self._factor) * y


class Exponential(Operator):
    r"""The pointwise exponential operator.

    Parameters
    ----------
    domain : regpy.vecsps.NumPyVectorSpaceBase
        The underlying vector space.
    """

    def __init__(self, domain):
        assert isinstance(domain,NumPyVectorSpace)
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._exponential_factor = np.exp(x)
            return self._exponential_factor
        return np.exp(x)

    def _derivative(self, x):
        return self._exponential_factor * x

    def _adjoint(self, y):
        return self._exponential_factor.conj() * y

###################### General Operators that require UniformGirdFcts or GridFcts ######################

class FourierTransform(Operator):
    """Fourier transform operator on UniformGridFcts implemented via numpy.fft.fftn.

    Parameters
    ----------
    domain : regpy.vecsps.UniformGridFcts
        The underlying vector space
    centered : bool, optional
            Whether the resulting grid will have its zero frequency in the center or not. The
            advantage is that the resulting grid will have strictly increasing axes, making it
            possible to define a `UniformGridFcts` instance in frequency space. The disadvantage is
            that `numpy.fft.fftshift` has to be used, which should generally be avoided for
            performance reasons. Defaults to `False`.
    axes : sequence of ints, optional
        Axes over which to compute the Fourier transform. If not given all axes are used.
        Defaults to None.
    """
    def __init__(self, domain, centered=False, axes=None):
        assert isinstance(domain, UniformGridFcts)
        self.is_complex = domain.is_complex
        frqs = self.frequencies(domain,centered=centered, axes=axes, rfft= not domain.is_complex)
        shape = domain.shape
        s = shape[-1]
        if centered or (not domain.is_complex and domain.ndim==1):
            codomain = UniformGridFcts(*frqs, dtype=complex)
        else:
            # In non-centered case, the frequencies are not ascencing, so even using GridFcts here is slighty questionable.
            codomain = GridFcts(*frqs, dtype=complex,use_cell_measure=False)
        super().__init__(domain, codomain, linear=True)
        self.centered = centered
        self.axes = axes
  
    def _eval(self, x):
        if self.centered:
            x = np.fft.ifftshift(x, axes=self.axes)
        if self.is_complex:
            y = np.fft.fftn(x, axes=self.axes, norm='ortho')
        else:
            y = np.fft.rfftn(x, axes=self.axes, norm='ortho')
        if self.centered:
            return np.fft.fftshift(y, axes=self.axes)
        else:
            return y

    def _adjoint(self, y):
        if self.centered:
            y = np.fft.ifftshift(y, axes=self.axes)
        if self.is_complex:
            x = np.fft.ifftn(y, axes=self.axes, norm='ortho')
        else:
            x = np.fft.irfftn(y, tuple(self.domain.shape[i] for i in self.axes),axes=self.axes, norm='ortho')
        if self.centered:
            x = np.fft.fftshift(x, axes=self.axes)
        if self.domain.is_complex:
            return x
        else:
            return np.real(x)
        
    def frequencies(self,domain,centered=False, axes=None, rfft=False):
        """Compute the grid of frequencies for an FFT on this grid instance.

        Parameters
        ----------
        centered : bool, optional
            Whether the resulting grid will have its zero frequency in the center or not. The
            advantage is that the resulting grid will have strictly increasing axes, making it
            possible to define a `UniformGridFcts` instance in frequency space. The disadvantage is
            that `numpy.fft.fftshift` has to be used, which should generally be avoided for
            performance reasons. Default: `False`.
        axes : tuple of ints, optional
            Axes for which to compute the frequencies. All other axes will be returned as-is.
            Intended to be used with the corresponding argument to `numpy.fft.fffn`. If `None`, all
            axes will be computed. Default: `None`.
        Returns
        -------
        array
        """
        if axes is None:
            axes = range(domain.ndim)
        axes = set(axes)
        frqs = []
        for i, (s, l) in enumerate(zip(domain.shape, domain.spacing)):
            if i in axes:
                # Use (spacing * shape) in denominator instead of extents, since the grid is assumed
                # to be periodic.
                shalf = s/2+1 if (s//2)*2==s else (s+1)/2
                if i==domain.ndim-1 and rfft==True:
                    frqs.append(np.arange(0,shalf) / (s*l))
                else:
                    if centered:
                        frqs.append(np.arange(-(s//2), (s+1)//2) / (s*l))
                    else:
                        frqs.append(np.concatenate((np.arange(0, (s+1)//2), np.arange(-(s//2), 0))) / (s*l))
            else:
                frqs.append(domain.axes[i])
        return np.asarray(np.broadcast_arrays(*np.ix_(*frqs)))
        

    @property
    def inverse(self):
        return self.adjoint

    def __repr__(self):
        return util.make_repr(self, self.domain)

