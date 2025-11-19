import string

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix, csc_array
import scipy.fft as spfft
import scipy.sparse._csc as CSC
import scipy.sparse.linalg as sla

from regpy.util import make_repr,memoized_property,Errors
from regpy.vecsps import NumPyVectorSpace,UniformGridFcts,GridFcts, MeasureSpaceFcts, DirectSum, Prod

from .base import Operator

__all__ = ["MatrixMultiplication","CholeskyInverse","SuperLUInverse","Power","Exponential","FourierTransform",
           "PtwMatrixVectorMultiplication","PtwScalarMultiplication","AddSingletonVectorDimension","OuterProduct"]

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
        if not isinstance(matrix,(np.ndarray,csc_matrix,csc_array)):
            try:
                self.log.warning(f"Casting the matrix {matrix} to an ndarray.")
                matrix = np.asarray(matrix)
            except Exception as e:
                raise TypeError(Errors.type_error("While initializing MatrixMultiplication the matrix could not be converted to numpy array!")) from e
        if len(matrix.shape) != 2:
            raise ValueError(Errors.value_error(f"While initializing MatrixMultiplication the matrix has to be two-dimensional. Was given \n matrix = {matrix} \n of shape {matrix.shape} and type {type(matrix)}"))
        
        if dtype == None:
            dtype = matrix.dtype

        if domain is None:
            domain = NumPyVectorSpace(matrix.shape[1],dtype = dtype)
        elif not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.type_error(f"While initializing MatrixMultiplication: Domain either None or NumPyVectorSpace given was {type(domain)}"))
        if codomain is None:
            codomain = NumPyVectorSpace(matrix.shape[0],dtype = dtype)
        elif not isinstance(codomain,NumPyVectorSpace):
            raise TypeError(Errors.type_error(f"While initializing MatrixMultiplication: Codomain either none or NumPyVectorSpace given was {type(codomain)}"))
        
        self.matrix = matrix
        
        super().__init__(
            domain=domain,
            codomain=codomain,
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
        
    def _adjoint_eval(self, x):
        if hasattr(self,'_MTM'):
            return self._MTM @ x
        self._MTM = self.matrix.conj().T @ self.matrix
        return self._MTM @ x

    @memoized_property
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
        raise NotImplementedError(Errors.generic_message(f"The inverse for this MatrixMultiplication {self} is not properly set and thus not implemented!"))

    def __repr__(self):
        return make_repr(self, self.matrix)


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
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,add_info="For a CholeskyInverse the operator need to be a RegPy operator!"))
        if not op.linear:
            raise ValueError(Errors.not_linear_op(op,add_info="To construct a CholeskyInverse the operator needs to be linear."))
        if op.domain != op.codomain:
            raise ValueError(Errors.not_equal(op.domain,op.codomain,add_info= "Domain  has to match codomain to construct a CholeskyInverse operator."))
        if not isinstance(op.domain,NumPyVectorSpace)
            raise TypeError(Errors.not_instance(op.domain,NumPyVectorSpace,add_info="Domain has to be a NumPyVectorSpace for the CholeskyInverse"))
        domain = op.domain
        if matrix is None:
            if isinstance(op,MatrixMultiplication):
                matrix = op.matrix
            else:
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
    
    def _adjoint_eval(self, x):
        return self.domain.fromflat(
            cho_solve(self.factorization,cho_solve(
                self.factorization, self.domain.flatten(x))
                )
            )

    @property
    def inverse(self):
        """Returns the original operator."""
        return self.op

    def __repr__(self):
        return make_repr(self, self.op)


class SuperLUInverse(Operator):
    """Implements the inverse of a MatrixMultiplication Operator given by a csc_matrix using SuperLU.

    Parameters
    ----------
        op : MatrixMultiplication
            The operator to be inverted.   
    """
    def __init__(self,op):
        if not isinstance(op,MatrixMultiplication):
            raise TypeError(Errors.not_instance(op,Operator,add_info="For a SuperLUInverse the operator need to be a RegPy MatrixMultiplication!"))
        if not isinstance(op.matrix,(csc_matrix,csc_array)):
            raise TypeError(Errors.type_error(f"To construct a SuperLUInverse of a Matrixmultiplication operator the matrix in that operator needs to be a csc_matrix or csc_array not a simple ndarray!"))
        super().__init__(
            domain=op.codomain, 
            codomain = op.domain,
            linear=True)
        self.op = op
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
        return make_repr(self, self.op)


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
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain, NumPyVectorSpace, add_info="Domain of a Power operator needs to be a NumPyVectorSpace!"))
        self.integer = integer
        if integer:
            if power<0 or int(power)!=power:
                raise ValueError(Errors.value_error(f"If specifying the power to be an integer the given power need to non-negative and its int casting need to be identical to the power. \n\t power = {power} \n\t int(power) = {int(power)}"))
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
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain, NumPyVectorSpace, add_info="Domain of a Exponential operator needs to be a NumPyVectorSpace!"))
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
        Axes over which to compute the Fourier transform. Only domain axes are allowed. 
        If not given, all domain axes are used. Defaults to None.
    """
    def __init__(self, domain, centered=False, axes=None):
        if not isinstance(domain,UniformGridFcts):
            raise TypeError(Errors.not_instance(domain, UniformGridFcts, add_info="Domain of a FourierTransform operator needs to be a UniformGridFcts!"))
        self.is_complex = domain.is_complex
        if axes is None:
            axes = tuple(np.arange(len(domain.shape_domain)))
        frqs = FourierTransform.frequencies(domain,centered=centered, axes=axes, rfft= not domain.is_complex)
        if centered or (not domain.is_complex and domain.ndim_domain==1):
            codomain = UniformGridFcts(*frqs, dtype=complex,shape_codomain=domain.shape_codomain)
        else:
            # In non-centered case, the frequencies are not ascencing, so using GridFcts here is slightly questionable.
            codomain = GridFcts(*frqs, dtype=complex,shape_codomain=domain.shape_codomain,use_cell_measure=False)
        super().__init__(domain, codomain, linear=True)
        self.centered = centered
        self.axes = axes
  
    def _eval(self, x):
        if self.centered:
            x = spfft.ifftshift(x, axes=self.axes)
        if self.is_complex:
            y = spfft.fftn(x, axes=self.axes, norm='ortho')
        else:
            y = spfft.rfftn(x, axes=self.axes, norm='ortho') # type: ignore
        if self.centered:
            return spfft.fftshift(y, axes=self.axes)
        else:
            return y

    def _adjoint(self, y):
        if self.centered:
            y = spfft.ifftshift(y, axes=self.axes)
        if self.is_complex:
            x = spfft.ifftn(y, axes=self.axes, norm='ortho')
        else:
            x = spfft.irfftn(y, tuple(self.domain.shape[i] for i in self.axes),axes=self.axes, norm='ortho')
        if self.centered:
            x = spfft.fftshift(x, axes=self.axes)
        if self.domain.is_complex:
            return x
        else:
            return x.real
        
    def _adjoint_eval(self, x):
        if self.domain.is_complex:
            return x
        else:
            return x.real
        
    @staticmethod
    def frequencies(domain,centered=False, axes=None, rfft=False):
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
            domain axes will be computed. Default: `None`.
        Returns
        -------
        array
        """
        if axes is not None:
            if not np.all([0 <= ax < len(domain.shape_domain) for ax in axes]):
                raise ValueError(Errors.value_error(f"Invalid axis specified: {axes}. Must be within [0, {len(domain.shape_domain)})",obj=FourierTransform,meth="frequencies"))
            if not len(axes) == len(set(axes)):
                raise ValueError(Errors.value_error(f"Axes contain duplicates: {axes}",obj=FourierTransform,meth="frequencies"))
        else:
            axes = np.arange(len(domain.shape_domain))
        frqs = []
        for i, (s, l) in enumerate(zip(domain.shape_domain, domain.spacing)):
            if i in axes:
                # Use (spacing * shape) in denominator instead of extents, since the grid is assumed
                # to be periodic.
                shalf = s/2+1 if (s//2)*2==s else (s+1)/2
                if i==axes[-1] and rfft==True:
                    frqs.append(np.arange(0,shalf) / (s*l))
                else:
                    if centered:
                        frqs.append(np.arange(-(s//2), (s+1)//2) / (s*l))
                    else:
                        frqs.append(np.concatenate((np.arange(0, (s+1)//2), np.arange(-(s//2), 0))) / (s*l))
            else:
                frqs.append(domain.axes[i])
        return tuple(frqs)
        

    @property
    def inverse(self):
        return self.adjoint

    def __repr__(self):
        return make_repr(self, self.domain)


class PtwMatrixVectorMultiplication(Operator):
    """
    Pointwise multiplication of a matrix-valued function with a vector-valued function.

    Parameters
    ----------
    domain : MeasureSpaceFcts
        The input grid function.
    matrixfct : np.ndarray
        The matrix-valued function to multiply with the vector-valued function.  
        The first dimensions must match the shape_domain of domain, the last dimensions  
        must match the shape_codomain of domain, and the middle dimensions define the output shape_codomain.
    """
    def __init__(self,domain,matrixfct):
        if not isinstance(domain, MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts, add_info='The domain of a PtwMultiplication must be at least of type MeasureSpaceFcts.'))
        domain_shape = domain.shape_domain
        codomain_shape = domain.shape_codomain

        if not isinstance(matrixfct,np.ndarray) or not matrixfct.dtype==domain.dtype:
            raise TypeError(Errors.type_error(f'To initialize a PtwMatrixVectorMultiplication the matrix functions matrixfct must be a numpy array of the same data type. Was given\n\t matrixfct = {matrixfct}\n\t and matrixfcts.dtype={matrixfct.dtype} and domain.dtype = {domain.dtype}'))
        if not matrixfct.shape[-len(codomain_shape):]==codomain_shape:
            raise ValueError(Errors.value_error(f'Initializing a PtwMatrixVectorMultiplication: shape of matrixfct does not match: {matrixfct.shape[-len(codomain_shape):]}, {codomain_shape}'))

        self.matrixfct= matrixfct
        remaining_codomain_shape = matrixfct.shape[len(domain_shape):-len(codomain_shape)]

        super().__init__(domain=domain,codomain=domain.vector_valued_space(remaining_codomain_shape),linear=True)

        letters_in = ''+string.ascii_letters[:len(codomain_shape)]
        letters_out = ''+string.ascii_letters[len(codomain_shape):len(codomain_shape)+len(remaining_codomain_shape)]
        self._einstein_string_mul = '...' + letters_out + letters_in + ',...' + letters_in + '->...'+ letters_out
        # e.g., '...ba,...a->...b'
        self._einstein_string_mulT =  '...' + letters_out + letters_in + ',...' + letters_out + '->...'+ letters_in
        # e.g., '...ba,...b->...a'

    def _eval(self, v):
        return  np.einsum(self._einstein_string_mul, self.matrixfct, v)
    
    def _adjoint(self, w):
        return np.einsum(self._einstein_string_mulT, np.conj(self.matrixfct), w)

    def __repr__(self):
        return make_repr(self, self.domain, self.codomain)

class PtwScalarMultiplication(Operator):
    """
    Pointwise multiplication of a scalar valued function with a vector-valued function.

    Parameters
    ----------
    domain : MeasureSpaceFcts
        The input grid function.
    scalarfct : np.ndarray
        The scalar valued function to multiply with the vector-valued function.
        The number of dimensions must match the number of dimensions of domain.   
        The first dimensions must match the shape_domain of domain, and the co-dimensions must be 1.
    """   

    def __init__(self, domain, multiplier):
        if not isinstance(domain,MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info=f'For a PtwScalarMultiplication the domain must be of type MeasureSpaceFcts. Got \n\t domain = {domain}'))
        if domain.ndim_codomain==0:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the domain must be vector valued!'))
        if multiplier in domain and (multiplier.shape == domain.shape_domain + (1,)*domain.ndim_codomain):
            self.multiplier = multiplier
            print("first option", self.multiplier.shape)
        elif multiplier in domain.scalar_space():
            self.multiplier = np.reshape(multiplier,domain.shape_domain+(1,)*domain.ndim_codomain)
        else:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the multiplier must be numpy array of matching size. Got \n\t domain = {domain}, \n\t multiplier = {multiplier}'))
        
        
        super().__init__(domain, domain, linear=True)

    def _eval(self, f):
        return f*self.multiplier
    
    def _adjoint(self, g):
        return g*self.multiplier.conj()

class AddSingletonVectorDimension(Operator):
    """Operater that adds a singleton dimension as codimension in MeasureSpaceFcts. 
    Wrapper to np.reshape(...,1).

    Parameters
    ----------
    grid: MeasureSpaceFcts    
    """
    def __init__(self, domain):
        if not isinstance(domain, MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info=f'The VectorSpace must be of type MeasureSpaceFcts. Got {type(domain)}'))
        if domain.ndim_codomain != 0:
            raise ValueError(Errors.value_error(f'TO add a SignletonVector dimension the domain must be scalar-valued. Got shape_codomain = {domain.shape_codomain}'))
        self.shape_domain = domain.shape_domain
        super().__init__(domain, domain.vector_valued_space((1,)), linear=True)

    def _eval(self,f):
        return np.expand_dims(f, axis=-1)
    
    def _adjoint(self,f):
        return np.squeeze(f, axis=-1)
    

class OuterProduct(Operator):
    r"""The operator \(x_1,x_2,\dots,x_n \mapsto x_1\otimes x_2\otimes\dots\otimes x_n\).

    Parameters
    ----------
    domains : regpy.vecsps.NumPyVectorSpace
        The underlying vector spaces. Have to be all real or all complex.
    """

    def __init__(self,*domains):
        domain=DirectSum(*domains)
        codomain=Prod(*domains)
        super().__init__(domain, codomain, linear=False)
        self._adjoint_summation_strings=self._calc_adjoint_summation_strings()
        """List of strings for calculation of adjoint in einsum."""

    def _calc_adjoint_summation_strings(self):
        li=[]
        characters=tuple(chr(k) for k in range(65,65+self.codomain.ndim))
        all_characters="".join(characters)
        for i,_ in enumerate(characters):
            without_i=",".join(characters[:i]+characters[i+1:])
            li.append(f"{all_characters},{without_i}")
        return li

    def _eval(self, x, differentiate=False):
        self._p=self.domain.split(x)
        if(differentiate):
            self._p_conj_flat=tuple(np.conj(p_j).flat for p_j in self._p)
        return self.codomain.product(*self._p)

    def _derivative(self, x):
        y=self.codomain.zeros()
        for j,x_j in enumerate(x):
            y+=self.codomain.product(*self._p[:j],x_j,*self._p[j+1:])
        return y
    
    def _adjoint(self, y):
        xs=[]
        for j,s in enumerate(self._adjoint_summation_strings):
            x_j=np.einsum(s,y,*self._p_conj_flat[:j],*self._p_conj_flat[j+1:],optimize=True)
            xs.append(x_j.reshape(self.domain[j].shape))
            # xs.append(self.domain[j].fromflat(x_j))
        return self.domain.join(*xs)
