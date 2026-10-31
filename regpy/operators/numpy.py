from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from numpy.typing import ArrayLike, DTypeLike
    from regpy.vecsps.base import ReadOnlyVector, Scalar, Vector, VectorSpaceBase
    from regpy.vecsps.numpy import GridFcts, UniformGridFcts
    from regpy.functionals.base import Functional
    from regpy.hilbert.base import HilbertSpace
    import scipy.sparse
    from regpy.vecsps.numpy import NumPyVectorSpace
import inspect
import string
from functools import partial

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix, csc_array, lil_matrix, linalg
import scipy.fft as spfft
import scipy.sparse._csc as CSC
import scipy.sparse.linalg as sla

from regpy.util import (
    Errors,
    complex2real as complex2real_array,
    make_repr,
    memoized_property,
    real2complex as real2complex_array,
)
from regpy.vecsps import *
from .base import Composition, Operator

__all__ = ["MatrixMultiplication", "Complex2Real", "complex2real",
           "CholeskyInverse","SuperLUInverse","Power","NemytskiiOperator","Exponential","FourierTransform",
           "RealFourierEmbedding", "RealFourierTransform",
           "PtwMatrixVectorMultiplication","PtwScalarMultiplication","AddSingletonVectorDimension","ForwardFDGradient","OuterProduct","EinSum"]

class MatrixMultiplication(Operator):
    r"""Implements an operator that does matrix-vector multiplication with a given matrix. Domain and codomain 
    are plain one dimensional `regpy.vecsps.NumPyVectorSpace` instances by default.
    
    Parameters
    ----------
    matrix: np.ndarray | csc_matrix | csc_array
        The matrix.
    inverse: Operator | np.ndarray | str | None [default: None]
        How to implement the inverse operator. If available, this should be given as `Operator`
        or array. If `inv`, `numpy.linalg.inv` will be used. If `cholesky` or `superLU`, a
        `CholeskyInverse` or `SuperLU` instance will be returned.
    domain: VectorSpaceBase | None [default: None]
        Spaces represented in their flatten/fromflat coordinates over the
        selected scalar field. With as_real_space=False, complex matrices act
        on complex coordinates. Native complex matrices on realified complex
        storage are converted to interleaved real block matrices.
        If omitted, matching NumPy spaces with the default real scalar field
        are constructed.
    codomain: VectorSpaceBase | None [default: None]
        Spaces represented in their flatten/fromflat coordinates over the
        selected scalar field. With as_real_space=False, complex matrices act
        on complex coordinates. Native complex matrices on realified complex
        storage are converted to interleaved real block matrices.
        If omitted, matching NumPy spaces with the default real scalar field
        are constructed.
    dtype: DTypeLike [default: None]
        Data type.
    
    Notes
    -----
    The matrix multiplication is done by applying numpy.dot to the matrix and an element of the domain. 
    The adjoint is implemented in the same way by multiplying with the adjoint matrix.
    Multidimensional NumPy and shape-free spaces use the same coordinate
    interface. The coordinate matrix has shape
    (codomain.dimension, domain.dimension).
    """

    def __init__(self, 
                 matrix:np.ndarray|csc_matrix|csc_array, 
                 inverse: Operator | np.ndarray | str | None=None, 
                 domain:VectorSpaceBase|None=None, 
                 codomain:VectorSpaceBase|None=None,
                 dtype: DTypeLike=None) -> None:
        if not isinstance(matrix,(np.ndarray,csc_matrix,csc_array)):
            try:
                self.log.warning(f"Casting the matrix {matrix} to an ndarray.")
                matrix = np.asarray(matrix)
            except Exception as e:
                raise TypeError(Errors.type_error("While initializing MatrixMultiplication the matrix could not be converted to NumPy array!")) from e
        if len(matrix.shape) != 2:
            raise ValueError(Errors.value_error(f"While initializing MatrixMultiplication the matrix has to be two-dimensional. Was given:"+"\n\t"+ f"matrix = {matrix}"+"\n\t"+ f"of shape {matrix.shape} and type {type(matrix)}"))
        
        if dtype == None:
            dtype = matrix.dtype

        if domain is None:
            domain = NumPyVectorSpace(matrix.shape[1],dtype = dtype)
        elif not isinstance(domain,VectorSpaceBase):
            raise TypeError(Errors.type_error(f"While initializing MatrixMultiplication: Domain either None or VectorSpaceBase given was {type(domain)}"))
        if codomain is None:
            codomain = NumPyVectorSpace(matrix.shape[0],dtype = dtype)
        elif not isinstance(codomain,VectorSpaceBase):
            raise TypeError(Errors.type_error(f"While initializing MatrixMultiplication: Codomain either none or VectorSpaceBase given was {type(codomain)}"))
        
        if domain.as_real_space != codomain.as_real_space:
            import warnings
            warnings.warn("Mixed matrix scalar fields; interpreting both spaces over R.",
                          RuntimeWarning, stacklevel=2)
            domain.as_real_space = codomain.as_real_space = True
        # Preserve native complex matrices on realified complex storage by
        # representing their action on interleaved real/imaginary coordinates.
        if (domain.is_complex and codomain.is_complex and domain.as_real_space
                and matrix.shape == (codomain.dimension // 2, domain.dimension // 2)):
            if isinstance(matrix, np.ndarray):
                real_matrix = np.empty((codomain.dimension, domain.dimension))
                real_matrix[0::2, 0::2] = matrix.real
                real_matrix[0::2, 1::2] = -matrix.imag
                real_matrix[1::2, 0::2] = matrix.imag
                real_matrix[1::2, 1::2] = matrix.real
                matrix = real_matrix
            else:
                from scipy.sparse import kron
                matrix = (kron(matrix.real, np.eye(2))
                          + kron(matrix.imag, np.array([[0., -1.], [1., 0.]]))).tocsc()
        if matrix.shape != (codomain.dimension, domain.dimension):
            raise ValueError("Matrix dimensions do not match the scalar-field dimensions.")
        if domain.as_real_space:
            values = matrix if isinstance(matrix, np.ndarray) else matrix.data
            if np.any(np.imag(values) != 0):
                raise ValueError("A matrix acting on real coordinates must be real.")
        complex_coordinates = not domain.as_real_space
        self.matrix = matrix
        self.complex_coordinates = complex_coordinates

        def domain_coordinates(vector: ReadOnlyVector) -> np.ndarray:
            r"""Domain coordinates.
            
            Parameters
            ----------
            vector: ReadOnlyVector
                Input vector; its coefficients are only read.
            
            
            Returns
            -------
            np.ndarray
                Computed array.
            """
            if self.complex_coordinates:
                return np.asarray(domain.flatten(vector)).reshape(-1)
            return np.asarray(domain.flatten(vector)).reshape(-1)

        def codomain_coordinates(vector: ReadOnlyVector) -> np.ndarray:
            r"""Codomain coordinates.
            
            Parameters
            ----------
            vector: ReadOnlyVector
                Input vector; its coefficients are only read.
            
            
            Returns
            -------
            np.ndarray
                Computed array.
            """
            if self.complex_coordinates:
                return np.asarray(codomain.flatten(vector)).reshape(-1)
            return np.asarray(codomain.flatten(vector)).reshape(-1)

        self._domain_coordinates = domain_coordinates
        self._codomain_coordinates = codomain_coordinates

        if isinstance(matrix,np.ndarray):
            self.matvec = lambda v: self.matrix @ self._domain_coordinates(v)
            self.rmatvec = lambda v: (
                self.matrix.conj().T @ self._codomain_coordinates(v)
            )
        else:
            lin_op = linalg.aslinearoperator(self.matrix)
            self.matvec = lambda v: lin_op.matvec(
                self._domain_coordinates(v)
            )
            self.rmatvec = lambda v: lin_op.rmatvec(
                self._codomain_coordinates(v)
            )

        self_adjoint = (
            domain == codomain
            and matrix.shape[0] == matrix.shape[1]
            and (
                np.allclose(matrix, matrix.conj().T)
                if isinstance(matrix, np.ndarray)
                else False
            )
        )
        
        super().__init__(
            domain=domain,
            codomain=codomain,
            linear=True,
            self_adjoint=self_adjoint,
        )
        self._inverse = inverse

    def _eval(self, x: ReadOnlyVector) -> Vector:
        result = np.asarray(self.matvec(x))
        if self.complex_coordinates:
            return self.codomain.fromflat(result)
        return self.codomain.fromflat(result)

    def _ieval(self, x: ReadOnlyVector, out: Vector) -> Vector:
        self.codomain.copyto(out, self._eval(x))
        return out

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        result = np.asarray(self.rmatvec(y))
        if self.complex_coordinates:
            return self.domain.fromflat(result)
        return self.domain.fromflat(result)

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        self.domain.copyto(out, self._adjoint(y))
        return out
        
    def _adjoint_eval(self, x: ReadOnlyVector) -> Vector:
        if hasattr(self,'_MTM'):
            matrix = self._MTM
        else:
            self._MTM = self.matrix.conj().T @ self.matrix
            matrix = self._MTM
        result = np.asarray(matrix @ self._domain_coordinates(x))
        if self.complex_coordinates:
            return self.domain.fromflat(result)
        return self.domain.fromflat(result)

    @memoized_property
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        if isinstance(self._inverse, Operator):
            return self._inverse
        elif isinstance(self._inverse, np.ndarray):
            return MatrixMultiplication(
                self._inverse,
                inverse=self,
                domain=self.codomain,
                codomain=self.domain,
            )
        elif isinstance(self._inverse, str):
            if self._inverse == 'inv':
                inverse = MatrixMultiplication(
                    np.linalg.inv(self.matrix),
                    inverse=self,
                    domain=self.codomain,
                    codomain=self.domain,
                )
                self._inverse = inverse
                return inverse
            if self._inverse == 'cholesky':
                return CholeskyInverse(self, matrix=self.matrix)
            if self._inverse == 'superLU':
                return SuperLUInverse(self)
        raise NotImplementedError(Errors.generic_message(f"The inverse for this MatrixMultiplication {self} is not properly set and thus not implemented!"))

    def __repr__(self) -> str:
        return make_repr(self, self.matrix)


class Complex2Real(Operator):
    r"""Identify a complex NumPy vector space with a real one.
    
    The operator appends an axis of length two (or inserts it at ``axis``)
    containing real and imaginary parts.  For example, with the default
    ``axis=-1`` it maps
    
    .. math::
        \mathbb C^{n_1\times\cdots\times n_d}
        \longrightarrow
        \mathbb R^{n_1\times\cdots\times n_d\times 2}.
    
    This is an isometric real-linear isomorphism for the standard real inner
    products.  Consequently its adjoint and inverse both apply
    :func:`regpy.util.real2complex`.
    
    Parameters
    ----------
    domain: NumPyVectorSpace
        Complex NumPy vector space to be realified.
    axis: int [default: -1]
        Position of the new real/imaginary axis in the codomain.
    """

    def __init__(self, domain: NumPyVectorSpace, axis: int = -1) -> None:
        if not isinstance(domain, NumPyVectorSpace):
            raise TypeError(Errors.not_instance(
                domain,
                NumPyVectorSpace,
                add_info="Complex2Real requires a NumPyVectorSpace."
            ))
        if not domain.is_complex:
            raise ValueError(Errors.value_error(
                "The domain of Complex2Real must be complex."
            ))
        if (isinstance(axis, (bool, np.bool_))
                or not isinstance(axis, (int, np.integer))):
            raise TypeError(Errors.type_error(
                f"axis must be an integer. Got {axis!r}."
            ))
        codomain_ndim = domain.ndim + 1
        axis = int(axis)
        if axis < 0:
            axis += codomain_ndim
        if not 0 <= axis < codomain_ndim:
            raise ValueError(Errors.value_error(
                f"axis must lie in [-{codomain_ndim}, {codomain_ndim}). "
                f"Got {axis}."
            ))

        shape = list(domain.shape)
        shape.insert(axis, 2)
        real_dtype = np.empty((), dtype=domain.dtype).real.dtype
        codomain = NumPyVectorSpace(tuple(shape), dtype=real_dtype)
        self.axis = axis
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x: ReadOnlyVector) -> Vector:
        # The utility may return a view.  Allocation-returning Operator
        # methods must return storage independent of their arguments.
        return np.array(
            complex2real_array(x, axis=self.axis),
            dtype=self.codomain.dtype,
            copy=True,
        )

    def _ieval(self, x: ReadOnlyVector, out: Vector) -> Vector:
        np.copyto(out, complex2real_array(x, axis=self.axis))
        return out

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        return np.array(
            real2complex_array(y, axis=self.axis),
            dtype=self.domain.dtype,
            copy=True,
        )

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        np.copyto(out, real2complex_array(y, axis=self.axis))
        return out

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return self.adjoint

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return True

    def __repr__(self) -> str:
        return make_repr(self, self.domain, f"axis={self.axis}")


# Lower-case alias matching the utility-function terminology and allowing
# ``complex2real(space)`` as a constructor.
complex2real = Complex2Real


class CholeskyInverse(Operator):
    r"""Implements the inverse of a linear, self-adjoint operator via Cholesky decomposition. Since
    it needs to assemble a full matrix, this should not be used for high-dimensional operators.
    
    Parameters
    ----------
    op: Operator
        The operator to be inverted.
    matrix: np.ndarray | None [default: None]
        If a matrix of `op` is already available, it can be passed in to avoid recomputation.
    """
    def __init__(self,
                op:Operator,
                matrix: np.ndarray | None =None
                ) -> None:
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,add_info="For a CholeskyInverse the operator need to be a RegPy operator!"))
        if not op.linear:
            raise ValueError(Errors.not_linear_op(op,add_info="To construct a CholeskyInverse the operator needs to be linear."))
        if op.domain != op.codomain:
            raise ValueError(Errors.not_equal(op.domain,op.codomain,add_info= "Domain  has to match codomain to construct a CholeskyInverse operator."))
        if not isinstance(op.domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(op.domain,NumPyVectorSpace,add_info="Domain has to be a NumPyVectorSpace for the CholeskyInverse"))
        domain = op.domain
        if matrix is None:
            if isinstance(op,MatrixMultiplication):
                matrix = op.matrix
            else:
                matrix = np.empty((domain.dimension,) * 2, dtype=float if domain.as_real_space else complex)
                for j, elm in enumerate(domain.iter_basis()):
                    matrix[:, j] = domain.flatten(op(elm))
        self.factorization = cho_factor(matrix)
        """The Cholesky factorization for use with `scipy.linalg.cho_solve`"""
        super().__init__(
            domain=domain,
            codomain=domain,
            linear=True
        )
        self.op = op

    def _eval(self, x: ReadOnlyVector) -> Vector:
        return self.domain.fromflat(
            cho_solve(self.factorization, self.domain.flatten(x)))

    def _ieval(self, x: ReadOnlyVector, out: Vector) -> Vector:
        flat_out = self.domain.flatten(out)
        np.copyto(flat_out, self.domain.flatten(x))
        result = cho_solve(self.factorization, flat_out, overwrite_b=True)
        if result is not flat_out:
            np.copyto(flat_out, result)
        # ``flatten`` normally returns a view. Retain correctness for vector
        # spaces whose flattening operation has to allocate.
        if not np.shares_memory(flat_out, out):
            np.copyto(out, self.domain.fromflat(flat_out))
        return out

    def _adjoint(self, x: ReadOnlyVector) -> Vector:
        return self._eval(x)

    def _iadjoint(self, x: ReadOnlyVector, out: Vector) -> Vector:
        return self._ieval(x, out)
    
    def _adjoint_eval(self, x: ReadOnlyVector) -> Vector:
        return self.domain.fromflat(
            cho_solve(self.factorization,cho_solve(
                self.factorization, self.domain.flatten(x))
                )
            )

    @property
    def inverse(self) -> Operator:
        r"""Returns the original operator.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return self.op

    def __repr__(self) -> str:
        return make_repr(self, self.op)


class SuperLUInverse(Operator):
    r"""Implements the inverse of a MatrixMultiplication Operator given by a csc_matrix using SuperLU.
    
    Parameters
    ----------
    op: MatrixMultiplication
        Operator used by this construction.
    """
    def __init__(self,op:MatrixMultiplication) -> None:
        if not isinstance(op,MatrixMultiplication):
            raise TypeError(Errors.not_instance(op,Operator,add_info="For a SuperLUInverse the operator need to be a RegPy MatrixMultiplication!"))
        if not isinstance(op.matrix,(csc_matrix,csc_array)):
            raise TypeError(Errors.type_error("To construct a SuperLUInverse of a MatrixMultiplication operator the matrix in that operator needs to be a csc_matrix or csc_array not a simple ndarray!"))
        super().__init__(
            domain=op.codomain, 
            codomain = op.domain,
            linear=True)
        self.op = op
        self.lu = sla.splu(op.matrix)

    def _solve_coordinates(self, rhs: np.ndarray, trans: str='N') -> np.ndarray:
        r"""Solve in the scalar-field coordinates used by op.matrix.
        
        Parameters
        ----------
        rhs: np.ndarray
            Rhs.
        trans: str [default: 'N']
            Trans.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if np.iscomplexobj(rhs) and not np.issubdtype(
                self.lu.U.dtype, np.complexfloating):
            # A real LU factorization can still act on complex coordinates.
            return (self.lu.solve(rhs.real, trans=trans)
                    + 1j * self.lu.solve(rhs.imag, trans=trans))
        return self.lu.solve(rhs, trans=trans)

    def _eval(self, x: ReadOnlyVector) -> Vector:
        rhs = self.domain.flatten(x)
        return self.codomain.fromflat(self._solve_coordinates(rhs))

    def _ieval(self, x: ReadOnlyVector, out: Vector) -> Vector:
        self.codomain.copyto(out, self._eval(x))
        return out

    def _adjoint(self, x: ReadOnlyVector) -> Vector:
        rhs = self.codomain.flatten(x)
        return self.domain.fromflat(self._solve_coordinates(rhs, trans='H'))

    def _iadjoint(self, x: ReadOnlyVector, out: Vector) -> Vector:
        self.domain.copyto(out, self._adjoint(x))
        return out

    @property
    def inverse(self) -> Operator:
        r"""Returns the original operator.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return self.op

    def __repr__(self) -> str:
        return make_repr(self, self.op)


class Power(Operator):
    r"""The operator \(x \mapsto x^n\).
    
    Parameters
    ----------
    power: float | int
        The exponent. If the flag integer is True, then power has to be a nonnegative integer. 
        In this case the result is computed by multiplications (based on values for power=2**n, n=1,2, ... ).
    domain: NumPyVectorSpace
        The underlying vector space
    integer: bool [default: False]
        Integer.
    """

    def __init__(self, 
                 power:float|int, 
                 domain:NumPyVectorSpace, 
                 integer:bool = False) -> None:
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain, NumPyVectorSpace, add_info="Domain of a Power operator needs to be a NumPyVectorSpace!"))
        self.integer = integer
        if integer:
            if power<0 or int(power)!=power:
                raise ValueError(Errors.value_error(f"If specifying the power to be an integer the given power need to non-negative and its int casting need to be identical to the power. "+"\n\t "+f"power = {power} "+"\n\t "+f"int(power) = {int(power)}"))
            power=int(power)
            self._power_bin = "{0:b}".format(power)
        self.power = power
        super().__init__(domain, domain)

    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        if self.integer:
            # ``powx`` is squared repeatedly and therefore always needs its own
            # storage. Taking the copy before initializing ``out`` also makes
            # the method safe for ``out is x``.
            powx = x.copy()
            out.fill(1)
            if differentiate:
                self._factor = np.empty_like(x)
                self._factor.fill(self.power)
                if self.power > 0:
                    self._dpow_bin = "{0:b}".format(self.power - 1)
                    if len(self._dpow_bin) < len(self._power_bin):
                        self._dpow_bin = '0' + self._dpow_bin
                else:
                    self._dpow_bin = "{0:b}".format(0)
            for k in reversed(range(len(self._power_bin))):
                if self._power_bin[k] == '1':
                    np.multiply(out, powx, out=out)
                if differentiate and self._dpow_bin[k] == '1':
                    np.multiply(self._factor, powx, out=self._factor)
                if k > 0:
                    np.multiply(powx, powx, out=powx)
        else:
            # The factor has to be computed before overwriting an aliased input.
            if differentiate:
                self._factor = np.empty_like(x)
                np.power(x, self.power - 1, out=self._factor)
                self._factor *= self.power
            np.power(x, self.power, out=out)
        return out

    def _iderivative(self, x: ReadOnlyVector, out: Vector) -> Vector:
        return np.multiply(self._factor, x, out=out)

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        if np.shares_memory(y, out):
            y = y.copy()
        np.conjugate(self._factor, out=out)
        return np.multiply(out, y, out=out)


class NemytskiiOperator(Operator):
    r"""Pointwise superposition operator induced by a NumPy function.
    
    Let ``domain`` be a :class:`regpy.vecsps.MeasureSpaceFcts` and let
    ``f`` map one function value in :math:`F_1^n` to a value in
    :math:`F_2^m`, where :math:`F_1,F_2` are either the real or complex
    numbers. This class implements the Nemytskii (superposition) operator
    
    .. math::
    
        (x_i)_{i\in I} \longmapsto (f(x_i))_{i\in I}.
    
    Complex vector spaces are viewed as real vector spaces throughout RegPy.
    Consequently, ``Df`` must return the **real** Jacobian. Complex coordinates
    are ordered componentwise as
    
    ``[Re z_0, Im z_0, Re z_1, Im z_1, ...]``.
    
    The function ``f`` and the Jacobian ``Df`` should be vectorized over all
    leading domain axes. They act only on the trailing value axes. Ordinary
    unary NumPy ufuncs such as ``np.exp`` are applied directly to the complete
    value array. A genuine multi-input or multi-output ufunc is interpreted as
    a map whose inputs and outputs are the flattened point-value components.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        Function space on which the superposition operator acts.
    f: Callable[..., ArrayLike]
        Vectorized point function. If it accepts an ``out`` keyword, it is used
        by :meth:`_ieval` to avoid an allocation.
    Df: Callable[..., ArrayLike]
        Vectorized real Jacobian. Its result must be real and broadcastable to
        ``domain.shape_domain + (real_output_size, real_input_size)``. For a
        real scalar-to-scalar map, returning only an array of shape
        ``domain.shape_domain`` is accepted as shorthand for a ``1 x 1``
        Jacobian.
    codomain_shape: tuple[int, ...] | None [default: None]
        Shape of one output value. By default it is inferred by evaluating
        ``f`` once on a zero point value.
    codomain_dtype: DTypeLike | None [default: None]
        Real or complex output dtype. By default it is inferred together with
        ``codomain_shape``.
    linear: bool [default: False]
        Declare the point function linear. In this case the Jacobian is
        computed once during initialization and the operator exposes an
        adjoint directly.
    
    Notes
    -----
    If ``A`` is a complex matrix representing a complex-linear derivative,
    :meth:`complex_matrix_to_real_jacobian` converts it to the real Jacobian
    convention required here. General non-holomorphic maps must provide their
    real Jacobian directly.
    """

    def __init__(
            self, domain: VectorSpaceBase, f: Callable[..., ArrayLike], Df: Callable[..., ArrayLike], codomain_shape: tuple[int, ...] | None=None,
            codomain_dtype: DTypeLike | None=None, linear: bool=False) -> None:
        if not isinstance(domain, MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(
                domain, MeasureSpaceFcts,
                add_info="The domain of a NemytskiiOperator must be a MeasureSpaceFcts."
            ))
        if not callable(f):
            raise TypeError(Errors.type_error("f must be callable."))
        if not callable(Df):
            raise TypeError(Errors.type_error("Df must be callable."))

        self.f = f
        self.Df = Df
        self._domain_shape = domain.shape_domain
        self._input_shape = domain.shape_codomain
        self._input_size = int(np.prod(self._input_shape)) if self._input_shape else 1
        self._split_ufunc = (
            isinstance(f, np.ufunc) and (f.nin != 1 or f.nout != 1)
        )
        if self._split_ufunc and f.nin != self._input_size:
            raise ValueError(Errors.value_error(
                f"The ufunc has {f.nin} inputs, but one domain value has "
                f"{self._input_size} flattened components."
            ))

        point = np.zeros(self._input_shape or (), dtype=domain.dtype)
        sample = self._evaluate_point(point)
        inferred_shape = sample.shape

        if codomain_shape is None:
            codomain_shape = inferred_shape
        elif isinstance(codomain_shape, (int, np.integer)):
            codomain_shape = (int(codomain_shape),)
        elif not (
                isinstance(codomain_shape, tuple)
                and all(isinstance(s, (int, np.integer)) and s > 0
                        for s in codomain_shape)):
            raise ValueError(Errors.value_error(
                "codomain_shape must be an int, a tuple of positive ints, or ()."
            ))
        codomain_shape = tuple(int(s) for s in codomain_shape)
        if sample.shape != codomain_shape:
            raise ValueError(Errors.value_error(
                f"f maps one point value to shape {sample.shape}, not to the "
                f"requested codomain_shape={codomain_shape}."
            ))

        codomain_dtype = np.dtype(
            sample.dtype if codomain_dtype is None else codomain_dtype
        )
        if not (
                np.issubdtype(codomain_dtype, np.floating)
                or np.issubdtype(codomain_dtype, np.complexfloating)):
            raise TypeError(Errors.type_error(
                f"codomain_dtype must be real or complex, got {codomain_dtype}."
            ))
        if not np.can_cast(sample.dtype, codomain_dtype, casting="same_kind"):
            raise TypeError(Errors.type_error(
                f"The values returned by f have dtype {sample.dtype}, which "
                f"cannot be cast to {codomain_dtype} with same-kind casting."
            ))

        if codomain_shape == ():
            codomain = domain.scalar_space()
        else:
            codomain = domain.vector_valued_space(codomain_shape)
        codomain.dtype = codomain_dtype
        codomain.is_complex = np.issubdtype(
            codomain_dtype, np.complexfloating
        )

        self._output_shape = codomain_shape
        self._output_size = (
            int(np.prod(codomain_shape)) if codomain_shape else 1
        )
        self._real_input_size = self._input_size * (2 if domain.is_complex else 1)
        self._real_output_size = self._output_size * (2 if codomain.is_complex else 1)
        self._jacobian_dtype = np.result_type(
            np.empty((), dtype=domain.dtype).real.dtype,
            np.empty((), dtype=codomain.dtype).real.dtype,
        )
        self._jacobian = np.empty(
            self._domain_shape
            + (self._real_output_size, self._real_input_size),
            dtype=self._jacobian_dtype,
        )
        self._input_real_buffer = (
            np.empty(
                self._domain_shape + (self._real_input_size,),
                dtype=self._jacobian_dtype,
            ) if domain.is_complex else None
        )
        self._output_real_buffer = (
            np.empty(
                self._domain_shape + (self._real_output_size,),
                dtype=self._jacobian_dtype,
            ) if codomain.is_complex else None
        )
        self._f_accepts_out = self._accepts_out(f)
        self._Df_accepts_out = (
            not isinstance(Df, np.ufunc) and self._accepts_out(Df)
        )

        super().__init__(domain, codomain, linear=linear)
        self._consts.update({'f', 'Df'})
        self._value_buffer = codomain.empty()
        if linear:
            self._update_jacobian(domain.zeros())

    @staticmethod
    def _accepts_out(func: Functional) -> bool:
        if isinstance(func, np.ufunc):
            return True
        try:
            parameters = inspect.signature(func).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == 'out'
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    @staticmethod
    def _stack_result(result: ArrayLike) -> np.ndarray:
        if isinstance(result, (tuple, list)):
            if len(result) == 1:
                result = result[0]
            else:
                result = np.stack(tuple(np.asarray(item) for item in result), axis=-1)
        return np.asarray(result)

    def _evaluate_point(self, point: ReadOnlyVector) -> np.ndarray:
        if self._split_ufunc:
            point_flat = np.reshape(point, (self._input_size,))
            result = self.f(*(point_flat[j] for j in range(self._input_size)))
        else:
            result = self.f(point)
        result = self._stack_result(result)
        if result.dtype == object:
            raise TypeError(Errors.type_error(
                "f returned an object array. It must return a numeric scalar or array."
            ))
        return result

    def _copy_function_result(self, result: ArrayLike, out: Vector) -> np.ndarray:
        result = self._stack_result(result)
        try:
            np.copyto(out, result, casting='same_kind')
        except (TypeError, ValueError) as exc:
            raise ValueError(Errors.value_error(
                f"f returned shape {result.shape} and dtype {result.dtype}; "
                f"expected an array broadcastable to shape {out.shape} with "
                f"dtype {out.dtype}."
            )) from exc
        return out

    def _apply_function(self, x: ReadOnlyVector, out: Vector) -> np.ndarray:
        if self._split_ufunc:
            x_flat = np.reshape(
                x, self._domain_shape + (self._input_size,)
            )
            args = tuple(x_flat[..., j] for j in range(self._input_size))
            if self.f.nout == 1:
                result = self.f(*args, out=out)
                if result is not out:
                    self._copy_function_result(result, out)
            else:
                out_flat = np.reshape(
                    out, self._domain_shape + (self._output_size,)
                )
                targets = tuple(out_flat[..., j] for j in range(self.f.nout))
                result = self.f(*args, out=targets)
                if result is not targets and not (
                        isinstance(result, tuple)
                        and all(a is b for a, b in zip(result, targets))):
                    self._copy_function_result(result, out)
            return out

        if self._f_accepts_out:
            result = self.f(x, out=out)
            if result is None or result is out:
                return out
            if isinstance(result, tuple) and len(result) == 1 and result[0] is out:
                return out
            return self._copy_function_result(result, out)
        return self._copy_function_result(self.f(x), out)

    def _update_jacobian(self, x: ReadOnlyVector) -> None:
        if self._Df_accepts_out:
            result = self.Df(x, out=self._jacobian)
            if result is None or result is self._jacobian:
                return
        else:
            result = self.Df(x)

        jacobian = np.asarray(result)
        if np.issubdtype(jacobian.dtype, np.complexfloating):
            raise TypeError(Errors.type_error(
                "Df must return the real Jacobian, not a complex derivative."
            ))
        if (
                self._real_input_size == 1
                and self._real_output_size == 1
                and jacobian.shape == self._domain_shape):
            jacobian = jacobian[..., None, None]
        try:
            jacobian = np.broadcast_to(jacobian, self._jacobian.shape)
            np.copyto(self._jacobian, jacobian, casting='same_kind')
        except (TypeError, ValueError) as exc:
            raise ValueError(Errors.value_error(
                f"Df returned shape {np.shape(result)}; expected a real array "
                f"broadcastable to {self._jacobian.shape}."
            )) from exc

    def _real_input(self, x: ReadOnlyVector) -> np.ndarray:
        x_flat = np.reshape(
            x, self._domain_shape + (self._input_size,)
        )
        if not self.domain.is_complex:
            return x_flat
        self._input_real_buffer[..., 0::2] = x_flat.real
        self._input_real_buffer[..., 1::2] = x_flat.imag
        return self._input_real_buffer

    def _real_output(self, y: ReadOnlyVector) -> np.ndarray:
        y_flat = np.reshape(
            y, self._domain_shape + (self._output_size,)
        )
        if not self.codomain.is_complex:
            return y_flat
        self._output_real_buffer[..., 0::2] = y_flat.real
        self._output_real_buffer[..., 1::2] = y_flat.imag
        return self._output_real_buffer

    def _apply_jacobian(self, h: ReadOnlyVector, out: Vector) -> Vector:
        if np.shares_memory(h, out):
            h = h.copy()
        h_real = self._real_input(h)
        out_flat = np.reshape(
            out, self._domain_shape + (self._output_size,)
        )
        if self.codomain.is_complex:
            np.einsum(
                '...oi,...i->...o', self._jacobian[..., 0::2, :],
                h_real, out=out_flat.real, optimize=True
            )
            np.einsum(
                '...oi,...i->...o', self._jacobian[..., 1::2, :],
                h_real, out=out_flat.imag, optimize=True
            )
        else:
            np.einsum(
                '...oi,...i->...o', self._jacobian,
                h_real, out=out_flat, optimize=True
            )
        return out

    def _apply_jacobian_transpose(self, y: ReadOnlyVector, out: Vector) -> Vector:
        if np.shares_memory(y, out):
            y = y.copy()
        y_real = self._real_output(y)
        out_flat = np.reshape(
            out, self._domain_shape + (self._input_size,)
        )
        if self.domain.is_complex:
            np.einsum(
                '...oi,...o->...i', self._jacobian[..., :, 0::2],
                y_real, out=out_flat.real, optimize=True
            )
            np.einsum(
                '...oi,...o->...i', self._jacobian[..., :, 1::2],
                y_real, out=out_flat.imag, optimize=True
            )
        else:
            np.einsum(
                '...oi,...o->...i', self._jacobian,
                y_real, out=out_flat, optimize=True
            )
        return out

    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        source = x.copy() if np.shares_memory(x, out) else x
        if differentiate and not self.linear:
            self._update_jacobian(source)
        return self._apply_function(source, out)

    def _iderivative(self, h: ReadOnlyVector, out: Vector) -> Vector:
        return self._apply_jacobian(h, out)

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        return self._apply_jacobian_transpose(y, out)

    def _iadjoint_eval(self, x: ReadOnlyVector, out: Vector) -> Vector:
        self._ieval(
            x, self._value_buffer, differentiate=not self.linear
        )
        return self._iadjoint(self._value_buffer, out)

    def _iadjoint_derivative(self, h: ReadOnlyVector, out: Vector) -> Vector:
        self._iderivative(h, self._value_buffer)
        return self._iadjoint(self._value_buffer, out)

    @staticmethod
    def complex_matrix_to_real_jacobian(matrix: Scalar | ArrayLike) -> np.ndarray:
        r"""Convert a complex matrix to its real Jacobian representation.
        
        ``matrix`` must have shape ``... + (m, n)``. The returned real array
        has shape ``... + (2*m, 2*n)`` and uses interleaved real/imaginary
        coordinates.
        
        Parameters
        ----------
        matrix: Scalar | ArrayLike
            Matrix.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        matrix = np.asarray(matrix)
        if matrix.ndim < 2:
            raise ValueError("matrix must have at least two dimensions (..., m, n).")
        m, n = matrix.shape[-2:]
        dtype = np.empty((), dtype=matrix.dtype).real.dtype
        result = np.empty(matrix.shape[:-2] + (2 * m, 2 * n), dtype=dtype)
        result[..., 0::2, 0::2] = matrix.real
        result[..., 0::2, 1::2] = -matrix.imag
        result[..., 1::2, 0::2] = matrix.imag
        result[..., 1::2, 1::2] = matrix.real
        return result

    def __repr__(self) -> str:
        return make_repr(
            self, self.domain, self.f, self.Df,
            f"codomain_shape={self.codomain.shape_codomain}",
            f"codomain_dtype={self.codomain.dtype}",
            f"linear={self.linear}",
        )


class Exponential(Operator):
    r"""The pointwise exponential operator.
    
    Parameters
    ----------
    domain: NumPyVectorSpace
        The underlying vector space.
    """

    def __init__(self, domain:NumPyVectorSpace) -> None:
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain, NumPyVectorSpace, add_info="Domain of a Exponential operator needs to be a NumPyVectorSpace!"))
        super().__init__(domain, domain)

    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        np.exp(x, out=out)
        if differentiate:
            # Reuse the value already computed in ``out`` instead of evaluating
            # the exponential a second time.
            self._exponential_factor = out.copy()
        return out

    def _iderivative(self, x: ReadOnlyVector, out: Vector) -> Vector:
        return np.multiply(self._exponential_factor, x, out=out)

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        if np.shares_memory(y, out):
            y = y.copy()
        np.conjugate(self._exponential_factor, out=out)
        return np.multiply(out, y, out=out)


class FourierTransform(Operator):
    r"""Discrete Fourier transform on selected axes of a ``UniformGridFcts``.
    
    Complex-valued domains use :func:`scipy.fft.fftn`.  Real-valued domains
    use :func:`scipy.fft.rfftn`, so only the non-negative frequencies of the
    last transformed axis are stored.
    
    Parameters
    ----------
    domain: UniformGridFcts
        The underlying function space. Its dtype must be floating or complex.
    centered: bool [default: False]
        Center the *frequency* ordering on every full FFT axis.  The compressed
        axis of an RFFT remains ordered from zero to the Nyquist frequency.
    axes: int | tuple[int, ...] | None [default: None]
        Domain axes over which to transform. Negative axes follow NumPy's
        convention. The order matters for real transforms because ``axes[-1]``
        is the compressed RFFT axis.
    centered_input: bool | None [default: None]
        Whether the samples are stored with the spatial zero at the center of
        every transformed axis. If true, an ``ifftshift`` is applied before the
        transform and the corresponding ``fftshift`` after the inverse or
        adjoint transform. ``None`` preserves the historical behavior and uses
        the value of ``centered``. Set this explicitly to ``False`` for grids
        whose zero sample is stored at index zero.
    
    Notes
    -----
    For a real domain the codomain is a :class:`GridFcts` carrying the
    Hermitian multiplicity weights of the stored half-spectrum. Its measure is
    the spatial volume element multiplied by ``1`` at zero and Nyquist
    frequencies and by ``2`` at all interior frequencies of the compressed
    axis. Consequently, if ``G_X`` and ``G_Y`` denote the L2 Gram operators
    induced by the domain and codomain measures, respectively, then
    
    ``G_X**(-1) * F.adjoint * G_Y``
    
    equals ``irfftn`` and is a left inverse of the real transform. The
    unweighted operator adjoint still uses factors ``1`` at zero and Nyquist
    and ``1/2`` at interior frequencies. Since the declared complex
    half-spectrum contains directions outside the RFFT range, the transform is
    not a two-sided isomorphism; :attr:`left_inverse` provides inversion on the
    range.
    """

    def __init__(self, domain: UniformGridFcts, centered: bool = False,
                 axes: int | tuple[int, ...] | None=None, centered_input: bool | None = None) -> None:
        if not isinstance(domain, UniformGridFcts):
            raise TypeError(Errors.not_instance(
                domain, UniformGridFcts,
                add_info="Domain of a FourierTransform operator needs to be a UniformGridFcts!"
            ))

        dtype = np.dtype(domain.dtype)
        if not (np.issubdtype(dtype, np.floating)
                or np.issubdtype(dtype, np.complexfloating)):
            raise TypeError(Errors.type_error(
                "FourierTransform requires a floating or complex domain dtype. "
                f"Got dtype={domain.dtype}."
            ))
        if not isinstance(centered, (bool, np.bool_)):
            raise TypeError(Errors.type_error(
                f"centered must be boolean. Got {centered!r}."
            ))
        if centered_input is None:
            centered_input = bool(centered)
        elif not isinstance(centered_input, (bool, np.bool_)):
            raise TypeError(Errors.type_error(
                f"centered_input must be boolean or None. Got {centered_input!r}."
            ))

        self.axes = self._normalize_axes(domain.ndim_domain, axes)
        if any(domain.spacing[axis] == 0 for axis in self.axes):
            raise ValueError(Errors.value_error(
                "FourierTransform cannot be constructed on a transformed axis "
                "with zero grid spacing."
            ))

        self.is_complex = domain.is_complex
        self.centered = bool(centered)
        self.centered_input = bool(centered_input)
        self._frequency_shift_axes = (
            self.axes if self.is_complex else self.axes[:-1]
        )

        frqs = self.frequencies(
            domain, centered=self.centered, axes=self.axes,
            rfft=not self.is_complex
        )
        complex_dtype = np.result_type(dtype, 1j)

        if not self.is_complex:
            # The RFFT half-spectrum has non-constant Hermitian multiplicity
            # weights and therefore must not be represented as a
            # UniformGridFcts, even when its coordinate axes are uniform.
            codomain = GridFcts(
                *frqs, dtype=complex_dtype,
                shape_codomain=domain.shape_codomain,
                use_cell_measure=False
            )

            n = domain.shape_domain[self.axes[-1]]
            n_half = n // 2 + 1
            real_dtype = np.empty((), dtype=dtype).real.dtype

            adjoint_weights_1d = np.full(
                n_half, 0.5, dtype=real_dtype
            )
            adjoint_weights_1d[0] = 1
            if n % 2 == 0:
                adjoint_weights_1d[-1] = 1

            value_broadcast_shape = [1] * len(codomain.shape)
            value_broadcast_shape[self.axes[-1]] = n_half
            self._rfft_adjoint_weights = adjoint_weights_1d.reshape(
                value_broadcast_shape
            )
            self._rfft_inverse_adjoint_weights = (
                1 / self._rfft_adjoint_weights
            )
            self._rfft_axis = self.axes[-1]
            self._rfft_spatial_size = n
            self._rfft_boundary_indices = (
                (0, n // 2) if n % 2 == 0 else (0,)
            )

            # The Nyquist mode is needed for direct evaluations of the
            # spatial normal operators. It is present only for even sizes.
            self._rfft_nyquist_pattern = None
            if n % 2 == 0:
                self._rfft_nyquist_pattern = (
                    1 - 2 * (np.arange(n) % 2)
                ).astype(real_dtype)

            # At zero and Nyquist frequency of the compressed axis, an RFFT
            # spectrum must be Hermitian with respect to all remaining FFT
            # axes. Store the index maps k -> -k in the actual (possibly
            # centered) frequency ordering.
            self._rfft_negative_frequency_indices = {}
            for axis in self.axes[:-1]:
                size = domain.shape_domain[axis]
                bins = np.rint(spfft.fftfreq(size) * size).astype(int)
                if self.centered:
                    bins = spfft.fftshift(bins)
                index_by_residue = {
                    int(frequency) % size: index
                    for index, frequency in enumerate(bins)
                }
                self._rfft_negative_frequency_indices[axis] = np.asarray([
                    index_by_residue[(-int(frequency)) % size]
                    for frequency in bins
                ], dtype=int)

            # Let A denote the Euclidean real adjoint of rfftn. Then
            # A(y) = irfftn(w_adj*y), where w_adj is 1 at zero/Nyquist
            # and 1/2 at interior frequencies. With the measure below,
            # G_X^{-1} A G_Y = irfftn, since G_X is multiplication by
            # domain.volume_elem and G_Y by domain.volume_elem/w_adj.
            measure_broadcast_shape = [1] * codomain.ndim_domain
            measure_broadcast_shape[self.axes[-1]] = n_half
            rfft_multiplicity = (1 / adjoint_weights_1d).reshape(
                measure_broadcast_shape
            )
            codomain.measure = np.broadcast_to(
                np.asarray(domain.volume_elem, dtype=real_dtype)
                * rfft_multiplicity,
                codomain.shape_domain
            )
        else:
            # A non-centered full FFT axis has wrap-around ordering and
            # therefore cannot in general be represented by
            # UniformGridFcts.
            frequencies_are_uniform = (
                self.centered
                or all(domain.shape_domain[axis] == 1
                       for axis in self.axes)
            )
            if frequencies_are_uniform:
                codomain = UniformGridFcts(
                    *frqs, dtype=complex_dtype,
                    shape_codomain=domain.shape_codomain
                )
            else:
                codomain = GridFcts(
                    *frqs, dtype=complex_dtype,
                    shape_codomain=domain.shape_codomain,
                    use_cell_measure=False
                )

        super().__init__(domain, codomain, linear=True)

    @staticmethod
    def _normalize_axes(ndim: int, axes: int | tuple[int, ...]) -> tuple[int, ...]:
        if axes is None:
            axes = tuple(range(ndim))
        elif isinstance(axes, (int, np.integer)) and not isinstance(
                axes, (bool, np.bool_)):
            axes = (int(axes),)
        else:
            try:
                axes = tuple(axes)
            except TypeError as exc:
                raise TypeError(Errors.type_error(
                    "axes must be an integer or a sequence of integers."
                )) from exc

        if len(axes) == 0:
            raise ValueError(Errors.value_error(
                "FourierTransform needs at least one transformed axis."
            ))

        normalized = []
        for axis in axes:
            if (isinstance(axis, (bool, np.bool_))
                    or not isinstance(axis, (int, np.integer))):
                raise TypeError(Errors.type_error(
                    "FourierTransform axes must be integers. "
                    f"Got axis={axis!r}."
                ))
            axis = int(axis)
            if axis < 0:
                axis += ndim
            if not 0 <= axis < ndim:
                raise ValueError(Errors.value_error(
                    f"Invalid FourierTransform axis {axis}. "
                    f"Expected an axis in [-{ndim}, {ndim})."
                ))
            normalized.append(axis)

        if len(normalized) != len(set(normalized)):
            raise ValueError(Errors.value_error(
                f"FourierTransform axes contain duplicates: {axes}."
            ))
        return tuple(normalized)

    def _eval(self, x: ReadOnlyVector) -> Vector:
        data = (
            spfft.ifftshift(x, axes=self.axes)
            if self.centered_input else x
        )
        if self.is_complex:
            result = spfft.fftn(data, axes=self.axes, norm='ortho')
        else:
            result = spfft.rfftn(data, axes=self.axes, norm='ortho')
        if self.centered and self._frequency_shift_axes:
            result = spfft.fftshift(
                result, axes=self._frequency_shift_axes
            )
        return result.astype(self.codomain.dtype, copy=False)

    def _unshift_frequency_data(self, y: ReadOnlyVector) -> Vector:
        if self.centered and self._frequency_shift_axes:
            return spfft.ifftshift(
                y, axes=self._frequency_shift_axes
            )
        return y

    def _finish_spatial_transform(self, x: ReadOnlyVector) -> Vector:
        if self.centered_input:
            x = spfft.fftshift(x, axes=self.axes)
        return x.astype(self.domain.dtype, copy=False)

    def _inverse_eval(self, y: ReadOnlyVector) -> Vector:
        r"""Apply the inverse FFT, which is only a left inverse for an RFFT.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        spectrum = self._unshift_frequency_data(y)
        if self.is_complex:
            result = spfft.ifftn(
                spectrum, axes=self.axes, norm='ortho'
            )
        else:
            result = spfft.irfftn(
                spectrum,
                s=tuple(self.domain.shape_domain[axis]
                        for axis in self.axes),
                axes=self.axes,
                norm='ortho'
            ).real
        return self._finish_spatial_transform(result)

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        spectrum = self._unshift_frequency_data(y)
        if self.is_complex:
            result = spfft.ifftn(
                spectrum, axes=self.axes, norm='ortho'
            )
        else:
            # irfftn is the inverse of rfftn on its range, not its adjoint with
            # respect to the unrestricted complex half-spectrum. The factor
            # 1/2 compensates for the conjugate frequency that irfftn inserts.
            weighted = np.multiply(
                spectrum, self._rfft_adjoint_weights
            )
            result = spfft.irfftn(
                weighted,
                s=tuple(self.domain.shape_domain[axis]
                        for axis in self.axes),
                axes=self.axes,
                norm='ortho'
            ).real
        return self._finish_spatial_transform(result)

    def _rfft_spatial_normal(self, x: ReadOnlyVector, interior_factor: float, out: Vector) -> Vector:
        r"""Apply an RFFT normal operator directly in the spatial domain.
        
        The corresponding frequency multiplier is ``interior_factor`` at
        interior frequencies and one at zero and Nyquist frequency.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        interior_factor: float
            Interior factor.
        out: Vector
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        axis = self._rfft_axis
        mean = np.mean(x, axis=axis, keepdims=True)

        nyquist_coefficient = None
        if self._rfft_nyquist_pattern is not None:
            moved = np.moveaxis(x, axis, -1)
            nyquist_coefficient = np.einsum(
                '...i,i->...', moved, self._rfft_nyquist_pattern,
                optimize=True
            ) / self._rfft_spatial_size

        np.multiply(x, interior_factor, out=out)
        correction = 1 - interior_factor
        out += correction * mean
        if nyquist_coefficient is not None:
            moved_out = np.moveaxis(out, axis, -1)
            moved_out += (
                correction
                * nyquist_coefficient[..., np.newaxis]
                * self._rfft_nyquist_pattern
            )
        return out

    def _rfft_hermitian_partner(self, boundary: np.ndarray) -> np.ndarray:
        partner = boundary
        for axis, indices in self._rfft_negative_frequency_indices.items():
            partner = np.take(partner, indices, axis=axis)
        return partner

    def _rfft_spectrum_normal(self, y: ReadOnlyVector, interior_factor: float, out: Vector) -> Vector:
        r"""Apply an RFFT normal operator in the stored half-spectrum.
        
        Interior frequencies are scaled directly. The zero and Nyquist
        slices are projected onto their Hermitian subspaces.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        interior_factor: float
            Interior factor.
        out: Vector
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        projected_boundaries = []
        for index in self._rfft_boundary_indices:
            boundary_slice = [slice(None)] * y.ndim
            boundary_slice[self._rfft_axis] = slice(index, index + 1)
            boundary_slice = tuple(boundary_slice)
            boundary = y[boundary_slice]
            partner = self._rfft_hermitian_partner(boundary)
            projected = 0.5 * (boundary + np.conjugate(partner))
            projected_boundaries.append((boundary_slice, projected))

        np.multiply(y, interior_factor, out=out)
        for boundary_slice, projected in projected_boundaries:
            np.copyto(out[boundary_slice], projected)
        return out

    def _adjoint_eval(self, x: ReadOnlyVector) -> Vector:
        if self.is_complex:
            # The allocating API must return an independent result.
            return np.array(x, dtype=self.domain.dtype, copy=True)
        out = self.domain.empty()
        return self._rfft_spatial_normal(x, 0.5, out)

    def _iadjoint_eval(self, x: ReadOnlyVector, out: Vector) -> Vector:
        if self.is_complex:
            if out is not x:
                self.domain.copyto(out, x)
            return out
        return self._rfft_spatial_normal(x, 0.5, out)

    def _eval_adjoint(self, y: ReadOnlyVector) -> Vector:
        r"""Efficiently evaluate ``self(self.adjoint(y))``.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if self.is_complex:
            return np.array(y, dtype=self.codomain.dtype, copy=True)
        out = self.codomain.empty()
        return self._rfft_spectrum_normal(y, 0.5, out)

    def _ieval_adjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        if self.is_complex:
            if out is not y:
                self.codomain.copyto(out, y)
            return out
        return self._rfft_spectrum_normal(y, 0.5, out)

    @staticmethod
    def frequencies(domain: VectorSpaceBase, centered: bool=False, axes: int | tuple[int, ...] | None=None, rfft: bool=False) -> tuple[np.ndarray, ...]:
        r"""Return coordinate axes for the selected discrete Fourier transform.
        
        Parameters
        ----------
        domain: VectorSpaceBase
            Domain vector space.
        centered: bool [default: False]
            Centered.
        axes: int | tuple[int, ...] | None [default: None]
            Axes.
        rfft: bool [default: False]
            Rfft.
        
        
        Returns
        -------
        tuple[np.ndarray, ...]
            Result described above.
        """
        if not isinstance(domain, UniformGridFcts):
            raise TypeError(Errors.not_instance(
                domain, UniformGridFcts,
                add_info="frequencies requires a UniformGridFcts instance."
            ))
        if not isinstance(centered, (bool, np.bool_)):
            raise TypeError(Errors.type_error(
                f"centered must be boolean. Got {centered!r}."
            ))
        if not isinstance(rfft, (bool, np.bool_)):
            raise TypeError(Errors.type_error(
                f"rfft must be boolean. Got {rfft!r}."
            ))

        axes = FourierTransform._normalize_axes(domain.ndim_domain, axes)
        if any(domain.spacing[axis] == 0 for axis in axes):
            raise ValueError(Errors.value_error(
                "Cannot compute FFT frequencies for a zero grid spacing."
            ))

        result = []
        rfft_axis = axes[-1] if rfft else None
        for axis, (size, spacing) in enumerate(
                zip(domain.shape_domain, domain.spacing)):
            if axis not in axes:
                result.append(np.array(domain.axes[axis], copy=True))
            elif axis == rfft_axis:
                result.append(spfft.rfftfreq(size, d=spacing))
            else:
                frequencies = spfft.fftfreq(size, d=spacing)
                if centered:
                    frequencies = spfft.fftshift(frequencies)
                result.append(frequencies)
        return tuple(result)

    @memoized_property
    def left_inverse(self) -> Operator:
        r"""Return the inverse on the range of the transform.
        
        For a real transform this is also the L2 adjoint induced by the
        measures of ``domain`` and ``codomain``, i.e.
        ``G_X**(-1) * self.adjoint * G_Y``.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        if self.is_complex:
            return self.adjoint
        return _FourierTransformLeftInverse(self)

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        if not self.is_complex:
            raise NotImplementedError(Errors.generic_message(
                "A real FourierTransform maps into an unrestricted complex "
                "half-spectrum and therefore has no two-sided inverse. Use "
                "left_inverse when inversion on the RFFT range is intended."
            ))
        return self.adjoint

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.is_complex

    def __repr__(self) -> str:
        return make_repr(
            self, self.domain,
            f"centered={self.centered}",
            f"axes={self.axes}",
            f"centered_input={self.centered_input}"
        )


class _FourierTransformLeftInverse(Operator):
    r"""Left inverse of a real ``FourierTransform`` on its RFFT range.
    
    Parameters
    ----------
    transform: FourierTransform
        Transform.
    """

    def __init__(self, transform: FourierTransform) -> None:
        self.transform = transform
        super().__init__(
            transform.codomain, transform.domain, linear=True
        )
        self._consts.add('transform')

    def _eval(self, y: ReadOnlyVector) -> Vector:
        return self.transform._inverse_eval(y)

    def _adjoint(self, x: ReadOnlyVector) -> Vector:
        result = self.transform._eval(x)
        return np.multiply(
            result, self.transform._rfft_inverse_adjoint_weights
        )

    def _adjoint_eval(self, y: ReadOnlyVector) -> Vector:
        out = self.domain.empty()
        return self.transform._rfft_spectrum_normal(y, 2., out)

    def _iadjoint_eval(self, y: ReadOnlyVector, out: Vector) -> Vector:
        return self.transform._rfft_spectrum_normal(y, 2., out)

    def _eval_adjoint(self, x: ReadOnlyVector) -> Vector:
        r"""Efficiently evaluate ``self(self.adjoint(x))``.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        out = self.codomain.empty()
        return self.transform._rfft_spatial_normal(x, 2., out)

    def _ieval_adjoint(self, x: ReadOnlyVector, out: Vector) -> Vector:
        return self.transform._rfft_spatial_normal(x, 2., out)

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return False


class RealFourierEmbedding(Operator):
    r"""Embed one-dimensional real Fourier coordinates into an RFFT space.
    
    Let ``F`` be a real, one-dimensional :class:`FourierTransform` of
    spatial length :math:`n`.  Its complex half-spectrum is represented by
    :math:`\widehat x_0,\ldots,\widehat x_{\lfloor n/2\rfloor}`.  This
    operator maps a real vector :math:`z\in\mathbb R^n` to a valid
    half-spectrum by
    
    .. math::
        \widehat x_0 &= z_0,\\
        \widehat x_k &= (z_{2k-1}+i z_{2k})/\sqrt{2},\\
        \widehat x_{n/2} &= z_{n-1}\quad\text{if }n\text{ is even}.
    
    The last formula replaces the second formula at the Nyquist frequency.
    Thus zero and Nyquist coefficients are real, while every interior complex
    coefficient is represented by two real coordinates.  The domain is a
    plain real :class:`~regpy.vecsps.NumPyVectorSpace` of dimension ``n``.
    
    :attr:`left_inverse` maps a valid half-spectrum back to real coordinates
    and multiplies the real and imaginary parts of interior coefficients by
    :math:`\sqrt{2}`.  Consequently
    
    .. math::
        R = C^{-1}F
    
    is an orthogonal real Fourier transform with respect to identity Gram
    operators, where ``C`` denotes this embedding.  The embedding itself is
    not onto the unrestricted complex half-spectrum; hence it has no
    two-sided inverse.
    
    Parameters
    ----------
    transform: FourierTransform
        A real Fourier transform of a scalar-valued one-dimensional grid,
        acting on its sole domain axis.
    """

    def __init__(self, transform: FourierTransform) -> None:
        if not isinstance(transform, FourierTransform):
            raise TypeError(Errors.not_instance(
                transform,
                FourierTransform,
                add_info="RealFourierEmbedding requires a FourierTransform."
            ))
        if transform.is_complex:
            raise ValueError(Errors.value_error(
                "RealFourierEmbedding requires a real Fourier transform."
            ))
        if (transform.domain.ndim_domain != 1
                or transform.axes != (0,)
                or transform.domain.shape_codomain != ()):
            raise NotImplementedError(Errors.generic_message(
                "Real Fourier coordinates are currently implemented only "
                "for scalar-valued one-dimensional grids transformed along "
                "their sole domain axis."
            ))

        self.transform = transform
        self.spatial_size = int(transform.domain.shape_domain[0])
        self.half_size = self.spatial_size // 2 + 1
        real_dtype = np.empty(
            (), dtype=transform.domain.dtype
        ).real.dtype
        coordinate_space = NumPyVectorSpace(
            self.spatial_size, dtype=real_dtype
        )
        self._sqrt_two = np.sqrt(np.asarray(2, dtype=real_dtype))
        self._interior_stop = (
            self.half_size - 1
            if self.spatial_size % 2 == 0
            else self.half_size
        )
        self._interior_frequencies = slice(1, self._interior_stop)
        self._real_coordinates = slice(
            1, 2 * self._interior_stop - 1, 2
        )
        self._imag_coordinates = slice(
            2, 2 * self._interior_stop, 2
        )

        super().__init__(
            coordinate_space,
            transform.codomain,
            linear=True,
        )
        self._consts.add('transform')

    @property
    def coordinate_space(self) -> NumPyVectorSpace:
        r"""The real Fourier-coordinate space.
        
        
        Returns
        -------
        NumPyVectorSpace
            Result described above.
        """
        return self.domain

    def _eval(self, coordinates: ReadOnlyVector) -> Vector:
        spectrum = self.codomain.zeros()
        spectrum[0] = coordinates[0]
        spectrum[self._interior_frequencies] = (
            coordinates[self._real_coordinates]
            + 1j * coordinates[self._imag_coordinates]
        ) / self._sqrt_two
        if self.spatial_size % 2 == 0:
            spectrum[-1] = coordinates[-1]
        return spectrum

    def _ieval(self, coordinates: ReadOnlyVector, out: Vector) -> Vector:
        # The spaces have different scalar fields and shapes, so aliasing is
        # impossible for valid arguments.
        out.fill(0)
        out[0] = coordinates[0]
        out[self._interior_frequencies] = (
            coordinates[self._real_coordinates]
            + 1j * coordinates[self._imag_coordinates]
        ) / self._sqrt_two
        if self.spatial_size % 2 == 0:
            out[-1] = coordinates[-1]
        return out

    def _adjoint(self, spectrum: Scalar | ArrayLike) -> Vector:
        coordinates = self.domain.zeros()
        coordinates[0] = spectrum[0].real
        interior = spectrum[self._interior_frequencies]
        coordinates[self._real_coordinates] = (
            interior.real / self._sqrt_two
        )
        coordinates[self._imag_coordinates] = (
            interior.imag / self._sqrt_two
        )
        if self.spatial_size % 2 == 0:
            coordinates[-1] = spectrum[-1].real
        return coordinates

    @memoized_property
    def left_inverse(self) -> Operator:
        r"""Return the coordinate map from a valid RFFT half-spectrum.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return _RealFourierEmbeddingLeftInverse(self)

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        raise NotImplementedError(Errors.generic_message(
            "RealFourierEmbedding is not onto the unrestricted complex "
            "half-spectrum. Use left_inverse on valid RFFT spectra."
        ))

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return False

    def __repr__(self) -> str:
        return make_repr(self, self.transform)

    def real_multiplier(self, half_spectrum_multiplier: np.ndarray) -> np.ndarray:
        r"""Represent a real diagonal half-spectrum multiplier in coordinates.
        
        ``half_spectrum_multiplier`` must be real-valued and have the shape of
        the RFFT half-spectrum.  Every interior value is duplicated for its
        real and imaginary coordinate.  This is particularly useful for the
        non-negative absolute values in a real Fourier SVD.
        
        Parameters
        ----------
        half_spectrum_multiplier: np.ndarray
            Half spectrum multiplier.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        values = np.asarray(half_spectrum_multiplier)
        try:
            values = np.broadcast_to(values, self.codomain.shape)
        except ValueError as error:
            raise ValueError(Errors.value_error(
                "The half-spectrum multiplier must be broadcastable to "
                f"shape {self.codomain.shape}. Got {values.shape}."
            )) from error
        if not np.all(np.isreal(values)):
            raise ValueError(Errors.value_error(
                "A multiplier represented in real Fourier coordinates must "
                "be real-valued."
            ))

        result = self.domain.empty()
        result[0] = values[0].real
        interior = values[self._interior_frequencies].real
        result[self._real_coordinates] = interior
        result[self._imag_coordinates] = interior
        if self.spatial_size % 2 == 0:
            result[-1] = values[-1].real
        return result

    def __repr__(self) -> str:
        return make_repr(self, self.transform)


class _RealFourierEmbeddingLeftInverse(Operator):
    r"""Left inverse of :class:`RealFourierEmbedding` on its range.
    
    Parameters
    ----------
    embedding: RealFourierEmbedding
        Embedding.
    """

    def __init__(self, embedding: RealFourierEmbedding) -> None:
        self.embedding = embedding
        super().__init__(
            embedding.codomain,
            embedding.domain,
            linear=True,
        )
        self._consts.add('embedding')

    def _eval(self, spectrum: Scalar | ArrayLike) -> Vector:
        coordinates = self.codomain.zeros()
        coordinates[0] = spectrum[0].real
        interior = spectrum[self.embedding._interior_frequencies]
        coordinates[self.embedding._real_coordinates] = (
            self.embedding._sqrt_two * interior.real
        )
        coordinates[self.embedding._imag_coordinates] = (
            self.embedding._sqrt_two * interior.imag
        )
        if self.embedding.spatial_size % 2 == 0:
            coordinates[-1] = spectrum[-1].real
        return coordinates

    def _adjoint(self, coordinates: ReadOnlyVector) -> Vector:
        spectrum = self.domain.zeros()
        spectrum[0] = coordinates[0]
        spectrum[self.embedding._interior_frequencies] = (
            self.embedding._sqrt_two
            * (
                coordinates[self.embedding._real_coordinates]
                + 1j * coordinates[self.embedding._imag_coordinates]
            )
        )
        if self.embedding.spatial_size % 2 == 0:
            spectrum[-1] = coordinates[-1]
        return spectrum

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return False


class RealFourierTransform(Composition):
    r"""Orthogonal one-dimensional Fourier transform in real coordinates.
    
    This is the composition ``C.left_inverse * F`` of a real
    :class:`FourierTransform` ``F`` and the coordinate map associated with a
    :class:`RealFourierEmbedding` ``C``.  Its domain is the original real grid
    and its codomain is a real vector space of the same dimension.  Hence its
    adjoint is its inverse.
    
    Parameters are the same as for :class:`FourierTransform`, subject to the
    current restriction to scalar-valued one-dimensional grids and ``axes=0``.
    
    Parameters
    ----------
    domain: UniformGridFcts
        Domain vector space.
    centered: bool [default: False]
        Centered.
    axes: int | tuple[int, ...] | None [default: None]
        Axes.
    centered_input: bool | None [default: None]
        Centered input.
    """

    def __init__(self, domain: UniformGridFcts, centered: bool = False,
                 axes: int | tuple[int, ...] | None=None, centered_input: bool | None = None) -> None:
        transform = FourierTransform(
            domain,
            centered=centered,
            axes=axes,
            centered_input=centered_input,
        )
        embedding = RealFourierEmbedding(transform)
        self.fourier_transform = transform
        self.embedding = embedding
        super().__init__(embedding.left_inverse, transform)
        self._consts.update({'fourier_transform', 'embedding'})

    @property
    def coordinate_space(self) -> NumPyVectorSpace:
        r"""The real Fourier-coordinate space.
        
        
        Returns
        -------
        NumPyVectorSpace
            Result described above.
        """
        return self.codomain

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return self.adjoint

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return True

    def __repr__(self) -> str:
        return make_repr(
            self,
            self.domain,
            f"centered={self.fourier_transform.centered}",
            f"centered_input={self.fourier_transform.centered_input}",
        )


class PtwMatrixVectorMultiplication(Operator):
    r"""Pointwise multiplication of a matrix-valued function with a vector-valued function.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        The input grid function.
    matrixfct: np.ndarray
        The matrix-valued function to multiply with the vector-valued function.  
        The first dimensions must match the shape_domain of domain, the last dimensions  
        must match the shape_codomain of domain, and the middle dimensions define the output shape_codomain.
    """
    def __init__(self,domain: VectorSpaceBase,matrixfct: np.ndarray) -> None:
        if not isinstance(domain, MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts, add_info='The domain of a PtwMultiplication must be at least of type MeasureSpaceFcts.'))
        domain_shape = domain.shape_domain
        codomain_shape = domain.shape_codomain

        if not isinstance(matrixfct,np.ndarray) or not matrixfct.dtype==domain.dtype:
            raise TypeError(Errors.type_error(f'To initialize a PtwMatrixVectorMultiplication the matrix functions matrixfct must be a NumPy array of the same data type. Was given\n\t matrixfct = {matrixfct}\n\t and matrixfcts.dtype={matrixfct.dtype} and domain.dtype = {domain.dtype}'))
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

    def _ieval(self, v: ReadOnlyVector, out: Vector) -> Vector:
        return  np.einsum(self._einstein_string_mul, self.matrixfct, v, out = out)
    
    def _iadjoint(self, w: Scalar | ArrayLike, out: Vector) -> Vector:
        return np.einsum(self._einstein_string_mulT, np.conj(self.matrixfct), w, out = out)

    def __repr__(self) -> str:
        return make_repr(self, self.domain, self.codomain)

class PtwScalarMultiplication(Operator):
    r"""Pointwise multiplication of a scalar valued function with a vector-valued function.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        The input grid function.
    multiplier: Scalar | ArrayLike
        Multiplier.
    """   

    def __init__(self, domain: VectorSpaceBase, multiplier: Scalar | ArrayLike) -> None:
        if not isinstance(domain,MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info=f'For a PtwScalarMultiplication the domain must be of type MeasureSpaceFcts. Got \n\t domain = {domain}'))
        if domain.ndim_codomain==0:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the domain must be vector valued!'))
        scalar_space = domain.scalar_space()
        if (isinstance(multiplier, np.ndarray)
                and multiplier.shape == domain.shape_domain + (1,)*domain.ndim_codomain
                and multiplier.reshape(domain.shape_domain) in scalar_space):
            self.multiplier = multiplier
        elif multiplier in scalar_space:
            self.multiplier = np.reshape(multiplier,domain.shape_domain+(1,)*domain.ndim_codomain)
        else:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the multiplier must be NumPy array of matching size. Got \n\t domain = {domain}, \n\t multiplier = {multiplier}'))
        
        
        super().__init__(domain, domain, linear=True)

    def _ieval(self, f: ReadOnlyVector, out: Vector) -> Vector:
        return np.multiply(f,self.multiplier,out = out)
    
    def _iadjoint(self, g: ReadOnlyVector, out: Vector) -> Vector:
        return np.multiply(g,self.multiplier.conj(), out = out)

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return PtwScalarDivision(self.domain,self.multiplier)

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return np.all(self.multiplier!=0)

class PtwScalarDivision(Operator):
    r"""Pointwise division of a vector-valued function by a scalar valued function
    
    Parameters
    ----------
    domain: VectorSpaceBase
        The input grid function.
    divisor: Scalar | ArrayLike
        Divisor.
    """   

    def __init__(self, domain: VectorSpaceBase, divisor: Scalar | ArrayLike) -> None:
        if not isinstance(domain,MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info=f'For a PtwScalarMultiplication the domain must be of type MeasureSpaceFcts. Got \n\t domain = {domain}'))
        if domain.ndim_codomain==0:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the domain must be vector valued!'))
        scalar_space = domain.scalar_space()
        if (isinstance(divisor, np.ndarray)
                and divisor.shape == domain.shape_domain + (1,)*domain.ndim_codomain
                and divisor.reshape(domain.shape_domain) in scalar_space):
            self.divisor = divisor
        elif divisor in scalar_space:
            self.divisor = np.reshape(divisor,domain.shape_domain+(1,)*domain.ndim_codomain)
        else:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the divisor must be NumPy array of matching size. Got \n\t domain = {domain}, \n\t divisor = {divisor}'))
        if np.any(self.divisor==0):
            raise ValueError(Errors.value_error('Divisor must not vanish anywhere.'))
        
        super().__init__(domain, domain, linear=True)

    def _ieval(self, f: ReadOnlyVector, out: Vector) -> Vector:
        return np.divide(f, self.divisor, out=out)

    def _iadjoint(self, g: ReadOnlyVector, out: Vector) -> Vector:
        return np.divide(g, self.divisor.conj(), out=out)

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return PtwScalarMultiplication(self.domain,self.divisor)

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return True

class AddSingletonVectorDimension(Operator):
    r"""Operator that adds a singleton dimension as codimension in `regpy.vecsps.MeasureSpaceFcts`. 
    Wrapper to np.reshape(...,1).
    
    Parameters
    ----------
    domain: VectorSpaceBase
        Domain vector space.
    """
    def __init__(self, domain: VectorSpaceBase) -> None:
        if not isinstance(domain, MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info=f'The VectorSpace must be of type MeasureSpaceFcts. Got {type(domain)}'))
        if domain.ndim_codomain != 0:
            raise ValueError(Errors.value_error(f'TO add a SignletonVector dimension the domain must be scalar-valued. Got shape_codomain = {domain.shape_codomain}'))
        self.shape_domain = domain.shape_domain
        super().__init__(domain, domain.vector_valued_space((1,)), linear=True)

    def _eval(self,f: ReadOnlyVector) -> Vector:
        return np.expand_dims(f, axis=-1)
    
    def _adjoint(self,f: ReadOnlyVector) -> Vector:
        return np.squeeze(f, axis=-1)
    
    @property
    def inverse(self) -> Operator:
        r"""Inverse.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return self.adjoint

    @property
    def invertible(self) -> bool:
        r"""Invertible.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return True   

class ForwardFDGradient(Operator):
    r"""Forward finite-difference gradient on ``UniformGridFcts``.

    For scalar functions the codomain has point-value shape ``(d,)``.
    For vector-valued functions with point-value shape ``C`` the codomain
    has shape ``C + (d,)``, where ``d = domain.ndim_domain``. Thus the
    final axis indexes spatial derivatives, while any preceding point-value
    axes (for example RGB channels) are preserved. All differences act only
    along spatial axes.

    Parameters
    ----------
    domain: UniformGridFcts
        Grid of scalar- or vector-valued functions.
    boundary_condition: str [default: 'Neum']
        'Neum', 'Diri', or 'per', for the right boundary of each spatial axis.
    """

    def __init__(self, domain: VectorSpaceBase,
                 boundary_condition: str = 'Neum') -> None:
        if not isinstance(domain, UniformGridFcts):
            raise TypeError('ForwardFDGradient only implemented for UniformGridFcts')
        if boundary_condition not in ('Neum', 'Diri', 'per'):
            raise ValueError("boundary_condition must be 'Neum', 'Diri', or 'per'.")
        self.boundary_condition = boundary_condition
        self.spatial_ndim = domain.ndim_domain
        self._value_slices = (slice(None),) * domain.ndim_codomain
        super().__init__(
            domain,
            domain.vector_valued_space(
                domain.shape_codomain + (self.spatial_ndim,)),
            linear=True,
        )

    def _ieval(self, v: ReadOnlyVector, out: Vector) -> Vector:
        out.fill(0.)
        value_slices = self._value_slices
        d = self.spatial_ndim
        for ax, h in enumerate(self.domain.spacing):
            mid = [slice(None)] * d
            mid[ax] = slice(0, -1)
            plus = mid.copy()
            plus[ax] = slice(1, None)
            out[tuple(mid) + value_slices + (ax,)] = (
                v[tuple(plus) + value_slices]
                - v[tuple(mid) + value_slices]
            ) / h

            if self.boundary_condition in ('Diri', 'per'):
                last = [slice(None)] * d
                last[ax] = -1
                out[tuple(last) + value_slices + (ax,)] = (
                    -v[tuple(last) + value_slices] / h
                )
                if self.boundary_condition == 'per':
                    first = last.copy()
                    first[ax] = 0
                    out[tuple(last) + value_slices + (ax,)] += (
                        v[tuple(first) + value_slices] / h
                    )
        return out

    def _iadjoint(self, v: ReadOnlyVector, out: Vector) -> Vector:
        out.fill(0.)
        value_slices = self._value_slices
        d = self.spatial_ndim
        for ax, h in enumerate(self.domain.spacing):
            mid = [slice(None)] * d
            mid[ax] = slice(1, -1)
            minus = mid.copy()
            minus[ax] = slice(None, -2)
            out[tuple(mid) + value_slices] += (
                v[tuple(minus) + value_slices + (ax,)]
                - v[tuple(mid) + value_slices + (ax,)]
            ) / h

            first = [slice(None)] * d
            first[ax] = 0
            out[tuple(first) + value_slices] -= (
                v[tuple(first) + value_slices + (ax,)] / h
            )
            if self.boundary_condition == 'per':
                last_as_first = first.copy()
                last_as_first[ax] = -1
                out[tuple(first) + value_slices] += (
                    v[tuple(last_as_first) + value_slices + (ax,)] / h
                )

            last = [slice(None)] * d
            last[ax] = -1
            penultimate = last.copy()
            penultimate[ax] = -2
            out[tuple(last) + value_slices] += (
                v[tuple(penultimate) + value_slices + (ax,)] / h
            )
            if self.boundary_condition in ('Diri', 'per'):
                out[tuple(last) + value_slices] -= (
                    v[tuple(last) + value_slices + (ax,)] / h
                )
        return out

    def matrices(self) -> list[scipy.sparse.csc_matrix]:
        r"""Sparse spatial partial-derivative matrices on flattened domain data.

        Each matrix has shape ``(domain.dimension, domain.dimension)``.
        For vector-valued data, the spatial difference is tensored with an
        identity on the flattened point-value axes (channel-first and
        derivative-first coefficients are never mixed).
        """
        d = self.spatial_ndim
        shape = self.domain.shape_domain
        n_spatial = int(np.prod(shape))
        matrices = [lil_matrix((n_spatial, n_spatial)) for _ in range(d)]
        for ax, h in enumerate(self.domain.spacing):
            for idx in np.ndindex(shape):
                row = np.ravel_multi_index(idx, shape)
                if idx[ax] < shape[ax] - 1:
                    matrices[ax][row, row] = -1 / h
                    neighbor = list(idx)
                    neighbor[ax] += 1
                    matrices[ax][row, np.ravel_multi_index(neighbor, shape)] = 1 / h
                elif self.boundary_condition in ('Diri', 'per'):
                    matrices[ax][row, row] = -1 / h
                    if self.boundary_condition == 'per':
                        neighbor = list(idx)
                        neighbor[ax] = 0
                        matrices[ax][row, np.ravel_multi_index(neighbor, shape)] = 1 / h
        result = [matrix.tocsc() for matrix in matrices]
        n_values = int(np.prod(self.domain.shape_codomain))
        if n_values > 1:
            from scipy.sparse import eye, kron
            identity = eye(n_values, format='csc')
            result = [kron(matrix, identity, format='csc') for matrix in result]
        return result

    def norm(self, h_domain: HilbertSpace | None = None,
             h_codomain: HilbertSpace | None = None,
             method: str | None = None,
             codomain_free: bool = False) -> float:
        r"""Analytic upper bound in the standard L2 geometry.

        The bound is independent of the number or shape of value channels,
        because the gradient acts on each component identically.
        """
        from regpy.hilbert import L2
        if ((h_domain is None or h_domain == L2)
                and (h_codomain is None or h_codomain == L2)
                and method is None):
            return 2 * np.linalg.norm(1. / self.domain.spacing)
        return super().norm(
            h_domain=h_domain, h_codomain=h_codomain,
            method=method, codomain_free=codomain_free,
        )

class OuterProduct(Operator):
    r"""The operator \(x_1,x_2,\dots,x_n \mapsto x_1\otimes x_2\otimes\dots\otimes x_n\).
    
    Parameters
    ----------
    *domains: NumPyVectorSpace
        The underlying vector spaces. Have to be all real or all complex.
    """

    def __init__(self,*domains: NumPyVectorSpace) -> None:
        domain=DirectSum(*domains)
        codomain=Prod(*domains)
        super().__init__(domain, codomain, linear=False)
        self._adjoint_summation_strings=self._calc_adjoint_summation_strings()
        """List of strings for calculation of adjoint in einsum."""
        characters=tuple(chr(k) for k in range(ord('A'),ord('A')+self.codomain.ndim))
        self._eval_summation_string = ",".join(characters) + "->" + "".join(characters)
        """String for writing the outer product directly into an output array."""
        self._consts = {*self._consts,"_p","_p_conj_flat"}

    def _calc_adjoint_summation_strings(self) -> list[str]:
        li=[]
        characters=tuple(chr(k) for k in range(ord('A'),ord('A')+self.codomain.ndim))
        all_characters="".join(characters)
        for i,_ in enumerate(characters):
            without_i=",".join(characters[:i]+characters[i+1:])
            li.append(f"{all_characters},{without_i}")
        return li

    def _eval(self, x: ReadOnlyVector, differentiate: bool=False) -> Vector:
        self._p=self.domain.split(x)
        if(differentiate):
            self._p_conj_flat=tuple(np.conj(p_j).flat for p_j in self._p)
        return self.codomain.product(*self._p)

    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        self._p = self.domain.split(x)
        if differentiate:
            self._p_conj_flat = tuple(np.conj(p_j).flat for p_j in self._p)
        operands = tuple(
            p_j.copy().reshape(-1) if np.shares_memory(p_j, out) else p_j.reshape(-1)
            for p_j in self._p
        )
        return np.einsum(
            self._eval_summation_string,
            *operands,
            optimize=True,
            out=out
        )

    def _iderivative(self, x: ReadOnlyVector, out: Vector) -> Vector:
        x_components = tuple(
            x_j.copy() if np.shares_memory(x_j, out) else x_j
            for x_j in self.domain.split(x)
        )
        flat_p = tuple(
            p_j.copy().reshape(-1) if np.shares_memory(p_j, out) else p_j.reshape(-1)
            for p_j in self._p
        )
        term = None
        for j, x_j in enumerate(x_components):
            operands = (*flat_p[:j], x_j.reshape(-1), *flat_p[j+1:])
            if j == 0:
                np.einsum(
                    self._eval_summation_string,
                    *operands,
                    optimize=True,
                    out=out
                )
            else:
                if term is None:
                    term = np.empty_like(out)
                np.einsum(
                    self._eval_summation_string,
                    *operands,
                    optimize=True,
                    out=term
                )
                np.add(out, term, out=out)
        return out

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        xs=[]
        for j,s in enumerate(self._adjoint_summation_strings):
            x_j=np.einsum(s,y,*self._p_conj_flat[:j],*self._p_conj_flat[j+1:],optimize=True)
            xs.append(x_j.reshape(self.domain[j].shape))
        return self.domain.join(*xs)

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        out_components = self.domain.split(out)
        if any(np.shares_memory(y, out_j) for out_j in out_components):
            y = y.copy()
        for j, (s, out_j) in enumerate(
            zip(self._adjoint_summation_strings, out_components)
        ):
            np.einsum(
                s,
                y,
                *self._p_conj_flat[:j],
                *self._p_conj_flat[j+1:],
                optimize=True,
                out=out_j.reshape(-1)
            )
        return out


class EinSum(Operator):
    r"""Operator that computes the numpy.einsum(subscripts,x1,x2,...,xn,t1,t2,...tn) for inputs x1...xn and fixed NumPy arrays t1...tn.
    Nearly all accepted subscripts are supported except ellipses and subscripts resulting in scalar values.
    
    Parameters
    ----------
    subscripts: str
        Subscripts used for the evaluation in numpy.einsum. See the documentation of numpy.einsum for further information.
    tensors: tuple[Any, ...] [default: ()]
        Tuple of constant NumPy arrays.
    codomain: VectorSpaceBase | None [default: None]
        Codomain of the operator. If it is not specified it is computed based on the input data.
    *domains: GridFcts
        The underlying vector spaces.
    """


    def __init__(self, subscripts: str,*domains: GridFcts,tensors: tuple[Any, ...]=(),codomain: VectorSpaceBase | None=None) -> None:
        EinSum._check_parameter_validity(subscripts,*domains,tensors=tensors)
        self.indoms,self.inconsts,self.out=EinSum._get_standard_subscript_info(subscripts,len(domains))
        '''Split information from subscripts corresponding to domains, tensors and output'''
        self.subscripts=subscripts
        '''Subscripts used in einsum in evaluation'''
        self.tensors=tensors
        '''Constant tensors'''
        self._adjoint_subscripts=self._calc_adjoint_subscripts()
        '''Information necessary for computation of adjoints via einsum'''
        if(codomain is None):
            codomain=self._calc_codomain(*domains)
        if(len(domains)==1):
            super().__init__(domains[0], codomain, linear=True)
        else:
            super().__init__(DirectSum(*domains), codomain, linear=False)
        self.conj_tensors=tuple(np.conj(tensor) for tensor in tensors)
        self._eval_opt,self._adjoint_opt=self._optimize_einsum_paths()
        '''Optimal paths for computation of einsum in evaluation and evaluation of adjoint.'''

    @staticmethod
    def _get_standard_subscript_info(subscripts: str,n_domains: int) -> tuple[Any, Any, Any]:
        in_out_split=subscripts.split("->")
        ins=in_out_split[0]
        if(len(in_out_split)==2):
            out=in_out_split[1]
        else:
            fullset=set()
            doubleset=set()
            for s in ins:
                if(s!=","):
                    if(s in fullset):
                        doubleset.add(s)
                    else:
                        fullset.add(s)
            out="".join(sorted(fullset.difference(doubleset)))
            if(out==""):
                raise ValueError(f"Invalid subscripts \'{subscripts}\'. Operations with scalar results are currently not supported.")
        fullins=ins.split(",")
        return fullins[:n_domains],fullins[n_domains:],out
    
    @staticmethod
    def _check_parameter_validity(subscripts: str,*domains: NumPyVectorSpace,tensors: tuple[np.ndarray, ...]) -> bool:
        if("." in subscripts):
            raise ValueError(f"EinSum operator currently does not support ellipses, so input:\'{subscripts}\' is invalid.")
        #TODO add further tests

    def _calc_codomain(self,*domains: NumPyVectorSpace) -> NumPyVectorSpace:
        axesinfo=[]
        for s in self.out:
            index=-1
            j=0
            while index==-1 and j<len(self.indoms):
                index=self.indoms[j].find(s)
                j+=1
            if(index!=-1):
                axesinfo.append(domains[j-1].axes[index])
            else:
                j=0
                while index==-1 and j<len(self.inconsts):
                    index=self.inconsts[j].find(s)
                    j+=1
                if(index!=-1):
                    axesinfo.append(self.tensors[j-1].shape[index])
                else:
                    raise ValueError(f"Output with index \'{s}\' could not be matched to input.")
        return GridFcts(*axesinfo,dtype=np.result_type(*(domain.dtype for domain in domains),*self.tensors))
    
    def _calc_adjoint_subscripts(self) -> list[tuple[bool, str, str]]:
        adj_subscripts=[]
        if(len(self.inconsts)!=0):
            constant_part=",".join(self.inconsts)+","
        else:
            constant_part=""
        if(len(self.indoms)==1):
            info=EinSum._calc_adjoint_assignment_info(f"{constant_part}{self.out}",self.indoms[0])
            return [info]
        for j,input in enumerate(self.indoms):
            indom_scr=",".join(self.indoms[:j]+self.indoms[j+1:])
            info=EinSum._calc_adjoint_assignment_info(f"{indom_scr},{constant_part}{self.out}",input)
            adj_subscripts.append(info)
        return adj_subscripts
    
    @staticmethod
    def _calc_adjoint_assignment_info(ins_str: str,out_str: str) -> tuple[bool, Any, Any]:
        outset=set(out_str)
        insset=set(ins_str.replace(",",""))
        if(len(out_str)==len(outset) and outset.issubset(insset)):
            return True,f"{ins_str}->{out_str}",f"{out_str}->{out_str}"
        unknowns="".join(outset.difference(insset))
        knowns="".join(outset.difference(unknowns))
        return False,f"{ins_str}->{knowns}",f"{out_str}->{unknowns}{knowns}"
        
    def _optimize_einsum_paths(self) -> tuple[Any, Any]:
        inputs=self.domain.zeros()
        output=self.codomain.zeros()
        if(self.linear):
            eval_opt=np.einsum_path(self.subscripts,inputs,*self.tensors)[0]
            adjoint_opt=np.einsum_path(self._adjoint_subscripts[0][1],*self.tensors,output)[0]
            return eval_opt,[adjoint_opt]
        eval_opt=np.einsum_path(self.subscripts,*inputs,*self.tensors)[0]
        adjoint_opt=[]
        for j,s in enumerate(self._adjoint_subscripts):
            opt_j=np.einsum_path(s[1],*inputs[:j],*inputs[j+1:],*self.tensors,output)[0]
            adjoint_opt.append(opt_j)
        return eval_opt,adjoint_opt

    
    def _ieval(self, x: ReadOnlyVector, out: Vector, differentiate: bool=False) -> Vector:
        self._p = (x,) if self.linear else self.domain.split(x)
        if differentiate:
            self._p_conj = tuple(np.conj(p_j) for p_j in self._p)
        operands = tuple(
            p_j.copy() if np.shares_memory(p_j, out) else p_j
            for p_j in self._p
        )
        return np.einsum(
            self.subscripts,
            *operands,
            *self.tensors,
            optimize=self._eval_opt,
            out=out
        )

    def _iderivative(self, x: ReadOnlyVector, out: Vector) -> Vector:
        x_components = (x,) if self.linear else self.domain.split(x)
        x_components = tuple(
            x_j.copy() if np.shares_memory(x_j, out) else x_j
            for x_j in x_components
        )
        evaluation_point = tuple(
            p_j.copy() if np.shares_memory(p_j, out) else p_j
            for p_j in self._p
        )
        term = None
        for j, x_j in enumerate(x_components):
            operands = (
                *evaluation_point[:j],
                x_j,
                *evaluation_point[j+1:],
                *self.tensors
            )
            if j == 0:
                np.einsum(
                    self.subscripts,
                    *operands,
                    optimize=self._eval_opt,
                    out=out
                )
            else:
                if term is None:
                    term = np.empty_like(out)
                np.einsum(
                    self.subscripts,
                    *operands,
                    optimize=self._eval_opt,
                    out=term
                )
                np.add(out, term, out=out)
        return out

    def _adjoint(self, y: ReadOnlyVector) -> Vector:
        if(self.linear):
            x=np.einsum(self._adjoint_subscripts[0][1],*self.conj_tensors,y,optimize=self._adjoint_opt[0])#np.conj
            if(np.issubdtype(self.domain.dtype, np.floating)):
                x=np.real(x)
            if(self._adjoint_subscripts[0][0]):
                return x
            x_full=self.domain.zeros()
            np.einsum(self._adjoint_subscripts[0][2],x_full)[:]=x
            return x_full
        xs=[]
        for j,s in enumerate(self._adjoint_subscripts):
            x_j=np.einsum(s[1],*self._p_conj[:j],*self._p_conj[j+1:],*self.conj_tensors,y,optimize=self._adjoint_opt[j])
            if(np.issubdtype(self.domain[j].dtype, np.floating)):
                x_j=np.real(x_j)
            if(s[0]):
                xs.append(x_j)
            else:
                x_j_full=self.domain[j].zeros()
                np.einsum(s[2],x_j_full)[:]=x_j
                xs.append(x_j_full)
        return self.domain.join(*xs)

    @staticmethod
    def _write_adjoint_component(subscripts: str, operands: list[np.ndarray], optimize: bool | str | list[Any], out: Vector) -> None:
        if not subscripts[0]:
            # Missing output indices have to be introduced by broadcasting.
            # ``einsum(..., out=target)`` requires an exact shape match, so this
            # branch necessarily retains the reduced temporary.
            out.fill(0)
            target = np.einsum(subscripts[2], out)
            value = np.einsum(
                subscripts[1],
                *operands,
                optimize=optimize
            )
            if np.issubdtype(out.dtype, np.floating):
                value = np.real(value)
            np.copyto(target, value)
            return

        if np.issubdtype(out.dtype, np.floating):
            value = np.einsum(
                subscripts[1],
                *operands,
                optimize=optimize
            )
            np.copyto(out, np.real(value))
        else:
            np.einsum(
                subscripts[1],
                *operands,
                optimize=optimize,
                out=out
            )

    def _iadjoint(self, y: ReadOnlyVector, out: Vector) -> Vector:
        if self.linear:
            if np.shares_memory(y, out):
                y = y.copy()
            self._write_adjoint_component(
                self._adjoint_subscripts[0],
                (*self.tensors, y),
                self._adjoint_opt[0],
                out
            )
            return out

        out_components = self.domain.split(out)
        if any(np.shares_memory(y, out_j) for out_j in out_components):
            y = y.copy()
        for j, (subscripts, out_j) in enumerate(
            zip(self._adjoint_subscripts, out_components)
        ):
            self._write_adjoint_component(
                subscripts,
                (
                    *self._p_conj[:j],
                    *self._p_conj[j+1:],
                    *self.tensors,
                    y
                ),
                self._adjoint_opt[j],
                out_j
            )
        return out

