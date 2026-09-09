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
    matrix : array-like
        The matrix.
    inverse : Operator, array-like, 'inv', 'cholesky' or None, optional
        How to implement the inverse operator. If available, this should be given as `Operator`
        or array. If `inv`, `numpy.linalg.inv` will be used. If `cholesky` or `superLU`, a
        `CholeskyInverse` or `SuperLU` instance will be returned.
    domain, codomain : regpy.vecsps.VectorSpaceBase, optional
        Spaces represented by the matrix. Real spaces use their
        :meth:`~regpy.vecsps.VectorSpaceBase.flatten` and ``fromflat``
        coordinates. Complex spaces implementing ``complex_flatten`` and
        ``complex_fromflat`` may instead use genuinely complex coordinates,
        avoiding realification. If omitted, one-dimensional NumPy spaces
        matching the matrix dimensions are constructed.
    dtype: type or None, optional
        Data type. 
        
    Notes
    -----
    The matrix multiplication is done by applying numpy.dot to the matrix and an element of the domain. 
    The adjoint is implemented in the same way by multiplying with the adjoint matrix.
    Multidimensional NumPy spaces and arbitrary real vector spaces are
    supported through their flattened coordinates. A genuinely complex
    representation is available for backends implementing
    ``complex_flatten`` and ``complex_fromflat``; other complex backends can
    still be represented by a real matrix of shape
    ``(codomain.realsize, domain.realsize)``.
    """

    def __init__(self, 
                 matrix:np.ndarray|csc_matrix|csc_array, 
                 inverse=None, 
                 domain:VectorSpaceBase|None=None, 
                 codomain:VectorSpaceBase|None=None,
                 dtype=None):
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
        
        complex_shape = (int(codomain.size), int(domain.size))
        real_shape = (int(codomain.realsize), int(domain.realsize))
        complex_coordinates = (
            domain.is_complex
            and codomain.is_complex
            and matrix.shape == complex_shape
        )
        if complex_coordinates:
            pass
        elif matrix.shape == real_shape:
            matrix_values = (
                matrix if isinstance(matrix, np.ndarray) else matrix.data
            )
            if not np.all(np.isreal(matrix_values)):
                raise ValueError(Errors.value_error(
                    "A matrix acting on real flattened coordinates must be "
                    "real-valued."
                ))
        else:
            raise ValueError(Errors.value_error(
                f"Matrix shape {matrix.shape} is incompatible with domain "
                f"and codomain coordinates; expected real shape "
                f"{real_shape}" + (
                    f" or complex shape {complex_shape}."
                    if domain.is_complex and codomain.is_complex
                    else "."
                )
            ))
        self.matrix = matrix
        self.complex_coordinates = complex_coordinates

        def domain_coordinates(vector):
            if self.complex_coordinates:
                return np.asarray(domain.complex_flatten(vector)).reshape(-1)
            return np.asarray(domain.flatten(vector)).reshape(-1)

        def codomain_coordinates(vector):
            if self.complex_coordinates:
                return np.asarray(codomain.complex_flatten(vector)).reshape(-1)
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

    def _eval(self, x):
        result = np.asarray(self.matvec(x))
        if self.complex_coordinates:
            return self.codomain.complex_fromflat(result)
        return self.codomain.fromflat(result)

    def _ieval(self, x, out):
        self.codomain.copyto(out, self._eval(x))
        return out

    def _adjoint(self, y):
        result = np.asarray(self.rmatvec(y))
        if self.complex_coordinates:
            return self.domain.complex_fromflat(result)
        return self.domain.fromflat(result)

    def _iadjoint(self, y, out):
        self.domain.copyto(out, self._adjoint(y))
        return out
        
    def _adjoint_eval(self, x):
        if hasattr(self,'_MTM'):
            matrix = self._MTM
        else:
            self._MTM = self.matrix.conj().T @ self.matrix
            matrix = self._MTM
        result = np.asarray(matrix @ self._domain_coordinates(x))
        if self.complex_coordinates:
            return self.domain.complex_fromflat(result)
        return self.domain.fromflat(result)

    @memoized_property
    def inverse(self):
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

    def __repr__(self):
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
    domain : regpy.vecsps.NumPyVectorSpace
        Complex NumPy vector space to be realified.
    axis : int [default: -1]
        Position of the new real/imaginary axis in the codomain.
    """

    def __init__(self, domain: NumPyVectorSpace, axis: int = -1):
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

    def _eval(self, x):
        # The utility may return a view.  Allocation-returning Operator
        # methods must return storage independent of their arguments.
        return np.array(
            complex2real_array(x, axis=self.axis),
            dtype=self.codomain.dtype,
            copy=True,
        )

    def _ieval(self, x, out):
        np.copyto(out, complex2real_array(x, axis=self.axis))
        return out

    def _adjoint(self, y):
        return np.array(
            real2complex_array(y, axis=self.axis),
            dtype=self.domain.dtype,
            copy=True,
        )

    def _iadjoint(self, y, out):
        np.copyto(out, real2complex_array(y, axis=self.axis))
        return out

    @Operator.inverse.getter
    def inverse(self):
        return self.adjoint

    @property
    def invertible(self):
        return True

    def __repr__(self):
        return make_repr(self, self.domain, f"axis={self.axis}")


# Lower-case alias matching the utility-function terminology and allowing
# ``complex2real(space)`` as a constructor.
complex2real = Complex2Real


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
    def __init__(self,
                op:Operator,
                matrix: np.ndarray | None =None
                ):
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

    def _ieval(self, x, out):
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

    def _adjoint(self, x):
        return self._eval(x)

    def _iadjoint(self, x, out):
        return self._ieval(x, out)
    
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
    def __init__(self,op:MatrixMultiplication):
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

    def _eval(self,x):
        if np.issubdtype(self.lu.U.dtype,np.complexfloating):
            return self.lu.solve(x)
        else: 
            if np.isrealobj(x):
                return self.lu.solve(x)
            else:
                return self.lu.solve(x.real) + 1j*self.lu.solve(x.imag) 

    def _ieval(self, x, out):
        if np.issubdtype(self.lu.U.dtype, np.complexfloating) or np.isrealobj(x):
            np.copyto(out, self.lu.solve(x))
        else:
            # A real SuperLU factorization cannot consume a complex right-hand
            # side directly. Fill the two components without constructing the
            # additional ``real + 1j * imag`` result array.
            out.real[...] = self.lu.solve(x.real)
            out.imag[...] = self.lu.solve(x.imag)
        return out

    def _adjoint(self, x):
        if np.issubdtype(self.lu.U.dtype, np.complexfloating) or np.isrealobj(x):
            return self.lu.solve(x, trans='H')
        return (
            self.lu.solve(x.real, trans='H')
            + 1j * self.lu.solve(x.imag, trans='H')
        )

    def _iadjoint(self, x, out):
        if np.issubdtype(self.lu.U.dtype, np.complexfloating) or np.isrealobj(x):
            np.copyto(out, self.lu.solve(x, trans='H'))
        else:
            out.real[...] = self.lu.solve(x.real, trans='H')
            out.imag[...] = self.lu.solve(x.imag, trans='H')
        return out

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
    power : float or int
        The exponent. If the flag integer is True, then power has to be a nonnegative integer. 
        In this case the result is computed by multiplications (based on values for power=2**n, n=1,2, ... ). 
    domain : regpy.vecsps.NumPyVectorSpace
        The underlying vector space
    integer: bool, optiomal
    """

    def __init__(self, 
                 power:float|int, 
                 domain:NumPyVectorSpace, 
                 integer:bool = False):
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

    def _ieval(self, x, out, differentiate=False):
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

    def _iderivative(self, x, out):
        return np.multiply(self._factor, x, out=out)

    def _iadjoint(self, y, out):
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
    domain : regpy.vecsps.MeasureSpaceFcts
        Function space on which the superposition operator acts.
    f : callable
        Vectorized point function. If it accepts an ``out`` keyword, it is used
        by :meth:`_ieval` to avoid an allocation.
    Df : callable
        Vectorized real Jacobian. Its result must be real and broadcastable to
        ``domain.shape_domain + (real_output_size, real_input_size)``. For a
        real scalar-to-scalar map, returning only an array of shape
        ``domain.shape_domain`` is accepted as shorthand for a ``1 x 1``
        Jacobian.
    codomain_shape : int or tuple of ints, optional
        Shape of one output value. By default it is inferred by evaluating
        ``f`` once on a zero point value.
    codomain_dtype : dtype, optional
        Real or complex output dtype. By default it is inferred together with
        ``codomain_shape``.
    linear : bool, optional
        Declare the point function linear. In this case the Jacobian is
        computed once during initialization and the operator exposes an
        adjoint directly. Defaults to ``False``.

    Notes
    -----
    If ``A`` is a complex matrix representing a complex-linear derivative,
    :meth:`complex_matrix_to_real_jacobian` converts it to the real Jacobian
    convention required here. General non-holomorphic maps must provide their
    real Jacobian directly.
    """

    def __init__(
            self, domain, f, Df, codomain_shape=None,
            codomain_dtype=None, linear=False):
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
    def _accepts_out(func):
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
    def _stack_result(result):
        if isinstance(result, (tuple, list)):
            if len(result) == 1:
                result = result[0]
            else:
                result = np.stack(tuple(np.asarray(item) for item in result), axis=-1)
        return np.asarray(result)

    def _evaluate_point(self, point):
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

    def _copy_function_result(self, result, out):
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

    def _apply_function(self, x, out):
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

    def _update_jacobian(self, x):
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

    def _real_input(self, x):
        x_flat = np.reshape(
            x, self._domain_shape + (self._input_size,)
        )
        if not self.domain.is_complex:
            return x_flat
        self._input_real_buffer[..., 0::2] = x_flat.real
        self._input_real_buffer[..., 1::2] = x_flat.imag
        return self._input_real_buffer

    def _real_output(self, y):
        y_flat = np.reshape(
            y, self._domain_shape + (self._output_size,)
        )
        if not self.codomain.is_complex:
            return y_flat
        self._output_real_buffer[..., 0::2] = y_flat.real
        self._output_real_buffer[..., 1::2] = y_flat.imag
        return self._output_real_buffer

    def _apply_jacobian(self, h, out):
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

    def _apply_jacobian_transpose(self, y, out):
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

    def _ieval(self, x, out, differentiate=False):
        source = x.copy() if np.shares_memory(x, out) else x
        if differentiate and not self.linear:
            self._update_jacobian(source)
        return self._apply_function(source, out)

    def _iderivative(self, h, out):
        return self._apply_jacobian(h, out)

    def _iadjoint(self, y, out):
        return self._apply_jacobian_transpose(y, out)

    def _iadjoint_eval(self, x, out):
        self._ieval(
            x, self._value_buffer, differentiate=not self.linear
        )
        return self._iadjoint(self._value_buffer, out)

    def _iadjoint_derivative(self, h, out):
        self._iderivative(h, self._value_buffer)
        return self._iadjoint(self._value_buffer, out)

    @staticmethod
    def complex_matrix_to_real_jacobian(matrix):
        r"""Convert a complex matrix to its real Jacobian representation.

        ``matrix`` must have shape ``... + (m, n)``. The returned real array
        has shape ``... + (2*m, 2*n)`` and uses interleaved real/imaginary
        coordinates.
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

    def __repr__(self):
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
    domain : regpy.vecsps.NumPyVectorSpaceBase
        The underlying vector space.
    """

    def __init__(self, domain:NumPyVectorSpace):
        if not isinstance(domain,NumPyVectorSpace):
            raise TypeError(Errors.not_instance(domain, NumPyVectorSpace, add_info="Domain of a Exponential operator needs to be a NumPyVectorSpace!"))
        super().__init__(domain, domain)

    def _ieval(self, x, out, differentiate=False):
        np.exp(x, out=out)
        if differentiate:
            # Reuse the value already computed in ``out`` instead of evaluating
            # the exponential a second time.
            self._exponential_factor = out.copy()
        return out

    def _iderivative(self, x, out):
        return np.multiply(self._exponential_factor, x, out=out)

    def _iadjoint(self, y, out):
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
    domain : regpy.vecsps.UniformGridFcts
        The underlying function space. Its dtype must be floating or complex.
    centered : bool, optional
        Center the *frequency* ordering on every full FFT axis.  The compressed
        axis of an RFFT remains ordered from zero to the Nyquist frequency.
        Defaults to ``False``.
    axes : int or sequence of ints, optional
        Domain axes over which to transform. Negative axes follow NumPy's
        convention. The order matters for real transforms because ``axes[-1]``
        is the compressed RFFT axis. Defaults to all domain axes.
    centered_input : bool or None, optional
        Whether the samples are stored with the spatial zero at the center of
        every transformed axis. If true, an ``ifftshift`` is applied before the
        transform and the corresponding ``fftshift`` after the inverse or
        adjoint transform. ``None`` preserves the historical behavior and uses
        the value of ``centered``. Set this explicitly to ``False`` for grids
        whose zero sample is stored at index zero. Defaults to ``None``.

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
                 axes=None, centered_input: bool | None = None):
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
    def _normalize_axes(ndim, axes):
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

    def _eval(self, x):
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

    def _unshift_frequency_data(self, y):
        if self.centered and self._frequency_shift_axes:
            return spfft.ifftshift(
                y, axes=self._frequency_shift_axes
            )
        return y

    def _finish_spatial_transform(self, x):
        if self.centered_input:
            x = spfft.fftshift(x, axes=self.axes)
        return x.astype(self.domain.dtype, copy=False)

    def _inverse_eval(self, y):
        """Apply the inverse FFT, which is only a left inverse for an RFFT."""
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

    def _adjoint(self, y):
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

    def _rfft_spatial_normal(self, x, interior_factor, out):
        r"""Apply an RFFT normal operator directly in the spatial domain.

        The corresponding frequency multiplier is ``interior_factor`` at
        interior frequencies and one at zero and Nyquist frequency.
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

    def _rfft_hermitian_partner(self, boundary):
        partner = boundary
        for axis, indices in self._rfft_negative_frequency_indices.items():
            partner = np.take(partner, indices, axis=axis)
        return partner

    def _rfft_spectrum_normal(self, y, interior_factor, out):
        r"""Apply an RFFT normal operator in the stored half-spectrum.

        Interior frequencies are scaled directly. The zero and Nyquist
        slices are projected onto their Hermitian subspaces.
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

    def _adjoint_eval(self, x):
        if self.is_complex:
            # The allocating API must return an independent result.
            return np.array(x, dtype=self.domain.dtype, copy=True)
        out = self.domain.empty()
        return self._rfft_spatial_normal(x, 0.5, out)

    def _iadjoint_eval(self, x, out):
        if self.is_complex:
            if out is not x:
                self.domain.copyto(out, x)
            return out
        return self._rfft_spatial_normal(x, 0.5, out)

    def _eval_adjoint(self, y):
        """Efficiently evaluate ``self(self.adjoint(y))``."""
        if self.is_complex:
            return np.array(y, dtype=self.codomain.dtype, copy=True)
        out = self.codomain.empty()
        return self._rfft_spectrum_normal(y, 0.5, out)

    def _ieval_adjoint(self, y, out):
        if self.is_complex:
            if out is not y:
                self.codomain.copyto(out, y)
            return out
        return self._rfft_spectrum_normal(y, 0.5, out)

    @staticmethod
    def frequencies(domain, centered=False, axes=None, rfft=False):
        """Return coordinate axes for the selected discrete Fourier transform."""
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
    def left_inverse(self):
        """Return the inverse on the range of the transform.

        For a real transform this is also the L2 adjoint induced by the
        measures of ``domain`` and ``codomain``, i.e.
        ``G_X**(-1) * self.adjoint * G_Y``.
        """
        if self.is_complex:
            return self.adjoint
        return _FourierTransformLeftInverse(self)

    @Operator.inverse.getter
    def inverse(self):
        if not self.is_complex:
            raise NotImplementedError(Errors.generic_message(
                "A real FourierTransform maps into an unrestricted complex "
                "half-spectrum and therefore has no two-sided inverse. Use "
                "left_inverse when inversion on the RFFT range is intended."
            ))
        return self.adjoint

    @property
    def invertible(self):
        return self.is_complex

    def __repr__(self):
        return make_repr(
            self, self.domain,
            f"centered={self.centered}",
            f"axes={self.axes}",
            f"centered_input={self.centered_input}"
        )


class _FourierTransformLeftInverse(Operator):
    """Left inverse of a real ``FourierTransform`` on its RFFT range."""

    def __init__(self, transform):
        self.transform = transform
        super().__init__(
            transform.codomain, transform.domain, linear=True
        )
        self._consts.add('transform')

    def _eval(self, y):
        return self.transform._inverse_eval(y)

    def _adjoint(self, x):
        result = self.transform._eval(x)
        return np.multiply(
            result, self.transform._rfft_inverse_adjoint_weights
        )

    def _adjoint_eval(self, y):
        out = self.domain.empty()
        return self.transform._rfft_spectrum_normal(y, 2., out)

    def _iadjoint_eval(self, y, out):
        return self.transform._rfft_spectrum_normal(y, 2., out)

    def _eval_adjoint(self, x):
        """Efficiently evaluate ``self(self.adjoint(x))``."""
        out = self.codomain.empty()
        return self.transform._rfft_spatial_normal(x, 2., out)

    def _ieval_adjoint(self, x, out):
        return self.transform._rfft_spatial_normal(x, 2., out)

    @property
    def invertible(self):
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
    transform : FourierTransform
        A real Fourier transform of a scalar-valued one-dimensional grid,
        acting on its sole domain axis.
    """

    def __init__(self, transform: FourierTransform):
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
    def coordinate_space(self):
        """The real Fourier-coordinate space."""
        return self.domain

    def _eval(self, coordinates):
        spectrum = self.codomain.zeros()
        spectrum[0] = coordinates[0]
        spectrum[self._interior_frequencies] = (
            coordinates[self._real_coordinates]
            + 1j * coordinates[self._imag_coordinates]
        ) / self._sqrt_two
        if self.spatial_size % 2 == 0:
            spectrum[-1] = coordinates[-1]
        return spectrum

    def _ieval(self, coordinates, out):
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

    def _adjoint(self, spectrum):
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
    def left_inverse(self):
        """Return the coordinate map from a valid RFFT half-spectrum."""
        return _RealFourierEmbeddingLeftInverse(self)

    @Operator.inverse.getter
    def inverse(self):
        raise NotImplementedError(Errors.generic_message(
            "RealFourierEmbedding is not onto the unrestricted complex "
            "half-spectrum. Use left_inverse on valid RFFT spectra."
        ))

    @property
    def invertible(self):
        return False

    def __repr__(self):
        return make_repr(self, self.transform)

    def real_multiplier(self, half_spectrum_multiplier):
        r"""Represent a real diagonal half-spectrum multiplier in coordinates.

        ``half_spectrum_multiplier`` must be real-valued and have the shape of
        the RFFT half-spectrum.  Every interior value is duplicated for its
        real and imaginary coordinate.  This is particularly useful for the
        non-negative absolute values in a real Fourier SVD.
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

    def __repr__(self):
        return make_repr(self, self.transform)


class _RealFourierEmbeddingLeftInverse(Operator):
    """Left inverse of :class:`RealFourierEmbedding` on its range."""

    def __init__(self, embedding):
        self.embedding = embedding
        super().__init__(
            embedding.codomain,
            embedding.domain,
            linear=True,
        )
        self._consts.add('embedding')

    def _eval(self, spectrum):
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

    def _adjoint(self, coordinates):
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
    def invertible(self):
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
    """

    def __init__(self, domain: UniformGridFcts, centered: bool = False,
                 axes=None, centered_input: bool | None = None):
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
    def coordinate_space(self):
        """The real Fourier-coordinate space."""
        return self.codomain

    @Operator.inverse.getter
    def inverse(self):
        return self.adjoint

    @property
    def invertible(self):
        return True

    def __repr__(self):
        return make_repr(
            self,
            self.domain,
            f"centered={self.fourier_transform.centered}",
            f"centered_input={self.fourier_transform.centered_input}",
        )


class PtwMatrixVectorMultiplication(Operator):
    """
    Pointwise multiplication of a matrix-valued function with a vector-valued function.

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
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

    def _ieval(self, v, out):
        return  np.einsum(self._einstein_string_mul, self.matrixfct, v, out = out)
    
    def _iadjoint(self, w, out):
        return np.einsum(self._einstein_string_mulT, np.conj(self.matrixfct), w, out = out)

    def __repr__(self):
        return make_repr(self, self.domain, self.codomain)

class PtwScalarMultiplication(Operator):
    """
    Pointwise multiplication of a scalar valued function with a vector-valued function.

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
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
        elif multiplier in domain.scalar_space():
            self.multiplier = np.reshape(multiplier,domain.shape_domain+(1,)*domain.ndim_codomain)
        else:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the multiplier must be NumPy array of matching size. Got \n\t domain = {domain}, \n\t multiplier = {multiplier}'))
        
        
        super().__init__(domain, domain, linear=True)

    def _ieval(self, f, out):
        return np.multiply(f,self.multiplier,out = out)
    
    def _iadjoint(self, g, out):
        return np.multiply(g,self.multiplier.conj(), out = out)

    @Operator.inverse.getter
    def inverse(self):
        return PtwScalarDivision(self.domain,self.multiplier)

    @property
    def invertible(self):
        return np.all(self.multiplier!=0)

class PtwScalarDivision(Operator):
    """
    Pointwise division of a vector-valued function by a scalar valued function

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        The input grid function.
    scalarfct : np.ndarray
        The scalar valued function to divide by.
        The number of dimensions must match the number of dimensions of domain.   
        The first dimensions must match the shape_domain of domain, and the co-dimensions must be 1.
    """   

    def __init__(self, domain, divisor):
        if not isinstance(domain,MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(domain,MeasureSpaceFcts,add_info=f'For a PtwScalarMultiplication the domain must be of type MeasureSpaceFcts. Got \n\t domain = {domain}'))
        if domain.ndim_codomain==0:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the domain must be vector valued!'))
        if divisor in domain and (divisor.shape == domain.shape_domain + (1,)*domain.ndim_codomain):
            self.divisor = divisor
            print("first option", self.divisor.shape)
        elif divisor in domain.scalar_space():
            self.divisor = np.reshape(divisor,domain.shape_domain+(1,)*domain.ndim_codomain)
        else:
            raise ValueError(Errors.value_error(f'For a PtwScalarMultiplication the divisor must be NumPy array of matching size. Got \n\t domain = {domain}, \n\t divisor = {divisor}'))
        if np.any(self.divisor==0):
            raise ValueError(Errors.value_error('Divisor must not vanish anywhere.'))
        
        super().__init__(domain, domain, linear=True)

    def _ieval(self, f, out):
        return np.divide(f, self.divisor, out=out)

    def _iadjoint(self, g, out):
        return np.divide(g, self.divisor.conj(), out=out)

    @Operator.inverse.getter
    def inverse(self):
        return PtwScalarMultiplication(self.domain,self.divisor)

    @property
    def invertible(self):
        return True

class AddSingletonVectorDimension(Operator):
    """Operator that adds a singleton dimension as codimension in `regpy.vecsps.MeasureSpaceFcts`. 
    Wrapper to np.reshape(...,1).

    Parameters
    ----------
    grid: regpy.vecsps.MeasureSpaceFcts    
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
    
    @property
    def inverse(self):
        return self.adjoint

    @property
    def invertible(self):
        return True   

class ForwardFDGradient(Operator):
    """Forward finite-difference gradient on ``UniformGridFcts``.

    The codomain is a vector-valued
    :class:`regpy.vecsps.UniformGridFcts` space.
    
    Parameters
    ----------
    domain: regpy.vecsps.UniformGridFcts
        The grid on which the gradient (with respect to the given coordinates) is defined.
    boundary_condition: Either 'Neum' or 'Diri' or 'per' (default: 'Neum')
        Boundary condition on the 'left' boundaries. The strings stand for Neumann, Dirichlet, and periodic, respectively
    """
    def __init__(self, domain,boundary_condition = 'Neum'):
        if not isinstance(domain,UniformGridFcts):
            raise TypeError(f'ForwardFDGradient only implemented for UniformGridFcts')
        if domain.ndim_codomain!=0:
            raise ValueError(f'domain must be scalar valued.')
        self.boundary_condition = boundary_condition
        super().__init__(domain, domain.vector_valued_space(domain.ndim), linear=True)

    def _ieval(self, v, out):
        out.fill(0.)
        for (ax,h) in zip(range(self.domain.ndim),self.domain.spacing):
            # Interior: forward differences: out[i1,...in,ax] = (v[... i_ax+1 ...] - v[...,i_ax ...])/h_ax
            slc_mid = [slice(None)] * self.domain.ndim
            slc_mid[ax] = slice(0, -1)
            slc_p1 = slc_mid.copy(); slc_p1[ax] = slice(1, None)
            out[tuple(slc_mid)+(ax,)] = (v[tuple(slc_p1)] - v[tuple(slc_mid)])/h

            # right boundary Dirichlet: out[... N_ax ...] = -v[....N_ax ...]/h_ax
            if self.boundary_condition in ['Diri','per']:
                slN = [slice(None)] * self.domain.ndim
                slN[ax] = -1
                out[tuple(slN)+(ax,)] = - v[tuple(slN)]/h
                if self.boundary_condition == 'per':
                    slN_p1 = slN.copy(); slN_p1[ax] = 0
                    out[tuple(slN)+(ax,)] += v[tuple(slN_p1)]/h
            # For Neumann boundary conditions the last row of the matrix (in 1D) is identically 0. 

        return out

    def _iadjoint(self, v, out):
        out.fill(0.)
        for ax,h in zip(range(self.domain.ndim),self.domain.spacing):
            # Interior: backward difference: (v[i_ax-1]-v[i_ax])/h_ax
            slc_mid = [slice(None)] * self.domain.ndim
            slc_mid[ax] = slice(1, -1)
            slc_m1 = slc_mid.copy(); slc_m1[ax] = slice(None, -2)
            out[tuple(slc_mid)] += (v[tuple(slc_m1)+(ax,)] - v[tuple(slc_mid)+(ax,)])/h
 
            # left boundary
            sl0 = [slice(None)] * self.domain.ndim
            sl0[ax] = 0
            out[tuple(sl0)] -=  v[tuple(sl0)+(ax,)] /h
            if self.boundary_condition == 'per':
                sl0_m1 = sl0.copy(); sl0_m1[ax] = -1
                out[tuple(sl0)] += v[tuple(sl0_m1)+(ax,)] /h

            slN =  [slice(None)] * self.domain.ndim
            slN[ax] = -1
            slN_m1 = slN.copy(); slN_m1[ax] = -2
            out[tuple(slN)] += v[tuple(slN_m1)+(ax,)]/h
            if self.boundary_condition in ['Diri','per']:
                out[tuple(slN)] -= v[tuple(slN)+(ax,)]/h

        return out

    def matrices(self):
        """ Returns a list of matrices of length domain.ndim containing matrix representations of the partial derivatives
        """
        N = np.prod(self.domain.shape)
        ndim = self.domain.ndim
        shape = self.domain.shape
        out = [lil_matrix((N,N)) for j in range(ndim)]
        for (ax,h) in zip(range(ndim),self.domain.spacing):
            for idx in np.ndindex(shape):
                i_flat = np.ravel_multi_index(idx,shape)
                if idx[ax] < shape[ax]-1:
                    out[ax][i_flat,i_flat] = -1/h
                    neigh = list(idx)
                    neigh[ax]+=1
                    neigh_flat = np.ravel_multi_index(tuple(neigh),shape)
                    out[ax][i_flat,neigh_flat] = 1/h
                elif self.boundary_condition in ['Diri','per']:
                    out[ax][i_flat,i_flat] = -1/h
                    if self.boundary_condition == 'per':
                        neigh = list(idx)
                        neigh[ax] = 0
                        neigh_flat = np.ravel_multi_index(tuple(neigh),shape)
                        out[ax][i_flat,neigh_flat] = 1/h 
        return [out[j].tocsc() for j in range(ndim)]
    
    def norm(self,h_domain=None,h_codomain=None,method=None,without_codomain_vectors=False):
        """ A good upper bound on the  norm with respect to the standard L2 inner product can be computed analytically.         
        """
        from regpy.hilbert import L2
        if (h_domain is None or h_domain==L2) and (h_codomain is None or h_codomain==L2) and method is None:
            return 2*np.linalg.norm(1./self.domain.spacing)
        else: 
            return super().norm(h_domain=h_domain,h_codomain=h_codomain,method=method,without_codomain_vectors=without_codomain_vectors)

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
        characters=tuple(chr(k) for k in range(ord('A'),ord('A')+self.codomain.ndim))
        self._eval_summation_string = ",".join(characters) + "->" + "".join(characters)
        """String for writing the outer product directly into an output array."""
        self._consts = {*self._consts,"_p","_p_conj_flat"}

    def _calc_adjoint_summation_strings(self):
        li=[]
        characters=tuple(chr(k) for k in range(ord('A'),ord('A')+self.codomain.ndim))
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

    def _ieval(self, x, out, differentiate=False):
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

    def _iderivative(self, x, out):
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

    def _adjoint(self, y):
        xs=[]
        for j,s in enumerate(self._adjoint_summation_strings):
            x_j=np.einsum(s,y,*self._p_conj_flat[:j],*self._p_conj_flat[j+1:],optimize=True)
            xs.append(x_j.reshape(self.domain[j].shape))
        return self.domain.join(*xs)

    def _iadjoint(self, y, out):
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
    subscripts : string
        Subscripts used for the evaluation in numpy.einsum. See the documentation of numpy.einsum for further information.
    domains : regpy.vecsps.GridFcts
        The underlying vector spaces.
    tensors : tuple of numpy.ndarray
        Tuple of constant NumPy arrays.
    codomain : regpy.vecsps.NumPyVectorSpace, optional
        Codomain of the operator. If it is not specified it is computed based on the input data.
    """


    def __init__(self, subscripts,*domains,tensors=(),codomain=None):
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
        self._eval_opt,self._adjoint_opt=self._optimize_einsum_paths()
        '''Optimal paths for computation of einsum in evaluation and evaluation of adjoint.'''

    @staticmethod
    def _get_standard_subscript_info(subscripts,n_domains):
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
    def _check_parameter_validity(subscripts,*domains,tensors):
        if("." in subscripts):
            raise ValueError(f"EinSum operator currently does not support ellipses, so input:\'{subscripts}\' is invalid.")
        #TODO add further tests

    def _calc_codomain(self,*domains):
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
    
    def _calc_adjoint_subscripts(self):
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
    def _calc_adjoint_assignment_info(ins_str,out_str):
        outset=set(out_str)
        insset=set(ins_str.replace(",",""))
        if(len(out_str)==len(outset) and outset.issubset(insset)):
            return True,f"{ins_str}->{out_str}",f"{out_str}->{out_str}"
        unknowns="".join(outset.difference(insset))
        knowns="".join(outset.difference(unknowns))
        return False,f"{ins_str}->{knowns}",f"{out_str}->{unknowns}{knowns}"
        
    def _optimize_einsum_paths(self):
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

    
    def _ieval(self, x, out, differentiate=False):
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

    def _iderivative(self, x, out):
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

    def _adjoint(self, y):
        if(self.linear):
            x=np.einsum(self._adjoint_subscripts[0][1],*self.tensors,y,optimize=self._adjoint_opt[0])
            if(np.issubdtype(self.domain.dtype, np.floating)):
                x=np.real(x)
            if(self._adjoint_subscripts[0][0]):
                return x
            x_full=self.domain.zeros()
            np.einsum(self._adjoint_subscripts[0][2],x_full)[:]=x
            return x_full
        xs=[]
        for j,s in enumerate(self._adjoint_subscripts):
            x_j=np.einsum(s[1],*self._p_conj[:j],*self._p_conj[j+1:],*self.tensors,y,optimize=self._adjoint_opt[j])
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
    def _write_adjoint_component(subscripts, operands, optimize, out):
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

    def _iadjoint(self, y, out):
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

