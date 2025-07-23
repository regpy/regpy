r"""
Forward operators
=================

This module provides the basis for defining forward operators, and implements some simple
auxiliary operators. Actual forward problems are implemented in submodules.

The base class is `Operator`.
"""

from collections import defaultdict
from copy import deepcopy
from re import A

import numpy as np
from numpy.core.numeric import zeros_like
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla

from regpy import functionals, util, vecsps


class _Revocable:
    def __init__(self, val):
        self.__val = val

    @classmethod
    def take(cls, other):
        return cls(other.revoke())

    def get(self):
        try:
            return self.__val
        except AttributeError:
            raise RuntimeError('Attempted to use revoked reference') from None

    def revoke(self):
        val = self.get()
        del self.__val
        return val

    @property
    def valid(self):
        try:
            self.__val
            return True
        except AttributeError:
            return False


class Operator:
    r"""Base class for forward operators. Both linear and non-linear operators are handled. Operator
    instances are callable, calling them with an array argument evaluates the operator.

    Subclasses implementing non-linear operators should implement the following methods:

        _eval(self, x, differentiate=False)
        _derivative(self, x)
        _adjoint(self, y)

    These methods are not intended for external use, but should be invoked indirectly via calling
    the operator or using the `Operator.linearize` method. They must not modify their argument, and
    should return arrays that can be freely modified by the caller, i.e. should not share data
    with anything. Usually, this means they should allocate a new array for the return value.

    Implementations can assume the arguments to be part of the specified vector spaces, and return
    values will be checked for consistency.

    In some cases a solver only requires the application of the composition of the adjoint with the 
    derivative. When this should be used i.e. in cases when setting up elements in the codomain is 
    not feasible one can implement the `_adjoint_derivative` method and when linearizing can set the
    flag `adjoint_derivative = True`.

    The mechanism for derivatives and their adjoints is this: whenever a derivative is to be
    computed, `_eval` will be called first with `differentiate=True` or `adjoint_derivative=True`, 
    and should produce the operator's value and perform any precomputation needed for evaluating 
    the derivative. Any subsequent invocation of `_derivative`, `_adjoint` and `_adjoint_derivative`
    should evaluate the  derivative, its adjoint or their composition at the same point `_eval` was 
    called. The reasoning is this

     * In most cases, the derivative alone is not useful. Rather, one needs a linearization of the
       operator around some point, so the value is almost always needed.
     * Many expensive computations, e.g. assembling of finite element matrices, need to be carried
       out only once per linearization point, and can be shared between the operator and the
       derivative, so they should only be computed once (in `_eval`).
    
    For callers, this means that since the derivative shares data with the operator, it can't be
    reliably called after the operator has been evaluated somewhere else, since shared data may
    have been overwritten. The `Operator`, `Derivative` and `Adjoint` classes ensure that an
    exception is raised when an invalidated derivative is called.

    If derivatives at multiple points are needed, a copy of the operator should be performed using
    `copy.deepcopy`. For efficiency, subclasses can add the names of attributes that are considered
    as constants and should not be deepcopied to `self._consts` (a `set`). By default, `domain` and
    `codomain` will not be copied, since `regpy.vecsps.VectorSpace` instances should never
    change in-place.

    If no derivative at some point is needed, `_eval` will be called with `differentiate=False`,
    allowing it to save on precomputations. It does not need to ensure that data shared with some
    derivative remains intact; all derivative instances will be invalidated regardless.

    Linear operators should implement

        _eval(self, x)
        _adjoint(self, y)

    Here the logic is simpler, and no sharing of precomputations is needed (unless it applies to the
    operator as a whole, in which case it should be performed in `__init__`).

    Note that the adjoint should be computed with respect to the standard real inner product on the
    domain / codomain, given as

        np.real(np.vdot(x, y))

    Other inner products on vector spaces are independent of both vector spaces and operators,
    and are implemented in the `regpy.hilbert` module.

    Basic operator algebra is supported:

        a * op1 + b * op2    # linear combination
        op1 * op2            # composition
        op * arr             # composition with array multiplication in domain
        op + arr             # operator shifted in codomain
        op + scalar          # dto.

    Parameters
    ----------
    domain, codomain : regpy.vecsps.VectorSpace or None
        The vector space on which the operator's arguements / values are defined. Using `None`
        suppresses some consistency checks and is intended for ease of development, but should 
        not be used except as a temporary measure. Some constructions like direct sums will fail
        if the vector spaces are unknown.
    linear : bool, optional
        Whether the operator is linear. Default: `False`.
    """

    log = util.classlogger

    def __init__(self, domain=None, codomain=None, linear=False):
        assert not domain or isinstance(domain, vecsps.VectorSpace)
        assert not codomain or isinstance(codomain, vecsps.VectorSpace)
        self.domain = domain
        r"""The vector space on which the operator is defined. Either a
        subclass of `regpy.vecsps.VectorSpace` or `None`."""
        self.codomain = codomain
        r"""The vector space on which the operator values are defined. Either
        a subclass of `regpy.vecsps.VectorSpace` or `None`."""
        self.linear = linear
        r"""Boolean indicating whether the operator is linear."""
        self._consts = {'domain', 'codomain'}

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._consts:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def attrs(self):
        r"""The set of all instance attributes. Useful for updating the `_consts` attribute via

            self._consts.update(self.attrs)

        to declare every current attribute as constant for deep copies.
        """
        return set(self.__dict__)

    def __call__(self, x):
        assert not self.domain or x in self.domain
        if self.linear:
            y = self._eval(x)
        else:
            self.__revoke()
            y = self._eval(x, differentiate=False)
        assert not self.codomain or y in self.codomain
        return y

    def linearize(self, x, adjoint_derivative = False):
        r"""Linearize the operator around some point.

        Parameters
        ----------
        x : array-like
            The point around which to linearize.

        adjoint_deriv : boolean (Default: False)
            Flag to determine if AdjointDerivative should be returned as additional output argument. 
            This can be used if AdjointDerivative has an a more efficient implementation than by composition 
            or if the image space of the operator is too large to store vectors in this space. 

        Returns
        -------
        if adjoint_deriv==True: 
          array, Derivative:
             The value and the derivative at `x`, the latter as an `Operator` instance.
        if adjoint_deriv ==False: 
           array, Derivative, AdjointDerivative
               array is Derivative.adjoint(self(x), Derivative is as above, and  
               AdjointDerivative is an efficient implementation of the composition Derivative.adjoint * Derivative
        """
        if self.linear:
            if not adjoint_derivative:
                return self(x), self
            else:
                return self.adjoint(self(x)), self, self.adjoint * self
        else:
            if not adjoint_derivative:
                assert not self.domain or x in self.domain
                self.__revoke()
                y = self._eval(x, differentiate=True)
                assert not self.codomain or y in self.codomain
                deriv = Derivative(self.__get_handle())
                return y, deriv
            else:
                assert not self.domain or x in self.domain
                self.__revoke()
                try:
                    Fstar_y = self._eval(x, differentiate=True, adjoint_derivative=True)
                except TypeError:
                    y = self._eval(x, differentiate=True)
                    assert not self.codomain or y in self.codomain
                    Fstar_y = self._adjoint(y)
                deriv = Derivative(self.__get_handle()) 
                adjoint_deriv = AdjointDerivative(self.__get_handle())
                return Fstar_y, deriv, adjoint_deriv
    @util.memoized_property
    def adjoint(self):
        r"""For linear operators, this is the adjoint as a linear `regpy.operators.Operator`
        instance. Will only be computed on demand and saved for subsequent invocations.

        Returns
        -------
        Adjoint
            The adjoint as an `Operator` instance.
        """
        return Adjoint(self)

    def __revoke(self):
        try:
            self.__handle = _Revocable.take(self.__handle)
        except AttributeError:
            pass

    def __get_handle(self):
        try:
            return self.__handle
        except AttributeError:
            self.__handle = _Revocable(self)
            return self.__handle

    def _eval(self, x, differentiate=False, adjoint_derivative = False):
        raise NotImplementedError

    def _derivative(self, x):
        raise NotImplementedError

    def _adjoint(self, y):
        raise NotImplementedError

    def _adjoint_derivative(self, x):
        if self.linear:
            return self._adjoint(self._eval(x))
        else:
            return self._adjoint(self._derivative(x))

    @property
    def inverse(self):
        r"""A property containing the  inverse as an `Operator` instance. In most cases this will
        just raise a `NotImplementedError`, but subclasses may override this if possible and useful.
        To avoid recomputing the inverse on every access, `regpy.util.memoized_property` may be
        useful."""
        raise NotImplementedError

    def as_linear_operator(self):
        r"""Creating a `scipy.linalg.LinearOperator` from the defined linear operator.  

        Returns
        -------
        scipy.linalg.LinearOperator 
            The linear operator as a scipy linear operator.

        Raises
        ------
        RuntimeError
            If operator flag `linear` is False. 
        """
        if self.linear:
            return SciPyLinearOperator(self)
        else:
            raise RuntimeError('Operator is not linear.')

    def __mul__(self, other):
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, Operator):
            return Composition(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return self * PtwMultiplication(self.domain, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            if other == 1:
                return self
            else:
                return LinearCombination((other, self))         
        elif isinstance(other, np.ndarray):
            return PtwMultiplication(self.codomain, other) * self
        elif isinstance(other, Operator):
            return Composition(other, self) 
        else:
            return NotImplemented

    def __add__(self, other):
        if np.isscalar(other) and other == 0:
            return self
        elif isinstance(other, Operator):
            return LinearCombination(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return OuterShift(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return self


class Adjoint(Operator):
    r"""An proxy class wrapping a linear operator. Calling it will evaluate the operator's
    adjoint. This class should not be instantiated directly, but rather through the
    `Operator.adjoint` property of a linear operator.
    """

    def __init__(self, op):
        assert op.linear
        self.op = op
        r"""The underlying operator."""
        super().__init__(op.codomain, op.domain, linear=True)

    def _eval(self, x):
        return self.op._adjoint(x)

    def _adjoint(self, x):
        return self.op._eval(x)

    @property
    def adjoint(self):
        return self.op

    @property
    def inverse(self):
        return self.op.inverse.adjoint

    def __repr__(self):
        return util.make_repr(self, self.op)


class Derivative(Operator):
    r"""An proxy class wrapping a non-linear operator. Calling it will evaluate the operator's
    derivative. This class should not be instantiated directly, but rather through the
    `Operator.linearize` method of a non-linear operator.
    """

    def __init__(self, op):
        if not isinstance(op, _Revocable):
            # Wrap plain operators in a _Revocable that will never be revoked to
            # avoid case distinctions below.
            op = _Revocable(op)
        self.op = op
        r"""The underlying operator."""
        _op = op.get()
        super().__init__(_op.domain, _op.codomain, linear=True)

    def _eval(self, x):
        return self.op.get()._derivative(x)

    def _adjoint(self, x):
        return self.op.get()._adjoint(x)

    def __repr__(self):
        return util.make_repr(self, self.op.get())


class AdjointDerivative(Operator):
    r"""A proxy class wrapping a non-linear operator :math:`F`. Calling it will evaluate the coposition of the operator's
    derivative adjoint with its derivative :math:`F'^\ast\circ F'`. This class should not be instantiated directly, 
    but rather through the `Operator.linearize` method of a non-linear operator with the flag `adjoint_derivitave = True`.
    The `_eval` and `_adjoint` require the implementation of `_adjoint_derivative` note that only one implimentation is 
    needed as it is a selfadjoint operator.   
    """

    def __init__(self, op):
        if not isinstance(op, _Revocable):
            # Wrap plain operators in a _Revocable that will never be revoked to
            # avoid case distinctions below.
            op = _Revocable(op)
        self.op = op
        r"""The underlying operator."""
        _op = op.get()
        super().__init__(_op.domain, _op.domain, linear=True)

    def _eval(self, x):
        return self.op.get()._adjoint_derivative(x)

    def _adjoint(self, x):
        return self.op.get()._adjoint_derivative(x)

    def __repr__(self):
        return util.make_repr(self, self.op.get())


class LinearCombination(Operator):
    r"""A linear combination of operators. This class should normally not be instantiated directly,
    but rather through adding and multiplying `Operator` instances and scalars.
    """

    def __init__(self, *args):
        coeff_for_op = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, op = arg
            else:
                coeff, op = 1, arg
            assert isinstance(op, Operator)
            assert (
                not np.iscomplex(coeff)
                or not op.codomain
                or op.codomain.is_complex
            )
            if isinstance(op, type(self)):
                for c, o in zip(op.coeffs, op.ops):
                    coeff_for_op[o] += coeff * c
            else:
                coeff_for_op[op] += coeff
        self.coeffs = []
        """List of coefficients of the combined operators."""
        self.ops = []
        """List of combined operators."""
        for op, coeff in coeff_for_op.items():
            self.coeffs.append(coeff)
            self.ops.append(op)

        domains = [op.domain for op in self.ops if op.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        codomains = [op.codomain for op in self.ops if op.codomain]
        if codomains:
            codomain = codomains[0]
            assert all(c == codomain for c in codomains)
        else:
            codomain = None

        super().__init__(domain, codomain, linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        y = self.codomain.zeros()
        if differentiate:
            self._derivs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if differentiate:
                tup = op.linearize(x,adjoint_deriv=adjoint_deriv)
                z = tup[0]
                self._derivs.append(tup[1])
                if adjoint_derivative:
                    self._adjoint_derivs.append(tup[2])
            else:
                z = op(x)
            y += coeff * z
        return y

    def _derivative(self, x):
        y = self.codomain.zeros()
        for coeff, deriv in zip(self.coeffs, self._derivs):
            y += coeff * deriv(x)
        return y

    def _adjoint(self, y):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        x = self.domain.zeros()
        for coeff, op in zip(self.coeffs, ops):
            x += np.conj(coeff) * op.adjoint(y)
        return x
    
    def _adjoint_derivative(self, x):
        y = self.domain.zeros()
        for coeff, adjoint_deriv in zip(self.coeffs, self._adjoint_derivs):
            y += np.abs(coeff)**2 * adjoint_deriv(x)
        return y

    @property
    def inverse(self):
        if len(self.ops) > 1:
            raise NotImplementedError
        return (1 / self.coeffs[0]) * self.ops[0].inverse

    def __repr__(self):
        return util.make_repr(self, *zip(self.coeffs, self.ops))

    def __str__(self):
        reprs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if coeff == 1:
                reprs.append(repr(op))
            else:
                reprs.append('{} * {}'.format(coeff, op))
        return ' + '.join(reprs)


class Composition(Operator):
    r"""A composition of operators. This class should normally not be instantiated directly,
    but rather through multiplying `Operator` instances.
    """

    def __init__(self, *ops):
        for f, g in zip(ops, ops[1:]):
            assert not f.domain or not g.codomain or f.domain == g.codomain
        self.ops = []
        """The list of composed operators."""
        for op in ops:
            assert isinstance(op, Operator)
            if isinstance(op, Composition):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        super().__init__(
            self.ops[-1].domain, self.ops[0].codomain,
            linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False, adjoint_derivative = False):
        y = x
        if differentiate:
            self._derivs = []
            for op in self.ops[:0:-1]:
                y, deriv = op.linearize(y)
                self._derivs.insert(0,deriv)
            tup = self.ops[0].linearize(y,adjoint_derivative=adjoint_derivative)
            y = tup[0]
            self._derivs.insert(0,tup[1])
            if adjoint_derivative:
                self._inner_adjoint_deriv = tup[2]
                for deriv in self._derivs[1:]:
                    y = deriv.adjoint(y)
        else:
            for op in self.ops[::-1]:
                y = op(y)
        return y

    def _derivative(self, x):
        y = x
        for deriv in self._derivs[::-1]:
            y = deriv(y)
        return y

    def _adjoint(self, y):
        x = y
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        for op in ops:
            x = op.adjoint(x)
        return x
    
    def _adjoint_derivative(self, x):
        y = x
        for deriv in self._derivs[:0:-1]:
            y = deriv(y)
        y = self._inner_adjoint_deriv(y)
        for deriv in self._derivs[1:]:
            y = deriv.adjoint(y)
        return y

    @util.memoized_property
    def inverse(self):
        return Composition(*(op.inverse for op in self.ops[::-1]))

    def __repr__(self):
        return util.make_repr(self, *self.ops)

class SciPyLinearOperator(sla.LinearOperator):
    r"""A class wrapping a linear operator :math:`F` into a scipy.sparse.linalg.LinearOperator so that it can be used conveniently in scipy methods.
    The domain and codomain are flattened.
    """
    def __init__(self, op2):
        self.op2 = op2
        r"""the wrapped operator"""
        # super().__init__(op2.domain.dtype, (np.prod(op2.codomain.shape),np.prod(op2.domain.shape)))
        domain_shape=np.prod(op2.domain.shape)
        codomain_shape=np.prod(op2.codomain.shape)
        if(op2.domain.is_complex):
            domain_shape*=2
        if(op2.codomain.is_complex):
            codomain_shape*=2
        super().__init__(np.float64, (codomain_shape,domain_shape))

    
    def _matvec(self, x):
        r"""Applies the operator.
        
        Parameters
        ----------
        x : numpy.ndarray
            Flattened element from domain of operator.
        
        Returns
        -------
        numpy.ndarray
        """
        op2 = self.op2
        return op2.codomain.flatten(op2(op2.domain.fromflat(x)))
    
    def _rmatvec(self, y):
        r"""Applies the adjoint operator.
        
        Parameters
        ----------
        y : numpy.ndarray
            Flattened element from codomain of operator.
        
        Returns
        -------
        numpy.ndarray
        """
        op2 = self.op2
        return op2.domain.flatten(op2.adjoint(op2.codomain.fromflat(y)))

class Pow(Operator):
    r"""Power of a linear operator A, mapping a domain into itself, i.e. 
       A * A * ... * A

       Parameters
       ----------
       op : operator
       exponent :  non-negative integer
    """

    def __init__(self, op, exponent):
        assert op.linear
        assert op.domain == op.codomain
        assert type(exponent)==int and exponent>=0
        super().__init__(op.domain,op.domain,linear=True)
        self.op = op
        self.exponent = exponent

    def _eval(self,x):
        res = x
        for j in range(self.exponent):
            res = self.op(res)
        return res

    def _adjoint(self,x):
        res = x
        for j in range(self.exponent):
            res = self.op.adjoint(res)
        return res
    
    @property
    def inverse(self):
        return Pow(self.op.inverse,self.exponent)

class Identity(Operator):
    r"""The identity operator on a vector space. 
    By default, a copy is performed to prevent callers from
    accidentally modifying the argument when modifying the return value.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space.
    """

    def __init__(self, domain, copy=True):
        self.copy = copy
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        if self.copy:
            return x.copy()
        else:
            return x

    def _adjoint(self, x):
        if self.copy:
            return x.copy()
        else:
            return x

    @property
    def inverse(self):
        return self

    def __repr__(self):
        return util.make_repr(self, self.domain)

class MatrixMultiplication(Operator):
    r"""Implements an operator that does matrix-vector multiplication with a given matrix. Domain and codomain 
    are plain one dimensional `regpy.vecsps.VectorSpace` instances by default.

    Parameters
    ----------
    matrix : array-like
        The matrix.
    inverse : Operator, array-like or None, optional
        How to implement the inverse operator. If available, this should be given as `Operator`
        or array. If `inv`\, `numpy.linalg.inv` will be used. If `cholesky` or `superLU`\, a
        `CholeskyInverse` or `SuperLU` instance will be returned.
    domain : regpy.vecsps.VectorSpace, optional
        The underlying vector space. If not given a `regpy.vecsps.VectorSpace` with same number of elements as
        matrix columns is used. Defaults to None.
    codomain : regpy.vecsps.VectorSpace, optional
        The underlying vector space. If not given a `regpy.vecsps.VectorSpace` with same number of elements as
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
        self.matrix = matrix
        if dtype == None:
            dtype = matrix.dtype
        super().__init__(
            domain=domain or vecsps.VectorSpace(matrix.shape[1],dtype = dtype),
            codomain=codomain or vecsps.VectorSpace(matrix.shape[0],dtype = dtype),
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
    r"""Implements the inverse of a linear, self-adjoint operator via Cholesky decomposition. Since
    it needs to assemble a full matrix, this should not be used for high-dimensional operators.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator to be inverted.
    matrix : array-like, optional
        If a matrix of `op` is already available, it can be passed in to avoid recomputation.
    """
    def __init__(self, op, matrix=None):
        assert op.linear
        assert op.domain and op.domain == op.codomain
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
    r"""Implements the inverse of a MatrixMultiplication Operator given by a csc_matrix using SuperLU.

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

class CoordinateProjection(Operator):
    r"""A projection operator onto a subset of the domain. The codomain is a one-dimensional
    `regpy.vecsps.VectorSpace` of the same dtype as the domain.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space
    mask : array-like
        Boolean mask of the subset onto which to project.
    """
    def __init__(self, domain, mask):
        mask = np.broadcast_to(mask, domain.shape)
        assert mask.dtype == bool
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=vecsps.VectorSpace(np.sum(mask), dtype=domain.dtype),
            linear=True
        )

    def _eval(self, x):
        return x[self.mask]

    def _adjoint(self, x):
        y = self.domain.zeros()
        y[self.mask] = x
        return y

    def __repr__(self):
        return util.make_repr(self, self.domain, self.mask)

class CoordinateMask(Operator):
    r"""A projection operator onto a subset of the domain. The remaining array elements are set to zero.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space
    mask : array-like
        Boolean mask of the subset onto which to project.
    """
    def __init__(self, domain, mask):
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=domain,
            linear=True
        )

    def _eval(self, x):
        return np.where(self.mask==False, 0, x)

    def _adjoint(self, x):
        return np.where(self.mask==False, 0, x)

    def __repr__(self):
        return util.make_repr(self, self.domain)


class PtwMultiplication(Operator):
    r"""A multiplication operator by a constant factor.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space
    factor : array-like
        The factor by which to multiply. Can be anything that can be broadcast to `domain.shape`.
    """
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if domain:
            factor = np.broadcast_to(factor, domain.shape)
            assert factor in domain
        self.factor = factor
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        if self.domain.is_complex:
            return np.conj(self.factor) * x
        else:
            return self.factor * x

    @util.memoized_property
    def inverse(self):
        sav = np.seterr(divide='raise')
        try:
            return PtwMultiplication(self.domain, 1 / self.factor)
        finally:
            np.seterr(**sav)

    def __repr__(self):
        return util.make_repr(self, self.domain)

class OuterShift(Operator):
    r"""Shift an operator by a constant offset in the codomain.

    Parameters
    ----------
    op : Operator
        The underlying operator.
    offset : array-like
        The offset by which to shift. Can be anything that can be broadcast to `op.codomain.shape`.
    """
    def __init__(self, op, offset):
        assert offset in op.codomain
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = np.copy(offset)

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        if differentiate:
            tup = self.op.linearize(x, adjoint_derivative= adjoint_derivative)
            y = tup[0]
            self._deriv = tup[1]
            if not adjoint_derivative:
                return y + self.offset
            else:
                self._adjoint_derivative = tup[2]
                return y+self._adjoint(self.offset)
        else:
            return self.op(x) + self.offset

    def _derivative(self, x):
        return self._deriv(x)

    def _adjoint(self, y):
        return self._deriv.adjoint(y)
    
    def _adjoint_derivative(self, x):
        return self._adjoint_deriv(x)

    def _adjoint_derivative(self,x):
        return self._adjoint_derivative(x)


class InnerShift(Operator):
    r"""Shift an operator by a constant offset in the domain.

    Parameters
    ----------
    op : Operator
        The underlying operator.
    offset : array-like
        The offset by which to shift. Can be anything that can be broadcast to `op.domain.shape`.
    """
    def __init__(self, op, offset):
        assert offset in op.domain
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = np.copy(offset)

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        if differentiate or adjoint_derivative:
            y, self._deriv = self.op.linearize(x-self.offset, adjoint_derivative=adjoint_derivative)
            return y 
        else:
            return self.op(x - self.offset)

    def _derivative(self, h):
        return self._deriv(h)

    def _adjoint(self, y):
        return self._deriv.adjoint(y)


class FourierTransform(Operator):
    r"""Fourier transform operator on UniformGridFcts implemented via numpy.fft.fftn.

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
        assert isinstance(domain, vecsps.UniformGridFcts)
        self.is_complex = domain.is_complex
        frqs = self.frequencies(domain,centered=centered, axes=axes, rfft= not domain.is_complex)
        shape = domain.shape
        s = shape[-1]
        if centered or (not domain.is_complex and domain.ndim==1):
            codomain = vecsps.UniformGridFcts(*frqs, dtype=complex)
        else:
            # In non-centered case, the frequencies are not ascencing, so even using GridFcts here is slighty questionable.
            codomain = vecsps.GridFcts(*frqs, dtype=complex,use_cell_measure=False)
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
        r"""Compute the grid of frequencies for an FFT on this grid instance.

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

class Power(Operator):
    r"""The operator :math:`x \mapsto x^n`.

    Parameters
    ----------
    power : float
        The exponent.
    domain : regpy.vecsps.VectorSpace
        The underlying vector space
    """

    def __init__(self, power, domain, integer = False):
        self.integer = integer
        if integer:
            assert(isinstance(power,np.uintc))
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

class DirectSum(Operator):
    r"""The direct sum of operators. For

    .. math::
        T_i \colon X_i \to Y_i 

    the direct sum

    .. math::
        T := DirectSum(T_i) \colon DirectSum(X_i) \to DirectSum(Y_i) 

    is given by :math:`T(x)_i := T_i(x_i)`. As a matrix, this is the block-diagonal
    with blocks :math:`(T_i)`.

    Parameters
    ----------
    *ops : tuple of Operator
    flatten : bool, optional
        If True, summands that are themselves direct sums will be merged with
        this one. Default: False.
    domain, codomain : vecsps.VectorSpace or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, *ops, flatten=False, domain=None, codomain=None):
        assert all(isinstance(op, Operator) for op in ops)
        self.ops = []
        r""" List of all operators :math:`(T_1,\dots,T_n)`"""
        for op in ops:
            if flatten and isinstance(op, type(self)):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)

        if domain is None:
            domain = vecsps.DirectSum
        if isinstance(domain, vecsps.VectorSpace):
            pass
        elif callable(domain):
            domain = domain(*(op.domain for op in self.ops))
        else:
            raise TypeError('domain={} is neither a VectorSpace nor callable'.format(domain))
        assert all(op.domain == d for op, d in zip(ops, domain))

        if codomain is None:
            codomain = vecsps.DirectSum
        if isinstance(codomain, vecsps.VectorSpace):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
        else:
            raise TypeError('codomain={} is neither a VectorSpace nor callable'.format(codomain))
        assert all(op.codomain == c for op, c in zip(ops, codomain))

        super().__init__(domain=domain, codomain=codomain, linear=all(op.linear for op in ops))

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        elms = self.domain.split(x)
        if differentiate:
            linearizations = [op.linearize(elm,adjoint_derivative=adjoint_derivative) for op, elm in zip(self.ops, elms)]
            self._derivs = [l[1] for l in linearizations]
            if adjoint_derivative:
                self._adjoint_derivs = [l[2] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        elif adjoint_derivative:
            linearizations = [op.linearize(elm,adjoint_derivative=True) for op, elm in zip(self.ops, elms)]
            self._adjoint_derivs = [l[1] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        else:
            return self.codomain.join(*(op(elm) for op, elm in zip(self.ops, elms)))

    def _derivative(self, x):
        elms = self.domain.split(x)
        return self.codomain.join(
            *(deriv(elm) for deriv, elm in zip(self._derivs, elms))
        )

    def _adjoint(self, y):
        elms = self.codomain.split(y)
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        return self.domain.join(
            *(op.adjoint(elm) for op, elm in zip(ops, elms))
        )
    
    def _adjoint_derivative(self, x):
        elms = self.domain.split(x)
        return self.domain.join(
            *(adjoint_deriv(elm) for adjoint_deriv, elm in zip(self._adjoint_derivs, elms))
        )

    @util.memoized_property
    def inverse(self):
        """The component-wise inverse as a `DirectSum`, if all of them exist."""
        return DirectSum(
            *(op.inverse for op in self.ops),
            domain=self.codomain,
            codomain=self.domain
        )

    def __repr__(self):
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)

class VectorOfOperators(Operator):
    r"""Vector of operators. For

    .. math::
        T_i \colon X \to Y_i

    we define

    .. math::
        T := VectorOfOperators(T_i) \colon X \to DirectSum(Y_i)

    by :math:`T(x)_i := T_i(x)`. 
    
    Parameters
    ----------
    *ops : tuple of Operator
    codomain : vecsps.VectorSpace or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, ops,  domain=None, codomain=None):
        assert all([isinstance(op, Operator) for op in ops])
        assert ops
        self.ops = ops
        r"""List of all Operators :math:`(T_1,\dots,T_n)`"""

        if domain is None:
            self.domain = self.ops[0].domain
        else:
            self.domain = domain
        assert all(op.domain == self.domain for op in self.ops)

        if codomain is None:
            codomain = vecsps.DirectSum(*tuple([op.codomain for op in ops]))
        if isinstance(codomain, vecsps.VectorSpace):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
        else:
            raise TypeError('codomain={} is neither a VectorSpace nor callable'.format(codomain))
        assert all(op.codomain == c for op, c in zip(ops, codomain))

        super().__init__(domain=self.domain, codomain=codomain, linear=all(op.linear for op in ops))

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        if differentiate:
            linearizations = [op.linearize(x,adjoint_derivative=adjoint_derivative) for op in self.ops]
            self._derivs = [l[1] for l in linearizations]
            if adjoint_derivative:
                self._adjoint_derivs = [l[2] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        else:
            return self.codomain.join(*(op(x) for op in self.ops))

    def _derivative(self, x):
        return self.codomain.join(
            *(deriv(x) for deriv in self._derivs)
        )

    def _adjoint(self, y):
        elms = self.codomain.split(y)
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        result = self.domain.zeros()    
        for op, elm in zip(ops, elms):
            result += op.adjoint(elm)
        return result
    
    def _adjoint_derivative(self, x):
        result = self.domain.zeros() 
        for adjoint_deriv in self._adjoint_derivs:
            result += adjoint_deriv(x)
        return result

    @util.memoized_property
    def __repr__(self):
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)

class MatrixOfOperators(Operator):
    r"""Matrix of operators. For

    .. math::
        T_ij \colon X_j \to Y_i

    we define

    .. math::
        T := MatrixOfOperators(T_ij) \colon DirectSum(X_j) \to DirectSum(Y_i)

    by :math:`T(x)_i := \sum_j T_ij(x_j)`. 
    
    Parameters
    ----------
    *ops : list of list of operators [[T_00, T_10, ...], [T_01, T_11, ...], ...]
           zero operators should be given by None's 
    domain, codomain : vecsps.VectorSpace or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, ops,  domain=None, codomain=None):
        ops_flat = [op for op_col in ops for op in op_col]
        assert all((isinstance(op, Operator) or op==None) for op in ops_flat)
        self.ops = ops
        r""" Matrix of Operators :math:`(T_ij)`"""

        domains = [None]*len(ops)
        for j in range(len(ops)):
            for i in range(len(ops[0])):
                if ops[j][i]:
                    if domains[j]:
                        assert domains[j] == ops[j][i].domain
                    else:    
                        domains[j] = ops[j][i].domain
        assert None not in domains

        if domain is None:
            domain = vecsps.DirectSum
        if isinstance(domain, vecsps.VectorSpace):
            pass
        elif callable(domain):
            domain = domain(*tuple(domains))
        else:
            raise TypeError('domain={} is neither a VectorSpace nor callable'.format(domain))

        codomains = [None]*len(ops[0])
        for i in range(len(ops[0])):
            for j in range(len(ops)):
                if ops[j][i]:
                    if codomains[i]:
                        assert codomains[i] == ops[j][i].codomain
                    else:
                        codomains[i] = ops[j][i].codomain
        assert None not in codomains

        if codomain is None:
            codomain = vecsps.DirectSum
        if isinstance(codomain, vecsps.VectorSpace):
            pass
        elif callable(codomain):
            codomain = codomain(*tuple(codomains))
        else:
            raise TypeError('codomain={} is neither a VectorSpace nor callable'.format(domain))
        
        super().__init__(domain=domain, codomain=codomain, linear=all(op==None or op.linear for op in ops_flat))

    def _eval(self, x, differentiate=False):
        x_comp = self.domain.split(x)
        res = self.codomain.split(self.codomain.zeros()) 
        Tprime = []
        Tadjprime = []
        for T_j, x_j in zip(self.ops,x_comp):
            Tprime_j = []
            Tadjprime_j = []
            for T_ij,res_i in zip(T_j,res):
                if differentiate:
                    if T_ij:
                        res,deriv = T_ij.linearize(x_j)
                        res_i += res
                        Tprime_ij = deriv
                    else:
                        Tprime_ij = None
                        Tadjprime_ij = None
                    Tprime_j.append(Tprime_ij)
                else:   
                    if T_ij:
                        res_i += T_ij(x_j)
            Tprime.append(Tprime_j)
            Tadjprime.append(Tadjprime_j)
        if differentiate:
            self._derivs = Tprime
        return self.codomain.join(*(res_i for res_i in res))

    def _derivative(self, x):
        res = self.codomain.split(self.codomain.zeros())
        x_comp = self.domain.split(x)
        for Tprime_j, x_j in zip(self._derivs,x_comp):
            for Tprime_ij,res_i in zip(Tprime_j,res):
                if Tprime_ij:
                    res_i += Tprime_ij(x_j)
        return self.codomain.join(*(res_i for res_i in res))

    def _adjoint(self, y):
        y_comp = self.codomain.split(y)
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        res_comp = self.domain.split(self.domain.zeros())    
        for Tprime_j, res_j in zip(ops, res_comp):
            for Tprime_ij, y_i in zip(Tprime_j,y_comp):
                if Tprime_ij:
                    res_j += Tprime_ij.adjoint(y_i)
        return self.domain.join(*(res_j for res_j in res_comp))

    @util.memoized_property
    def __repr__(self):
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)


class Exponential(Operator):
    r"""The pointwise exponential operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space.
    """

    def __init__(self, domain):
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        if differentiate or adjoint_derivative:
            self._exponential_factor = np.exp(x)
            return self._exponential_factor
        return np.exp(x)

    def _derivative(self, x):
        return self._exponential_factor * x

    def _adjoint(self, y):
        return self._exponential_factor.conj() * y


class RealPart(Operator):
    r"""The pointwise real part operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpace.real_space`.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.real.copy()

    def _adjoint(self, y):
        return y.copy()


class ImaginaryPart(Operator):
    r"""The pointwise imaginary part operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpace.real_space`.
    """

    def __init__(self, domain):
        if domain:
            assert domain.is_complex
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.imag.copy()

    def _adjoint(self, y):
        return 1j * y


class SquaredModulus(Operator):
    r"""The pointwise squared modulus operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpace.real_space`.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        if differentiate or adjoint_derivative:
            self._factor = 2 * x
        return x.real**2 + x.imag**2

    def _derivative(self, h):
        return (self._factor.conj() * h).real

    def _adjoint(self, y):
        return self._factor * y


class Zero(Operator):
    r"""The constant zero operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space.
    codomain : regpy.vecsps.VectorSpace, optional
        The vector space of the codomain. Defaults to `domain`.
    """
    def __init__(self, domain, codomain=None):
        if codomain is None:
            codomain = domain
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return self.codomain.zeros()

    def _adjoint(self, x):
        return self.domain.zeros()

class ApproximateHessian(Operator):
    r"""An approximation of the Hessian of a `regpy.functionals.Functional` at some point, computed
    using finite differences of it `gradient` if it is implemented for that functional.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    x : array-like
        The point at which to evaluate the Hessian.
    stepsize : float, optional
        The stepsize for the finite difference approximation.
    """
    def __init__(self, func, x, stepsize=1e-8):
        assert isinstance(func, functionals.Functional)
        assert hasattr(func,"gradient")
        self.gradx = func.gradient(x)
        """The gradient at `x`"""
        self.func = func
        self.x = x.copy()
        self.stepsize = stepsize
        # linear=True is a necessary lie
        super().__init__(func.domain, func.domain, linear=True)
        self.log.info('Using approximate Hessian of functional {}'.format(self.func))

    def _eval(self, h):
        grad = self.func.gradient(self.x + self.stepsize * h)
        return grad - self.gradx

    def _adjoint(self, x):
        return self._eval(x)


