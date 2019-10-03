from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .. import spaces, util
from ..spaces import discrs


class Revocable:
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
    log = util.classlogger

    def __init__(self, domain=None, codomain=None, linear=False):
        assert not domain or isinstance(domain, spaces.Discretization)
        assert not codomain or isinstance(codomain, spaces.Discretization)
        self.domain, self.codomain = domain, codomain
        self.linear = linear
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

    def linearize(self, x):
        if self.linear:
            return self(x), self
        else:
            assert not self.domain or x in self.domain
            self.__revoke()
            y = self._eval(x, differentiate=True)
            assert not self.codomain or y in self.codomain
            deriv = Derivative(self.__get_handle())
            return y, deriv

    @util.memoized_property
    def adjoint(self):
        assert self.linear
        return Adjoint(self)

    def __revoke(self):
        try:
            self.__handle = Revocable.take(self.__handle)
        except AttributeError:
            pass

    def __get_handle(self):
        try:
            return self.__handle
        except AttributeError:
            self.__handle = Revocable(self)
            return self.__handle

    def _eval(self, x, differentiate=False):
        raise NotImplementedError

    def _derivative(self, x):
        raise NotImplementedError

    def _adjoint(self, y):
        raise NotImplementedError

    @property
    def inverse(self):
        raise NotImplementedError

    def norm(self, iterations=10):
        assert self.linear
        h = self.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for _ in range(iterations):
            h = h / norm
            # TODO gram matrices
            h = self.adjoint(self(h))
            norm = np.sqrt(np.real(np.vdot(h, h)))
        return np.sqrt(norm)

    def __mul__(self, other):
        if isinstance(other, Operator):
            return Composition(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return Composition(other, self)
        elif other == 1:
            return self
        elif (
            np.isreal(other) or
            (np.iscomplex(other) and (not self.codomain or self.codomain.is_complex))
        ):
            return LinearCombination((other, self))
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Operator):
            return LinearCombination(self, other)
        else:
            return NotImplemented


# NonlinearOperator and LinearOperator are added for compatibility, will be
# removed when possible.

class NonlinearOperator(Operator):
    def __init__(self, domain=None, codomain=None):
        super().__init__(domain, codomain)


class LinearOperator(Operator):
    def __init__(self, domain=None, codomain=None):
        super().__init__(domain, codomain, linear=True)


class Adjoint(Operator):
    def __init__(self, op):
        assert op.linear
        self.op = op
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
    def __init__(self, op):
        if not isinstance(op, Revocable):
            # Wrap plain operators in a Revocable that will never be revoked to
            # avoid case distinctions below.
            op = Revocable(op)
        self.op = op
        _op = op.get()
        super().__init__(_op.domain, _op.codomain, linear=True)

    def _eval(self, x):
        return self.op.get()._derivative(x)

    def _adjoint(self, x):
        return self.op.get()._adjoint(x)

    def __repr__(self):
        return util.make_repr(self, self.op.get())


class LinearCombination(Operator):
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
        self.ops = []
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

    def _eval(self, x, differentiate=False):
        y = self.codomain.zeros()
        if differentiate:
            self._derivs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if differentiate:
                z, deriv = op.linearize(x)
                self._derivs.append(deriv)
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
        x = self.codomain.zeros()
        for coeff, op in zip(self.coeffs, ops):
            x += np.conj(coeff) * op.adjoint(y)
        return x

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
    def __init__(self, *ops):
        for f, g in zip(ops, ops[1:]):
            assert not f.domain or not g.codomain or f.domain == g.codomain
        self.ops = []
        for op in ops:
            assert isinstance(op, Operator)
            if isinstance(op, Composition):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        super().__init__(
            self.ops[-1].domain, self.ops[0].codomain,
            linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False):
        y = x
        if differentiate:
            self._derivs = []
            for op in self.ops[::-1]:
                y, deriv = op.linearize(y)
                self._derivs.insert(0, deriv)
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

    @util.memoized_property
    def inverse(self):
        return Composition(*(op.inverse for op in self.ops[::-1]))

    def __repr__(self):
        return util.make_repr(self, *self.ops)


class Identity(Operator):
    def __init__(self, domain):
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return x.copy()

    def _adjoint(self, x):
        return x.copy()

    @property
    def inverse(self):
        return self

    def __repr__(self):
        return util.make_repr(self, self.domain)


class CholeskyInverse(Operator):
    def __init__(self, op, matrix=None):
        assert op.linear
        assert op.domain and op.domain == op.codomain
        domain = op.domain
        if matrix is None:
            matrix = np.empty((domain.size,) * 2, dtype=float)
            for j, elm in enumerate(domain.iter_basis()):
                matrix[j, :] = domain.flatten(op(elm))
        self.factorization = cho_factor(matrix)
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
        return self.op

    def __repr__(self):
        return util.make_repr(self, self.op)


class CoordinateProjection(Operator):
    def __init__(self, domain, mask):
        mask = np.asarray(mask)
        assert mask.dtype == bool
        assert mask.shape == domain.shape
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=spaces.Discretization(np.sum(mask), dtype=domain.dtype),
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


class Multiplication(Operator):
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if domain:
            assert factor.ndim <= domain.ndim
            for sf, sd in zip(factor.shape[::-1], domain.shape[::-1]):
                assert sf == sd or sf == 1
            assert domain.is_complex or not util.is_complex_dtype(factor)
        self.factor = factor
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        return np.conj(self.factor) * x

    @util.memoized_property
    def inverse(self):
        sav = np.seterr(divide='raise')
        try:
            return Multiplication(self.domain, 1 / self.factor)
        finally:
            np.seterr(**sav)

    def __repr__(self):
        return util.make_repr(self, self.domain, self.factor)


class FourierTransform(Operator):
    def __init__(self, domain, centered=False, axes=None):
        assert isinstance(domain, spaces.UniformGrid)
        frqs = domain.frequencies(centered=centered, axes=axes)
        if centered:
            codomain = discrs.UniformGrid(*frqs, dtype=complex)
        else:
            # In non-centered case, the frequencies are not ascencing, so even using Grid here is slighty questionable.
            codomain = discrs.Grid(*frqs, dtype=complex)
        super().__init__(domain, codomain, linear=True)
        self.centered = centered
        self.axes = axes

    def _eval(self, x):
        y = np.fft.fftn(x, axes=self.axes, norm='ortho')
        if self.centered:
            return np.fft.fftshift(y, axes=self.axes)
        else:
            return y

    def _adjoint(self, y):
        if self.centered:
            y = np.fft.ifftshift(y, axes=self.axes)
        x = np.fft.ifftn(y, axes=self.axes, norm='ortho')
        if self.domain.is_complex:
            return x
        else:
            return np.real(x)

    @property
    def inverse(self):
        return self.adjoint

    def __repr__(self):
        return util.make_repr(self, self.domain)


class MatrixMultiplication(Operator):
    """
    Implements a matrix multiplication with a given matrix.
    """

    # TODO complex case
    def __init__(self, matrix, inverse=None):
        self.matrix = matrix
        super().__init__(
            domain=spaces.Discretization(matrix.shape[1]),
            codomain=spaces.Discretization(matrix.shape[0]),
            linear=True
        )
        self._inverse = inverse

    def _eval(self, x):
        return self.matrix @ x

    def _adjoint(self, y):
        return self.matrix.T @ y

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
                # TODO LU, QR
                return CholeskyInverse(self, matrix=self.matrix)
        raise NotImplementedError

    def __repr__(self):
        return util.make_repr(self, self.matrix)


class Power(NonlinearOperator):
    # TODO complex case
    def __init__(self, power, domain):
        self.power = power
        super().__init__(domain, domain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = self.power * x**(self.power - 1)
        return x ** self.power

    def _derivative(self, x):
        return self._factor * x

    def _adjoint(self, y):
        return self._factor * y


class DirectSum(Operator):
    """The direct sum of operators. For

        T_i : X_i -> Y_i

    the direct sum

        T := DirectSum(T_i) : DirectSum(X_i) -> DirectSum(Y_i)

    is given by `T(x)_i := T_i(x_i)`. As a matrix, this is the block-diagonal
    with blocks (T_i).

    Parameters
    ----------
    *ops : Operator tuple
    flatten : bool, optional
        If True, summands that are themselves direct sums will be merged with
        this one. Default: False.
    domain, codomain : discrs.Discretization or callable, optional
        Either the underlying discretizations or factory functions that will be
        called with all summands' discretizations passed as arguments and should
        return a discrs.DirectSum instance. Default: discrs.DirectSum.
    """

    def __init__(self, *ops, flatten=False, domain=discrs.DirectSum, codomain=discrs.DirectSum):
        assert all(isinstance(op, Operator) for op in ops)
        self.ops = []
        for op in ops:
            if flatten and isinstance(op, type(self)):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)

        if isinstance(domain, discrs.Discretization):
            pass
        elif callable(domain):
            domain = domain(*(op.domain for op in self.ops))
        else:
            raise TypeError('domain={} is neither a Discretization nor callable'.format(domain))
        assert all(op.domain == d for op, d in zip(ops, domain))

        if isinstance(codomain, discrs.Discretization):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
        else:
            raise TypeError('codomain={} is neither a Discretization nor callable'.format(codomain))
        assert all(op.codomain == c for op, c in zip(ops, codomain))

        super().__init__(domain=domain, codomain=codomain, linear=all(op.linear for op in ops))

    def _eval(self, x, differentiate=False):
        elms = self.domain.split(x)
        if differentiate:
            linearizations = [op.linearize(elm) for op, elm in zip(self.ops, elms)]
            self._derivs = [l[1] for l in linearizations]
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

    @util.memoized_property
    def inverse(self):
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


class Exponential(Operator):
    """The pointwise exponential operator exp.

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.

    Notes
    -----
    The pointwise exponential operator :math:`exp` is defined as

    .. math:: exp(f)(x) = exp(f(x)).

    Its discrete form is

    .. math:: exp(x)_i = exp(x_i).
    """

    def __init__(self, domain):
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


class RealPart(Operator):
    """The pointwise real part operator

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.
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
    """The pointwise imaginary part operator

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.
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
    """The pointwise squared modulus operator.

    Parameters
    ----------
    domain : :class:`~itreg.spaces.Space`
        The domain on which the operator is defined.

    Notes
    -----
    The pointwise exponential operator :math:`exp` is defined as

    .. math:: (|f|^2)(x) = |f(x)|^2 = Re(f(x))^2 + Im(f(x))^2

    where :math:`Re` and :math:`Im` denote the real- and imaginary parts.
    Its discrete form is

    .. math:: (|x|^2)_i = Re(x_i)^2 + Im(x_i)^2.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = 2 * x
        return x.real**2 + x.imag**2

    def _derivative(self, x):
        return (self._factor.conj() * x).real

    def _adjoint(self, y):
        return self._factor * y


from .mediumscattering import MediumScatteringFixed, MediumScatteringOneToMany
from .volterra import Volterra, NonlinearVolterra
