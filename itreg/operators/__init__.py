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

    def __init__(self, domain, codomain, linear=False):
        assert isinstance(domain, spaces.Discretization)
        assert isinstance(codomain, spaces.Discretization)
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
        assert x in self.domain
        if self.linear:
            y = self._eval(x)
        else:
            self.__revoke()
            y = self._eval(x, differentiate=False)
        assert y in self.codomain
        return y

    def linearize(self, x):
        if self.linear:
            return self(x), self
        else:
            assert x in self.domain
            self.__revoke()
            y = self._eval(x, differentiate=True)
            assert y in self.codomain
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

    def norm(self, iterations=10):
        assert self.linear
        h = self.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for i in range(iterations):
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
        else:
            return NotImplemented


# NonlinearOperator and LinearOperator are added for compatibility, will be
# removed when possible.

class NonlinearOperator(Operator):
    def __init__(self, domain, codomain):
        super().__init__(domain, codomain)


class LinearOperator(Operator):
    def __init__(self, domain, codomain):
        super().__init__(domain, codomain, linear=True)


class Adjoint(Operator):
    def __init__(self, op):
        self.op = op
        super().__init__(op.codomain, op.domain, linear=True)

    def _eval(self, x):
        return self.op._adjoint(x)

    def _adjoint(self, x):
        return self.op._eval(x)

    @property
    def adjoint(self):
        return self.op

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
    """
    Implements a linear combination of any number of operators with sepcified scalars.
    Domains and ranges must be the same.
    E.g. : F(x) = a * f(x) + b * g(x) + c * h(x)
    """
    # TODO split into Scaled and Sum classes, allow nonlinear operators, add __mul__ etc
    # TODO nonlinear case
    def __init__(self, operators, scalars):
        assert all(op.linear for op in operators)
        for i in range(len(operators)-1):
            f = operators[i]
            g = operators[i+1]
            assert f.domain == g.domain and f.range == g.range, "Domains and Ranges of Operators must be the same"
        self.operators = operators
        self.scalars = scalars
        super().__init__(f.domain, f.range, linear=True)

    def _eval(self, x):
        res = 0
        for i in range(len(self.operators)-1):
            res += self.paramas.scalar[i] * self.operators[i](x)
        return res

    def _adjoint(self, x):
        res = 0
        for i in range(len(self.operators)-1):
            res += self.paramas.scalar[i] * self.operators[i].adjoint(x)
        return res


class Composition(Operator):
    def __init__(self, *ops):
        for f, g in zip(ops, ops[1:]):
            assert f.domain == g.codomain
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

    def __repr__(self):
        return util.make_repr(self, *self.ops)


class Identity(Operator):
    def __init__(self, domain):
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return x

    def _adjoint(self, x):
        return x

    def __repr__(self):
        return util.make_repr(self, self.domain)


class CholeskyInverse(Operator):
    def __init__(self, op):
        assert op.linear
        assert op.domain == op.codomain
        domain = op.domain
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


class PointwiseMultiplication(Operator):
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
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

    def __repr__(self):
        return util.make_repr(self, self.domain, self.factor)


class FourierTransform(Operator):
    def __init__(self, domain):
        assert isinstance(domain, spaces.UniformGrid)
        super().__init__(domain, domain.dualgrid, linear=True)

    def _eval(self, x):
        return self.domain.fft(x)

    def _adjoint(self, y):
        return self.domain.ifft(y)

    def __repr__(self):
        return util.make_repr(self, self.domain)


class MatrixMultiplication(Operator):
    """
    Implements a matrix multiplication with a given matrix.
    """

    # TODO complex case
    def __init__(self, matrix):
        self.matrix = matrix
        super().__init__(
            domain=spaces.Discretization(matrix.shape[1]),
            codomain=spaces.Discretization(matrix.shape[0]),
            linear=True
        )

    def _eval(self, x):
        return self.params.matrix @ x

    def _adjoint(self, y):
        return self.params.matrix.T @ y


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


class Scale(Operator):
    # TODO complex case
    def __init__(self, scale, domain):
        self.scale = scale
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.scale * x

    def _adjoint(self, y):
        return self.scale * y


class BlockDiagonal(Operator):
    def __init__(self, *blocks, flatten=True):
        assert all(isinstance(block, Operator) for block in blocks)
        self.blocks = []
        for block in blocks:
            if flatten and isinstance(block, type(self)):
                self.blocks.extend(block.blocks)
            else:
                self.blocks.append(block)
        super().__init__(
            domain=discrs.Product(*(block.domain for block in self.blocks), flatten=False),
            codomain=discrs.Product(*(block.codomain for block in self.blocks), flatten=False),
            linear=all(block.linear for block in blocks)
        )

    def _eval(self, x, differentiate=False):
        elms = self.domain.split(x)
        if differentiate:
            linearizations = [block.linearize(elm) for block, elm in zip(self.blocks, elms)]
            self._derivs = [l[1] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        else:
            return self.codomain.join(*(block(elm) for block, elm in zip(self.blocks, elms)))

    def _derivative(self, x):
        elms = self.domain.split(x)
        return self.codomain.join(
            *(deriv(elm) for deriv, elm in zip(self._derivs, elms)))

    def _adjoint(self, y):
        elms = self.codomain.split(y)
        if self.linear:
            blocks = self.blocks
        else:
            blocks = self._derivs
        return self.domain.join(
            *(block.adjoint(elm) for block, elm in zip(blocks, elms)))


from .mediumscattering import MediumScattering
from .volterra import Volterra, NonlinearVolterra
