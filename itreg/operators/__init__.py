import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .. import spaces
from .. import util


class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


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
            raise RuntimeError('Attempted to use revoked reference')

    def revoke(self):
        val = self.get()
        del self.__val
        return val


class BaseOperator:
    log = util.classlogger

    def __init__(self, domain, codomain, **kwargs):
        self.__set_params(Params(domain=domain, codomain=codomain, **kwargs))
        self._alloc()

    def __set_params(self, params):
        self._params = params
        self.__dict__.update(params.__dict__)

    def _alloc(self):
        pass

    def clone(self):
        cls = type(self)
        instance = cls.__new__(cls)
        instance.__set_params(self._params)
        instance._alloc()
        return instance

    def __call__(self, x):
        raise NotImplementedError

    def linearize(self, x):
        raise NotImplementedError


class NonlinearOperator(BaseOperator):
    def __call__(self, x):
        assert x in self.domain
        self.__revoke()
        y = self._eval(x, differentiate=False)
        assert y in self.codomain
        return y

    def linearize(self, x):
        assert x in self.domain
        self.__revoke()
        y = self._eval(x, differentiate=True)
        assert y in self.codomain
        deriv = Derivative(self.__get_handle())
        return y, deriv

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

    def __mul__(self, other):
        if isinstance(other, BaseOperator):
            return NonlinearOperatorComposition(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, BaseOperator):
            return NonlinearOperatorComposition(other, self)
        else:
            return NotImplemented


class LinearOperator(BaseOperator):
    def __call__(self, x):
        assert x in self.domain
        y = self._eval(x)
        assert y in self.codomain
        return y

    def linearize(self, x):
        return self(x), self

    @util.memoized_property
    def adjoint(self):
        return Adjoint(self)

    def norm(self, iterations=10):
        h = self.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for i in range(iterations):
            h = h / norm
            h = self.hermitian(self(h))
            norm = np.sqrt(np.real(np.vdot(h, h)))
        return np.sqrt(norm)

    def _eval(self, x):
        raise NotImplementedError

    def _adjoint(self, x):
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, LinearOperator):
            return LinearOperatorComposition(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, LinearOperator):
            return LinearOperatorComposition(other, self)
        else:
            return NotImplemented


class Adjoint(LinearOperator):
    def __init__(self, op):
        super().__init__(op.codomain, op.domain, op=op)

    def _eval(self, x):
        return self.op._adjoint(x)

    def _adjoint(self, x):
        return self.op._eval(x)

    @property
    def adjoint(self):
        return self.op


class Derivative(LinearOperator):
    def __init__(self, op):
        if not isinstance(op, Revocable):
            # Wrap plain operators in a Revocable that will never be revoked to
            # avoid case distinctions below.
            op = Revocable(op)
        super().__init__(op.get().domain, op.get().codomain, op=op)

    def clone(self):
        raise RuntimeError("Derivatives can't be cloned")

    def _eval(self, x):
        return self.op.get()._derivative(x)

    def _adjoint(self, x):
        return self.op.get()._adjoint(x)


class NonlinearOperatorComposition(NonlinearOperator):
    def __init__(self, f, g):
        assert f.domain == g.codomain
        super().__init__(g.domain, f.codomain, f=f, g=g)

    def _eval(self, x, differentiate=False):
        if differentiate:
            gx, dg_x = self.g.linearize(x)
            y, df_gx = self.f.linearize(gx)
            self._dg_x = dg_x
            self._df_gx = df_gx
            return y
        else:
            return self.f(self.g(x))

    def _derivative(self, x):
        return self._df_gx(self._dg_x(x))

    def _adjoint(self, y):
        return self._dg_x.adjoint(self._df_gx.adjoint(y))


class LinearOperatorComposition(LinearOperator):
    def __init__(self, f, g):
        assert f.domain == g.codomain
        super().__init__(g.domain, f.codomain, f=f, g=g)

    def _eval(self, x):
        return self.f(self.g(x))

    def _adjoint(self, y):
        return self.g.adjoint(self.f.adjoint(y))


class Identity(LinearOperator):
    def __init__(self, domain):
        super().__init__(domain, domain)

    def _eval(self, x):
        return x

    def _adjoint(self, x):
        return x


class CholeskyInverse(LinearOperator):
    def __init__(self, domain, matrix):
        matrix = np.asarray(matrix)
        assert matrix.shape == (domain.size,) * 2
        assert util.is_real_dtype(matrix)
        super().__init__(
            domain=domain,
            codomain=domain,
            factorization=cho_factor(matrix))

    def _eval(self, x):
        return self.domain.fromflat(
            cho_solve(self.factorization, self.domain.flatten(x)))

    def _adjoint(self, x):
        return self._eval(x)


class CoordinateProjection(LinearOperator):
    def __init__(self, domain, mask):
        mask = np.asarray(mask)
        assert mask.dtype == bool
        assert mask.shape == domain.shape
        super().__init__(
            domain=domain,
            codomain=spaces.GenericDiscretization(np.sum(mask), dtype=domain.dtype),
            mask=mask)

    def _eval(self, x):
        return x[self.mask]

    def _adjoint(self, x):
        y = self.domain.zeros()
        y[self.mask] = x
        return y


class PointwiseMultiplication(LinearOperator):
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        assert factor.ndim <= domain.ndim
        for sf, sd in zip(factor.shape[::-1], domain.shape[::-1]):
            assert sf == sd or sf == 1
        assert domain.is_complex or not util.is_complex_dtype(factor)
        super().__init__(domain, domain, factor=factor)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        return np.conj(self.factor) * x


class FourierTransform(LinearOperator):
    def __init__(self, domain):
        assert isinstance(domain, spaces.UniformGrid)
        super().__init__(domain, domain.dualgrid)

    def _eval(self, x):
        return self.domain.fft(x)

    def _adjoint(self, y):
        return self.domain.ifft(y)


from .mediumscattering import MediumScattering
from .volterra import Volterra, NonlinearVolterra
