class Params:
    def __init__(self, domain, range, **kwargs):
        self.domain = domain
        self.range = range
        self.__dict__.update(kwargs)


class GenericData:
    def __init__(self, params):
        pass


class RevokedError(Exception):
    pass


class Revocable:
    def __init__(self, val):
        self.__val = val

    @classmethod
    def take(cls, other):
        return cls(other.revoke())

    def __enter__(self):
        if self.__val is None:
            raise RevokedError
        return self.__val

    def __exit__(self, *args):
        pass

    def revoke(self):
        with self as val:
            self.__val = None
            return val


class NotImplemented:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Operator:
    from itreg.util import classlogger as log

    Data = GenericData
    Derivative = NotImplemented
    nocopy = frozenset(['params'])
    reuse_data = True

    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        if self.reuse_data:
            if not hasattr(self, '_owned_data'):
                self._owned_data = self.Data(self.params)
            data = self._owned_data
        else:
            data = self.Data(self.params)
        return self._eval(x, data=data, differentiate=False)

    def _eval(self, x, data, differentiate=False, **kwargs):
        raise NotImplementedError

    def linearize(self, x):
        try:
            self._shared_data = Revocable.take(self._shared_data)
        except AttributeError:
            self._shared_data = Revocable(self.Data(self.params))
        with self._shared_data as data:
            y = self._eval(x, data=data, differentiate=True)
        deriv = self.Derivative(self.params, self._shared_data)
        return y, deriv

    def __deepcopy__(self, memo):
        from copy import deepcopy
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(result)] = result
        for k, v in self.__dict__.items():
            if k not in self.nocopy:
                v = deepcopy(v, memo)
            setattr(result, k, v)
        return result

    def __str__(self):
        return 'Operator({}, {})'.format(type(self).__qualname__, self.params)

    @property
    def domain(self):
        return self.params.domain

    @property
    def range(self):
        return self.params.range


class LinearOperator(Operator):
    Data = NotImplemented

    @property
    def Derivative(self):
        return type(self)

    def __init__(self, params, context=None):
        super().__init__(params)
        self.context = context

    def __call__(self, x):
        if self.context is not None:
            with self.context as data:
                return self._eval(x, data=data)
        else:
            return self._eval(x)

    def adjoint(self, y):
        if self.context is not None:
            with self.context as data:
                return self._adjoint(y, data=data)
        else:
            return self._adjoint(y)

    def hermitian(self, y):
        return self.domain.gram_inv(self.adjoint(self.range.gram(y)))

    def _eval(self, x, data=None, **kwargs):
        raise NotImplementedError

    def _adjoint(self, x, data=None, **kwargs):
        raise NotImplementedError

    def linearize(self, x):
        return self(x), self

    def norm(self, iterations=10):
        import numpy as np
        h = self.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for i in range(iterations):
            h = h / norm
            h = self.hermitian(self(h))
            norm = np.sqrt(np.real(np.vdot(h, h)))
        return np.sqrt(norm)


class Adjoint(LinearOperator):
    def __init__(self, op):
        self.op = op

    def _eval(self, y, **kwargs):
        return self.op._adjoint(self, y)

    def _adjoint(self, y, **kwargs):
        return self.op._eval(self, y)

    @property
    def domain(self):
        return self.op.range

    @property
    def range(self):
        return self.op.domain
