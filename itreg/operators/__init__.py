from itreg.util import emptycontext, classlogger

from copy import deepcopy
import numpy as np


class OperatorImplementation:
    log = classlogger

    def eval(self, params, x, *kwargs):
        raise NotImplementedError

    def adjoint(self, params, y, *kwargs):
        raise NotImplementedError

    def abs_squared(self, params, x, *kwargs):
        aux = self.eval(params, x, **kwargs)
        aux = params.range.gram(aux)
        aux = self.adjoint(params, aux, **kwargs)
        return params.domain.gram_inv(aux)


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


class BaseOperator:
    nocopy = {'params'}

    def __init__(self, params):
        self.params = params

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(result)] = result
        for k, v in self.__dict__.items():
            if k not in self.nocopy:
                v = deepcopy(v, memo)
            setattr(result, k, v)
        return result

    def __str__(self):
        return 'Operator({}, {})'.format(
            type(self.operator).__qualname__, self.params)

    @property
    def domain(self):
        return self.params.domain

    @property
    def range(self):
        return self.params.range


class NonlinearOperator(BaseOperator):
    Data = GenericData

    def __init__(self, params):
        super().__init__(params)
        self.__owned_data = None
        self.__shared_data = None

    def __call__(self, x):
        self.__owned_data = self.__owned_data or self.Data(self.params)
        return self.operator.eval(
            self.params, x, data=self.__owned_data, differentiate=False)

    def linearize(self, x):
        if self.__shared_data is None:
            self.__shared_data = Revocable(self.Data(self.params))
        else:
            self.__shared_data = Revocable.take(self.__shared_data)
        with self.__shared_data as data:
            y = self.operator.eval(
                self.params, x, data=data, differentiate=True)
        deriv = Derivative(self.derivative, self.params, self.__shared_data)
        return y, deriv


class LinearOperator(BaseOperator):
    handle = emptycontext

    def __call__(self, x):
        with self.handle as data:
            return self.operator.eval(self.params, x, data=data)

    def adjoint(self, x):
        with self.handle as data:
            return self.operator.adjoint(self.params, x, data=data)

    def abs_squared(self, x):
        with self.handle as data:
            return self.operator.abs_squared(self.params, x, data=data)

    def linearize(self, x):
        return self(x), self

    @property
    def derivative(self):
        return self.operator

    def norm(self, iterations=10):
        h = self.domain.rand()
        norm = np.sqrt(np.real(np.vdot(h, h)))
        for i in range(iterations):
            h = h / norm
            h = self.abs_squared(h)
            norm = np.sqrt(np.real(np.vdot(h, h)))
        return np.sqrt(norm)


class Derivative(LinearOperator):
    def __init__(self, operator, params, handle):
        super().__init__(params)
        self.operator = operator
        self.handle = handle


from .volterra import Volterra, NonlinearVolterra
from .weighted import Weighted
