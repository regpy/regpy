from itreg.util import classlogger, memoized_property


class Params:
    def __init__(self, domain, range, **kwargs):
        self.domain = domain
        self.range = range
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
    log = classlogger

    def __init__(self, params):
        self.params = params

    @property
    def domain(self):
        return self.params.domain

    @property
    def range(self):
        return self.params.range

    def clone(self):
        cls = type(self)
        instance = cls.__new__(cls)
        instance.params = self.params
        return instance

    def __call__(self, x):
        raise NotImplementedError

    def linearize(self, x):
        raise NotImplementedError


class NonlinearOperator(BaseOperator):
    def __call__(self, x):
        self.__revoke()
        return self._eval(self.params, x, differentiate=False)

    def linearize(self, x):
        self.__revoke()
        y = self._eval(self.params, x, differentiate=True)
        deriv = Derivative(self.__get_handle())
        return y, deriv

    def __revoke(self):
        try:
            self.__handle = Revocable.take(self.__handle)
            self.log.info('revoked')
        except AttributeError:
            pass

    def __get_handle(self):
        try:
            return self.__handle
        except AttributeError:
            self.__handle = Revocable(self)
            return self.__handle

    def _eval(self, params, x, differentiate=False):
        raise NotImplementedError

    def _deriv(self, params, x):
        raise NotImplementedError

    def _adjoint(self, params, y):
        raise NotImplementedError


class LinearOperator(BaseOperator):
    def __call__(self, x):
        return self._eval(self.params, x)

    def linearize(self, x):
        return self(x), self

    @memoized_property
    def adjoint(self):
        return Adjoint(self)

    def hermitian(self, y):
        # TODO Once gram matrices are turned into operators and operator
        # composition works, this can be made into a memoized property that
        # returns a composed operator instance.
        # return self.domain.gram_inv * self.adjoint * self.range.gram
        return self.domain.gram_inv(self.adjoint(self.range.gram(y)))

    def norm(self):
        # TODO
        pass

    def _eval(self, params, x):
        raise NotImplementedError

    def _adjoint(self, params, x):
        raise NotImplementedError


class Adjoint(LinearOperator):
    def __init__(self, op):
        self.__op = op
        super().__init__(self.op.params)

    @property
    def op(self):
        if isinstance(self.__op, Revocable):
            return self.__op.get()
        else:
            return self.__op

    @property
    def domain(self):
        return self.op.range

    @property
    def range(self):
        return self.op.domain

    def _eval(self, params, y):
        return self.op._adjoint(params, y)

    def _adjoint(self, params, x):
        return self.op._eval(params, x)

    @property
    def adjoint(self):
        return self.op


class Derivative(LinearOperator):
    def __init__(self, op):
        self.__op = op
        super().__init__(self.op.params)

    @property
    def op(self):
        if isinstance(self.__op, Revocable):
            return self.__op.get()
        else:
            return self.__op

    def _eval(self, params, x):
        return self.op._deriv(params, x)

    def _adjoint(self, params, x):
        return self.op._adjoint(params, x)
