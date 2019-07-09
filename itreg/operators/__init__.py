import numpy as np

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
        self._alloc(params)

    def _alloc(self, params):
        pass

    @property
    def domain(self):
        return self.params.domain

    @property
    def range(self):
        return self.params.range

    def clone(self):
        cls = type(self)
        instance = cls.__new__(cls)
        BaseOperator.__init__(instance, self.params)
        return instance

    def __call__(self, x):
        raise NotImplementedError

    def linearize(self, x):
        raise NotImplementedError


class NonlinearOperator(BaseOperator):
    def __call__(self, x):
        self.__revoke()
        return self._eval(x, differentiate=False)

    def linearize(self, x):
        self.__revoke()
        y = self._eval(x, differentiate=True)
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


class LinearOperator(BaseOperator):
    def __call__(self, x):
        return self._eval(x)

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


class Adjoint(LinearOperator):
    def __init__(self, op):
        super().__init__(Params(op.range, op.domain, op=op))

    def _eval(self, x):
        return self.params.op._adjoint(x)

    def _adjoint(self, x):
        return self.params.op._eval(x)

    @property
    def adjoint(self):
        return self.params.op


class Derivative(LinearOperator):
    def __init__(self, op):
        if not isinstance(op, Revocable):
            # Wrap plain operators in a Revocable that will never be revoked to
            # avoid case distinctions below.
            op = Revocable(op)
        super().__init__(Params(op.get().domain, op.get().range, op=op))

    def _eval(self, x):
        return self.params.op.get()._derivative(x)

    def _adjoint(self, x):
        return self.params.op.get()._adjoint(x)


class LinearCombination(LinearOperator):
    """
    Implements a linear combination of any number of operators with sepcified scalars.
    Domains and ranges must be the same.
    E.g. : F(x) = a * f(x) + b * g(x) + c * h(x)
    """
    # TODO Domain and range query for scalars missing
    def __init__(self, operators, scalars):
        for i in range(len(operators)-1):
            f = operators[i]
            g = operators[i+1]
            assert f.domain == g.domain and f.range == g.range, "Domains and Ranges of Operators must be the same"
        super().__init__(Params(f.domain, f.range, operators=operators, scalars=scalars))

    """
    Computes the result of the linear combination evaluated at x.

    Keyword arguments:
    x -- point of evaluation
    """
    def _eval(self, x):
        res = 0
        for i in range(len(self.params.operators)-1):
            res += self.paramas.scalar[i] * self.params.operators[i].__call__(x)
        return res

    """
    Computes the adjoint of the linear combination evaluated at x.

    Keyword arguments:
    x -- point of evaluation
    """
    def _adjoint(self, x):
        res = 0
        for i in range(len(self.params.operators)-1):
            res += self.paramas.scalar[i] * self.params.operators[i].adjoint(x)
        return res


class Composition(NonlinearOperator):
    """
    Implements a composition of any number of operators.
    Domains and ranges of operators must match, in the sense that for any two
    given operators f,g either f.domain == g.range or g.domain == f.range
    """
    def __init__(self, operators):
        assert len(operators) > 1, "Number of operators must be greater than 1"
        for i in range(len(operators)-1):
            f = operators[i]
            g = operators[i+1]
            assert f.range == g.domain, "Domains and ranges must match"
        super().__init__(Params(operators[len(operators)-1].domain, operators[0].domain, operators=operators))

    """
    Computes the result of the composition. If differentiate flag is set, also
    computes and stores linearizations of operators for later uses.

    Keyword arguments:
    x -- point at which to evaluate
    differentiate -- flag to determine wether linearizations are needed
                     (default False)
    """
    def _eval(self, x, differentiate=False):
        _fx = x
        self.derivatives = []
        for i in range(len(self.params.operators)-1, -1, -1):
            _f = self.params.operators[i]
            if(differentiate):
                # compute linearization of f in x and store it
                _fx, _dfx = _f.linearize(_fx)
                self.derivatives.append(_dfx)
            else:
                _fx = _f.__call__(_fx)
        self._fx = _fx
        return self._fx

    """
    Computes the evaluation of the adjoint of the linearization at
    x (given in _eval()) of the composition at h.

    Keyword arguments:
    h -- point of evaluation
    """
    def _adjoint(self, h):
        _fh = h
        for i in range(len(self.derivatives)-1, -1, -1):
            # evaluate adjoints of linearizations saved earlier
            _df = self.derivatives[i]
            _fh = _df._adjoint(_fh)
        return _fh

    """
    Computes the evaluation of the linearization at x (given in _eval()) at h.

    Keyword arguments:
    h -- point of evaluation
    """
    def _derivative(self, h):
        _dfh = h
        # evaluate linearizations saved earlier
        for i in range(len(self.derivatives)):
            _df = self.derivatives[i]
            _dfh = _df.__call__(_dfh)
        return _dfh


class MatrixMultiplication(LinearOperator):
    """
    Implements a matrix multiplication with a given matrix.
    """
    def __init__(self, matrix):
        super().__init__(Params(None, None, matrix=matrix))
    """
    Computes A∘x.

    Keyword arguments:
    x -- point of evaluation
    """
    def _eval(self, x):
        return self.params.matrix @ x

    """
    Computes the adjoint of the matrix composed with y.

    Keyword arguments:
    y -- point of evaluation
    """
    def _adjoint(self, y):
        return self.params.matrix.T @ y
