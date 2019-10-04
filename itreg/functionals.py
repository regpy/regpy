from collections import defaultdict

import numpy as np

from .spaces import discrs, hilbert
from . import util, operators


class Functional:
    def __init__(self, domain):
        # TODO implement domain=None case
        assert isinstance(domain, discrs.Discretization)
        self.domain = domain

    def __call__(self, x):
        assert x in self.domain
        y = self._eval(x)
        assert isinstance(y, float)
        return y

    def linearize(self, x):
        assert x in self.domain
        y, grad = self._linearize(x)
        assert isinstance(y, float)
        assert grad in self.domain
        return y, grad

    def _eval(self, x):
        raise NotImplementedError

    def _linearize(self, x):
        raise NotImplementedError

    def __mul__(self, other):
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return Composed(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return self * operators.Multiplication(self.domain, other)
        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            if other == 1:
                return self
            elif util.is_real_dtype(other):
                return LinearCombination((other, self))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Functional):
            return LinearCombination(self, other)
        elif np.isscalar(other):
            return Shifted(self, other)
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


class Composed(Functional):
    def __init__(self, func, op):
        assert isinstance(func, Functional)
        assert isinstance(op, operators.Operator)
        assert func.domain == op.codomain
        super().__init__(op.domain)
        if isinstance(func, type(self)):
            op = func.op * op
            func = func.func
        self.func = func
        self.op = op

    def _eval(self, x):
        return self.func(self.op(x))

    def _linearize(self, x):
        y, deriv = self.op.linearize(x)
        z, grad = self.func.linearize(y)
        return z, deriv.adjoint(grad)


class LinearCombination(Functional):
    def __init__(self, *args):
        coeff_for_func = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, func = arg
            else:
                coeff, func = 1, arg
            assert isinstance(func, Functional)
            assert np.isscalar(coeff) and util.is_real_dtype(coeff)
            if isinstance(func, type(self)):
                for c, f in zip(func.coeffs, func.funcs):
                    coeff_for_func[f] += coeff * c
            else:
                coeff_for_func[func] += coeff
        self.coeffs = []
        self.funcs = []
        for func, coeff in coeff_for_func.items():
            self.coeffs.append(coeff)
            self.funcs.append(func)

        domains = [op.domain for op in self.funcs if op.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        super().__init__(domain)

    def _eval(self, x):
        y = 0
        for coeff, func in zip(self.coeffs, self.funcs):
            y += coeff * func(x)
        return y

    def _linearize(self, x):
        y = 0
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            f, g = func.linearize(x)
            y += coeff * f
            grad += coeff * g
        return y, grad


class Shifted(Functional):
    def __init__(self, func, offset):
        assert isinstance(func, Functional)
        assert np.isscalar(offset) and util.is_real_dtype(offset)
        super().__init__(func.domain)
        self.func = func
        self.offset = offset

    def _eval(self, x):
        return self.func(x) + self.offset

    def _linearize(self, x):
        return self.func.linearize(x)


class HilbertNorm(Functional):
    def __init__(self, hspace):
        assert isinstance(hspace, hilbert.HilbertSpace)
        super().__init__(hspace.discr)
        self.hspace = hspace

    def _eval(self, x):
        return np.real(np.vdot(x, self.hspace.gram(x))) / 2

    def _linearize(self, x):
        gx = self.hspace.gram(x)
        y = np.real(np.vdot(x, gx)) / 2
        return y, gx
