from collections import defaultdict

import numpy as np

from regpy import operators, util, discrs, hilbert


class Functional:
    def __init__(self, domain):
        # TODO implement domain=None case
        assert isinstance(domain, discrs.Discretization)
        self.domain = domain

    def __call__(self, x):
        assert x in self.domain
        try:
            y = self._eval(x)
        except NotImplementedError:
            y, _ = self._linearize(x)
        assert isinstance(y, float)
        return y

    def linearize(self, x):
        assert x in self.domain
        try:
            y, grad = self._linearize(x)
        except NotImplementedError:
            y = self._eval(x)
            grad = self._gradient(x)
        assert isinstance(y, float)
        assert grad in self.domain
        return y, grad

    def gradient(self, x):
        assert x in self.domain
        try:
            grad = self._gradient(x)
        except NotImplementedError:
            _, grad = self._linearize(x)
        assert grad in self.domain
        return grad

    def hessian(self, x):
        assert x in self.domain
        h = self._hessian(x)
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def _eval(self, x):
        raise NotImplementedError

    def _linearize(self, x):
        raise NotImplementedError

    def _gradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        return operators.ApproximateHessian(self, x)

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

    def __truediv__(self, other):
        return (1 / other) * self

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

    def _gradient(self, x):
        y, deriv = self.op.linearize(x)
        return deriv.adjoint(self.func.gradient(y))

    def _hessian(self, x):
        if self.op.linear:
            return self.op.adjoint * self.func.hessian(x) * self.op
        else:
            # TODO this can be done slightly more efficiently
            return super()._hessian(x)


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

    def _gradient(self, x):
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            grad += coeff * func.gradient(x)
        return grad

    def _hessian(self, x):
        return operators.LinearCombination(
            *((coeff, func.hessian(x)) for coeff, func in zip(self.coeffs, self.funcs))
        )


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

    def _gradient(self, x):
        return self.func.gradient(x)

    def _hessian(self, x):
        return self.func.hessian(x)


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

    def _gradient(self, x):
        return self.hspace.gram(x)

    def _hessian(self, x):
        return self.hspace.gram


class Indicator(Functional):
    def __init__(self, domain, predicate):
        super().__init__(domain)
        self.predicate = predicate

    def _eval(self, x):
        if self.predicate(x):
            return 0
        else:
            return np.inf

    def _gradient(self, x):
        # This is of course not correct, but lets us use an Indicator functional to force
        # rejecting an MCMC proposals without altering the gradient.
        return self.domain.zeros()

    def _hessian(self, x):
        return operators.Zero(self.domain)


class ErrorToInfinity(Functional):
    def __init__(self, func):
        super().__init__(func.domain)
        self.func = func

    def _eval(self, x):
        try:
            return self.func(x)
        except:
            return np.inf

    def _gradient(self, x):
        try:
            return self.func.gradient(x)
        except:
            return self.domain.zeros()


class L1Norm(Functional):
    def _eval(self, x):
        return np.sum(np.abs(x))

    def _gradient(self, x):
        return np.sign(x)

    def _hessian(self, x):
        # Even approximate Hessians don't work here.
        raise NotImplementedError
