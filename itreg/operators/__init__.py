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


#TODO Testing, need to define some simple cases first
class LinearCombination(LinearOperator):
	#TODO Domain and range query for scalars missing
	def __init__(self, f, g, scalar_f, scalar_g):
		assert f.domain == g.domain and f.range == g.range, "Domains and Ranges of Operators must be the same"
		super().__init__(Params(f.domain, f.range, f=f, g=g, scalar_f = scalar_f, scalar_g = scalar_g))
	
	def _eval(self, x):
		return self.params.scalar_f * self.params.f.__call__(x) + self.params.scalar_g * self.params.g.__call__(x)

	def _adjoint(self, x):
		return self.params.scalar_f * self.params.f._adjoint(x) + self.params.scalar_g * self.params.g._adjoint(x)


class Composition(NonlinearOperator):
	def __init__(self,f,g):
		assert f.domain == g.range, "For f∘g, domain of f and range of g must be the same"
		super().__init__(Params(g.domain, f.range, f=f, g=g))
	
	def _eval(self, x, differentiate = False):
		#if differentiate save linearizations in x of g and f∘g now
		if (differentiate):
			_gx,_dgx = self.params.g.linearize(x)
			self._gx = _gx
			self._dgx = _dgx
			_fgx, _dfgx = self.params.f.linearize(_gx)
			self._fgx = _fgx
			self._dfgx = _dfgx
		else:
			self._gx = self.params.g.__call__(x)
			self._fgx = self.params.f.__call__(self._gx)
		return self._fgx
	
	def _adjoint(self, h):
		return self._dgx._adjoint(h) * self._dfgx._adjoint(h)
	
	#evaluate linearizations saved earlier in a possibly distinct point h
	def _derivative(self, h):
		print(self._dgx.__call__(h))
		return self._dgx.__call__(h) * self._dfgx.__call__(h)
		
	














