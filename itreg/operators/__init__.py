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
    
    #Jakob
    def getArg(key):
    	return self.__dict__.get(key)


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

################################################################################

#Fragen:
#1. Woher kommt das handle in NonlinearOperator und was macht das?
#2. Was genau soll bei NonlinearOperator passieren wenn differentiate=True?
#3. Meines Erachtens nach berechnet die _adjoint Funktion in der
#	Derivative-Klasse aktuell die adjungierte des originalen Operators,
#	müsste die nicht eigentlich die adjungierte der Ableitung berechnen?
#4. Wie sieht im Allgemeinen die Adjungierte der Ableitung der Komposition
#	zweier nicht-linearer Operatoren aus?
#5. Derivative für lineare Operatoren nicht gebraucht? Was ist mit so Sachen wie
#	k*x linearer Operator, aktuell würde linearize für die ableitung widerum
#	k*x speichern, was dann zu falschen Ergebnissen führt.
#6.	NonlinearOperator hat keine property adjoint, soll das so?
#7. Kann derivative auch ne property werden? Dann wäre das ganze einheitlich


#TODO Testing, need to define some simple cases first
class LinearCombination(LinearOperator):
	#TODO Domain and range query for scalars missing
	def __init__(self, f, g, scalar_f, scalar_g):
		if(f.domain() != g.domain() or f.range() != g.range()):
			raise ValueError('Domains and Ranges of Operators must be the same')
		args = {f:f, g:g, scalar_f:scalar_f, scalar_g:scalar_g}
		super().__init__(Params(f.domain(), f.range(), args))
	
	def _eval(self, x):
		return (self.getArg(scalar_f) * self.getArg(f)._eval(x) +
				self.getArg(scalar_g) * self.getArg(g)._eval(x))
	
	#f._adjoint(x) oder f.adjoint._eval(x) verwenden?
	def _adjoint(self, x):
		return self.getArg(scalar_f) * self.getArg(f).adjoint._eval(x) + self.getArg(scalar_g) * self.getArg(g).adjoint._eval(x)

class Composition(NonlinearOperator):
	def __init__(self,f,g):
		if(f.domain() != g.range()):
			raise ValueError('For f∘g, domain of f and' +
							 'range of g must be the same')
		super().__init__(Params(g.domain(), f.range(), {f:f, g:g}))
	
	def _eval(self, x, differentiate = False):
		return self.getArg(f)._eval(self.getArg(g)._eval(x))
	
	def _adjoint(self, x):
		return self.getArg(g)._adjoint(self.getArg(f)._adjoint(x))
	
	#not sure if this works
	def _derivative(self, x):
		g, dg = self.getArg(g).linearize(x)
		fg, dfg = self.getArg(f).linearize(g)
		return dg*dfg
		
	














