import itreg.operators as op

class Power(op.NonlinearOperator):
	def __init__(self, p):
		super().__init__(op.Params(None, None, p=p))
	
	def _eval(self, x, differentiate = False):
		self.params.__dict__.update(x = x)
		self.params.__dict__.update(fx = x**self.params.p)
		self.params.__dict__.update(dfx = self.params.p*x**(self.params.p-1))
		return self.params.fx
	
	def _derivative(self, h):
		return self.params.fx + self.params.dfx(h-self.params.x)
	def _adjoint(self, x):
		return self.__call__(x)

class Scale(op.LinearOperator):
	def __init__(self, s):
		super().__init__(op.Params(None, None, s=s))
	
	def _eval(self, x):
		return self.params.s*x
	
	def _adjoint(self, x):
		return self.__call__(x)

p = Power(2)
s = Scale(2)

LC = op.LinearCombination(p,s,1,1)
d = 2
print(LC._eval(d))
print(LC._adjoint(d))
C = op.Composition(p,s)
print(C._eval(d))
print(C._adjoint(d))
print(C._derivative(d))
