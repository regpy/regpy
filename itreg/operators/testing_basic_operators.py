import itreg.operators as op

class Power(op.NonlinearOperator):
	def __init__(self, p):
		super().__init__(op.Params(None, None, p=p))
	
	def _eval(self, x, differentiate = False):
		self.params.x = x
		self.params.fx = x**self.params.p
		if(differentiate):
			self.params.dfx = self.params.p*x**(self.params.p-1)
		return self.params.fx
	
	#erklärung für h!=x brauche ich noch.
	def _derivative(self, h):
		return self.params.dfx
		
	def _adjoint(self, x):
		return self.__call__(x)

class Scale(op.LinearOperator):
	def __init__(self, s):
		super().__init__(op.Params(None, None, s=s))
	
	def _eval(self, x):
		return self.params.s*x
	
	def _adjoint(self, x):
		return self.__call__(x)

po = 3
sc = 2
d = 3
p = Power(po)
s = Scale(sc)

print(str(d) + '**' + str(po) + ' = ' + str(p.__call__(d)))
print(str(d) + '*' + str(sc) + ' = ' + str(s.__call__(d)))
y,dy = p.linearize(d)
print(str(po) + '*' + str(d) + '**' + str(po-1) + ' = ' + str(dy.__call__(d)))

LC = op.LinearCombination(p,s,1,1)
print(str(d) + '*' + str(sc) + ' + ' + str(d) + '**' + str(po) + ' = ' + str(LC.__call__(d)))
A = LC.adjoint
#print(A._adjoint(d))
#print(A.__call__(d))

C = op.Composition(p,s)
print('(' + str(sc) + '*' + str(d) + ')**' + str(po) + ' = ' + str(C.__call__(d)))
y,dy = C.linearize(d)
print(str(sc) + '*' + str(po) + '*' + str(s.__call__(d)) + '**' + str(po-1) + ' = ' + str(dy.__call__(d)))
#print(dy._adjoint(d))
