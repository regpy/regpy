from regpy.operators import PtwMultiplication,SquaredModulus, Exponential
from regpy import vecsps
from regpy.operators.graph_operator import get_operators_and_edges,merge_operators

dom=vecsps.UniformGridFcts(2,4)
A=PtwMultiplication(dom,2)
B=SquaredModulus(dom)
C=Exponential(dom)
D=PtwMultiplication(dom+dom,3)

E=D@(A@B@C,B)
ops,eds,N_in,N_out=get_operators_and_edges(E)
print(ops)
for ed in eds:
    print(f"{ed[0][0]}{ed[0][1]}-->{ed[1][1]}{ed[1][0]}")
print(N_in)
print(N_out)
x=3*E.domain.ones()
y,deriv=E.linearize(x)
print(y)
print(deriv(E.domain.ones()))
print(deriv.adjoint(E.codomain.ones()))

from regpy.operators import Sum,Product
import numpy as np
dom1=vecsps.UniformGridFcts(2,4,dtype=np.complex128)
dom2=vecsps.UniformGridFcts(2,4)
dom3=vecsps.UniformGridFcts(2,4)

dom_sum=dom1+dom2+dom3

# sum_op=Sum(dom1+dom2+dom3)
# print(sum_op(sum_op.domain.ones()))
# from regpy.util.operator_tests import test_operator
# test_operator(sum_op)
# print(dom_sum.ones())

prod_op=Product(dom1+dom2+dom3)

y,deriv=prod_op.linearize(2*prod_op.domain.ones())

print(y)
print(deriv(prod_op.domain.ones()))
print(prod_op.deriv_data)
print(prod_op(prod_op.domain.ones()))
from regpy.util.operator_tests import test_operator
test_operator(prod_op)