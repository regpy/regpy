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