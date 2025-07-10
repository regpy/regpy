from regpy.operators import Exponential,Product,FourierTransform,PtwMultiplication
from regpy.operators.graph_operator import OperatorGraph
from regpy.vecsps import UniformGridFcts
import numpy as np
from matplotlib import pyplot as plt

#this is an unfinished stub for a future split step example

def get_div_operator(domain,Nx,dt,L,half=False):
    ft=FourierTransform(domain)
    ift=FourierTransform(domain).adjoint
    flip1=PtwMultiplication(domain,(-1)**np.arange(Nx))
    flip2=PtwMultiplication(domain,(-1)**np.arange(Nx))
    kernel_const= 0.25 if half else 0.5 
    diff_kernel=np.exp(-kernel_const*1j*dt*((-np.pi*Nx+np.arange(Nx)*np.pi*2)/L)**2)
    kernel_mult=PtwMultiplication(ft.codomain,diff_kernel)
    return PtwMultiplication(domain,(-1)**np.arange(Nx))*ift*kernel_mult*ft*PtwMultiplication(domain,(-1)**np.arange(Nx))



dt=0.2 #time step
Nt=24 #time step number
Nx=200 #domain discretization
L=8 #domain width

domain=UniformGridFcts((-L/2,L/2,Nx),dtype=np.complex128)

exp_op=Exponential(domain)*PtwMultiplication(domain,-1j*dt)#operator for exponential of potential

sign_flip=(-1)**np.arange(Nx)

half_start=get_div_operator(domain,Nx,dt,L,half=True)
half_end=get_div_operator(domain,Nx,dt,L,half=True)

diff_ops=[get_div_operator(domain,Nx,dt,L,half=False) for _ in range(Nt-1)]
prod_ops=[Product(domain+domain) for _ in range(Nt)]

edges=[((exp_op,[0]),(prod_op,1)) for prod_op in prod_ops]
edges.append(((None,[0]),(exp_op,0)))
edges+=[((prod_ops[i],[0]),(diff_op,0)) for i,diff_op in enumerate(diff_ops)]
edges+=[((diff_op,[0]),(prod_ops[i+1],0)) for i,diff_op in enumerate(diff_ops)]
edges.append(((half_start,[0]),(prod_ops[0],0)))
edges.append(((prod_ops[-1],[0]),(half_end,0)))
edges.append(((None,[1]),(half_start,0)))
edges.append(((half_end,[0]),(None,0)))


full_op=OperatorGraph([exp_op,half_start,half_end]+diff_ops+prod_ops,edges)

xs=np.linspace(-L/2,L/2,Nx)
s=xs**2
s=(xs**4/16-2*xs**2/4+1)
g=np.exp(-((xs))**2)

w=full_op.domain.zeros()
w[0:2*Nx:2]=s
w[2*Nx:4*Nx:2]=g

res=full_op(w)


xs=np.linspace(-L/2,L/2,Nx)
plt.ylim((-2,np.max(s)+1))
plt.plot(xs,s)
# plt.plot(xs,g)
plt.plot(xs,np.abs(res))
plt.plot(xs,np.imag(res))
plt.plot(xs,np.real(res))
plt.show()
