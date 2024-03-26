import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
import regpy.stoprules as rules
from regpy.operators import Exponential, SquaredModulus
#from regpy.operators.fresnel import fresnel_propagator
import matplotlib.animation as animation
import time

from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop, Ptw_Multiplication, Mat, Theta, Tau, Proj

import numpy as np
from math import floor
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)


N=256  #N^2 is the pixel number
M=10   # Shots per frame
N_frame=4000  # frames
T=10**12       # the observation time or the number of photon counts
fresnel_number=30 # not properly scaled
coherence_len = 0.3 # coherence length
N_b=4
Newton_steps = 10
CG_steps=10
sobolev_index=1.2

xsample=np.arange(-1,1-1/N,2/N)
ysample=xsample
grid=UniformGrid(xsample, ysample, dtype=complex)

freq = grid.frequencies()

fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))
    
create_type='fourier_random'
if create_type=='spatial':
    vec=np.zeros((N_b, N, N), dtype=complex)
    for i in range(0, N_b):
        vec[i, :, :]=1/np.sqrt(2)*(np.random.randn(N**2).reshape(N, N)+complex(0,1)*np.random.randn(N**2).reshape(N, N))
        #vec[i, :, :] = 1/np.sqrt(2)*grid.randn()
    
    #Multiply with a rapid decaying function
    sigma=1
    vec=vec*np.exp(-(xsample)**2/(2*sigma**2)).reshape(N, 1)*np.exp(-(ysample)**2/(2*sigma**2)).reshape(1, N)
    U, S, V=np.linalg.svd(vec.reshape(N_b, N**2), full_matrices=False)
    Vcov=V.T.conj()*S
elif create_type=='some_fourier':
    from regpy.operators import FourierTransform

    fourier=FourierTransform(grid, centered=True)
    #Multiply with Gaussian in Fourier domain
    sigma=1
    func_freq=np.exp(-np.linalg.norm(fourier.codomain.coords, axis=0)**2/sigma**2)

    def cutoff_function_2d(x, y):
        xx,yy = np.meshgrid(x,y)
        mask=((1-xx**2-yy**2)>0)
        func=np.zeros(xx.shape)
        func[mask]=(np.exp(1)*np.exp(-1/(1-xx**2-yy**2)))[mask]
        return func


    vec_cut=cutoff_function_2d(xsample,ysample)
    
    vec_cutted=np.zeros((N_b, N, N), dtype=complex)
    for i in range(0, N_b):
        N_x=np.random.randint(N)
        N_y=np.random.randint(N)
        
        coords_0=fourier.codomain.coords[:, N_x, N_y]
        coords=fourier.codomain.coords-coords_0.reshape(2, 1, 1)
        func_freq=np.exp(-np.linalg.norm(coords, axis=0)**2/sigma**2)
        
        vec_cutted[i, :, :]=fourier.adjoint(func_freq)*vec_cut
        
    U, S, V=np.linalg.svd(vec_cutted.reshape(N_b, N**2), full_matrices=False)
    Vcov=V.T.conj()*S
    
elif create_type=='Fresnelprop':
    col=fresnel_prop(grid, coherence_len**2)
    conv=np.zeros((N_b, N, N), dtype=complex)
    for i in range(0, N_b):    
        fjsq=np.random.randn(N**2).reshape(N, N)
        conv[i, :, :]=col(fjsq)
        
    U, S, V=np.linalg.svd(conv.reshape(N_b, N**2), full_matrices=False)
    Vcov=V.T.conj()*S
    
elif create_type=='fourier_random':
    from regpy.operators import FourierTransform

    fourier=FourierTransform(grid, centered=True)
    #Multiply with Gaussian in Fourier domain
    sigma=1
    func_freq=np.exp(-np.linalg.norm(fourier.codomain.coords, axis=0)**2/sigma**2)

    def cutoff_function_2d(x, y):
        xx,yy = np.meshgrid(x,y)
        mask=((1-xx**2-yy**2)>0)
        func=np.zeros(xx.shape)
        func[mask]=(np.exp(1)*np.exp(-1/(1-xx**2-yy**2)))[mask]
        return func


    vec_cut=cutoff_function_2d(xsample,ysample)
    
    vec_cutted=np.zeros((N_b, N, N), dtype=complex)
    for i in range(0, N_b):
        vec_cutted[i, :, :]=fourier.adjoint(fourier.domain.randn()*func_freq)*vec_cut
        
    U, S, V=np.linalg.svd(vec_cutted.reshape(N_b, N**2), full_matrices=False)
    Vcov=V.T.conj()*S
    
else:
    raise ValueError('No method specified')
    

X,Y = np.meshgrid(xsample, ysample, sparse=False)

absorp=np.load('cell1.npy')
phase=np.load('cell2.npy')
support_mask=((abs(X)<=0.801)*(abs(Y)<=0.801)).astype('int')
contrast = support_mask*(0.1*absorp + 0.1*complex(0,1) * phase)

grid=UniformGrid(xsample, ysample, dtype=complex)
grid_codomain=UniformGrid(N, N, N_b, dtype=complex)
grid_codomain_2=UniformGrid(N, N, N_b, N_b, dtype=complex)
grid_codomain_3=UniformGrid(2, N, N, N_b, N_b, dtype=complex)

mask=(contrast!=0)
#projection=CoordinateMask(grid, mask)
Mat_op=Mat(grid, grid_codomain, Vcov, fp)
Tau_op=Tau(Mat_op.codomain, grid_codomain_2)
Proj_op=Proj(Tau_op.codomain, grid_codomain_3)
Theta_op=Theta(N, N_b)
#op=Proj_op*Tau_op*Mat_op
op=Tau_op*Mat_op

ptw_detection= SquaredModulus(grid)
mult=Ptw_Multiplication(grid, np.exp(contrast))
ptw_op=ptw_detection*fp*mult

taumat_0=Mat_op(contrast) 
intens_tot=np.zeros((N, N))

intensities=np.zeros((N_frame, N, N))

for i in range(0, N_frame):
    print(i)
    signal=np.zeros((N, N))
    for j in range(0, M):
        random=1/np.sqrt(2)*(np.random.randn(N_b)+complex(0,1)*np.random.randn(N_b))
        uinc=np.tensordot(taumat_0, random, axes=([-1], [0]))
        signal+=ptw_detection(uinc)
    
    #Cox-processes
    signal=(1/T)*np.random.poisson(lam=T*signal.flatten(), size=(N**2)).reshape(N, N)
    intens_tot+=signal
    
    intensities[i, :, :]=signal
    

intensities-=intens_tot/N_frame

gram_type='L2'

if gram_type=='L2':
    def _gram_inv(x):
        return x
    
    def _gram(x):
        return x
    
elif gram_type=='Sobolev':
    from regpy.hilbert import SobolevUniformGrid
    
    sobolev_space=SobolevUniformGrid(op.domain, index=sobolev_index, axes=None)
    
    _gram=sobolev_space.gram
    _gram_inv=sobolev_space.gram_inv
        
    #def _gram(x):
    #    return projection._adjoint(sobolev_space.gram(projection(x)))
    
    #def _gram_inv(x):
    #    return projection._adjoint(sobolev_space.gram_inv(projection(x)))
    


def _norm(taumat, deriv):
    h=np.random.randn(N**2).reshape(N, N)
    norm = np.sqrt(np.real(np.vdot(h, h)))
    for count in range(10):
        print(count)
        h = h / norm
        
        derivh=deriv(h)
        derivh=M**2*Theta_op._deriv_adjoint(taumat, taumat, Proj_op(derivh))
        h=deriv._adjoint(Proj_op._adjoint(derivh))
        
        norm = np.sqrt(np.real(np.vdot(_gram_inv(h), _gram_inv(h))))
    return np.sqrt(norm)


def CG_op(backprop, op, taumat, Theta_op, Proj_op, max_iter=5, reg=0, tol=10**(-50), print_residual=True):
    r=_gram_inv(backprop)
    d = r
    counter=0
    x=op.domain.zeros()
    while counter<=max_iter and np.linalg.norm(r)>=tol:
        mat=op(d)
        z_adj_1=M**2*Theta_op._deriv_adjoint(taumat, taumat, Proj_op(mat))
        z_adj=op._adjoint(Proj_op._adjoint(z_adj_1))
        z=_gram_inv(z_adj)+reg*d
        
        normsq_r_old=np.vdot(r, _gram(r)).real
        scalar=np.vdot(d, _gram(z)).real
        alpha=normsq_r_old/scalar
        x = x + alpha*d
        r = r - alpha*z
        beta=np.vdot(r, _gram(r)).real/normsq_r_old
        d = r + beta*d
        if print_residual:
            print('residual=\n',np.linalg.norm(r))
        counter+=1
    return x

def _deriv_adjoint(h, deriv, taumat):        
    derivh=deriv(h)
    derivh=M**2*Theta_op._deriv_adjoint(taumat, taumat, Proj_op(derivh))
    h=deriv._adjoint(Proj_op._adjoint(derivh)) 
    return h       
        
def _ADMM(backprop, gamma, deriv, taumat, N_ADMM=10):
    x=deriv.domain.zeros()
    Tstarv1=deriv.domain.zeros()
    v2=deriv.domain.zeros()
    Tstarp1=deriv.domain.zeros()
    p2=deriv.domain.zeros()
    for i in range(0, N_ADMM):
        x=CG_op(Tstarv1+Tstarp1+v2+p2, deriv, taumat, Theta_op, Proj_op, max_iter=5, reg=1, tol=10**(-50))
        adjx=_deriv_adjoint(x, deriv, taumat)
        Tstarv1 = 1/(1+1/gamma)*(adjx-backprop-Tstarp1)+backprop
        Tstarp1= Tstarp1-gamma*adjx+gamma*Tstarv1
        v2=_proximal_penalty(x-p2, 1/gamma)
        p2=p2-gamma*(x-v2)
    return x
    
def _proximal_data(y, tau):
    return 1/(1+tau)*y

def _proximal_penalty(y, tau):
    return np.maximum(y.real, 0*y.real)+complex(0,1)*np.maximum(y.imag, 0*y.imag)

sol_it = np.zeros((N, N), dtype=complex)
for Newton_it in range(Newton_steps):
    print(Newton_it)
    # evaluate forward operator at sol_it
    taumat, deriv=op.linearize(sol_it)
    # apply adjoint to rhs of Newton equation
    backprop_1=M**2*Theta_op._eval_adjoint(taumat, taumat, Proj_op(taumat))
    backprop_2=M/N_frame*Theta_op._backprop(taumat, taumat, intensities.transpose([1, 2, 0]), intensities.transpose([1, 2, 0]), k_G=N_frame)
    backprop=backprop_2-backprop_1
    backprop=deriv._adjoint(Proj_op._adjoint(backprop))
    #compute approximation to operator norm of linearized forward operator T
    # by power method (we compute the largest eigenvalue 1/mu of T'*T)
    gamma=1
    norm=_norm(taumat, deriv)
    Newton_up=_ADMM(1/norm**2*backprop, gamma, 1/norm*deriv, taumat, N_ADMM=10)
    
    sol_it = sol_it + Newton_up


plt.figure()
plt.imshow(Vcov[:, 0].reshape(N, N).real)
plt.colorbar()
plt.title('Sample incident')
plt.show()

plt.figure()
plt.imshow(intens_tot)
plt.colorbar()
plt.title('Total intensity')
plt.show()


plt.figure()
plt.imshow(sol_it.real)
plt.colorbar()
plt.title('Recovered absorption')
plt.show()

plt.figure()
plt.imshow(contrast.real)
plt.colorbar()
plt.title('Exact absorption')
plt.show()

plt.figure()
plt.imshow(sol_it.imag)
plt.colorbar()
plt.title('Recovered phase')
plt.show()

plt.figure()
plt.imshow(contrast.imag)
plt.colorbar()
plt.title('Exact phase')
plt.show()

plt.figure()
plt.plot(sol_it[int(N/2), :].imag, label='Recovered phase')
plt.plot(contrast[int(N/2), :].imag, label='Exact phase')
plt.legend()
plt.show()

plt.figure()
plt.plot(sol_it[int(N/2), :].real, label='Recovered absorption')
plt.plot(contrast[int(N/2), :].real, label='Exact absorption')
plt.legend()
plt.show()

