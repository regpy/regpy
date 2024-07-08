import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
import regpy.stoprules as rules
from regpy.operators import Exponential, SquaredModulus, CoordinateMask
#from regpy.operators.fresnel import fresnel_propagator
import matplotlib.animation as animation
import time

from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop, Ptw_Multiplication, Mat, Theta, Tau, Proj, Positivity

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
N_frame=3000  # frames
T=10**12       # the observation time or the number of photon counts
fresnel_number=30 # not properly scaled
coherence_len = 0.3 # coherence length
N_b=2
Newton_steps = 10
CG_steps=10
sobolev_index=1.2

xsample=np.arange(-1,1-1/N,2/N)
ysample=xsample
grid=UniformGrid(xsample, ysample, dtype=complex)

freq = grid.frequencies()

fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))
col=fresnel_prop(grid, coherence_len**2)
    
spatial=False
if spatial:
    vec=np.zeros((N_b, N, N), dtype=complex)
    for i in range(0, N_b):
        vec[i, :, :]=1/np.sqrt(2)*(np.random.randn(N**2).reshape(N, N)+complex(0,1)*np.random.randn(N**2).reshape(N, N))
        #vec[i, :, :] = 1/np.sqrt(2)*grid.randn()
    
    #Multiply with a rapid decaying function
    sigma=1
    vec=vec*np.exp(-(xsample)**2/(2*sigma**2)).reshape(N, 1)*np.exp(-(ysample)**2/(2*sigma**2)).reshape(1, N)
    U, S, V=np.linalg.svd(vec.reshape(N_b, N**2), full_matrices=False)
    Vcov=V.T.conj()*S

else:
    from regpy.operators import FourierTransform

    fourier=FourierTransform(grid, centered=True)
    #Multiply with Gaussian in Fourier domain
    sigma=10000
    func_freq=np.exp(-(np.linalg.norm(fourier.codomain.coords, axis=0))**2/sigma**2)

    def cutoff_function_2d(x, y):
     xx,yy = np.meshgrid(x,y)
     mask=((1-xx**2-yy**2)>0)
     func=np.zeros(xx.shape)
     func[mask]=(np.exp(1)*np.exp(-1/(1-xx**2-yy**2)))[mask]
     return func


    vec_cut=cutoff_function_2d(xsample,ysample)
    
    vec_cutted=np.zeros((N_b, N, N))
    for i in range(0, N_b):
        vec_cutted[i, :, :]=fourier.adjoint(fourier.domain.randn()*func_freq)*vec_cut
        
    U, S, V=np.linalg.svd(vec_cutted.reshape(N_b, N**2), full_matrices=False)
    Vcov=V.T.conj()*S
#Inverse Fourier transform
"""vec_fft=np.zeros((N_b, N, N), dtype=complex)
for i in range(0, N_b):
    vec_fft[i, :, :]=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vec[i, :, :])))

# multipling with a cut-off function
smoothness=5
threshold=1
def cutoff_function_2d(x, y, threshold, smoothness):
    xx,yy = np.meshgrid(x,y)
    return 1 / (1+np.exp(-smoothness * (np.maximum(np.abs(xx), np.abs(yy)) - threshold)))

vec_cut=cutoff_function_2d(xsample,ysample,threshold,smoothness)


vec_cutted=vec*(1-vec_cut)"""

X,Y = np.meshgrid(xsample, ysample, sparse=False)

absorp=np.load('/home/milad-karimi/Documents/Code_for_milad/New_code/avoid_covariance/cell1.npy')
phase=np.load('/home/milad-karimi/Documents/Code_for_milad/New_code/avoid_covariance/cell2.npy')


support_mask=((abs(X)<=0.801)*(abs(Y)<=0.801)).astype('int')
contrast = support_mask*(0.1*absorp + 0.1*complex(0,1) * phase)


grid=UniformGrid(xsample, ysample, dtype=complex)
grid_codomain=UniformGrid(N, N, N_b, dtype=complex)
grid_codomain_2=UniformGrid(N, N, N_b, N_b, dtype=complex)
grid_codomain_3=UniformGrid(2, N, N, N_b, N_b, dtype=complex)

mask=(contrast!=0)
pos=Positivity(grid)
projection=CoordinateMask(pos.codomain, mask)
Mat_op=Mat(projection.codomain, grid_codomain, Vcov, fp)
Tau_op=Tau(Mat_op.codomain, grid_codomain_2)
Proj_op=Proj(Tau_op.codomain, grid_codomain_3)
Theta_op=Theta(N, N_b)
#op=Proj_op*Tau_op*Mat_op
op=Tau_op*Mat_op*projection

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

def _gram_inv(x):
    return x

def _gram(x):
    return x

"""from regpy.hilbert import SobolevUniformGrid

sobolev_space=SobolevUniformGrid(op.domain, index=sobolev_index, axes=None)

_gram=sobolev_space.gram
_gram_inv=sobolev_space.gram_inv

def _gram(x):
    return projection._adjoint(sobolev_space.gram(projection(x)))

def _gram_inv(x):
    return projection._adjoint(sobolev_space.gram_inv(projection(x)))"""

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


def CG(backprop, op, taumat, Theta_op, Proj_op, max_iter=5, reg=0, tol=10**(-50)):
    s_old_tilde=backprop
    s_old=_gram_inv(s_old_tilde)
    d=s_old
    d_tilde=s_old_tilde    
    kappa=1
    x=op.domain.zeros()
    counter=0
    while counter<=max_iter and np.linalg.norm(s_old)>np.sqrt(kappa)*reg*tol:
        mat=op(d)
        z_adj_1=M**2*Theta_op._deriv_adjoint(taumat, taumat, Proj_op(mat))
        z_adj=op._adjoint(Proj_op._adjoint(z_adj_1))
        z_scalar=np.vdot(z_adj, d)
        s_scalar=np.vdot(s_old_tilde, s_old)
        d_scalar=np.vdot(d_tilde, d)
        gamma=s_scalar/(reg*d_scalar+z_scalar)
        x+=gamma*d
        s_new_tilde=s_old_tilde-gamma*(z_adj+reg*d_tilde)
        s_new=_gram_inv(s_new_tilde)
        beta=np.vdot(s_new, s_new)/s_scalar
        kappa=1+beta*kappa
        d=s_new+beta*d
        s_old=s_new.copy()
        s_old_tilde=s_new_tilde.copy()
        counter+=1
    return x


def CG_op(backprop, op, taumat, Theta_op, Proj_op, max_iter=5, reg=0, tol=10**(-50)):
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
        print('residual=\n',np.linalg.norm(r))
        counter+=1
    return x

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
    
    if Newton_it==0:
        reg=_norm(taumat, deriv)**2
        #reg=1e5
    else:
        reg*=0.6
    #reg=10**5
    print('reg=\n',reg)
    
    Newton_up=CG_op(backprop, deriv, taumat, Theta_op, Proj_op, max_iter=CG_steps, reg=reg, tol=10**(-50))
    
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
plt.plot(-sol_it[int(N/2), :].real, label='Recovered absorption')
plt.plot(-contrast[int(N/2), :].real, label='Exact absorption')
plt.legend()
plt.show()