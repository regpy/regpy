import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from regpy.hilbert import L2, Sobolev
from regpy.vecsps import UniformGridFcts
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

#%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%
N=200  #N^2 is the pixel number
M=10   # Shots per frame
N_frame=1000  # frames
T=10**12        # the observation time or the number of photon counts
fresnel_number=40 # not properly scaled
coherence_len = 0.3 # coherence length
N_b=4
Newton_steps = 10
FISTA_steps = 10
FISTA_ub_absorp = 0
FISTA_lb_absorp = -1000
FISTA_ub_phase = 1000
FISTA_lb_phase = 0
row_ab = floor(N/2)
row_ph = floor(N/3)
#%%%%%%%%%%%%%%%%%%%%%%%%% Create test image, Fresnel propagator matrix%%%%%%%%%%%%%%%%%%%%%%%
xsample=np.arange(-1,1-1/N,2/N)
ysample=xsample
grid=UniformGridFcts(xsample, ysample, dtype=complex)

cov_u=np.eye(N**2)

fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))
col=fresnel_prop(grid, coherence_len**2)
    
    
"""import random
#Create i.i.d Gaussians
mus=np.zeros((N_b, 2))
for i in range(0, N_b):
    mus[i, 0]=random.uniform(-1, 1)
    mus[i, 1]=random.uniform(-1, 1)

sigma=0.1
vec=np.zeros((N_b, N, N))
for i in range(0, N_b):
    vec[i, :, :]=np.exp(-(xsample-mus[i, 0])**2/2).reshape(N, 1)*np.exp(-(ysample-mus[i, 1])**2/2).reshape(1, N)

#Multiplize with a rapid decaying function
sigma=1
vec=vec*np.exp(-(xsample)**2/(2*sigma**2)).reshape(N, 1)*np.exp(-(ysample)**2/(2*sigma**2)).reshape(1, N)

U, S, V=np.linalg.svd(vec.reshape(N_b, N**2), full_matrices=False)
Vcov=V.T.conj()*S"""

vec=np.zeros((N_b, N, N), dtype=complex)
for i in range(0, N_b):
    vec[i, :, :]=1/np.sqrt(2)*(np.random.randn(N**2).reshape(N, N)+complex(0,1)*np.random.randn(N**2).reshape(N, N))

#Multiplize with a rapid decaying function
sigma=1
vec=vec*np.exp(-(xsample)**2/(2*sigma**2)).reshape(N, 1)*np.exp(-(ysample)**2/(2*sigma**2)).reshape(1, N)

#Inverse Fourier transform

vec_fft=np.zeros((N_b, N, N), dtype=complex)
for i in range(0, N_b):
    vec_fft[i, :, :]=np.fft.ifft2(np.fft.fftshift(vec[i, :, :]))
    
U, S, V=np.linalg.svd(vec.reshape(N_b, N**2), full_matrices=False)
Vcov=V.T.conj()*S


grid=UniformGridFcts(xsample, ysample, dtype=complex)
grid_codomain=UniformGridFcts(N, N, N_b, dtype=complex)
grid_codomain_2=UniformGridFcts(N, N, N_b, N_b, dtype=complex)
grid_codomain_3=UniformGridFcts(2, N, N, N_b, N_b, dtype=complex)

Mat_op=Mat(grid, grid_codomain, Vcov, fp)
Tau_op=Tau(Mat_op.codomain, grid_codomain_2)
Proj_op=Proj(Tau_op.codomain, grid_codomain_3)
Theta_op=Theta(N, N_b)
#op=Proj_op*Tau_op*Mat_op
op=Tau_op*Mat_op


"""dom=Mat_op.domain.randn()
codom=Mat_op.codomain.randn()
y, deriv=Mat_op.linearize(contrast)
first=np.vdot(deriv(dom), codom)
second=np.vdot(dom, deriv._adjoint(codom))


dom=Tau_op.domain.randn()
codom=Tau_op.codomain.randn()
_, deriv=Tau_op.linearize(y)
first=np.vdot(deriv(dom), codom)
second=np.vdot(dom, deriv._adjoint(codom))"""

    
X,Y = np.meshgrid(xsample, ysample, sparse=False)


absorp_0=(abs(X)<0.8)*(abs(Y)<0.199)+(abs(X)<0.299)*(abs(Y)<0.7)+(X**2+Y**2<=0.5**2)*(X**2+Y**2>=0.45**2)
absorp_0=absorp_0.astype('int')

absorp_1=(X**2+Y**2<=0.501**2)*(X**2+Y**2>=0.45**2)
absorp_1=absorp_1.astype('int')
absorp=absorp_0+absorp_1

phase = ((abs(X+Y) <=0.101)+(abs(X-Y) <= 0.101)).astype('int')
support_mask=((abs(X)<=0.801)*(abs(Y)<=0.801)).astype('int')
contrast = support_mask*(-0.1*absorp + 0.1*complex(0,1) * phase)

FISTA_ub_ab = FISTA_ub_absorp*support_mask
FISTA_lb_ab = FISTA_lb_absorp*support_mask
FISTA_ub_ph = FISTA_ub_phase*support_mask
FISTA_lb_ph = FISTA_lb_phase*support_mask


ptw_detection= SquaredModulus(grid)
#ptw_op=ptw_detection*fp*ex
mult=Ptw_Multiplication(grid, np.exp(contrast))
#basis_op= small_rank_basis(random_coeffs, codomain=mult.domain)
#ptw_op=ptw_detection*fp*mult*basis_op
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

#intensities=np.load('intensities.npy')


def _norm(taumat):
    h=np.random.randn(N**2).reshape(N, N)
    norm = np.sqrt(np.real(np.vdot(h, h)))
    for count in range(10):
        print(count)
        h = h / norm
        
        derivh=deriv(h)
        derivh=M**2*Theta_op._deriv_adjoint(taumat, taumat, Proj_op(derivh))
        h=deriv._adjoint(Proj_op._adjoint(derivh))
        
        norm = np.sqrt(np.real(np.vdot(h, h)))
    return np.sqrt(norm)
        


#correlation_data=1/N_frame*intensities.reshape(N_frame, N**2).T.dot(intensities.reshape(N_frame, N**2).conj())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate forward operator %%%%%%%%%%%%%%%%
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
    
    mu=1/_norm(taumat)**2
    print('mu=\n',mu)
    
    #%%%%%%%%%%%%solve normal equation of Newton's equation by FISTA %%%%%%%%%%%%%%#
    Newton_up = 0*sol_it
    Newton_up_old = Newton_up
    t=0
    for iteration in range(1, FISTA_steps):
        # extra-gradient step
        told = t
        t = (1 + np.sqrt(1+4*t*t))/2
        beta = (told-1)/t
        y = Newton_up + beta*(Newton_up-Newton_up_old)
        derivy=deriv(y)
        derivy=M**2*Theta_op._deriv_adjoint(taumat, taumat, Proj_op(derivy))
        FISTAup=deriv._adjoint(Proj_op._adjoint(derivy))
        Newton_up_old = Newton_up
        Newton_up = y-  mu*(FISTAup-backprop)
        Newton_up = np.minimum(FISTA_ub_ab, np.maximum(FISTA_lb_ab,Newton_up.real)) \
            + complex(0,1)*np.minimum(FISTA_ub_ph, np.maximum(FISTA_lb_ph,Newton_up.imag))
    sol_it = sol_it + Newton_up



#error[Newton_it]=np.linalg.norm(sol_it-exact_solution)/np.linalg.norm(exact_solution)
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot reaults %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# 
  
 
#plt.semilogy([np.linalg.norm(x-exact_solution) for x in sol_it])


plt.figure()
plt.imshow(intens_tot)
plt.colorbar()
plt.title('Total intensity')
plt.show()


plt.figure()
plt.imshow(-sol_it.real)
plt.colorbar()
plt.title('recon absorption')
plt.show()

plt.figure()
plt.imshow(-contrast.real)
plt.colorbar()
plt.title('Exact absorption')
plt.show()

plt.figure()
plt.imshow(sol_it.imag)
plt.colorbar()
plt.title('recon phase')
plt.show()

plt.figure()
plt.imshow(contrast.imag)
plt.colorbar()
plt.title('Exact phase')
plt.show()

plt.figure()
plt.plot(sol_it[10, :].imag, label='recon phase')
plt.plot(contrast[10, :].imag, label='true phase')
plt.legend()
plt.show()

plt.figure()
plt.plot(-sol_it[10, :].real, label='recon absorption')
plt.plot(-contrast[10, :].real, label='true absorption')
plt.legend()
plt.show()


