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

from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop, Ptw_Multiplication

import numpy as np
from math import floor
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

"""from regpy.operators import Operator
from regpy.discrs import UniformGrid

class small_rank_basis(Operator):
    
    def __init__(self, random_coeffs, codomain):
        self.random_coeffs=random_coeffs
        domain=UniformGrid(len(self.random_coeffs), dtype=complex)
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        vec=self.codomain.zeros().flatten()
        vec[self.random_coeffs]=x
        return vec.reshape(self.codomain.shape)
    
    def _adjoint(self, y):
        return y.flatten()[self.random_coeffs]"""

#%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%
N=20  #N^2 is the pixel number
M=10   # Shots per frame
N_frame=1000  # frames
T=10**12        # the observation time or the number of photon counts
fresnel_number=40 # not properly scaled
coherence_len = 0.3 # coherence length
N_b=100
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
grid=UniformGrid(xsample, ysample, dtype=complex)

cov_u=np.eye(N**2)


#fp=_build_fresnel_2(grid, number=complex(0, 1)/(2*fresnel_number))
#col=_build_fresnel_2(grid, coherence_len**2)
fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))
col=fresnel_prop(grid, coherence_len**2)



"""Fresnelprop=np.zeros((N**2, N_b),dtype=complex)
convmat2d=np.zeros((N_b,N_b))
random_coeffs=np.zeros(N_b)
from random import randrange
for i in range(0, N_b):
    random_coeffs[i]=randrange(N**2)
random_coeffs=random_coeffs.astype(int)
random_coeffs=np.sort(random_coeffs)
random_coeffs=np.arange(0, N**2)
for i in range(0,N_b):
    fj=grid.zeros().flatten()
    fj[random_coeffs[i]]=1
    fjsq=fj.reshape(grid.shape)
    Fresnelprop[:, i]=fp(fjsq).flatten()
    vec=col(fjsq).real.flatten()/N**2
    convmat2d[:, i]=vec[random_coeffs]

W, V=np.linalg.eigh(convmat2d)
Vcov=V*np.sqrt(np.maximum(W, np.zeros((N_b))))"""
    
    
import random
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
Vcov=V.T.conj()*S
    

def _convmat2d(x):
    #return basis_op(convmat2d.dot(basis_op._adjoint(x)))
    return (Vcov.dot(Vcov.T.conj().dot(x.flatten()))).reshape(N, N)
    
def _convmat2d_conj(y):
    #return basis_op(convmat2d.T.conj().dot(basis_op._adjoint(y)))
    return (Vcov.dot(Vcov.T.conj().dot(y.flatten()))).reshape(N, N)

    
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

#%%%%%%%%%%%%%%%%%%%%% Creating shot noise %%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
#corr_signal=np.zeros((N**2, N**2))
intens_tot=np.zeros((N, N))
#uincmat=np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame)
#uincmat=np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame*M)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame*M)
#uincmat=uincmat.reshape(N_frame, M, N**2)

intensities=np.zeros((N_frame, N, N))

for i in range(0, N_frame):
    print(i)
    signal=ptw_op.codomain.zeros()
    for j in range(0, M):
        random=(np.random.randn(N_b)+complex(0,1)*np.random.randn(N_b))
        uinc=Vcov.dot(random)
        #uinc=np.random.multivariate_normal(np.zeros(N**2), convmat2d)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d)
        #uinc=uincmat[i,j, :].reshape(N, N)
        #uinc=uinc.reshape(N, N)
        #Shot noise
        signal+=ptw_op(uinc.reshape(N, N))
    
    #Cox-processes
    signal=(1/T)*np.random.poisson(lam=T*signal.flatten(), size=(N**2)).reshape(N, N)
    intens_tot+=signal
    
    intensities[i, :, :]=signal
    

intensities-=intens_tot/N_frame

#intensities=np.load('intensities.npy')

def _deriv_adjoint(x, h):
    #Computation of F^'[x]^* F^'[x] h
    
    adj=np.zeros((N, N), dtype=complex)
    mult_x=np.exp(x)
    for i in range(0, N):
        for j in range(0, N):
            #Compute the covariance for one rhs.
            arr=np.zeros((N, N))
            arr[i, j]=1
            first=mult_x.T.conj()*fp._adjoint(arr)
            second=_convmat2d(first)
            third=2*fp._eval(mult_x.T*second)
            
            #Compute the derivative, from the variable h
            deriv_1=2*fp._eval((mult_x*h).T*second)
            rhs=_convmat2d((first*h.T.conj()))
            deriv_1+=2*fp._eval(mult_x.T*rhs)
            deriv=2*M*(third.conj()*deriv_1).real
                       
            #Adjoint of the derivative, factor of 2 coming from the adjoint
            deriv_adj=2*M*(third*deriv)
            deriv_1_adj=2*fp._adjoint(deriv_adj)*mult_x.T.conj()
            
            rhs_adj=(first.conj()*(_convmat2d_conj(deriv_1_adj))).T.conj()
            
            deriv_adj_2=(fp._adjoint(2*deriv_adj)*second.conj()).T*mult_x.conj()
            adj+=rhs_adj+deriv_adj_2
    return adj
    
def _backprop(x):
    #Computation of F^'[x]^* (F[x]-g^{obs})
    
    backprop=np.zeros((N, N), dtype=complex)
    mult_x=np.exp(x)
            
    print('Compute back-propagation')
    for i in range(0, N):
        print(i)
        for j in range(0, N):
            #Compute the covariance for one rhs.
            arr=np.zeros((N, N))
            arr[i, j]=1
            first=mult_x.T.conj()*fp._adjoint(arr)
            second=_convmat2d(first)
            third=2*fp._eval(mult_x.T*second)
            res=M*abs(third)**2
            
            #Store the measurements as vector: [intensities]=N_frame, N, N
            signal=np.mean(intensities*intensities[:, i, j].conj().reshape(N_frame, 1, 1), axis=0)
                       
            #Adjoint of the data-cov_it
            deriv_adj=2*M*(third*(signal-res))
            deriv_1_adj=2*fp._adjoint(deriv_adj)*mult_x.T.conj()
            
            rhs_adj=(first.conj()*(_convmat2d_conj(deriv_1_adj))).T.conj()
            
            deriv_adj_2=(fp._adjoint(2*deriv_adj)*second.conj()).T*mult_x.conj()
            backprop+=rhs_adj+deriv_adj_2
    return backprop


def _norm(x):
    h=np.random.randn(N**2).reshape(N, N)
    norm = np.sqrt(np.real(np.vdot(h, h)))
    for count in range(10):
        print(count)
        h = h / norm
        h = _deriv_adjoint(x, h)
        norm = np.sqrt(np.real(np.vdot(h, h)))
    return np.sqrt(norm)
        


#correlation_data=1/N_frame*intensities.reshape(N_frame, N**2).T.dot(intensities.reshape(N_frame, N**2).conj())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate forward operator %%%%%%%%%%%%%%%%
sol_it = np.zeros((N, N), dtype=complex)

error=np.zeros(Newton_steps)

for Newton_it in range(Newton_steps):
    print('Newton iteration'+str(Newton_it))
    # evaluate forward operator at sol_it
    if Newton_it==0:
        mu=1/_norm(sol_it)**2
        print('mu=\n',mu)
    
    backprop=_backprop(sol_it)
    
    #compute approximation to operator norm of linearized forward operator T
    # by power method (we compute the largest eigenvalue 1/mu of T'*T)
    
    #mu=1818635001956990.0
    
    #mu=22200630025657.832
    mu=6138805595708.638
    
    #%%%%%%%%%%%%solve normal equation of Newton's equation by FISTA %%%%%%%%%%%%%%#
    Newton_up = 0*sol_it
    Newton_up_old = Newton_up
    t=0
    print('FISTA inner step')
    for iteration in range(1, FISTA_steps):
        # extra-gradient step
        told = t
        t = (1 + np.sqrt(1+4*t*t))/2
        beta = (told-1)/t
        y = Newton_up + beta*(Newton_up-Newton_up_old)
        
        #FISTAup=deriv._adjoint(deriv(y))
        
        FISTAup=_deriv_adjoint(sol_it, y)
        Newton_up_old = Newton_up
        Newton_up = y-  mu*(FISTAup-backprop)
        Newton_up = np.minimum(FISTA_ub_ab, np.maximum(FISTA_lb_ab,Newton_up.real)) \
            + complex(0,1)*np.minimum(FISTA_ub_ph, np.maximum(FISTA_lb_ph,Newton_up.imag))
    sol_it+=Newton_up
    
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


fontsize=2
levels=40
plt.figure(figsize=(12, 12))
fig, axs = plt.subplots(3, 3)
#axs[0, 0].contourf(xsample, ysample, uinc.real, levels=levels)
#axs[0, 0].set_title('Sample Incident Field', pad=fontsize)
axs[1, 0].contourf(xsample, ysample, contrast.imag, levels=levels)
axs[2, 0].set_title('Exact absorption', pad=fontsize)
axs[2, 0].contourf(xsample, ysample, -contrast.real, levels=levels)
axs[1, 0].set_title('Exact phase contrast', pad=fontsize)
axs[0, 1].contourf(xsample, ysample, intens_tot, levels=levels)
axs[0, 1].set_title('Total Intensity', pad=fontsize)
axs[1, 1].contourf(xsample, ysample, sol_it.imag, levels=levels)
axs[1, 1].set_title('Reconstructed phase', pad=fontsize)
axs[2, 1].contourf(xsample, ysample, -sol_it.real, levels=levels)
axs[2, 1].set_title('Reconstructed absorption', pad=fontsize)
#axs[0, 2].contourf(np.arange(N**2), np.arange(N**2), abs(exact_data), levels=levels)
#axs[0, 2].set_title('Covariance matrix', pad=fontsize)
axs[1, 2].plot(sol_it[10, :].imag, label='Approx phase')
axs[1, 2].plot(contrast[10, :].imag, label='True phase')
axs[1, 2].legend()
axs[2, 2].plot(-sol_it[10, :].real, label='Approx. absorp')
axs[2, 2].plot(-contrast[10, :].real, label='True absorp')
axs[2, 2].legend()
#fig.tight_layout()
plt.show()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.3,
                    wspace=0.17)


np.save("reco.npy", sol_it)
reco=np.load("reco.npy")