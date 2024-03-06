from regpy.hilbert import L2, Sobolev
from regpy.discrs import UniformGrid
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
import regpy.stoprules as rules
from regpy.operators import Exponential, SquaredModulus, Ptw_Multiplication
#from regpy.operators.fresnel import fresnel_propagator

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop

import numpy as np
from math import floor
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)
#%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%
N=30  #N^2 is the pixel number
M=10   # Shots per frame
N_frame=10000  # frames
T=10**12         # the observation time or the number of photon counts
fresnel_number=40 # not properly scaled
coherence_len = 0.3 # coherence length
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
grid_codomain=UniformGrid(N**2, N**2, dtype=complex)

cov_u=np.eye(N**2)

#fp=_build_fresnel_2(grid, number=complex(0, 1)/(2*fresnel_number))
#col=_build_fresnel_2(grid, coherence_len**2)
fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))
col=fresnel_prop(grid, coherence_len**2)
Fresnelprop=np.zeros((N**2,N**2),dtype=complex)
convmat2d=np.zeros((N**2,N**2))
for i in range(0,N*N):
    fj=grid.zeros().flatten()
    fj[i]=1
    fjsq=fj.reshape(grid.shape)
    Fresnelprop[:, i]=fp(fjsq).flatten()
    convmat2d[:, i]=col(fjsq).real.flatten()/N**2
    
X,Y = np.meshgrid(xsample, ysample, sparse=False)


absorp_0=(abs(X)<0.8)*(abs(Y)<0.199)+(abs(X)<0.299)*(abs(Y)<0.7)+(X**2+Y**2<=0.5**2)*(X**2+Y**2>=0.45**2)
absorp_0=absorp_0.astype('int')

absorp_1=(X**2+Y**2<=0.501**2)*(X**2+Y**2>=0.45**2)
absorp_1=absorp_1.astype('int')
absorp=absorp_0+absorp_1

phase = ((abs(X+Y) <=0.101)+(abs(X-Y) <= 0.101)).astype('int')
support_mask=((abs(X)<=0.801)*(abs(Y)<=0.801)).astype('int')
contrast = support_mask*(-0.1*absorp + 0.1*complex(0,1) * phase)

FISTA_ub_ab = FISTA_ub_absorp*support_mask;
FISTA_lb_ab = FISTA_lb_absorp*support_mask;
FISTA_ub_ph = FISTA_ub_phase*support_mask;
FISTA_lb_ph = FISTA_lb_phase*support_mask;

ex=Exponential(grid)
corr_op=Corr(ex.codomain, grid_codomain, Fresnelprop, convmat2d)
detection_op = SquaredModulus(corr_op.codomain)
Mmult=Ptw_Multiplication(detection_op.codomain, M*detection_op.codomain.ones())
op=Mmult*detection_op*corr_op*ex
y=op(contrast)

ptw_detection= SquaredModulus(grid)
#ptw_op=ptw_detection*fp*ex
mult=Ptw_Multiplication(grid, np.exp(contrast))
ptw_op=ptw_detection*fp*mult

#frqs = grid.coords*0.5*np.pi*N
#FTcovui=np.fft.fftshift(np.exp(-coherence_len**2 * (frqs[0]**2 + frqs[1]**2)))
#%%%%%%%%%%%%%%%%%%%%% Creating shot noise %%%%%%%%%%%%%%%%%%%%%%%%%%%%       
corr_signal=np.zeros((N**2, N**2))
intens_tot=np.zeros((N, N))
uincmat=np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame)
uincmat=np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame*M)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame*M)
uincmat=uincmat.reshape(N_frame, M, N**2)
for i in range(0, N_frame):
    signal=ptw_op.codomain.zeros()
    for j in range(0, M):
        #random=(np.random.randn(N**2)+complex(0,1)*np.random.randn(N**2))
        #uinc=A.dot(random).reshape(N, N)
        #uinc=np.random.multivariate_normal(np.zeros(N**2), convmat2d)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d)
        uinc=uincmat[i,j, :].reshape(N, N)
        #Shot noise
        signal+=ptw_op(uinc)
    
    #Cox-processes
    signal=(1/T)*np.random.poisson(lam=T*signal.flatten(), size=(N**2)).reshape(N, N)
    intens_tot+=signal
    corr_signal+=signal.reshape(N**2, 1).dot(signal.conj().reshape(1, N**2))
    
crosscor=(corr_signal-intens_tot.reshape(N**2, 1).dot(intens_tot.reshape(1, N**2))/N_frame)/N_frame
    
exact_solution = contrast
exact_data, deriv = op.linearize(exact_solution)
regpar=deriv.norm()
#noise = 10**(-12) * op.codomain.randn()
#data = exact_data + noise
data=crosscor
noise=crosscor-exact_data
init=op.domain.zeros()

data=crosscor.copy()
#data=exact_data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate forward operator %%%%%%%%%%%%%%%%
print('norm(data)=\n', np.linalg.norm(data))
sol_it = np.zeros((N, N))

error=np.zeros(Newton_steps)

for Newton_it in range(Newton_steps):
    print(Newton_it)
    # evaluate forward operator at sol_it
    cov_it, deriv=op.linearize(sol_it)
    # apply adjoint to rhs of Newton equation
    backprop=deriv._adjoint(data-cov_it)
    #compute approximation to operator norm of linearized forward operator T
    # by power method (we compute the largest eigenvalue 1/mu of T'*T)
    mu=1/deriv.norm()**2
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
        FISTAup=deriv._adjoint(deriv(y))
        Newton_up_old = Newton_up
        Newton_up = y-  mu*(FISTAup-backprop)
        Newton_up = np.minimum(FISTA_ub_ab, np.maximum(FISTA_lb_ab,Newton_up.real)) \
            + complex(0,1)*np.minimum(FISTA_ub_ph, np.maximum(FISTA_lb_ph,Newton_up.imag))
    sol_it = sol_it + Newton_up
    error[Newton_it]=np.linalg.norm(sol_it-exact_solution)/np.linalg.norm(exact_solution)
    
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot reaults %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#  
plt.figure()
plt.imshow(uinc.real)
plt.colorbar()
plt.title('Sample Incident Field')
plt.show()


plt.figure()
plt.imshow(intens_tot)
plt.colorbar()
plt.title('Total intensity')
plt.show()


plt.figure()
plt.imshow(abs(exact_data))
plt.colorbar()
plt.title('Covariance matrix ')
plt.show()


plt.figure()
plt.imshow(np.transpose(-sol_it.real))
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
axs[0, 0].contourf(xsample, ysample, uinc.real, levels=levels)
axs[0, 0].set_title('Sample Incident Field', pad=fontsize)
axs[1, 0].contourf(xsample, ysample, contrast.imag, levels=levels)
axs[2, 0].set_title('Exact absorption', pad=fontsize)
axs[2, 0].contourf(xsample, ysample, -contrast.real, levels=levels)
axs[1, 0].set_title('Exact phase contrast', pad=fontsize)
axs[0, 1].contourf(xsample, ysample, intens_tot, levels=levels)
axs[0, 1].set_title('Total Intensity', pad=fontsize)
axs[1, 1].contourf(xsample, ysample, sol_it.imag, levels=levels)
axs[1, 1].set_title('Reconstructed phase', pad=fontsize)
axs[2, 1].contourf(xsample, ysample, np.transpose(-sol_it.real), levels=levels)
axs[2, 1].set_title('Reconstructed absorption', pad=fontsize)
axs[0, 2].contourf(np.arange(N**2), np.arange(N**2), abs(exact_data), levels=levels)
axs[1, 2].plot(sol_it[10, :].imag, label='Approx phase')
axs[0, 2].set_title('Covariance matrix', pad=fontsize)
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
plt.savefig("image.png")

