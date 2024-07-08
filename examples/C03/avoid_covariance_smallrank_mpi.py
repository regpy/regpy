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

from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop, Ptw_Multiplication

import numpy as np
from math import floor
import logging

from mpi4py import MPI
# initiate MPI for python, note the openmpi should from anaconda!
host = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from regpy.operators import Operator

class small_rank_basis(Operator):
    
    def __init__(self, random_coeffs, codomain):
        self.random_coeffs=random_coeffs
        domain=UniformGridFcts(len(self.random_coeffs), dtype=complex)
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        vec=self.codomain.zeros().flatten()
        vec[self.random_coeffs]=x
        return vec.reshape(self.codomain.shape)
    
    def _adjoint(self, y):
        return y.flatten()[self.random_coeffs]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)
#%%%%%%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%
N=200  #N^2 is the pixel number
M=10   # Shots per frame
N_frame=1000  # frames
T=10**12        # the observation time or the number of photon counts
fresnel_number=30 # not properly scaled
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

N_s=int(N/size)

cov_u=np.eye(N**2)


#fp=_build_fresnel_2(grid, number=complex(0, 1)/(2*fresnel_number))
#col=_build_fresnel_2(grid, coherence_len**2)
fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))
col=fresnel_prop(grid, coherence_len**2)
Fresnelprop=np.zeros((N**2, N_b),dtype=complex)
convmat2d=np.zeros((N_b,N_b))
random_coeffs=np.zeros(N_b)
if rank==0:
    random_coeffs=np.zeros(N_b)
    from random import randrange
    for i in range(0, N_b):
        random_coeffs[i]=randrange(N**2)
    random_coeffs=random_coeffs.astype(int)

    for i in range(0,N_b):
        fj=grid.zeros().flatten()
        fj[random_coeffs[i]]=1
        fjsq=fj.reshape(grid.shape)
        Fresnelprop[:, i]=fp(fjsq).flatten()
        vec=col(fjsq).real.flatten()/N**2
        convmat2d[:, i]=vec[random_coeffs]
        
convmat2d=comm.bcast(convmat2d, root=0)
Fresnelprop=comm.bcast(Fresnelprop, root=0)
random_coeffs=comm.bcast(random_coeffs, root=0)

def _convmat2d(x):
    return basis_op(convmat2d.dot(basis_op._adjoint(x)))
    
def _convmat2d_conj(y):
    return basis_op(convmat2d.T.conj().dot(basis_op._adjoint(y)))

    
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
basis_op= small_rank_basis(random_coeffs, codomain=mult.domain)
ptw_op=ptw_detection*fp*mult*basis_op

#%%%%%%%%%%%%%%%%%%%%% Creating shot noise %%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
#corr_signal=np.zeros((N**2, N**2))
intens_tot=np.zeros((N, N))
#uincmat=np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame)
#uincmat=np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame*M)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d, N_frame*M)
#uincmat=uincmat.reshape(N_frame, M, N**2)

intensities=np.zeros((N_frame, N, N))
if rank==0:
    
    W, V=np.linalg.eigh(convmat2d)
    
    
    A=V*np.sqrt(np.maximum(W, np.zeros((N_b))))
    
    for i in range(0, N_frame):
        signal=ptw_op.codomain.zeros()
        for j in range(0, M):
            random=(np.random.randn(N_b)+complex(0,1)*np.random.randn(N_b))
            uinc=A.dot(random)
            #uinc=np.random.multivariate_normal(np.zeros(N**2), convmat2d)+complex(0,1)*np.random.multivariate_normal(np.zeros(N**2), convmat2d)
            #uinc=uincmat[i,j, :].reshape(N, N)
            #uinc=uinc.reshape(N, N)
            #Shot noise
            signal+=ptw_op(uinc)
        
        #Cox-processes
        signal=(1/T)*np.random.poisson(lam=T*signal.flatten(), size=(N**2)).reshape(N, N)
        intens_tot+=signal
        
        intensities[i, :, :]=signal
        
    
    intensities-=intens_tot/N_frame
    
intensities=comm.bcast(intensities, root=0)

#np.save(save_path, intensities)
#intensities=np.load(save_path)


#correlation_data=1/N_frame*intensities.reshape(N_frame, N**2).T.dot(intensities.reshape(N_frame, N**2).conj())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate forward operator %%%%%%%%%%%%%%%%
sol_it = np.zeros((N, N))

error=np.zeros(Newton_steps)
ims=[]
fontsize=2
levels=40
for Newton_it in range(Newton_steps):
    if rank==0:
        print('Newton iteration'+str(Newton_it))
    # evaluate forward operator at sol_it
    
    
    backprop=np.zeros((N, N), dtype=complex)
    mult_sol_it=np.exp(sol_it)
    """mat=np.zeros((N, N, N_b), dtype=complex)
    for i in range(0, N_b):
        arr=np.zeros(N_b)
        arr[i]=1
        mat[:, :, i]=fp(mult_sol_it*basis_op(A.dot(arr)))
     
    print('Compute back-propagation')    
    for i in range(0, N):
        print(i)
        t0=time.time()
        for j in range(0, N):
            third=2*np.tensordot(mat, mat[i, j, :].conj(), axes=([-1], [0]))
            res=M*abs(third)**2
            
            #Store the measurements as vector: [intensities]=N_frame, N, N
            signal=np.mean(intensities*intensities[:, i, j].conj().reshape(N_frame, 1, 1), axis=0)
                       
            #Adjoint of the data-cov_it
            deriv_adj=2*M*(third*(signal-res))
            deriv_1_adj=2*fp._adjoint(deriv_adj)*mult_sol_it.T.conj()
            
            first=basis_op(mat[i, j, :].conj())
            second=_convmat2d(first)
            #rhs_adj=(first.conj()*(convmat2d.T.conj().dot(deriv_1_adj.flatten())).reshape(N, N)).T.conj()
            rhs_adj=(first.conj()*(_convmat2d_conj(deriv_1_adj))).T.conj()
            
            deriv_adj_2=(fp._adjoint(2*deriv_adj)*second.conj()).T*mult_sol_it.conj()
            backprop+=rhs_adj+deriv_adj_2 
        t1=time.time()
        print(t1-t0)"""
    
    if rank==0:        
        print('Compute back-propagation')
    for i in range(0, N_s):
        if rank==0:
            print(i)
        for j in range(0, N):
            #Compute the covariance for one rhs.
            arr=np.zeros((N, N))
            arr[i+N_s*rank, j]=1
            first=mult_sol_it.T.conj()*fp._adjoint(arr)
            second=_convmat2d(first)
            third=2*fp._eval(mult_sol_it.T*second)
            res=M*abs(third)**2
            
            #Store the measurements as vector: [intensities]=N_frame, N, N
            signal=np.mean(intensities*intensities[:, i+rank*N_s, j].conj().reshape(N_frame, 1, 1), axis=0)
                       
            #Adjoint of the data-cov_it
            deriv_adj=2*M*(third*(signal-res))
            deriv_1_adj=2*fp._adjoint(deriv_adj)*mult_sol_it.T.conj()
            
            #rhs_adj=(first.conj()*(convmat2d.T.conj().dot(deriv_1_adj.flatten())).reshape(N, N)).T.conj()
            rhs_adj=(first.conj()*(_convmat2d_conj(deriv_1_adj))).T.conj()
            
            deriv_adj_2=(fp._adjoint(2*deriv_adj)*second.conj()).T*mult_sol_it.conj()
            backprop+=rhs_adj+deriv_adj_2   
            
    backprop=comm.reduce(backprop, op=MPI.SUM, root=0)
    backprop=comm.bcast(backprop, root=0)
    
    #compute approximation to operator norm of linearized forward operator T
    # by power method (we compute the largest eigenvalue 1/mu of T'*T)
    
    #mu=1/deriv.norm()**2
    #print('mu=\n',mu)
    
    mu=1818635001956990.0
    
    mu=1805985772025204.0
    
    #%%%%%%%%%%%%solve normal equation of Newton's equation by FISTA %%%%%%%%%%%%%%#
    Newton_up = 0*sol_it
    Newton_up_old = Newton_up
    t=0
    if rank==0:
        print('FISTA inner step')
    for iteration in range(1, FISTA_steps):
        # extra-gradient step
        told = t
        t = (1 + np.sqrt(1+4*t*t))/2
        beta = (told-1)/t
        y = Newton_up + beta*(Newton_up-Newton_up_old)
        
        #FISTAup=deriv._adjoint(deriv(y))
        
        
        FISTAup=np.zeros((N, N), dtype=complex)
        mult_sol_it=np.exp(sol_it)
        for i in range(0, N_s):
            if rank==0:
                print(i)
            for j in range(0, N):
                #Compute the covariance for one rhs.
                arr=np.zeros((N, N))
                arr[i+rank*N_s, j]=1
                first=mult_sol_it.T.conj()*fp._adjoint(arr)
                #Convmat2d can not be stored anymore. Use low order approximations instead.
                
                #second=convmat2d.dot(first.flatten()).reshape(N, N)
                second=_convmat2d(first.flatten()).reshape(N, N)
                
                
                third=2*fp._eval(mult_sol_it.T*second)
                res=M*abs(third)**2
                
                #Compute the derivative, from the variable h
                deriv_1=2*fp._eval((mult_sol_it*y).T*second)
                #rhs=convmat2d.dot((first*y.T.conj()).flatten()).reshape(N, N)
                rhs=_convmat2d((first*y.T.conj()))
                deriv_1+=2*fp._eval(mult_sol_it.T*rhs)
                deriv=2*M*(third.conj()*deriv_1).real
                           
                #Adjoint of the derivative, factor of 2 coming from the adjoint
                deriv_adj=2*M*(third*deriv)
                deriv_1_adj=2*fp._adjoint(deriv_adj)*mult_sol_it.T.conj()
                
                #rhs_adj=(first.conj()*(convmat2d.T.conj().dot(deriv_1_adj.flatten())).reshape(N, N)).T.conj()
                rhs_adj=(first.conj()*(_convmat2d_conj(deriv_1_adj.flatten())).reshape(N, N)).T.conj()
                
                deriv_adj_2=(fp._adjoint(2*deriv_adj)*second.conj()).T*mult_sol_it.conj()
                FISTAup+=rhs_adj+deriv_adj_2
                
        
        FISTAup=comm.reduce(FISTAup, op=MPI.SUM, root=0)
        FISTAup=comm.bcast(FISTAup, root=0)
        
        Newton_up_old = Newton_up
        Newton_up = y-  mu*(FISTAup-backprop)
        Newton_up = np.minimum(FISTA_ub_ab, np.maximum(FISTA_lb_ab,Newton_up.real)) \
            + complex(0,1)*np.minimum(FISTA_ub_ph, np.maximum(FISTA_lb_ph,Newton_up.imag))


if rank==0:
    np.save("reco.npy", sol_it)