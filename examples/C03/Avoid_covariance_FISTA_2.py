import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from regpy.hilbert import L2, Sobolev
from regpy.vecsps import UniformGridFcts
from regpy.vecsps import VectorSpace
from regpy.solvers import TikhonovRegularizationSetting
from regpy.solvers.nonlinear.fista import FISTA
from regpy.operators import FourierTransform
import regpy.stoprules as rules
from regpy.operators import Exponential, SquaredModulus, VectorOfOperators, Identity, RealPart, ImaginaryPart
from regpy.solvers.nonlinear.fista import FISTA
from regpy.functionals import QuadraticNonneg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from regpy.operators.fresnel import fresnel_propagator
import matplotlib.animation as animation
import time

from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop, Ptw_Multiplication, Mat, Theta_2, Tau, Proj, Reshape, Real_to_complex
from LowRank import HilbertSchmidtLowRank
from create_Vcov import _create_Vcov

import numpy as np
from math import floor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

################################################ Initialization parameters ################################
N=256             # Pixel numbers
N_frame=1000       # Number of frames ( or realizations)
T=10**12            # the observation time or the number of photon counts
fresnel_number=1e-3   # not properly scaled
coherence_len = 0.3 # coherence length
N_b=2             # denotes the rank of the matrix V
Newton_steps =5    # Number of Newton updates
CG_steps=15         # Number of Conjugate gradient steps
N_ADMM=10          # Number of ADMM iterations
sigma=0.5           # parameter used in the rapid deacaying function
gamma=2           # proximity parameter
 ############################################################################
# perform a uniform grid
xsample=np.arange(-1,1-1/N,2/N)
ysample=xsample
grid=UniformGridFcts((-1, 1, N), (-1, 1, N), dtype=complex)
# perform Fresenel propagation operator
fp=fresnel_prop(grid, number=complex(0, 1)/(2*fresnel_number))

create_types=['spatial', 'Fresnelprop', 'fourier_random']
create_type='fourier_random'

Vcov=_create_Vcov(N, N_b, create_type=create_type, grid=grid, sigma=sigma, xsample=xsample, ysample=ysample)
    
# create test images
X,Y = np.meshgrid(xsample, ysample, sparse=False)

#absorp=np.load('cell1.npy')
#phase=np.load('cell2.npy')
absorp=np.load('cell256.npy')
phase=absorp
#support_mask=((abs(X)<=0.801)*(abs(Y)<=0.801)).astype('int')  #constant rectangular bump
support_mask=((abs(X)**2+abs(Y)**2)<=0.6).astype('int')   # constant circular bump
contrast = (0.01+0.01*complex(0,1))*support_mask+(0.1*absorp + 0.2*complex(0,1) * phase)
contrast=contrast.real


grid_codomain= VectorSpace((N, N, N_b), dtype=complex)
grid_codomain_2=VectorSpace((N, N, N_b, N_b), dtype=complex)
grid_codomain_3=VectorSpace((2, N, N, N_b, N_b), dtype=complex)
grid_codomain_4=VectorSpace((N_frame, N, N))
grid_codomain_5=VectorSpace((N, N, N_b**2), dtype=complex)
grid_dom=UniformGridFcts(np.arange(2), xsample, ysample)


# defines the support of the contrast 
mask=(contrast!=0)
#Rtc=Real_to_complex(grid_dom, grid)
Mat_op=Mat(grid, grid_codomain, Vcov, fp)
Tau_op=Tau(Mat_op.codomain, grid_codomain_2)
Proj_op=Proj(grid_codomain_2, grid_codomain_3)
Theta_op=Theta_2(grid_codomain_3, grid_codomain_4, N, N_b) #The codomain has the dimension of the intensities
Resh=Reshape(grid_codomain_2, grid_codomain_5)
op=Resh*Tau_op*Mat_op

#Correct the code from here

ptw_detection= SquaredModulus(grid)
taumat_0=Mat_op(contrast) 
# create noisy intensity data
intens_tot=np.zeros((N, N))
intensities=np.zeros((N_frame, N, N))

for i in range(0, N_frame):
    print(i)
    random=1/np.sqrt(2)*(np.random.randn(N_b)+complex(0,1)*np.random.randn(N_b))
    uinc=np.tensordot(taumat_0, random, axes=([-1], [0]))
    signal=ptw_detection(uinc)
    
    #Cox-processes
    signal=(1/T)*np.random.poisson(lam=T*signal.flatten(), size=(N**2)).reshape(N, N)
    intens_tot+=signal
    intensities[i, :, :]=signal
    

intensities-=intens_tot/N_frame

#data=list(intensities/np.sqrt(N_frame))
data=intensities.transpose([1, 2, 0])/np.sqrt(N_frame)
dom = op.codomain
Sfun = HilbertSchmidtLowRank(domain = dom+dom, data=[data])

taumat=op(contrast)
taumat_join=Sfun.domain.join(taumat, taumat)
Sfun.getLipschitz(taumat_join)

#test=Sfun.eval(taumat_join)
#test_2=Sfun.eval(taumat_join, arguments_identical=True)
#print(test)
#print(test_2)

#corr_2=np.tensordot(intensities, intensities, axes=([0], [0]))/N_frame
#corr=np.tensordot(data, data, axes=([-1], [-1]))
#taumat=op(contrast)
#cov=np.tensordot(taumat, taumat.conj(), axes=([-1], [-1]))

#print(1/2*np.linalg.norm((cov.real-corr).reshape(N**2, N**2))**2)

#print(np.linalg.norm(cov-corr)/np.linalg.norm(cov))


Double = VectorOfOperators([Identity(op.codomain), Identity(op.codomain)])

Re=RealPart(grid)

Im=ImaginaryPart(op.domain)

#penalty=QuadraticNonneg(op.domain)

penalty=QuadraticNonneg(Re.codomain)

setting = TikhonovRegularizationSetting(op=Double*op*Re.adjoint, penalty=penalty, data_fid = Sfun,regpar=1e-20)

FISTA_solver = FISTA(setting)

stoprule = (rules.CountIterations(10))

reco, reco_data=FISTA_solver.run(stoprule)

#it = iter(FISTA_solver)

#x,_ = next(it)
#print(np.linalg.norm(x-contrast)/np.linalg.norm(contrast))

