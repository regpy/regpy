# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:27:43 2019

@author: Hendrik Müller
"""

import setpath


from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.util import test_adjoint
from itreg.operators.MRI import parallel_MRI
import itreg.stoprules as rules

import numpy as np
import scipy as scp
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')
Nx=128
Ny=96
#nr_coils=3
P = np.ones((Nx,Ny))
samplingIndx = np.nonzero(np.reshape(P, (Nx*Ny, 1)))[0]
#op =parallel_MRI(L2(4*2*4), Nx=Nx, Ny=Ny, samplingIndx=samplingIndx, nr_coils=nr_coils) 
op=parallel_MRI(L2(128*96*13))

exact_solution=np.ones(128*96*13)+1j*np.zeros(128*96*13)
exact_data=op(exact_solution)
yscale=100/np.linalg.norm(exact_data)
data=yscale*exact_data
        
init = op.domain.one()+1j*op.domain.zero()
#init=np.ones(128*96*13)+1j*np.zeros(128*96*13)
_, deriv = op.linearize(init)
test_adjoint(deriv)
