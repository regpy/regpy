#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:46:07 2019

@author: agaltsov
"""

import setpath
import itreg

import yaml
import numpy as np

import matplotlib.pyplot as plt

from itreg.operators import MediumScatteringOneToMany, CoordinateProjection
from itreg.solvers import IterativeBorn

#from itreg.util.nfft_ewald import NFFT, Rep
from itreg.util import get_directions, potentials
#from itreg import stoprules as rules

# Set parameter file
parameter_file = 'parameter_files/default.yaml'

# Load parameters from file
with open(parameter_file,'r') as stream:
    par = yaml.load(stream,Loader=yaml.CLoader)
    p = par['p']
    m = par['m']

# Computing the incident and measurement directions
inc_directions, farfield_directions = get_directions(p)

# Initializing the problem
scattering = MediumScatteringOneToMany(
    gridshape=p['GRID_SHAPE'],
    radius=p['SUPPORT_RADIUS'],
    wave_number=p['WAVE_NUMBER'],
    inc_directions=inc_directions,
    farfield_directions=farfield_directions,
    equation='SCHROEDINGER')

# Initializing the contrast
v = getattr(potentials,p['POTENTIAL'])
contrast = v(scattering.domain)

# Defining the projection operator onto the contrast support
projection = CoordinateProjection(
    scattering.domain,
    scattering.support)

# Composition
op = scattering * projection.adjoint
proj = projection

exact_solution = projection(contrast)
exact_data = op(exact_solution)     # Forward map applied texacto exact solution
noise = p['NOISE_LVL'] * op.codomain.randn() * np.max(np.abs(exact_data))
data = exact_data + noise            # Noisy data for inversions

# Initializing the inversion method
BornSolver = IterativeBorn(op,proj,data,inc_directions,farfield_directions,p,m)

for x,y in BornSolver:
    r = y-BornSolver.rhs
    e = x - exact_solution
    r_norm = BornSolver.NFFT.norm(r) / BornSolver.NFFT.norm(BornSolver.rhs)
    e_norm = np.linalg.norm(e)/np.linalg.norm(exact_solution)

    print('{:2d}: |r|={:.2g}, |e|={:.2g}'.format(BornSolver.iteration,r_norm,e_norm))


plt.figure()
plt.imshow(np.abs(contrast))
plt.title('potential')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(proj.adjoint(x)))
plt.title('solution')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(proj.adjoint(e)))
plt.title('error')
plt.colorbar()
plt.show()
