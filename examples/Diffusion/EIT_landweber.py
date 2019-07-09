import setpath

from itreg.operators.NGSolveProblems.EIT import EIT
from itreg.spaces import NGSolveDiscretization
from itreg.solvers import Landweber, HilbertSpaceSetting

import itreg.stoprules as rules

import numpy as np
import logging
#import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

from ngsolve import *
mesh=Mesh('..\..\itreg\meshes_ngsolve\meshes\circle.vol.gz')

fes_domain = L2(mesh, order=2)
domain= NGSolveDiscretization(fes_domain)

fes_codomain = H1(mesh, order=2)
codomain= NGSolveDiscretization(fes_codomain)

g=0.1*(x-0.5)*(y-0.5)
op = EIT(domain, g, codomain=codomain)
pts=np.array(op.pts)
nr_points=pts.shape[0]

exact_solution_coeff = sin(y)
gfu_exact_solution=GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

init=0.5*y
init_gfu=GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_sol=init_solution.copy()
init_data=op(init_solution)

_, deriv=op.linearize(exact_solution)
adj=deriv.adjoint(exact_data)

