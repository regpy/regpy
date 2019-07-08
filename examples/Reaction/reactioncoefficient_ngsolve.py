import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.spaces import L2, NGSolveDiscretization
from itreg.solvers import Landweber, HilbertSpaceSetting

from ngsolve.meshes import MakeQuadMesh

import itreg.stoprules as rules

import numpy as np
import logging
#import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')


meshsize_domain=10
meshsize_codomain=10

from ngsolve import *
mesh = MakeQuadMesh(meshsize_domain)
fes_domain = L2(mesh, order=2, dirichlet="left|top|right|bottom")
domain= NGSolveDiscretization(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain)
fes_codomain = H1(mesh, order=2, dirichlet="left|top|right|bottom")
codomain= NGSolveDiscretization(fes_codomain)

rhs=10*sin(x)*sin(y)
op = Coefficient(domain, codomain=codomain, rhs=rhs, bc_left=0, bc_right=0, bc_bottom=0, bc_top=0, diffusion=False, reaction=True, dim=2)


exact_solution_coeff = x+1
gfu_exact_solution=GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

init=1+x**2
init_gfu=GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

from itreg.spaces import NGSolveSpace
setting = HilbertSpaceSetting(op=op, domain=NGSolveSpace, codomain=NGSolveSpace)

landweber = Landweber(setting, data, init_solution, stepsize=3)
stoprule = (
    rules.CountIterations(3000) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

Draw (gfu_exact_solution)
Draw (init, op.fes_domain.mesh, "init")

#Draw recondtructed solution
gfu_reco=GridFunction(op.fes_domain)
gfu_reco.vec.FV().NumPy()[:]=reco
coeff_reco=CoefficientFunction(gfu_reco)

Draw (coeff_reco, op.fes_domain.mesh, "reco")
    

#Draw data space
gfu_data=GridFunction(op.fes_codomain)
gfu_reco_data=GridFunction(op.fes_codomain)

gfu_data.vec.FV().NumPy()[:]=data
coeff_data = CoefficientFunction(gfu_data)

gfu_reco_data.vec.FV().NumPy()[:]=reco_data
coeff_reco_data = CoefficientFunction(gfu_reco_data)

Draw(coeff_data, op.fes_codomain.mesh, "data")
Draw(coeff_reco_data, op.fes_codomain.mesh, "reco_data")






