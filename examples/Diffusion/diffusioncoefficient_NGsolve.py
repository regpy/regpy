import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.spaces import NGSolveDiscretization
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
fes_domain = H1(mesh, order=3)
domain= NGSolveDiscretization(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain)
fes_codomain = H1(mesh, order=3, dirichlet="left|top|right|bottom")
codomain= NGSolveDiscretization(fes_codomain)

rhs=10*sin(x)*sin(y)
op = Coefficient(domain, rhs, codomain=codomain, bc_left=0, bc_right=1, bc_bottom=sin(y), bc_top=sin(y), dim=2)

exact_solution_coeff = cos(x)*sin(y)
gfu_exact_solution=GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

init=cos(x)
init_gfu=GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

from itreg.spaces import NGSolveSpace_L2, NGSolveSpace_H1
setting = HilbertSpaceSetting(op=op, domain=NGSolveSpace_H1, codomain=NGSolveSpace_H1)

landweber = Landweber(setting, data, init_solution, stepsize=0.001)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(300) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

Draw (exact_solution_coeff, op.fes_domain.mesh, "exact")
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
