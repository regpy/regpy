import setpath

import logging
import ngsolve as ngs
from ngsolve.meshes import MakeQuadMesh

import itreg.stoprules as rules
from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.landweber import Landweber
from itreg.spaces import Sobolev
from itreg.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

meshsize_domain = 10
meshsize_codomain = 10

mesh = MakeQuadMesh(meshsize_domain)
fes_domain = ngs.H1(mesh, order=3)
domain = NgsSpace(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain)
fes_codomain = ngs.H1(mesh, order=3, dirichlet="left|top|right|bottom")
codomain = NgsSpace(fes_codomain)

rhs = 10 * ngs.sin(ngs.x) * ngs.sin(ngs.y)
op = Coefficient(
    domain, rhs, codomain=codomain,
    bc_left=0, bc_right=1, bc_bottom=ngs.sin(ngs.y), bc_top=ngs.sin(ngs.y),
    dim=2
)

exact_solution_coeff = ngs.cos(ngs.x) * ngs.sin(ngs.y)
gfu_exact_solution = ngs.GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data = exact_data

init = ngs.cos(ngs.x)
init_gfu = ngs.GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_data = op(init_solution)

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=Sobolev)

landweber = Landweber(setting, data, init_solution, stepsize=0.001)
# irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
        rules.CountIterations(300) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, op.fes_domain.mesh, "exact")
ngs.Draw(init, op.fes_domain.mesh, "init")

# Draw recondtructed solution
gfu_reco = ngs.GridFunction(op.fes_domain)
gfu_reco.vec.FV().NumPy()[:] = reco
coeff_reco = ngs.CoefficientFunction(gfu_reco)

ngs.Draw(coeff_reco, op.fes_domain.mesh, "reco")

# Draw data space
gfu_data = ngs.GridFunction(op.fes_codomain)
gfu_reco_data = ngs.GridFunction(op.fes_codomain)

gfu_data.vec.FV().NumPy()[:] = data
coeff_data = ngs.CoefficientFunction(gfu_data)

gfu_reco_data.vec.FV().NumPy()[:] = reco_data
coeff_reco_data = ngs.CoefficientFunction(gfu_reco_data)

ngs.Draw(coeff_data, op.fes_codomain.mesh, "data")
ngs.Draw(coeff_reco_data, op.fes_codomain.mesh, "reco_data")
