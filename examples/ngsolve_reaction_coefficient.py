# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
from ngsolve.meshes import MakeQuadMesh
import numpy as np

import regpy.stoprules as rules
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import L2, Sobolev
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

meshsize_domain = 10
meshsize_codomain = 10

mesh = MakeQuadMesh(meshsize_domain)
fes_domain = ngs.L2(mesh, order=2)
domain = NgsSpace(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain)
fes_codomain = ngs.H1(mesh, order=3, dirichlet="left|top|right|bottom")
codomain = NgsSpace(fes_codomain)

rhs = 10 * ngs.sin(ngs.x) * ngs.sin(ngs.y)
op = Coefficient(
    domain, rhs, codomain=codomain, bc_left=0, bc_right=0, bc_bottom=0, bc_top=0, diffusion=False,
    reaction=True, dim=2
)

exact_solution_coeff = ngs.x + 1
gfu_exact_solution = ngs.GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)

fes_noise=ngs.L2(fes_codomain.mesh, order=1)
gfu_noise_order1=ngs.GridFunction(fes_noise)
gfu_noise_order1.vec.FV().NumPy()[:]=0.0001*np.random.randn(fes_noise.ndof)
gfu_noise=ngs.GridFunction(fes_codomain)
gfu_noise.Set(gfu_noise_order1)
noise=gfu_noise.vec.FV().NumPy()

data = exact_data+noise

init = 1 + ngs.x ** 2
init_gfu = ngs.GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_data = op(init_solution)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=Sobolev)

landweber = Landweber(setting, data, init_solution, stepsize=1)
stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

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
