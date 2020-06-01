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

#WARNING: Only works if grad(exact_data) != 0 everywhere
#This example file implements: exact_data = ngs.exp(ngs.x+ngs.y)

meshsize_domain = 10
meshsize_codomain = 10

mesh = MakeQuadMesh(meshsize_domain)
fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain)
fes_codomain = ngs.H1(mesh, order=3, dirichlet="left|top|right|bottom")
codomain = NgsSpace(fes_codomain)

rhs = -2*ngs.exp(ngs.x+ngs.y)
op = Coefficient(
    domain, rhs, codomain=codomain, bc_left=ngs.exp(ngs.x+ngs.y), bc_right= ngs.exp(ngs.x+ngs.y),
    bc_bottom=ngs.exp(ngs.x+ngs.y), bc_top=ngs.exp(ngs.x+ngs.y), diffusion=True, reaction=False, dim=2
)

exact_solution_coeff = 1
exact_solution = domain.fromcoefficientfunction( exact_solution_coeff )
exact_data = op(exact_solution)

noise = 0 * 0.0001 * codomain.randn()

data = exact_data+noise

init = domain.fromcoefficientfunction ( 1.2 )
init_data = op(init)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=Sobolev)

landweber = Landweber(setting, data, init, stepsize=0.001)
stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, op.fes_domain.mesh, "exact")

# Draw reconstructed solution
domain.draw(reco, "reco")

# Draw data space
codomain.draw(data, "data")
codomain.draw(reco_data, "reco_data")


codomain.draw(reco_data, "reco_data")
codomain.draw(data, "data")
