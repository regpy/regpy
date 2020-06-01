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
fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

mesh = MakeQuadMesh(meshsize_codomain)
fes_codomain = ngs.H1(mesh, order=3, dirichlet="left|top|right|bottom")
codomain = NgsSpace(fes_codomain)

rhs = 1 * ngs.sin(ngs.x) * ngs.sin(ngs.y)
op = Coefficient(
    domain, rhs, codomain=codomain, bc_left=0.1, bc_right=0.1, bc_bottom=0.1, bc_top=0.1, diffusion=False,
    reaction=True, dim=2
)

exact_solution_coeff = 1+0.8*ngs.sin(2*np.pi*ngs.x) * ngs.sin(2*np.pi*ngs.y)
exact_solution = domain.from_ngs( exact_solution_coeff )
exact_data = op(exact_solution)

noise = 0.0001 * codomain.randn()

data = exact_data+noise

init = domain.from_ngs ( 1 )
init_data = op(init)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=Sobolev)

landweber = Landweber(setting, data, init, stepsize=1)
stoprule = (
        rules.CountIterations(50000) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, op.fes_domain.mesh, "exact")

# Draw reconstructed solution
domain.draw(reco, "reco")

# Draw data space
codomain.draw(data, "data")
codomain.draw(reco_data, "reco_data")

