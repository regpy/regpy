# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

import regpy.stoprules as rules
from regpy.operators.ngsolve import EIT
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import Sobolev, SobolevBoundary
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
geo.AddCircle((0, 0), r=1, bc="cyc", maxh=0.05)
mesh = ngs.Mesh(geo.GenerateMesh())

fes_domain = ngs.L2(mesh, order=4)
domain = NgsSpace(fes_domain)

fes_codomain = ngs.H1(mesh, order=4)
codomain = NgsSpace(fes_codomain)

g = 1#0.1 * (ngs.x - 0.5) * (ngs.y - 0.5)
op = EIT(domain, g, codomain=codomain, alpha=10**(-2))

exact_solution_coeff = 1+0.1*ngs.sqrt(ngs.y**2+ngs.x**2)
gfu_exact_solution = ngs.GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)

fes_noise=ngs.L2(fes_codomain.mesh, order=1)
gfu_noise_order1=ngs.GridFunction(fes_noise)
gfu_noise_order1.vec.FV().NumPy()[:]=0.001*np.random.randn(fes_noise.ndof)
gfu_noise=ngs.GridFunction(fes_codomain)
gfu_noise.Set(gfu_noise_order1)
noise=op._get_boundary_values(gfu_noise)

data = exact_data+noise

init = 1
init_gfu = ngs.GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_sol = init_solution.copy()
init_data = op(init_solution)

gfu_init_data=ngs.GridFunction(op.fes_codomain)
gfu_init_data.vec.FV().NumPy()[:] = init_data
coeff_init_data = ngs.CoefficientFunction(gfu_init_data)
ngs.Draw(coeff_init_data, op.fes_codomain.mesh, 'init_data')

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=SobolevBoundary)

#Discrepancy Principle usually stops very early
landweber = Landweber(setting, data, init_solution, stepsize=0.001)
stoprule = (
        rules.CountIterations(300) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise)*0, tau=1)
)

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, op.fes_domain.mesh, "exact")
ngs.Draw(init, op.fes_domain.mesh, "init")

# Draw reconstructed solution
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