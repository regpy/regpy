import setpath

import logging
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

import itreg.stoprules as rules
from itreg.operators.NGSolveProblems.EIT import EIT
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.landweber import Landweber
from itreg.spaces.hilbert import Sobolev, SobolevBoundary
from itreg.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
geo.AddCircle((0, 0), r=1, bc="cyc", maxh=0.2)
mesh = ngs.Mesh(geo.GenerateMesh())

fes_domain = ngs.L2(mesh, order=2)
domain = NgsSpace(fes_domain)

fes_codomain = ngs.H1(mesh, order=2)
codomain = NgsSpace(fes_codomain)

g = 0.1 * (ngs.x - 0.5) * (ngs.y - 0.5)
op = EIT(domain, g, codomain=codomain)
pts = np.array(op.pts)
nr_points = pts.shape[0]

exact_solution_coeff = ngs.sin(ngs.y)
gfu_exact_solution = ngs.GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data = exact_data

init = 0.5 * ngs.y
init_gfu = ngs.GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_sol = init_solution.copy()
init_data = op(init_solution)

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=SobolevBoundary)

landweber = Landweber(setting, data, init_solution, stepsize=0.001)
stoprule = (
        rules.CountIterations(300) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=0, tau=1.1)
)

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


def der(x):
    val2 = op(res1 + x * res2)
    val1 = op(res1)
    der = x * op._derivative(res2)
    return setting.Hcodomain.norm(1 / x * (val2 - val1 - der))


res1 = 0.001 * np.random.randn(804)
res2 = 0.001 * np.random.randn(804)

print(der(0.1))
print(der(0.01))
print(der(0.001))
print(der(0.0001))


def adj():
    res1 = 0.001 * np.random.randn(804)
    res2 = 0.001 * np.random.randn(804)
    v = op(res1)
    toret1 = setting.Hcodomain.inner(op._derivative(res1), v)
    toret2 = 5.08 * setting.Hdomain.inner(res1, op._adjoint(v))
    return [toret1, toret2]


print(adj())
print(adj())
print(adj())
print(adj())
print(adj())
