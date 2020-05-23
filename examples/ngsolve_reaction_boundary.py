# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

import regpy.stoprules as rules
from regpy.operators.ngsolve import ReactionBoundary
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import L2, SobolevBoundary
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
bc = "cyc"
geo.AddCircle((0, 0), r=1, bc=bc, maxh=0.2)
mesh = ngs.Mesh(geo.GenerateMesh())

fes_domain = ngs.H1(mesh, order=2)
domain = NgsSpace(fes_domain)

fes_codomain = ngs.H1(mesh, order=2)
codomain = NgsSpace(fes_codomain)

g = ngs.x ** 2 * ngs.y
op = ReactionBoundary(domain, g, bc, codomain=codomain)

exact_solution_coeff = ngs.sin(ngs.y) + 2
gfu_exact_solution = ngs.GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)

fes_noise=ngs.L2(fes_codomain.mesh, order=1)
gfu_noise_order1=ngs.GridFunction(fes_noise)
gfu_noise_order1.vec.FV().NumPy()[:]=0.0005*np.random.randn(fes_noise.ndof)
gfu_noise=ngs.GridFunction(fes_codomain)
gfu_noise.Set(gfu_noise_order1)
noise=op._get_boundary_values(gfu_noise)

data = exact_data+noise

init = 2
init_gfu = ngs.GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_sol = init_solution.copy()
init_data = op(init_solution)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=SobolevBoundary)

landweber = Landweber(setting, data, init_solution, stepsize=0.001)
# irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
        rules.CountIterations(5) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, op.fes_domain.mesh, "exact")
ngs.Draw(init, op.fes_domain.mesh, "init")

# Draw reconstructed solution
gfu_reco = ngs.GridFunction(op.fes_domain)
gfu_reco.vec.FV().NumPy()[:] = reco
coeff_reco = ngs.CoefficientFunction(gfu_reco)

ngs.Draw(coeff_reco, op.fes_domain.mesh, "reco")

# Draw data space
gfu_data = ngs.GridFunction(op.fes_bdr)
gfu_reco_data = ngs.GridFunction(op.fes_bdr)
gfu_init_data = ngs.GridFunction(op.fes_bdr)

gfu_data.vec.FV().NumPy()[:] = data
coeff_data = ngs.CoefficientFunction(gfu_data)

gfu_reco_data.vec.FV().NumPy()[:] = reco_data
coeff_reco_data = ngs.CoefficientFunction(gfu_reco_data)

gfu_init_data.vec.FV().NumPy()[:] = init_data
coeff_init_data = ngs.CoefficientFunction(gfu_init_data)

ngs.Draw(coeff_data, op.fes_codomain.mesh, "data")
ngs.Draw(coeff_reco_data, op.fes_codomain.mesh, "reco_data")
ngs.Draw(coeff_init_data, op.fes_codomain.mesh, "init_data")

def der(x):
    val2 = op(res1 + x * res2)
    val1 = op(res1)
    der = x * op._derivative(res2)
    return setting.Hcodomain.norm((1 / x) * (val2 - val1 - der))


res1 = 0.001 * np.random.randn(op.domain.shape[0])
res2 = 0.001 * np.random.randn(op.domain.shape[0])

der1=der(0.1)
der2=der(0.01)
der3=der(0.001)
der4=der(0.0001)
der5=der(0.00001)

print(der1, der2, der3, der4, der5)

def adj():
    res1 = 0.001 * np.random.randn(op.domain.shape[0])
    #res1=exact_solution
    v = op._eval(res1, differentiate=True)
    toret1 = setting.Hcodomain.inner(op._derivative(res1), v)
    toret2 = setting.Hdomain.inner(res1, op._adjoint(v))
    s = 0.001 * np.random.randn(op.domain.shape[0])
    h = 0.001 * np.random.randn(op.domain.shape[0])

    #Create vector in codomain
    create_data = 0.001 * np.random.randn(op.domain.shape[0])
    q = op._eval(create_data, differentiate=False)

    #Initailize operator with s
    u = op._eval(s, differentiate=True)

    #(F'[s]h, q)
    toret1 = setting.Hcodomain.inner(op._derivative(h), q)

    #(h, F'[s]^* q)
    toret2 = setting.Hdomain.inner(h, op._adjoint(q))
    return [toret1, toret2]

print(adj(), adj(), adj())