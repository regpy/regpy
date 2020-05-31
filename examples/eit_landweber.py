# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

import regpy.stoprules as rules
from regpy.operators.ngsolve import EIT, ProjectToBoundary
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import Sobolev, SobolevBoundary
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
bc = "cyc"
geo.AddCircle((0, 0), r=1, bc=bc, maxh=0.05)
mesh = ngs.Mesh(geo.GenerateMesh())

fes_domain = ngs.L2(mesh, order=4)
domain = NgsSpace(fes_domain)

fes_codomain = ngs.H1(mesh, order=4)
codomain = NgsSpace(fes_codomain, bdr=bc)

g = 1#0.1 * (ngs.x - 0.5) * (ngs.y - 0.5)
eit = EIT(domain, g, codomain=codomain, alpha=10**(-2))
proj = ProjectToBoundary(codomain)
op = proj * eit

exact_solution_coeff = 1+0.1*ngs.y#0.1*ngs.sqrt(ngs.y**2+ngs.x**2)
exact_solution = domain.fromcoefficientfunction( exact_solution_coeff )
exact_data = op(exact_solution)

noise = proj( codomain.randn() )

data = exact_data+noise

init = domain.fromcoefficientfunction(1)

setting = HilbertSpaceSetting(op=op, Hdomain=Sobolev, Hcodomain=SobolevBoundary)

#Discrepancy Principle usually stops very early
landweber = Landweber(setting, data, init, stepsize=0.001)
stoprule = (
        rules.CountIterations(3) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise)*0, tau=1)
)

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, fes_domain.mesh, "exact")

# Draw reconstructed solution
domain.draw(reco, "reco")

# Draw data space
codomain.draw(reco_data, "reco_data")
codomain.draw(data, "data")

###############################################################################
test_function_1_coeff = 0.1+0.1*ngs.x
gfu_test_function_1 = ngs.GridFunction(fes_domain)
gfu_test_function_1.Set(test_function_1_coeff)
test_function_1 = gfu_test_function_1.vec.FV().NumPy()

test_function_2_coeff = 1+0.1*ngs.y
gfu_test_function_2 = ngs.GridFunction(fes_domain)
gfu_test_function_2.Set(test_function_2_coeff)
test_function_2 = gfu_test_function_2.vec.FV().NumPy()

q = op(test_function_2)

#Initialize operator with s
u, deriv = op.linearize(exact_solution)

#(F'[s]h, q)
toret1 = setting.Hcodomain.inner(deriv(test_function_1), q)

#(h, F'[s]^* q)
toret2 = setting.Hdomain.inner(test_function_1, deriv.adjoint(q))
print(toret1, toret2)

def der(x):
    val2 = op(test_function_2 + x * test_function_1)
    val1, deriv = op.linearize(test_function_2)
    der = x * deriv(test_function_1)
    return setting.Hcodomain.norm(1 / x * (val2 - val1 - der))

der0=der(1)
der1=der(0.1)
der2=der(0.01)
der3=der(0.001)
der4=der(0.0001)
der5=der(0.00001)

print(der0, der1, der2, der3, der4, der5)
print(toret1, toret2)