import setpath

from itreg.operators.NGSolveProblems.EIT import EIT
from itreg.spaces import NGSolveDiscretization, NGSolveBoundaryDiscretization
from itreg.solvers import Landweber, HilbertSpaceSetting

import itreg.stoprules as rules

import numpy as np
import logging
#import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

from ngsolve import *

#import netgen.gui
#from netgen.geom2d import SplineGeometry
#geo=SplineGeometry()
#geo.AddRectangle((0,0), (2,2), bcs=["b","r","t","l"])
#geo.AddCircle ( (0, 0), r=1, bc="cyc", maxh=0.2)
#ngmesh = geo.GenerateMesh(maxh=0.2)
#mesh=Mesh(ngmesh)
mesh=Mesh('..\..\itreg\meshes_ngsolve\meshes\circle.vol.gz')

fes_domain = L2(mesh, order=2)
domain= NGSolveDiscretization(fes_domain)

fes_codomain = H1(mesh, order=2)
#fes_bdr = H1(mesh, order=1)
#pts=[v.point for v in mesh.vertices]
#ind=[np.linalg.norm(np.array(p))>0.95 for p in pts]
#pts_bdr=np.array(pts)[ind]
#codomain= NGSolveBoundaryDiscretization(fes_codomain, fes_bdr, ind)
codomain=NGSolveDiscretization(fes_codomain)

g=0.1*(x-0.5)*(y-0.5)
op = EIT(domain, g, codomain=codomain)
pts=np.array(op.pts)
nr_points=pts.shape[0]

exact_solution_coeff = sin(y)
gfu_exact_solution=GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

init=0.5*y
init_gfu=GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_sol=init_solution.copy()
init_data=op(init_solution)

#_, deriv=op.linearize(exact_solution)
#adj=deriv.adjoint(exact_data)

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

def der(x):
    val2=op(res1+x*res2)
    val1=op(res1)
    der=x*op._derivative(res2)
    return setting.codomain.norm( 1/x*(val2-val1-der) )

res1=0.001*np.random.randn(804)
res2=0.001*np.random.randn(804)

print(der(0.1))
print(der(0.01))
print(der(0.001))
print(der(0.0001))

def adj():
    res1=0.001*np.random.randn(804)
    res2=0.001*np.random.randn(804)
    v=op(res1)
    toret1=setting.codomain.inner(op._derivative(res1), v)
    toret2=5.08*setting.domain.inner(res1, op._adjoint(v))
    return [toret1, toret2]

print(adj())
print(adj())
print(adj())
print(adj())
print(adj())





