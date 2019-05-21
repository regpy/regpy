import setpath

from itreg.operators.NGSolveProblems.Coefficient_1D import Coefficient
from itreg.spaces import L2
from itreg.solvers import Landweber
from itreg.solvers import IRGNM_CG
from itreg.util import test_adjoint
import itreg.stoprules as rules
from itreg.grids import User_Defined
from itreg.spaces import H1_NGSolve
from ngsolve.meshes import Make1DMesh
from ngsolve import CoefficientFunction, GridFunction

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 1, 201)

grid=User_Defined(xs, xs.shape)

domain=L2(grid)
meshsize=100

from ngsolve import *
rhs=10*x**2
op = Coefficient(domain, meshsize, rhs=rhs, bc_left=1, bc_right=1.1, diffusion=False, reaction=True)

#exact_solution = np.linspace(1, 2, 201)
exact_solution_coeff = 1+x
gfu_exact_solution=GridFunction(op.params.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

gfu=GridFunction(op.params.fes)
for i in range(201):
    gfu.vec[i]=data[i]
    
Symfunc=CoefficientFunction(gfu)
func=np.zeros(201)
for i in range(0, 201):
    mip=op.params.mesh(op.params.domain.coords[i])
    func[i]=Symfunc(mip)
    
plt.plot(func)
plt.show()







_, deriv = op.linearize(exact_solution)
adj=deriv.adjoint(np.linspace(1, 2, 201))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=1+x**3
init_gfu=GridFunction(op.params.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

landweber = Landweber(op, data, init_solution, stepsize=1)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(op.range.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.plot(reco, label='reco')
plt.plot(exact_solution, label='exact')
#plt.plot(init_solution, label='init')
plt.legend()
plt.show()

gfu=GridFunction(op.params.fes)
gfu2=GridFunction(op.params.fes)
gfu3=GridFunction(op.params.fes)
for i in range(201):
    gfu.vec[i]=reco[i]
    gfu2.vec[i]=exact_solution[i]
    gfu3.vec[i]=init_solution[i]
    
Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
Symfunc3=CoefficientFunction(gfu3)
func=np.zeros(201)
func2=np.zeros(201)
func3=np.zeros(201)
for i in range(0, 201):
    mip=op.params.mesh(op.params.domain.coords[i])
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)
    
plt.plot(func, label='reco')
plt.plot(func2, label='exact')
#plt.plot(func3, label='init')
plt.legend()
plt.show()






gfu=GridFunction(op.params.fes)
gfu2=GridFunction(op.params.fes)
gfu3=GridFunction(op.params.fes)
for i in range(201):
    gfu.vec[i]=reco_data[i]
    gfu2.vec[i]=exact_data[i]
    gfu3.vec[i]=init_data[i]
    
Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
Symfunc3=CoefficientFunction(gfu3)
func=np.zeros(201)
func2=np.zeros(201)
func3=np.zeros(201)
for i in range(0, 201):
    mip=op.params.mesh(op.params.domain.coords[i])
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)
    
plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.legend()
plt.show()





