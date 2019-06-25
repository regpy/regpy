import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.spaces import UniformGrid
from itreg.solvers import Landweber, HilbertSpaceSetting

from ngsolve.meshes import Make1DMesh

import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 1, 201)

grid = UniformGrid(xs)

#domain=L2(grid)
meshsize=100

from ngsolve import *

mesh = Make1DMesh(meshsize)
fes = H1(mesh, order=2, dirichlet="left|right")

rhs=10*sin(x)
op = Coefficient(grid, fes, rhs, bc_left=1, bc_right=1.1, diffusion=True, reaction=False)

#exact_solution = np.linspace(1, 2, 201)
exact_solution_coeff = cos(x)
gfu_exact_solution=GridFunction(op.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

gfu=GridFunction(op.fes)
for i in range(201):
    gfu.vec[i]=data[i]
    
Symfunc=CoefficientFunction(gfu)
func=np.zeros(201)
for i in range(0, 201):
    mip=op.mesh(i/200)
    func[i]=Symfunc(mip)
    
plt.plot(func)
plt.show()







_, deriv = op.linearize(exact_solution)
adj=deriv.adjoint(np.linspace(1, 2, 201))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=cos(0.1*x)
init_gfu=GridFunction(op.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

from itreg.spaces import L2
setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=1)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(1000) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

plt.plot(reco, label='reco')
plt.plot(exact_solution, label='exact')
#plt.plot(init_solution, label='init')
plt.legend()
plt.show()

gfu=GridFunction(op.fes)
gfu2=GridFunction(op.fes)
gfu3=GridFunction(op.fes)
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
    mip=op.mesh(i/200)
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)
    
plt.plot(func, label='reco')
plt.plot(func2, label='exact')
#plt.plot(func3, label='init')
plt.legend()
plt.show()






gfu=GridFunction(op.fes)
gfu2=GridFunction(op.fes)
gfu3=GridFunction(op.fes)
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
    mip=op.mesh(i/200)
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)
    
plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.legend()
plt.show()





