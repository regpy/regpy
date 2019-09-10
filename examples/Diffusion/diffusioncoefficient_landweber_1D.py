import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.spaces import UniformGrid, NGSolveDiscretization
from itreg.solvers import Landweber, HilbertSpaceSetting

from ngsolve.meshes import Make1DMesh

#import netgen.gui

import itreg.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

meshsize_domain=100
meshsize_codomain=100

from ngsolve import *

mesh = Make1DMesh(meshsize_domain)
fes_domain = H1(mesh, order=3, dirichlet="left|right")
domain = NGSolveDiscretization(fes_domain)

mesh = Make1DMesh(meshsize_codomain)
fes_codomain = H1(mesh, order=3, dirichlet="left|right")
codomain = NGSolveDiscretization(fes_codomain)

rhs=0
op = Coefficient(domain, rhs, codomain=codomain, bc_left=1, bc_right=1.1, diffusion=True, reaction=False)

exact_solution_coeff = x
gfu_exact_solution=GridFunction(op.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data=exact_data

#Draw(gfu_exact_solution, name="exact")

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init= 1
init_gfu=GridFunction(op.fes)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_data=op(init_solution)

from itreg.spaces import NGSolveSpace, L2
setting = HilbertSpaceSetting(op=op, domain=L2, codomain=L2)


landweber = Landweber(setting, data, init_solution, stepsize=1)
#irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
    rules.CountIterations(200) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

reco, reco_data = landweber.run(stoprule)

#print(reco_data)

plt.plot(reco, label='reco')
plt.plot(exact_solution, label='exact')
plt.plot(init_solution, label='init')
plt.legend()
plt.show()

N=301

gfu=GridFunction(op.fes)
gfu2=GridFunction(op.fes)
gfu3=GridFunction(op.fes)
for i in range(N):
    gfu.vec[i]=reco[i]
    gfu2.vec[i]=exact_solution[i]
    gfu3.vec[i]=init_solution[i]

Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
Symfunc3=CoefficientFunction(gfu3)
func=np.zeros(N)
func2=np.zeros(N)
func3=np.zeros(N)
for i in range(0, N):
    mip=op.fes.mesh(i/N)
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
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
func=np.zeros(N)
func2=np.zeros(N)
func3=np.zeros(N)
for i in range(0, N):
    mip=op.fes.mesh(i/N)
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.legend()
plt.show()
