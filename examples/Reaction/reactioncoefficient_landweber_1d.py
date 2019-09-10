import setpath

from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.spaces import L2, NGSolveDiscretization
from itreg.solvers import Landweber, HilbertSpaceSetting

from ngsolve.meshes import Make1DMesh

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
fes_domain = L2(mesh, order=2, dirichlet="left|right")
domain = NGSolveDiscretization(fes_domain)

mesh = Make1DMesh(meshsize_codomain)
fes_codomain = H1(mesh, order=2, dirichlet="left|right")
codomain = NGSolveDiscretization(fes_codomain)

rhs=10*x**2
op = Coefficient(domain, codomain=codomain, rhs=rhs, bc_left=1, bc_right=1.1, diffusion=False, reaction=True)

#exact_solution = np.linspace(1, 2, 201)
exact_solution_coeff = 1+x
gfu_exact_solution=GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution=gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
noise = 0.01 * op.codomain.randn()
data=exact_data+noise

gfu=GridFunction(op.fes_codomain)
for i in range(201):
    gfu.vec[i]=data[i]

Symfunc=CoefficientFunction(gfu)
func=np.zeros(201)
for i in range(0, 201):
    mip=op.fes_codomain.mesh(i/200)
    func[i]=Symfunc(mip)

plt.plot(func)
plt.show()







_, deriv = op.linearize(exact_solution)
adj=deriv.adjoint(np.linspace(1, 2, 201))

#init=np.concatenate((np.linspace(1, 2, 101), np.ones(100)))
init=1+x**3
init_gfu=GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution=init_gfu.vec.FV().NumPy().copy()
init_plot=init_solution.copy()
init_data=op(init_solution)

from itreg.spaces import L2, NGSolveSpace
setting = HilbertSpaceSetting(op=op, domain=NGSolveSpace, codomain=NGSolveSpace)

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

N=op.fes_domain.ndof

gfu=GridFunction(op.fes_domain)
gfu2=GridFunction(op.fes_domain)
gfu3=GridFunction(op.fes_domain)
for i in range(N):
    gfu.vec[i]=reco[i]
    gfu2.vec[i]=exact_solution[i]
    gfu3.vec[i]=init_plot[i]

Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
Symfunc3=CoefficientFunction(gfu3)
func=np.zeros(N)
func2=np.zeros(N)
func3=np.zeros(N)
for i in range(0, N):
    mip=op.fes_domain.mesh(i/N)
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
#plt.plot(func3, label='init')
plt.legend()
plt.show()






gfu=GridFunction(op.fes_codomain)
gfu2=GridFunction(op.fes_codomain)
gfu3=GridFunction(op.fes_codomain)
gfu4=GridFunction(op.fes_codomain)
for i in range(201):
    gfu.vec[i]=reco_data[i]
    gfu2.vec[i]=exact_data[i]
    gfu3.vec[i]=init_data[i]
    gfu4.vec[i]=data[i]

Symfunc=CoefficientFunction(gfu)
Symfunc2=CoefficientFunction(gfu2)
Symfunc3=CoefficientFunction(gfu3)
Symfunc4=CoefficientFunction(gfu4)
func=np.zeros(201)
func2=np.zeros(201)
func3=np.zeros(201)
func4=np.zeros(201)
for i in range(0, 201):
    mip=op.fes_codomain.mesh(i/200)
    func[i]=Symfunc(mip)
    func2[i]=Symfunc2(mip)
    func3[i]=Symfunc3(mip)
    func4[i]=Symfunc4(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.plot(func4, label='data')
plt.legend()
plt.show()
