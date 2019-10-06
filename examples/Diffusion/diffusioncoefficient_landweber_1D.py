import logging
import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np
from ngsolve.meshes import Make1DMesh

import itreg.stoprules as rules
from itreg.operators.NGSolveProblems.Coefficient import Coefficient
from itreg.solvers import HilbertSpaceSetting
from itreg.solvers.landweber import Landweber
from itreg.hilbert import L2
from itreg.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

meshsize_domain = 100
meshsize_codomain = 100

mesh = Make1DMesh(meshsize_domain)
fes_domain = ngs.H1(mesh, order=3, dirichlet="left|right")
domain = NgsSpace(fes_domain)

mesh = Make1DMesh(meshsize_codomain)
fes_codomain = ngs.H1(mesh, order=3, dirichlet="left|right")
codomain = NgsSpace(fes_codomain)

rhs = 0
op = Coefficient(domain, rhs, codomain=codomain, bc_left=1, bc_right=1.1, diffusion=True, reaction=False)

exact_solution_coeff = ngs.x
gfu_exact_solution = ngs.GridFunction(op.domain.fes)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
data = exact_data

init = 1
init_gfu = ngs.GridFunction(op.domain.fes)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_data = op(init_solution)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=1)
stoprule = (
        rules.CountIterations(200) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=0, tau=1.1)
)

reco, reco_data = landweber.run(stoprule)

plt.plot(reco, label='reco')
plt.plot(exact_solution, label='exact')
plt.plot(init_solution, label='init')
plt.legend()
plt.show()

N = 301

gfu = ngs.GridFunction(op.fes)
gfu2 = ngs.GridFunction(op.fes)
gfu3 = ngs.GridFunction(op.fes)
for i in range(N):
    gfu.vec[i] = reco[i]
    gfu2.vec[i] = exact_solution[i]
    gfu3.vec[i] = init_solution[i]

Symfunc = ngs.CoefficientFunction(gfu)
Symfunc2 = ngs.CoefficientFunction(gfu2)
Symfunc3 = ngs.CoefficientFunction(gfu3)
func = np.zeros(N)
func2 = np.zeros(N)
func3 = np.zeros(N)
for i in range(0, N):
    mip = op.fes.mesh(i / N)
    func[i] = Symfunc(mip)
    func2[i] = Symfunc2(mip)
    func3[i] = Symfunc3(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.legend()
plt.show()

gfu = ngs.GridFunction(op.fes)
gfu2 = ngs.GridFunction(op.fes)
gfu3 = ngs.GridFunction(op.fes)
for i in range(201):
    gfu.vec[i] = reco_data[i]
    gfu2.vec[i] = exact_data[i]
    gfu3.vec[i] = init_data[i]

Symfunc = ngs.CoefficientFunction(gfu)
Symfunc2 = ngs.CoefficientFunction(gfu2)
Symfunc3 = ngs.CoefficientFunction(gfu3)
func = np.zeros(N)
func2 = np.zeros(N)
func3 = np.zeros(N)
for i in range(0, N):
    mip = op.fes.mesh(i / N)
    func[i] = Symfunc(mip)
    func2[i] = Symfunc2(mip)
    func3[i] = Symfunc3(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.legend()
plt.show()
