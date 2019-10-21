import logging
import matplotlib.pyplot as plt
import numpy as np
import ngsolve as ngs
from ngsolve.meshes import Make1DMesh

import regpy.stoprules as rules
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import L2
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

meshsize_domain = 100
meshsize_codomain = 100

mesh = Make1DMesh(meshsize_domain)
fes_domain = ngs.L2(mesh, order=2, dirichlet="left|right")
domain = NgsSpace(fes_domain)

mesh = Make1DMesh(meshsize_codomain)
fes_codomain = ngs.H1(mesh, order=2, dirichlet="left|right")
codomain = NgsSpace(fes_codomain)

rhs = 10 * ngs.x ** 2
op = Coefficient(domain, codomain=codomain, rhs=rhs, bc_left=1, bc_right=1.1, diffusion=False, reaction=True)

N_domain = op.fes_domain.ndof
N_codomain = op.fes_codomain.ndof

exact_solution_coeff = 1 + ngs.x
gfu_exact_solution = ngs.GridFunction(op.fes_domain)
gfu_exact_solution.Set(exact_solution_coeff)
exact_solution = gfu_exact_solution.vec.FV().NumPy()
exact_data = op(exact_solution)
noise = 0.01 * op.codomain.randn()
data = exact_data + noise

init = 1 + ngs.x ** 3
init_gfu = ngs.GridFunction(op.fes_domain)
init_gfu.Set(init)
init_solution = init_gfu.vec.FV().NumPy().copy()
init_plot = init_solution.copy()
init_data = op(init_solution)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

landweber = Landweber(setting, data, init_solution, stepsize=1)
# irgnm_cg = IRGNM_CG(op, data, init, cgmaxit = 50, alpha0 = 1, alpha_step = 0.9, cgtol = [0.3, 0.3, 1e-6])
stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)

gfu = ngs.GridFunction(op.fes_domain)
gfu2 = ngs.GridFunction(op.fes_domain)
gfu3 = ngs.GridFunction(op.fes_domain)
for i in range(N_domain):
    gfu.vec[i] = reco[i]
    gfu2.vec[i] = exact_solution[i]
    gfu3.vec[i] = init_plot[i]

Symfunc = ngs.CoefficientFunction(gfu)
Symfunc2 = ngs.CoefficientFunction(gfu2)
Symfunc3 = ngs.CoefficientFunction(gfu3)
func = np.zeros(N_domain)
func2 = np.zeros(N_domain)
func3 = np.zeros(N_domain)
for i in range(0, N_domain):
    mip = op.fes_domain.mesh(i / N_domain)
    func[i] = Symfunc(mip)
    func2[i] = Symfunc2(mip)
    func3[i] = Symfunc3(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.legend()
plt.show()

gfu = ngs.GridFunction(op.fes_codomain)
gfu2 = ngs.GridFunction(op.fes_codomain)
gfu3 = ngs.GridFunction(op.fes_codomain)
gfu4 = ngs.GridFunction(op.fes_codomain)
for i in range(N_codomain):
    gfu.vec[i] = reco_data[i]
    gfu2.vec[i] = exact_data[i]
    gfu3.vec[i] = init_data[i]
    gfu4.vec[i] = data[i]

Symfunc = ngs.CoefficientFunction(gfu)
Symfunc2 = ngs.CoefficientFunction(gfu2)
Symfunc3 = ngs.CoefficientFunction(gfu3)
Symfunc4 = ngs.CoefficientFunction(gfu4)
func = np.zeros(N_codomain)
func2 = np.zeros(N_codomain)
func3 = np.zeros(N_codomain)
func4 = np.zeros(N_codomain)
for i in range(0, N_codomain):
    mip = op.fes_codomain.mesh(i / N_codomain)
    func[i] = Symfunc(mip)
    func2[i] = Symfunc2(mip)
    func3[i] = Symfunc3(mip)
    func4[i] = Symfunc4(mip)

plt.plot(func, label='reco')
plt.plot(func2, label='exact')
plt.plot(func3, label='init')
plt.plot(func4, label='data')
plt.legend()
plt.show()
