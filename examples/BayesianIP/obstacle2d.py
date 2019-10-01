# -*- coding: utf-8 -*-


import setpath

import itreg

from itreg.operators.obstacle2d import PotentialOp




from itreg.operators.obstacle2d import plots
from itreg.operators.obstacle2d.PotentialOp import create_synthetic_data
from itreg.operators.obstacle2d.PotentialOp import create_impulsive_noise


#import itreg

from itreg.spaces import L2, HilbertPullBack, UniformGrid
from itreg.spaces import H1, HilbertPullBack, UniformGrid
from itreg.solvers import Landweber, HilbertSpaceSetting
#from itreg.util import test_adjoint
import itreg.stoprules as rules


from itreg.BIP.mcmc import Settings
from itreg.BIP.prior_distribution.prior_distribution import gaussian as gaussian_prior
from itreg.BIP.likelihood_distribution.likelihood_distribution import gaussian as gaussian_likelihood

from itreg.BIP.MonteCarlo_basics import fixed_stepsize
from itreg.BIP.MonteCarlo_basics import adaptive_stepsize
from itreg.BIP.MonteCarlo_basics import statemanager
from itreg.BIP.MonteCarlo_basics import RandomWalk
#from itreg.BIP.MonteCarlo_basics import AdaptiveRandomWalk
from itreg.BIP.MonteCarlo_basics import HamiltonianMonteCarlo
from itreg.BIP.MonteCarlo_basics import GaussianApproximation

from itreg.BIP.prior_distribution.prior_distribution import l1 as l1_prior
from itreg.BIP.prior_distribution.prior_distribution import tikhonov
from itreg.BIP.likelihood_distribution.likelihood_distribution import l1 as l1_likelihood
from itreg.BIP.likelihood_distribution.likelihood_distribution import unity
from itreg.BIP import HMCState
from itreg.BIP import State

import numpy as np
import logging
import matplotlib.pyplot as plt
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

xs = np.linspace(0, 2 * np.pi, 64)
spacing = xs[1] - xs[0]
ys=np.linspace(0, 63, 64)

grid=UniformGrid(xs)
grid_codomain=UniformGrid(ys)

op=PotentialOp(grid, codomain=grid_codomain)

#op = PotentialOp(L2(grid))

#exact_solution = np.ones(200)
#exact_data = op(exact_solution)
#noise = 0.03 * op.domain.rand(np.random.randn)
#data = exact_data
#noise = 0.03 * op.codomain.rand(np.random.randn)
#data = exact_data + noise
#noiselevel = op.codomain.norm(noise)

init = 1*op.domain.ones()

#_, deriv = op.linearize(init)
#test_adjoint(deriv)

setting=HilbertSpaceSetting(op=op, domain=partial(H1, index=2), codomain=L2)
#setting=HilbertSpaceSetting(op=op, domain=L2, codomain=L2)
exact_data=create_synthetic_data(setting, noiselevel=0)
data=exact_data
#exact_solution=op.obstacle.xdag_pts_plot
#exact_solution=op.obstacle.bd_ex.z
#apple=create_synthetic_data(setting, 0)

solver = Landweber(setting, data, init)
stopping_rule = (
    rules.CountIterations(1e4) +
    rules.Discrepancy(setting.codomain.norm, data, noiselevel=0, tau=1.1))

n_iter   = 1e5
stepsize = [1e-2, 1e-1, 5e-1, 7e-1, 1e0, 1.2, 1.5, 2.5, 10, 20][5]
Temperature=1e-10
reg_parameter=0



#prior=gaussian_prior(1/reg_parameter*np.eye(200), setting, np.zeros(200))
#likelihood=gaussian_likelihood(setting, np.eye(64), exact_data)
prior=tikhonov(setting, reg_parameter, exact_data, )
likelihood=unity(setting)


#sampler=['RandomWalk', 'AdaptiveRandomWalk', 'HamiltonianMonteCarlo', 'GaussianApproximation'][0]


stepsize_rule=partial(adaptive_stepsize, stepsize_factor=1.05)
#stepsize_rule=fixed_stepsize

bip=Settings(setting, data, prior, likelihood, solver, stopping_rule, Temperature,
              n_iter=n_iter, stepsize_rule=stepsize_rule)

statemanager=statemanager(bip.initial_state)
#sampler=[RandomWalk(bip, stepsize=stepsize), AdaptiveRandomWalk(bip, stepsize=stepsize), \
#         HamiltonianMonteCarlo(bip, stepsize=stepsize), GaussianApproximation(bip)][0]
sampler=RandomWalk(bip, statemanager, stepsize_rule=stepsize_rule)
#sampler=HamiltonianMonteCarlo(bip, statemanager, stepsize=1, stepsize_rule=stepsize_rule)
#sampler=GaussianApproximation(bip)

bip.run(sampler, statemanager)

from itreg.BIP.plot_functions import plot_lastiter
from itreg.BIP.plot_functions import plot_mean
from itreg.BIP.plot_functions import plot_verlauf
from itreg.BIP.plot_functions import plot_iter

#plot_lastiter(bip, exact_solution, exact_data, data)
#plot_mean(statemanager, exact_solution, n_iter=15000)
plot_verlauf(statemanager, pdf=bip, exact_solution=bip.setting.op.obstacle.bd_ex.q[0, :], plot_real=True)
plot_iter(bip, statemanager, 10)

a = np.array([s.positions for s in statemanager.states[-300000:]])
v = a.std(axis=0)
m = a.mean(axis=0)

reco=m
upper=m+v
lower=m-v
reco_data=op(m)

plt.figure()
#plt.plot(statemanager.states[-1].positions, label='statemanager')
plt.plot(m, label='mean')
plt.plot(m+v, label='mean+variance')
plt.plot(m-v, label='mean-variance')
plt.plot(op.obstacle.bd_ex.q[0, :], label='exact')
plt.plot(bip.first_state, label='Landweber')
plt.legend()
plt.show()

plt.figure()
plt.plot(op(statemanager.states[-1].positions),label='statemanager')
plt.plot(op(op.obstacle.bd_ex.q[0, :]), label='exact')
plt.plot(op(bip.first_state), label='landweber')
plt.legend()
plt.show()


plt.figure()
plt.plot(reco_data, label='mean over last 9e4 states')
plt.plot(op(upper), label='mean+variance')
plt.plot(op(lower), label='mean -variance')
plt.plot(exact_data, label='data')
plt.plot(op(op.obstacle.bd_ex.q[0, :]), label='exact data')
plt.title('2e4 states')
plt.legend()
plt.show()

#plt.plot(np.linspace(0, 1, 2), op.obstacle.bd_ex.q[0, :])
#plt.plot(np.linspace(0, 1, 128), m)

from matplotlib.patches import Polygon
"""n=128
fig, axs = plt.subplots(1, 2,sharey=True,figsize=(9, 6))
axs[0].set_title('Domain')
axs[1].set_title('Heat source')
axs[1].plot(exact_data)
axs[1].plot(data)
axs[1].plot(reco_data)
ymin=0.7*min(reco_data.min(), data.min(), exact_data.min())
ymax=1.3*max(reco_data.max(), data.max(), exact_data.max())
axs[1].set_ylim((ymin, ymax))
bd=op.bd
pts=bd.coeff2Curve(m+v, n)
pts_2=bd.coeff2Curve(exact_solution, n)
pts_3=bd.coeff2Curve(m-v, n)
poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), animated=True, fill=False)
poly_2=Polygon(np.column_stack([pts_2[0, :], pts_2[1, :]]), animated=True, fill=False)
poly_3=Polygon(np.column_stack([pts_3[0, :], pts_3[1, :]]), animated=True, fill=False)
axs[0].add_patch(poly)
axs[0].add_patch(poly_2)
axs[0].add_patch(poly_3)
xmin=1.5*min(pts[0, :].min(), pts_2[0, :].min(), pts[0, :].min())
xmax=1.5*max(pts[0, :].max(), pts_2[0, :].max(), pts[0, :].min())
ymin=1.5*min(pts[1, :].min(), pts_2[1, :].min(), pts[0, :].min())
ymax=1.5*max(pts[1, :].max(), pts_2[1, :].max(), pts[0, :].min())
axs[0].set_xlim((xmin, xmax))
axs[0].set_ylim((ymin, ymax))
plt.show()"""

fig, ax=plt.subplots(1, figsize=(9,6))
ax.set_title('Domain')
bd=op.bd
n=200
pts=bd.coeff2Curve(m+v, n)
#pts_2=bd.coeff2Curve(exact_solution, n)
pts_2=op.obstacle.xdag_pts_plot
pts_3=bd.coeff2Curve(m-v, n)
poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), label='mean +2*variance',  animated=True, fill=False)
poly_2=Polygon(np.column_stack([pts_2[0, :], pts_2[1, :]]), label='exact_solution', animated=True, fill=False)
poly_3=Polygon(np.column_stack([pts_3[0, :], pts_3[1, :]]), label='mean-2*variance', animated=True, fill=False)
poly.set_color([1, 0, 0])
poly_2.set_color([0, 1, 0])
poly_3.set_color([0, 0, 1])
ax.add_patch(poly)
ax.add_patch(poly_2)
ax.add_patch(poly_3)
xmin=1.5*min(pts[0, :].min(), pts_2[0, :].min(), pts[0, :].min())
xmax=1.5*max(pts[0, :].max(), pts_2[0, :].max(), pts[0, :].min())
ymin=1.5*min(pts[1, :].min(), pts_2[1, :].min(), pts[0, :].min())
ymax=1.5*max(pts[1, :].max(), pts_2[1, :].max(), pts[0, :].min())
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.legend()
plt.show()

#plotting=plots(op, m, reco_data, data, exact_data, exact_solution)
#plotting.plotting()


data=create_synthetic_data(setting, noiselevel=0)
n=128
fig, ax=plt.subplots(1, figsize=(9,6))
ax.set_title('Domain')
#bd_ex=op.obstacle.bd_ex
#pts=bd_ex.coeff2Curve(apple, n)
bd=op.obstacle.bd
#pts=bd.coeff2Curve(op.obstacle.xdag_pts_plot, n)
#pts=op.obstacle.xdag_pts_plot
pts=bd.coeff2Curve(op.obstacle.bd_ex.q[0,:],  n)
poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), label='peanut',  animated=True, fill=False)
poly.set_color([1, 0, 0])
ax.add_patch(poly)
xmin=1.5*pts[0, :].min()
xmax=1.5*pts[0, :].max()
ymin=1.5*pts[1, :].min()
ymax=1.5*pts[1, :].max()
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.legend()
plt.show()
