import logging

import numpy as np
from numpy.linalg import norm
from math import isfinite
from scipy.io import savemat
from scipy.optimize import lsq_linear
from copy import deepcopy
import regpy.stoprules as rules
from regpy.vecsps import UniformGridFcts, DirectSum
from regpy.vecsps.tensor_bases import legendre_basis
from regpy.hilbert import L2, HmDomain
from regpy.operators import CoordinateProjection, Zero, InnerShift, OuterShift
from regpy.operators import DirectSum as opDirectSum
from operators import get_op_g_to_data
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.solvers.nonlinear.newton import NewtonCG
from plotting import plot_exact_solution_data,plot_reco, plot_stats, init_plot_stats
from PINEM_setup import setup_simulated_g
from extensions import harmonic_extension
import matplotlib.pyplot as plt

################################ set parameters 

# intermediate results will be written to file names starting with output_filename
output_filename = r'./test'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)-20s :: %(message)s",
    handlers=[
        logging.FileHandler(output_filename+'.log',mode='w'),
        logging.StreamHandler()
    ]
)


# Iabs(g) is assumed to be known outside of the nanotip.
# this is imposed as contraint. 

# If using_polynomial_basis_for_phase==True, the phase will be parameterized by 
# a tensor-product polynomial basis of degrees pol_degrees in x and y directions. 
# Starting with small polynomial degrees helps to avoid getting trapped in local minimima.
using_polynomial_basis_for_phase = False #not working for True
pol_degrees = (20,7)
# total number of counts for gain and loss data
total_nr_counts = 2e9
# turn off/on all plots
do_plottings = True
# use log(|g|) instead of |g| for darkness in phase plots of g and g_rec. Makes phase visible everywhere
plot_log_g = True
# number of modes used for evaluations of forward operator and generation of simulated data
N_data = 30
# values of N used for evaluation of the derivative of the forward operator
# This value should gradually be increased to save computation time. 
N_deriv = [4,8,16,30]
# solver type: If True, NewtonCG is used, otherwise IrgnmCG
use_NewtonCG = True



sobolev_index_phase = 2; sobolev_index_ampl = 2
IRGNM_regpar = 1e-15
IRGNM_regpar_step = 2/3
IRGNM_cgstop = 1000
NewtonCG_rho = 0.95
NewtonCG_cgmaxit = 50
# If the norm of the residual (=data-predicted data) decreases by less than minimal_residual_reduction, 
# then N is increased to the next value in N_deriv
minimal_residual_reduction = 0.98 # NewtonCG_rho**0.5
# Maximum number of Newton iterations for each value of N
max_Newton_its = 20

op, grid, exact_solution, g_map, mask_a, mask_p, opdata \
    = setup_simulated_g(N=N_data,parallel=True)


############################## routines for reconstruction error evaluation  
def reconstruction_error(_exact, _reconstruction):
    def fnorm(arr):
        return norm(arr[:])
    log_ex_amp, ex_phase = op.domain.split(_exact)
    reco_amp, reco_phase = op.domain.split(_reconstruction)
    ex_amp = np.exp(log_ex_amp)
    reco_error1 = fnorm(reco_amp-ex_amp)/fnorm(ex_amp)
    reco_error2 = fnorm(np.exp(1j*reco_phase)-np.exp(1j*ex_phase))/np.sqrt(np.prod(ex_phase.shape))
    reco_error3 = fnorm(reco_amp*np.exp(1j*reco_phase) -ex_amp*np.exp(1j*ex_phase))/fnorm(ex_amp)
    return reco_error1, reco_error2, reco_error3

def plot_write_save(reco,reco_data,fig1,fig2,axs3, newton_step,
    do_plottings=do_plottings, stats =None,output_filename = None,N=None
    ):

    ereco = extension(reco)

    # fix unidentified constant global phase 
    _, ex_phase = op.domain.split(exact_solution)
    reco_amp, reco_phase = op.domain.split(ereco)
    reco_amp = np.exp(reco_amp)
    phase_correction = np.median(ex_phase[~mask_a])-np.median(reco_phase[~mask_a]) 
    reco_phase += phase_correction
    ereco = op.domain.join(reco_amp,reco_phase)
        
    reco_error1, reco_error2, reco_error3 = reconstruction_error(exact_solution, ereco)
    stats['Newton step'].append(newton_step)
    stats['ampl_err'].append(reco_error1)
    stats['phase_err'].append(reco_error2)
    stats['complex_err'].append(reco_error3)
    stats['residuals'].append(setting.h_codomain.norm(reco_data-data))
    stats['N'].append(N)
    if hasattr(solver, "nr_inner_its") and callable(solver.nr_inner_its):
        stats['nr_inner_steps'].append(solver.nr_inner_its())

    if output_filename:
        savemat(output_filename+'{}.mat'.format(newton_step),
        {'reco_amp':reco_amp, 'reco_phase': reco_phase, **stats}
        )
    if newton_step>0:
        residual_reduction = stats['residuals'][-1]/stats['residuals'][-2]
        logging.info('it.{}, N={}, modulus: {:1.4f}, phase: {:1.4f}, norm: {:1.4f}, resid.reduct: {:1.4f}'.format(
            newton_step, N,reco_error1, reco_error2, reco_error3,residual_reduction))
    else:
        residual_reduction = 1
        logging.info('it.{}, N={}, modulus: {:1.4f}, phase: {:1.4f}, norm: {:1.4f}'.format(
            newton_step, N,reco_error1, reco_error2, reco_error3))

    reco_data_comp = flat_codomain.split(reco_data)
    ex_data_comp = flat_codomain.split(data)

    if do_plottings:
        plot_reco(fig1,fig2,reco_amp,reco_phase,reco_data_comp,g_map,ex_data_comp,newton_step,mask_a = mask_a)
        plot_stats(axs3,stats,plot_inner_its = hasattr(solver, "nr_inner_its") and callable(solver.nr_inner_its))

    return residual_reduction

################################################   initialize forward operator
print(f"using_polynomial_basis_for_phase:{using_polynomial_basis_for_phase}")
print(f"isfinite(total_nr_counts):{isfinite(total_nr_counts)}")
print(f"output_filename:{output_filename}")
print(f"do_plottings:{do_plottings}")


if mask_a.any():
    prior_ampl = harmonic_extension(~mask_a,np.log(np.abs(g_map)),damping =0)
    ampl_proj = CoordinateProjection(grid, mask_a)
    ampl_domain = HmDomain(grid.real_space(),mask_a, index = sobolev_index_ampl)
else: # amplitude is known everywhere
    prior_ampl = np.log(np.abs(g_map))
    ampl_proj = Zero(grid)
    ampl_domain = L2(grid) # irrelevant, only needed formally

ampl_projection = InnerShift(ampl_proj,prior_ampl)
ampl_extension = OuterShift(ampl_proj.adjoint,prior_ampl)

if using_polynomial_basis_for_phase:
    coeff_grid = UniformGridFcts(np.arange(pol_degrees[0]),np.arange(pol_degrees[1]))
    phase_domain = L2(coeff_grid)         
    phase_extension =  legendre_basis(coeff_grid,grid)
    phase_projection = phase_extension.adjoint
else:
    # outer boundary values of phase must also be fixed for use of Sobolev norm
    #prior_phase = harmonic_extension(~mask_p,np.unwrap(np.angle(g_map.T)).T,damping =0)
    prior_phase = np.unwrap(np.angle(g_map.T)).T
    phase_proj = CoordinateProjection(grid, mask_p)
    phase_projection = InnerShift(phase_proj,prior_phase)
    phase_extension = OuterShift(phase_proj.adjoint,prior_phase)
    weight = (0.02+np.exp(prior_ampl)/np.exp(np.max(prior_ampl)))
    phase_domain = HmDomain(grid.real_space(),mask_p, index = sobolev_index_phase, weight = weight)

h_domain = ampl_domain + phase_domain
projection =  opDirectSum(ampl_projection, phase_projection)
extension = opDirectSum(ampl_extension,phase_extension)

# as extension is also needed for plotting, a copy is required to avoid errors 
# on the use of revoked copies 
extension2 = deepcopy(extension)
op_ext = op * extension2

############compute synthetic data generated by simulation, i.e. application of the forward operator to g and adding noise.
flat_codomain = DirectSum(*op.codomain.summands, flatten=True)
exact_data = op(exact_solution)
if isfinite(total_nr_counts):
    scal = np.sum(exact_data)/total_nr_counts
    data = scal*np.random.poisson(exact_data/scal)
else:
    data = exact_data
    scal =1
data_comp = op.codomain.split(data)
savemat(output_filename+'_data.mat',{'data':data})


################################## define Hilbert space setting

# define codomain Gram matrix based on observed data to approximate log-likelihood
h_codomain = L2(grid, weights=1/(scal**2+scal*data_comp[0])) 
for j in range(1, len(data_comp)):
    h_codomain = h_codomain + L2(grid, weights=1/(scal**2 + scal*data_comp[j]))


##################### define initial guess
# If None, the initial guess is either zero (for synthetic g) or constructed from initial 
# partial knowledge of |g| and phase(g)
angle = np.deg2rad(10)
X, Y = np.meshgrid(np.linspace(0, 1, np.size(mask_a, 1)), np.linspace(0, 1, np.size(mask_a, 0)))
init_phase = -1 * (np.sin(angle)*X + np.cos(angle)*Y) * 2*np.pi * 3 + 2.5
if using_polynomial_basis_for_phase:
    T = phase_extension.as_linear_operator() # Legendre basis
    init_phase_coeff_flat = lsq_linear(T,grid.flatten(init_phase)).x
    init_phase_coeff = phase_extension.domain.fromflat(init_phase_coeff_flat)
    init_vec_proj = op_ext.domain.join(ampl_projection(prior_ampl),init_phase_coeff)
else:
    init_vec = op.domain.join(prior_ampl,init_phase)
    init_vec_proj = projection(init_vec)

############################### initialize stopping rule

sqrtdata = np.sqrt(scal*data) 

discrepancy_rule = rules.Discrepancy(
        h_codomain.norm,
        data,
        noiselevel= h_codomain.norm(sqrtdata),
        tau=1
    )
stoprule = (discrepancy_rule + rules.CountIterations(max_iterations=max_Newton_its,while_type=True))
    

########################################## perform inversion

setting = HilbertSpaceSetting(op=op_ext, h_domain=h_domain, h_codomain=h_codomain)
if N_deriv:
    N_current = N_deriv[0]
    op_simple = get_op_g_to_data(*opdata, N=N_current) * deepcopy(extension)
else:
    N_current = N_data
    op_simple = None
if use_NewtonCG:
    solver = NewtonCG(
        setting, data, init=init_vec_proj,
        cgmaxit=NewtonCG_cgmaxit, rho=NewtonCG_rho,
        simplified_op = op_simple
        )
else:
    solver = IrgnmCG(
        setting, data, init=init_vec_proj,
        regpar=IRGNM_regpar, regpar_step=IRGNM_regpar_step,
        cgstop=IRGNM_cgstop,
        inner_it_logging_level=logging.INFO,
        simplified_op = op_simple
        )

fig1,fig2 = plot_exact_solution_data(g_map,data_comp,plot_log_g = plot_log_g)       
fig3, axs3 = init_plot_stats()

stats = {'ampl_err': [], 'phase_err': [], 'complex_err': [], 'residuals': [], 'nr_inner_steps': [], \
    'N': [], 'Newton step' : []}
newton_step=0
plot_write_save(solver.x,solver.y,fig1,fig2,axs3,newton_step,
    do_plottings=do_plottings,stats=stats,output_filename= output_filename,N=N_current)
    
for newton_step, [reco, reco_data] in enumerate(solver.while_(stoprule),1):
    residual_reduction = plot_write_save(solver.x,solver.y,fig1,fig2,axs3,newton_step,
        do_plottings=do_plottings,stats=stats,output_filename= output_filename,N=N_current)
    if residual_reduction > minimal_residual_reduction:
        break

for N_current in N_deriv[1:]:
    residual_last_N= stats['residuals'][-1]
    op_simple = get_op_g_to_data(*opdata, N=N_current) * deepcopy(extension)
    if use_NewtonCG:
        solver = NewtonCG(
            setting, data, init=stoprule.x,
            cgmaxit=NewtonCG_cgmaxit, rho=NewtonCG_rho,
            simplified_op = op_simple
        )
    else:
        solver = IrgnmCG(
            setting, data, init=stoprule.x,
            regpar=IRGNM_regpar*IRGNM_regpar_step**(newton_step-1), 
            regpar_step=IRGNM_regpar_step,
            cgstop=IRGNM_cgstop,
            inner_it_logging_level=logging.INFO,
            simplified_op = op_simple
        )

    stoprule = (discrepancy_rule + rules.CountIterations(max_iterations=max_Newton_its,while_type=True))
    plot_write_save(solver.x,solver.y,fig1,fig2,axs3,newton_step,
        do_plottings=True,stats=stats,output_filename= output_filename,N=N_current)

    for newton_step, [reco, reco_data] in enumerate(solver.while_(stoprule),newton_step+1):
        residual_reduction = plot_write_save(solver.x,solver.y,fig1,fig2,axs3,newton_step,
            do_plottings=True,stats=stats,output_filename= output_filename,N=N_current)
        if residual_reduction > minimal_residual_reduction and stats['residuals'][-1] < residual_last_N*NewtonCG_rho:
            break
if do_plottings:
    plt.show(block=True)

