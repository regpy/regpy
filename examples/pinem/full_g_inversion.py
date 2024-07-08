import logging

import numpy as np
from numpy.linalg import norm
from math import isfinite
from scipy.io import savemat
from copy import deepcopy
import regpy.stoprules as rules
from regpy.vecsps import DirectSum
from regpy.hilbert import L2, HmDomain
from regpy.operators import CoordinateProjection, Zero, InnerShift, OuterShift
from regpy.operators import DirectSum as opDirectSum
from operators import get_op_g_to_data
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
from regpy.solvers.nonlinear.newton import NewtonCG
from plotting import plot_exact_solution_data,plot_reco, plot_stats, init_plot_stats
from regpy.operators.parallel_operators import ParallelExecutionManager
from setup import setup_simulated_g
from extensions import harmonic_extension
import matplotlib.pyplot as plt
import os

################################ set parameters 
# intermediate results will be written to file names starting with output_prefix
output_prefix='example'

# abs(g) is assumed to be known outside of the nanotip.
# this is imposed as contraint. 

# total number of counts for gain and loss data
total_nr_counts = 2e9
# turn off/on all plots
do_plottings = True
# turn off/on saving of results
save_results=False
# use log(|g|) instead of |g| for darkness in phase plots of g and g_rec. Makes phase visible everywhere
plot_log_g = True
# number of modes used for evaluations of forward operator and generation of simulated data
N_data = 30
# values of N used for evaluation of the derivative of the forward operator
# This value should gradually be increased to save computation time. 
N_deriv = [4,8,16,30]
# solver type: If True, NewtonCG is used, otherwise IrgnmCG
use_NewtonCG = True

# Indices of sobolev spaces for regularization
sobolev_index_phase = 2; sobolev_index_ampl = 2
# Parameters for IRGNM
IRGNM_regpar = 1e-15
IRGNM_regpar_step = 2/3
IRGNM_cgstop = 1000
# Parameters for NewtonCG
NewtonCG_rho = 0.95
NewtonCG_cgmaxit = 50

# If the norm of the residual (=data-predicted data) decreases by less than minimal_residual_reduction, 
# then N is increased to the next value in N_deriv
minimal_residual_reduction = 0.98 # NewtonCG_rho**0.5
# Maximum number of Newton iterations for each value of N
max_Newton_its = 20

##############################ROUTINES FOR ERROR CALCULATION AND UPDATES OF STATS  

def calc_reco_amp_phase(reco,extension,op_domain,mask_a):
    r"""Splits reconstructed data into amplitude and phase

    Parameters
    ----------
        reco : numpy.ndarray
            the reconstruction
        extension : regpy.operators.Operator
            extension operator
        op_domain : regpy.vecsps.VectorSpace
            domain of op
        mask_a : numpy.ndarray
            mask on domain of operator

    Returns 
    ----------
        reco_amp : numpy.ndarray
            amplitude of reco
        reco_phase : numpy.ndarray
            phase of reco
    """
    ereco = extension(reco)
    # fix unidentified constant global phase 
    _, ex_phase = op_domain.split(exact_solution)
    reco_amp, reco_phase = op_domain.split(ereco)
    reco_amp = np.exp(reco_amp)
    phase_correction = np.median(ex_phase[~mask_a])-np.median(reco_phase[~mask_a]) 
    reco_phase += phase_correction
    return reco_amp,reco_phase

def calc_reco_errors(reco_amp,reco_phase,exact_solution,op_domain):
    r"""Calculates errors between reconstruction and exact solution

    Parameters
    ----------
        reco_amp : numpy.ndarray
            amplitude of reco
        reco_phase : numpy.ndarray
            phase of reco
        exact_solution : numpy.ndarray
            exact solution
        op_domain : regpy.vecsps.VectorSpace
            domain of op

    Returns 
    ----------
        reco_error_amp : numpy.ndarray
            error in amplitude
        reco_error_phase : numpy.ndarray
            error in phase
        reco_error_comp : numpy.ndarray
            total error
    """
    def fnorm(arr):
        return norm(arr[:])
    log_ex_amp, ex_phase = op_domain.split(exact_solution)
    ex_amp = np.exp(log_ex_amp)
    reco_error_amp = fnorm(reco_amp-ex_amp)/fnorm(ex_amp)
    reco_error_phase = fnorm(np.exp(1j*reco_phase)-np.exp(1j*ex_phase))/np.sqrt(np.prod(ex_phase.shape))
    reco_error_complex = fnorm(reco_amp*np.exp(1j*reco_phase) -ex_amp*np.exp(1j*ex_phase))/fnorm(ex_amp)
    return reco_error_amp, reco_error_phase, reco_error_complex

def calc_residual(reco_data,data,setting):
    r"""Calculates the residual"""
    return setting.h_codomain.norm(reco_data-data)

def update_stats(reco_errors,residual,newton_step,stats =None,N=None):
    r"""Updates the statistics"""
    stats['Newton step'].append(newton_step)
    stats['ampl_err'].append(reco_errors[0])
    stats['phase_err'].append(reco_errors[1])
    stats['complex_err'].append(reco_errors[2])
    stats['residuals'].append(residual)
    stats['N'].append(N)
    stats['nr_inner_steps'].append(solver.nr_inner_its())
    residual_reduction=1
    if newton_step>0:
        residual_reduction=stats['residuals'][-1]/stats['residuals'][-2]
    stats['residual_reduction'].append(residual_reduction)
    logging.info('it.{}, N={}, modulus: {:1.4f}, phase: {:1.4f}, norm: {:1.4f}, resid.reduct: {:1.4f}'.format(
        newton_step, N,reco_errors[0], reco_errors[1], reco_errors[2],residual_reduction))

if __name__ == '__main__':
    ##############################SETUP 
    with ParallelExecutionManager():
        #generate operator, grid, exact solution etc. from simulated data

        op, grid, exact_solution, g_map, mask_a, mask_p, opdata = setup_simulated_g(N=N_data,parallel=True)

        #set path for logging and output
        current_directory=os.path.dirname(os.path.realpath(__file__))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)-20s :: %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(current_directory,f"{output_prefix}.log"),mode='w'),
                logging.StreamHandler()
            ]
        )
        output_path=None
        if save_results:
            output_path = os.path.join(current_directory,'data','results',output_prefix)
            print(f"Files will be saved to {os.path.join(current_directory,'data','results')}")


        ########################INITIALIZE FORWARD OPERATOR

        if mask_a.any():
            prior_ampl = harmonic_extension(~mask_a,np.log(np.abs(g_map)),damping =0)
            ampl_proj = CoordinateProjection(grid, mask_a)
            ampl_domain = HmDomain(grid.real_space(),mask_a, index = sobolev_index_ampl)
        else: # amplitude is known everywhere
            prior_ampl = np.log(np.abs(g_map))
            ampl_proj = Zero(grid)
            ampl_domain = L2(grid) # only needed formally

        ampl_projection = InnerShift(ampl_proj,prior_ampl)
        ampl_extension = OuterShift(ampl_proj.adjoint,prior_ampl)

        # outer boundary values of phase must also be fixed for use of Sobolev norm
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

        ############COMPUTE SYNTHETIC DATA
        # generated by simulation, i.e. application of the forward operator to g and adding poisson noise.
        flat_codomain = DirectSum(*op.codomain.summands, flatten=True)
        exact_data = op(exact_solution)
        if isfinite(total_nr_counts):
            scal = np.sum(exact_data)/total_nr_counts
            data = scal*np.random.poisson(exact_data/scal)
        else:
            data = exact_data
            scal =1
        data_comp = op.codomain.split(data)
        if output_path:
            savemat(output_path+'_data.mat',{'data':data})


        #############DEFINE REGULARIZATION SETTING
        # define codomain Gram matrix based on observed data to approximate log-likelihood
        h_codomain = L2(grid, weights=1/(scal**2+scal*data_comp[0])) 
        for j in range(1, len(data_comp)):
            h_codomain = h_codomain + L2(grid, weights=1/(scal**2 + scal*data_comp[j]))


        ##################### DEFINE INITIAL GUESS
        # the initial guess is constructed from initial 
        # partial knowledge of |g| and phase(g)
        angle = np.deg2rad(10)
        X, Y = np.meshgrid(np.linspace(0, 1, np.size(mask_a, 1)), np.linspace(0, 1, np.size(mask_a, 0)))
        init_phase = -1 * (np.sin(angle)*X + np.cos(angle)*Y) * 2*np.pi * 3 + 2.5
        init_vec = op.domain.join(prior_ampl,init_phase)
        init_vec_proj = projection(init_vec)

        ############################### INITIALIZE STOPPING RULE

        sqrtdata = np.sqrt(scal*data) 

        discrepancy_rule = rules.Discrepancy(
                h_codomain.norm,
                data,
                noiselevel= h_codomain.norm(sqrtdata),
                tau=1
            )
        stoprule = (discrepancy_rule + rules.CountIterations(max_iterations=max_Newton_its,while_type=True))
            

        ########################################## PERFORM INVERSION

        setting = RegularizationSetting(op=op_ext, penalty=h_domain, data_fid=h_codomain)
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
            'N': [], 'Newton step' : [], 'residual_reduction':[]}
        newton_step=0

        def update_and_plot(newton_step,reco,reco_data):
            reco_amp,reco_phase=calc_reco_amp_phase(reco,extension,op.domain,mask_a)
            reco_errors=calc_reco_errors(reco_amp,reco_phase,exact_solution,op.domain)
            residual=calc_residual(reco_data,data,setting)
            update_stats(reco_errors,residual,newton_step,stats,N_current)
            reco_data_comp = flat_codomain.split(reco_data)
            ex_data_comp = flat_codomain.split(data)
            if(do_plottings):
                plot_reco(fig1,fig2,reco_amp,reco_phase,reco_data_comp,g_map,ex_data_comp,newton_step,mask_a = mask_a)
                plot_stats(axs3,stats,plot_inner_its = hasattr(solver, "nr_inner_its") and callable(solver.nr_inner_its))
            if output_path:
                savemat(output_path+'{}.mat'.format(newton_step),
                {'reco_amp':reco_amp, 'reco_phase': reco_phase, **stats}
                )

        update_and_plot(newton_step,solver.x,solver.y)   
        for newton_step, [reco, reco_data] in enumerate(solver.while_(stoprule),1):
            update_and_plot(newton_step,reco,reco_data) 
            if stats['residual_reduction'][-1] > minimal_residual_reduction:
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
            update_and_plot(newton_step,solver.x,solver.y)
            for newton_step, [reco, reco_data] in enumerate(solver.while_(stoprule),newton_step+1):
                update_and_plot(newton_step,reco,reco_data)
                if stats['residual_reduction'][-1] > minimal_residual_reduction and stats['residuals'][-1] < residual_last_N*NewtonCG_rho:
                    break
        if do_plottings:
            plt.show(block=True)

