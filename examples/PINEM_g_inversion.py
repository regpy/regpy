from ast import operator
import logging
from multiprocessing.spawn import get_command_line
from operator import ge

import matplotlib.pyplot as plt
import numpy as np
import regpy.stoprules as rules
from numpy.core.numeric import ones_like
from regpy.discrs import UniformGrid, DirectSum
from regpy.hilbert import L2, Sobolev
import regpy.hilbert as hilbert
from regpy.operators import Operator, SquaredModulus, Vector_of_operators
from regpy.operators.PINEM import PINEM_g_to_data, complex_PINEM_g_to_data
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
from regpy.solvers.newton import NewtonCG
from scipy.io import loadmat
from scipy.misc import ascent
from numpy.linalg import norm
from regpy.util.imshow_fig import imshow_fig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)


class FixAmplitude(Operator):
    # Operator (a,phase) |-> (ampl, phase) for some fixed ampl
    # (This could be replaced by the operator phase |-> (ampl, phase), but this would
    # require more if statements in main.)
    def __init__(self, ampl, domain):
        self.ampl = ampl
        super().__init__(DirectSum(domain, domain), DirectSum(domain, domain))

    def _eval(self, x, differentiate):
        _, phase = self.domain.split(x)
        return self.codomain.join(self.ampl, phase)

    def _derivative(self, x):
        ampl, phase = self.domain.split(x)
        return self.codomain.join(0*ampl, phase)

    def _adjoint(self, x):
        ampl, phase = self.domain.split(x)
        return self.codomain.join(0*ampl, phase)


def ampphase2complex(amp, phase):
    return amp * np.exp(1j * phase)


def load_experimental_data(filename):
    mat = loadmat(filename)
    mask = mat['mask']
    # mask_binary = mat['mask_binary'].astype(bool) # mask_binary does not mean the same in the context of the simulation as in the reconstruction
    # define simulated FOV as mask for now
    # mask_binary = ones_like(mask).astype(bool)
    mask_binary = mask > 0.1
    px_size = 1e-9 * np.median(np.diff(mat['y_v'], axis=0))  # m/px
    # * 4 # for stronger interaction; linear combination with mat['beta_s'] for rotated polarization
    g_map = mat['beta_p']
    pad_amount = (100, 100)
    g_map = np.pad(g_map, pad_amount, 'constant', constant_values=1e-5)
    mask = np.pad(mask, pad_amount, 'constant', constant_values=1)
    mask_binary = np.pad(mask_binary, pad_amount, 'constant', constant_values=False)
    return g_map, mask, mask_binary, px_size


def simulated_data(complex_g=True, amplitude_known=False, parallel=True, N=30):
    filename = r"./data/01_javier.mat"
    g_map, mask, mask_binary, px_size = load_experimental_data(filename)
    fov = tuple(x*px_size for x in mask.shape)
    lambda_electron = 2.51e-12
    defocus = 900e-6
    fresnelNumber = np.prod(fov)/(defocus * lambda_electron)
    # Uniform grid
    Xdim = mask.shape[0]
    Ydim = mask.shape[1]
    A_Psi0_Multiplier = mask.astype(complex)
    if complex_g:
        grid = UniformGrid(np.linspace(0, 1, Xdim, endpoint=False),
                           np.linspace(0, 1, Ydim, endpoint=False))
        op = complex_PINEM_g_to_data(grid, fresnelNumber, mask_binary,
                                     A_Psi0_Multiplier, N=N, parallel=parallel)
        if amplitude_known:
            op2 = SquaredModulus(grid.complex_space()) - np.abs(g_map)**2
            op = Vector_of_operators([op2, op])
        return op, grid, g_map, g_map, mask
    else:
        grid = UniformGrid(np.linspace(0, 1, Xdim, endpoint=False),
                           np.linspace(0, 1, Ydim, endpoint=False))
        op = PINEM_g_to_data(grid, fresnelNumber, mask_binary,
                             A_Psi0_Multiplier, N=N, parallel=parallel)
        exact_solution = op.domain.join(np.log(np.abs(g_map)),
                                        np.unwrap(np.angle(g_map.T)).T)

        return op, grid, exact_solution, g_map, mask_binary, A_Psi0_Multiplier


def synthetic_data(complex_g=True, amplitude_known=False, parallel=True):
    # Example parameters
    fresnelNumber = 5e3    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
    # in grid (i.e. with the size of one pixel for the above choice of grid)
    # noise_level = 0.0001      # Noise level in the simulated data

    N = 8
    # define grid
    Xdim = 256
    Ydim = 256
    grid = UniformGrid(np.linspace(0, 1, Xdim, endpoint=False),
                       np.linspace(0, 1, Ydim, endpoint=False))

    # define forward operator and its domain
    [Xco, Yco] = np.meshgrid(np.arange(-1, 1, 2/Xdim), np.arange(-1, 1, 2/Ydim))
    mask = (abs(Xco+0.2) <= 0.2) & (abs(Yco) <= 0.4)
    mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35)) <= 0.01)
    A_Psi0_Multiplier = np.ones(grid.shape, complex)

    # Create phantom image (= padded example-image)
    picture = ascent()
    g_map_amp = picture[-Xdim//2:, -Ydim//2:].astype(np.float64)/255
    g_map_amp *= 2
    g_map_phase = picture[:Xdim//2, :Ydim//2].astype(np.float64)/255
    g_map_phase *= 0.5*np.pi
    pad_amount = tuple([(grid.shape[0] - g_map_amp.shape[0])//2,
                       (grid.shape[1] - g_map_amp.shape[1])//2])
    g_map_amp = np.pad(g_map_amp, pad_amount, 'constant', constant_values=0)
    g_map_phase = np.pad(g_map_phase, pad_amount, 'constant', constant_values=0)
    g_map_phase *= mask
    g_map = ampphase2complex(g_map_amp, g_map_phase)
    # g_map /= 10*abs(g_map).max()
    # g_map += ones_like(g_map)
    g_map = g_map.astype(complex)*mask

    if complex_g:
        op = complex_PINEM_g_to_data(grid, fresnelNumber, mask,
                                     A_Psi0_Multiplier, N=N, parallel=parallel)
        if amplitude_known:
            op2 = SquaredModulus(grid.complex_space()) - np.abs(g_map)**2
            op = Vector_of_operators([op2, op])
    else:
        op = PINEM_g_to_data(grid, fresnelNumber, mask, A_Psi0_Multiplier, N=N, parallel=True)
        if amplitude_known:
            ampl_fix = FixAmplitude(np.abs(g_map), grid)
            op = op * ampl_fix

    if complex_g:
        exact_solution = g_map
    else:
        exact_solution = op.domain.join(np.abs(g_map), np.angle(g_map))
    return op, grid, exact_solution, g_map, mask, A_Psi0_Multiplier


def main():
    real_data = False
    complex_g = False
    amplitude_known = True
    intensity = 1e6
    if real_data:
        op, grid, exact_solution, g_map, mask, A_Psi0 = simulated_data(complex_g=complex_g,
                                                                       amplitude_known=amplitude_known)
        # regpar = 5e-2; sobolev_index = 2
        regpar = 1e-0
        sobolev_index = 0
        regpar_step = 1/3
    else:
        op, grid, exact_solution, g_map, mask, A_Psi0 = synthetic_data(complex_g=complex_g,
                                                                       amplitude_known=amplitude_known)
        regpar = 5e-4
        sobolev_index = 2
        regpar_step = 1/3

    if complex_g:
        Hdomain = Sobolev(grid.complex_space(), index=sobolev_index)
    else:
        Hdomain = 0.5 * Sobolev(grid, index=sobolev_index) + L2(grid)  # Sobolev(grid, index=0.5)

    flat_codomain = DirectSum(*op.codomain.summands, flatten=True)
    if complex_g and amplitude_known:
        exact_data = op(exact_solution)
        exact_data_comp = op.codomain.split(exact_data)
        noisy_data = np.random.poisson(intensity * exact_data_comp[1])/intensity
        data = op.codomain.join(np.zeros_like(exact_data_comp[0]), noisy_data)
        Hcodomain0 = L2(grid)
        data_comp = flat_codomain.split(data)
        Hcodomain1 = L2(grid, weights=(1+intensity*data_comp[1])/intensity)
        for j in range(2, len(flat_codomain)):
            Hcodomain1 = Hcodomain1 + L2(grid, weights=(1+intensity*data_comp[j])/intensity)
        Hcodomain = hilbert.DirectSum(Hcodomain0, Hcodomain1)
    else:
        exact_data = op(exact_solution)
        data = np.random.poisson(intensity * exact_data)/intensity
        data_comp = op.codomain.split(data)
        # define codomain Gram matrix based on observed data to approximate log-likelihood
        Hcodomain = L2(grid, weights=(1+intensity*data_comp[0])/intensity)
        for j in range(1, len(data_comp)):
            Hcodomain = Hcodomain + L2(grid, weights=(1+intensity*data_comp[j])/intensity)

    # Image-reconstruction using the IRGNM method
    setting = HilbertSpaceSetting(op=op, Hdomain=Hdomain, Hcodomain=Hcodomain)
    if complex_g:
        if amplitude_known:
            init_vec = abs(g_map).astype(complex)
        else:
            init_vec = grid.complex_space().ones() * mask
    else:
        angle = np.deg2rad(10)
        X, Y = np.meshgrid(np.linspace(0, 1, np.size(mask, 1)), np.linspace(0, 1, np.size(mask, 0)))
        init_phase_gradient = -1 * (np.sin(angle)*X + np.cos(angle)*Y) * 2*np.pi * 3 + 2.5
        if amplitude_known:
            # init_vec = op.domain.join(np.abs(g_map), \
            #         norm(np.angle(g_map).ravel(),1)/norm(mask.ravel(),1) * mask)
            init_vec = op.domain.join(np.abs(g_map),
                                      init_phase_gradient * mask)
        else:
            # init_vec = op.domain.join(mask, \
            #     norm(np.angle(g_map).ravel(),1)/norm(mask.ravel(),1) * mask)
            init_vec = op.domain.join(mask,
                                      init_phase_gradient * mask)

    solver = IrgnmCG(
        setting, data, init=init_vec,
        regpar=regpar, regpar_step=regpar_step,
        inner_it_logging_level=logging.INFO)
    stoprule = (
        rules.CountIterations(max_iterations=15) +
        rules.Discrepancy(
            setting.Hcodomain.norm,
            data,
            noiselevel=setting.Hcodomain.norm(np.sqrt(data/intensity)),
            tau=1.0
        )
    )

    # plot exact solution and data
    if not complex_g:
        ex_abs, ex_phase = op.domain.split(exact_solution)
    else:
        ex_abs = np.abs(g_map)
        ex_phase = np.angle(g_map)
    fig1 = imshow_fig(2, 2)
    if not amplitude_known:
        plotdata = [{'pos': (0, 0), 'data': np.abs(g_map), 'title': 'Exact |g|'}]
    else:
        plotdata = [{'pos': (1, 0), 'data': ex_phase, 'title': 'arg(g)'}]
    fig1.plot(plotdata)

    data_comp = flat_codomain.split(data)
    nr_data = len(data_comp)
    fig2 = imshow_fig(2, nr_data+1)
    plot_data = [{'pos': (0, j), 'data': data_comp[j], 'title':'Simulated data'}
                 for j in range(nr_data)]
    plot_data.append(
        {'pos': (0, nr_data), 'data': data_comp[nr_data-1]-data_comp[nr_data-2], 'title': 'diff'})
    fig2.plot(plot_data)

    if hasattr(solver, "nr_inner_its"):
        fig3, axs3 = plt.subplots(3, 1, sharex=False, sharey=False)
    else:
        fig3, axs3 = plt.subplots(2, 1, sharex=False, sharey=False)

    def reconstruction_error(_exact, _reconstruction):
        nonlocal complex_g
        nonlocal mask
        if complex_g:
            reco_error1 = norm(mask*(np.abs(_reconstruction)-np.abs(_exact)))/norm(mask*np.abs(_exact))
            reco_error2 = norm(np.abs(1-np.exp(1j*mask*np.angle(_reconstruction)-1j*mask*np.angle(_exact))))
        else:
            ex_abs, ex_phase = op.domain.split(_exact)
            reco_abs, reco_phase = op.domain.split(_reconstruction)
            reco_error1 = norm(mask*(reco_abs-ex_abs))/norm(mask*ex_abs)
            reco_error2 = norm(mask*(reco_phase-ex_phase))/norm(mask*ex_phase)
        return reco_error1, reco_error2

    # if complex_g:
    #     reco_error1 = norm(np.abs(init_vec)-np.abs(g_map))/norm(np.abs(g_map))
    #     reco_error2 = norm(np.abs(1-np.exp(1j*np.angle(init_vec)-1j*np.angle(g_map))))
    # else:
    #     reco_abs, reco_phase = op.domain.split(init_vec)
    #     reco_error1 = norm(mask*(reco_abs-ex_abs))/norm(mask*ex_abs)
    #     reco_error2 = norm(mask*(reco_phase-ex_phase))/norm(mask*ex_phase)
    reco_error1, reco_error2 = reconstruction_error(exact_solution, init_vec)
    
    print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
        0, reco_error1, reco_error2))
    stats = {'ampl_err': [reco_error1],
             'phase_err': [reco_error2],
             'residuals': [norm(solver.y-exact_data)/norm(exact_data)],
             'nr_inner_steps': [0]}

    for reco, reco_data in solver.until(stoprule):
        if not stoprule.triggered:
            Newton_step = solver.iteration_step_nr
            reco_error1, reco_error2 = reconstruction_error(exact_solution, reco)
            # if complex_g:
            #     reco_error1 = norm(np.abs(reco)-np.abs(exact_solution))/norm(np.abs(exact_solution))
            #     reco_error2 = norm(np.abs(1-np.exp(1j*np.angle(reco)-1j*np.angle(g_map))))
            # else:
            #     reco_abs, reco_phase = op.domain.split(reco)
            #     reco_error1 = norm(reco_abs-ex_abs)/norm(ex_abs)
            #     reco_error2 = norm(mask*(reco_phase-ex_phase))/norm(mask*ex_phase)
            print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
                Newton_step, reco_error1, reco_error2))
            stats['ampl_err'].append(reco_error1)
            stats['phase_err'].append(reco_error2)
            stats['residuals'].append(norm(reco_data-exact_data)/norm(exact_data))
            if hasattr(solver, "nr_inner_its") and callable(solver.nr_inner_its):
                stats['nr_inner_steps'].append(solver.nr_inner_its())

        if Newton_step % 2 == 0 or (stoprule.triggered and ((Newton_step-1) % 2 != 0)):
            if complex_g:
                reco_amp = np.abs(reco)
                reco_phase = np.angle(reco)
            else:
                reco_amp, reco_phase = op.domain.split(reco)
            reco_data_comp = flat_codomain.split(reco_data)

            plotdata = []
            if amplitude_known and not complex_g:
                plotdata.append({'pos': (0, 0), 'data': np.abs(1-np.exp(1j*reco_phase-1j*np.angle(g_map))),
                                 'title': '|1-exp(i arg(g_rec)-i arg g)|, step {}'.format(Newton_step)})
            else:
                plotdata.append({'pos': (1, 0), 'data': reco_amp,
                                'title': 'Reco |g|, step {}'.format(Newton_step)})
                plotdata.append({'pos': (0, 0), 'data': reco_amp-np.abs(g_map),
                                 'title': 'Error |g|, step {}'.format(Newton_step)})
            plotdata.append({'pos': (1, 1), 'data': reco_phase,
                            'title': 'arg(g_rec), step {}'.format(Newton_step)})
            plotdata.append({'pos': (0, 1), 'data': reco_phase-np.angle(g_map),
                             'title': 'arg(g_rec)-arg g), step {}'.format(Newton_step)})
            fig1.plot(plotdata)

            plotdata = [{'pos': (1, j), 'data': reco_data_comp[j],
                         'title':'recon. data step {}'.format(Newton_step)}
                        for j in range(nr_data)]
            plotdata.append({'pos': (1, nr_data), 'data': reco_data_comp[nr_data-1]-reco_data_comp[nr_data-2],
                             'title': 'diff reco'})
            fig2.plot(plotdata)

            axs3[0].cla()
            if not amplitude_known:
                axs3[0].plot(stats['ampl_err'], label='amplitude error')
            axs3[0].plot(stats['phase_err'], label='phase error')
            axs3[0].legend()
            axs3[1].cla()
            axs3[1].semilogy(stats['residuals'], label='residuals')
            axs3[1].legend()
            if hasattr(solver, "nr_inner_its") and callable(solver.nr_inner_its):
                axs3[2].cla()
                axs3[2].plot(stats['nr_inner_steps'], label='number of inner CG steps')
                axs3[2].legend()
            plt.show(block=False)
            plt.pause(0.1)
    plt.show(block=True)
    reco = stoprule.x
    reco_data = stoprule.y
    # np.save('PINEM_tests/reco',reco)


if __name__ == '__main__':
    main()
