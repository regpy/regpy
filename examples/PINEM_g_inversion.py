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
from regpy.operators import Operator,SquaredModulus,Vector_of_operators
from regpy.operators.PINEM import PINEM_g_to_data, complex_PINEM_g_to_data
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
from scipy.io import loadmat
from scipy.misc import ascent
from numpy.linalg import norm 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

class FixAmplitude(Operator):
# Operator (a,phase) |-> (ampl, phase) for some fixed ampl    
# (This could be replaced by the operator phase |-> (ampl, phase), but this would 
# require more if statements in main.)
    def __init__(self,ampl,domain):
        self.ampl = ampl
        super().__init__(DirectSum(domain,domain), DirectSum(domain,domain))

    def _eval(self,x,differentiate):
        _, phase = self.domain.split(x)
        return self.codomain.join(self.ampl,phase)

    def _derivative(self, x):
        ampl, phase = self.domain.split(x)
        return self.codomain.join(0*ampl,phase)

    def _adjoint(self, x):
        ampl, phase = self.domain.split(x)
        return self.codomain.join(0*ampl,phase)

def ampphase2complex(amp, phase):
    return amp * np.exp(1j * phase)

def load_experimental_data(filename):
    mat = loadmat(filename)
    mask = mat['mask']
    # mask_binary = mat['mask_binary'].astype(bool) # mask_binary does not mean the same in the context of the simulation as in the reconstruction
    # define simulated FOV as mask for now
    mask_binary = ones_like(mask).astype(bool)
    px_size = 1e-9 * np.median(np.diff(mat['y_v'], axis=0))  # m/px
    g_map = mat['beta_p'] # * 4 # for stronger interaction; linear combination with mat['beta_s'] for rotated polarization
    pad_amount = (100, 100)
    g_map = np.pad(g_map, pad_amount, 'constant', constant_values=1e-5)
    mask = np.pad(mask, pad_amount, 'constant', constant_values=1)
    mask_binary = np.pad(mask_binary, pad_amount, 'constant', constant_values=False)
    return g_map, mask, mask_binary, px_size


def simulated_data(complex_g=True,amplitude_known=False):
    N = 30
    parallel = True
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
            op = Vector_of_operators([op2,op])
        return op, grid, g_map, g_map,mask
    else:
        grid = UniformGrid(np.linspace(0, 1, Xdim, endpoint=False),
                           np.linspace(0, 1, Ydim, endpoint=False))
        op = PINEM_g_to_data(grid, fresnelNumber, mask_binary,
                             A_Psi0_Multiplier, N=N, parallel=parallel)
        exact_solution = op.domain.join(np.log(np.abs(g_map)), np.angle(g_map))

        return op, grid, exact_solution, g_map, mask

def synthetic_data(complex_g=True,amplitude_known=False):
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
                                     A_Psi0_Multiplier, N=N, parallel=True)
        if amplitude_known:
            op2 = SquaredModulus(grid.complex_space()) - np.abs(g_map)**2
            op = Vector_of_operators([op2,op])
    else:
        op = PINEM_g_to_data(grid, fresnelNumber, mask, A_Psi0_Multiplier, N=N, parallel=True)
        if amplitude_known:
            ampl_fix = FixAmplitude(np.abs(g_map),grid)
            op = op * ampl_fix

    if complex_g:
        exact_solution = g_map
    else:
        exact_solution = op.domain.join(np.abs(g_map), np.angle(g_map))
    return op, grid, exact_solution, g_map, mask


def main():
    real_data = False
    complex_g = True
    amplitude_known = True
    intensity = 1e8
    if real_data:
        op, grid, exact_solution, g_map, mask = simulated_data(complex_g = complex_g, \
            amplitude_known = amplitude_known)
    else:
        op, grid, exact_solution, g_map, mask = synthetic_data(complex_g = complex_g, \
            amplitude_known = amplitude_known)

    if complex_g:
        Hdomain = Sobolev(grid.complex_space(), index=0.5)
    else:
        Hdomain = 0.5 * Sobolev(grid, index=0.5) + Sobolev(grid, index=0.5)

    flat_codomain = DirectSum(*op.codomain.summands, flatten=True)    
    if complex_g and amplitude_known:
        exact_data = op(exact_solution)
        exact_data_comp = op.codomain.split(exact_data)
        noisy_data = np.random.poisson(intensity * exact_data_comp[1])/intensity
        data = op.codomain.join(np.zeros_like(exact_data_comp[0]), noisy_data)
        Hcodomain0 = L2(grid)
        data_comp = flat_codomain.split(data)
        Hcodomain1 = L2(grid, weights=(1+intensity*data_comp[1])/intensity)
        for j in range(2,len(flat_codomain)):
            Hcodomain1 = Hcodomain1 + L2(grid, weights=(1+intensity*data_comp[j])/intensity)
        Hcodomain = hilbert.DirectSum(Hcodomain0,Hcodomain1)
    else: 
        exact_data = op(exact_solution)
        data = np.random.poisson(intensity * exact_data)/intensity    
        data_comp = op.codomain.split(data)
        # define codomain Gram matrix based on observed data to approximate log-likelihood
        Hcodomain = L2(grid,weights=(1+intensity*data_comp[0])/intensity)
        for j in range(1,len(data_comp)):
            Hcodomain = Hcodomain + L2(grid, weights=(1+intensity*data_comp[j])/intensity)


    # Image-reconstruction using the IRGNM method
    setting = HilbertSpaceSetting(op=op, Hdomain=Hdomain, Hcodomain=Hcodomain)
    if complex_g:
        if amplitude_known:
            init_vec = abs(g_map).astype(complex)
        else:
            init_vec = grid.complex_space().ones() * mask
    else:
        if amplitude_known:
            init_vec = op.domain.join(np.abs(g_map), np.zeros_like(np.angle(g_map)))
        else:
            init_vec = op.domain.join(mask, grid.zeros())
    
    solver = IrgnmCG(
        setting, data, init=init_vec,
        regpar=5e-4, regpar_step=2/3,
        inner_it_logging_level=logging.INFO)
    stoprule = (
        rules.CountIterations(max_iterations=20) +
        rules.Discrepancy(
            setting.Hcodomain.norm,
            data,
            noiselevel=setting.Hcodomain.norm(np.sqrt(data/intensity)),
            tau=1.0
        )
    )

    # plot exact solution and data
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    if not amplitude_known:
        axs[0, 0].set_title('Exact |g|')
        im = axs[0, 0].imshow(np.abs(g_map))
        fig.colorbar(im, ax=axs[0, 0])
        #axs[0, 1].set_title('Exact  arg(g)')
        #im = axs[0, 1].imshow(np.angle(g_map))
        fig.colorbar(im, ax=axs[0, 1])

    data_comp = flat_codomain.split(data)
    fig2, axs2 = plt.subplots(2, len(data_comp), sharex=True, sharey=True)
    for j in range(len(data_comp)):
        im = axs2[0, j].imshow(data_comp[j])
        fig2.colorbar(im, ax=axs2[0, j])
        axs2[0, j].set_title('Simulated data')

    fig3, axs3 = plt.subplots(2,1, sharex=False, sharey=False)

    #reco, reco_data = solver.run(stoprule)
    stats ={'ampl_err':[], \
        'phase_err':[], \
            'residuals' : []}
    for reco, reco_data in solver.until(stoprule):
        Newton_step = solver.iteration_step_nr
        # Print reconstruction error
        if complex_g: 
            reco_error1 = norm(np.abs(reco)-np.abs(exact_solution))/norm(np.abs(exact_solution))
            #reco.real-exact_solution.real
            reco_error2 = norm(np.angle(reco)-np.angle(exact_solution))/norm(np.angle(exact_solution))
            #reco.imag-exact_solution.imag
        else:
            reco_err1, reco_err2 = op.domain.split(reco-exact_solution)
            reco_error1 = norm(reco_err1)/norm(np.abs(g_map))
            reco_error2 = norm(reco_err2)/norm(np.angle(g_map))
        print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
            Newton_step, reco_error1, reco_error2))
        stats['ampl_err'].append(reco_error1)
        stats['phase_err'].append(reco_error2)
        stats['residuals'].append(norm(reco_data-exact_data)/norm(exact_data))
        # Plot results
        if Newton_step == 2  or stoprule.triggered:
            if complex_g:
                reco_amp = np.abs(reco)
                reco_phase = np.angle(reco)
            else:
                reco_amp, reco_phase = op.domain.split(reco)
            reco_data_comp = flat_codomain.split(reco_data)

            if amplitude_known and not complex_g:
                axs[0, 0].set_title('1-|exp(i arg(g_rec)-i arg g)|, step {}'.format(Newton_step))
                plot_map = 1-np.abs(np.exp(1j*reco_phase-1j*np.angle(g_map)))
                im = axs[0, 0].imshow(plot_map)
                if Newton_step == 2:
                    fig.colorbar(im, ax=axs[0,0])
            
                axs[1,0].set_title('arg(g)')
                im = axs[1,0].imshow(np.angle(g_map))
                if Newton_step == 2:
                    fig.colorbar(im, ax=axs[1, 0])
            else:
                axs[1, 0].set_title('Reco |g|, step {}'.format(Newton_step))
                im = axs[1, 0].imshow(reco_amp)
                axs[0, 0].set_title('Error |g|, step {}'.format(Newton_step))
                im = axs[0, 0].imshow(reco_amp-np.abs(g_map)) 
                if Newton_step == 2:
                    fig.colorbar(im, ax=axs[1, 0])

            axs[1, 1].set_title('arg(g_rec), step {}'.format(Newton_step))
            im = axs[1, 1].imshow(mask)
            if Newton_step == 2:
                fig.colorbar(im, ax=axs[1, 1])

            axs[0, 1].set_title('arg(g_rec)-arg g), step {}'.format(Newton_step))
            im = axs[0, 1].imshow(reco_phase-np.angle(g_map))
            if Newton_step == 2:
                fig.colorbar(im, ax=axs[0, 1])

            for j in range(len(reco_data_comp)):
                im = axs2[1, j].imshow(reco_data_comp[j])
                if Newton_step == 2:
                    fig2.colorbar(im, ax=axs2[1, j])
                axs2[1, j].set_title('recon. data step {}'.format(Newton_step))

            axs3[0].cla()
            if not amplitude_known:
                axs3[0].plot(stats['ampl_err'],label = 'amplitude error')
            axs3[0].plot(stats['phase_err'], label = 'phase error')
            axs3[0].legend()
            axs3[1].cla()
            axs3[1].semilogy(stats['residuals'], label = 'residuals')
            axs3[1].legend()
            plt.show(block=False)
            plt.pause(0.1)
    plt.show(block=True)


if __name__ == '__main__':
    main()
