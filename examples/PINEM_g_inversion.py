import logging

import matplotlib.pyplot as plt
import numpy as np
import regpy.stoprules as rules
from numpy.core.numeric import ones_like
from regpy.discrs import UniformGrid
from regpy.hilbert import L2, Sobolev
from regpy.operators.PINEM import PINEM_g_to_data, complex_PINEM_g_to_data
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
from scipy.io import loadmat
from scipy.misc import ascent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)


def log2complex(log_z):
    return np.exp(log_z.real) * np.exp(1j * log_z.imag)


def complex2log(z):
    return np.log(np.abs(z)) + 1j * np.angle(z)


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


def simulated_data(complex_g=True):
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
    N = 30
    parallel = True
    if complex_g:
        # TODO Shouldn't the domain be complex?
        grid = UniformGrid(np.linspace(0, 1, Xdim, endpoint=False),
                           np.linspace(0, 1, Ydim, endpoint=False))
        op = complex_PINEM_g_to_data(grid, fresnelNumber, mask_binary,
                                     A_Psi0_Multiplier, N=N, parallel=parallel)
        return op, grid, g_map, g_map
    else:
        grid = UniformGrid(np.linspace(0, 1, Xdim, endpoint=False),
                           np.linspace(0, 1, Ydim, endpoint=False))
        op = PINEM_g_to_data(grid, fresnelNumber, mask_binary,
                             A_Psi0_Multiplier, N=N, parallel=parallel)
        log_g = np.log(np.abs(g_map)) + 1j * np.angle(g_map)
        exact_solution = op.domain.join(np.log(np.abs(g_map)), np.angle(g_map))

        return op, grid, exact_solution, log_g


def synthetic_data(complex_g=True):
    # Example parameters
    fresnelNumber = 5e3    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
    # in grid (i.e. with the size of one pixel for the above choice of grid)
    # noise_level = 0.0001      # Noise level in the simulated data

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
    N = 8
    if complex_g:
        op = complex_PINEM_g_to_data(grid, fresnelNumber, mask,
                                     A_Psi0_Multiplier, N=N, parallel=True)
    else:
        op = PINEM_g_to_data(grid, fresnelNumber, mask, A_Psi0_Multiplier, N=N, parallel=True)

    # Create phantom image (= padded example-image)
    picture = ascent()
    g_map_amp = picture[-Xdim//2:, -Ydim//2:].astype(np.float64)/255
    g_map_amp *= 2
    g_map_phase = picture[:Xdim//2, :Ydim//2].astype(np.float64)/255
    g_map_phase *= 2*np.pi
    pad_amount = tuple([(grid.shape[0] - g_map_amp.shape[0])//2,
                       (grid.shape[1] - g_map_amp.shape[1])//2])
    g_map_amp = np.pad(g_map_amp, pad_amount, 'constant', constant_values=0)
    g_map_phase = np.pad(g_map_phase, pad_amount, 'constant', constant_values=0)
    g_map_phase *= mask
    g_map = ampphase2complex(g_map_amp, g_map_phase)
    # g_map /= 10*abs(g_map).max()
    # g_map += ones_like(g_map)
    g_map = g_map.astype(complex)*mask

    # Create exact and noisy data
    if complex_g:
        exact_solution = g_map
    else:
        exact_solution = op.domain.join(np.abs(g_map), np.angle(g_map))
        g_map = complex2log(g_map)
    return op, grid, exact_solution, g_map


def main():
    real_data = True
    complex_g = True
    if real_data:
        op, grid, exact_solution, log_g = simulated_data(complex_g)
    else:
        op, grid, exact_solution, log_g = synthetic_data(complex_g)

    exact_data = op(exact_solution)

    intensity = 1e5
    data = np.random.poisson(intensity * exact_data)/intensity

    # Here we are weighting the penalty for the modulus a bit less
    if complex_g:
        Hdomain = Sobolev(grid.complex_space(), index=0.5)
    else:
        Hdomain = 0.5 * Sobolev(grid, index=0.5) + Sobolev(grid, index=0.5)
    # define codomain Gram matrix based on observed data to approximate log-likelihood
    Hcodomain0 = L2(grid, weights=(1+intensity*data[0])/intensity)
    Hcodomain1 = L2(grid, weights=(1+intensity*data[1])/intensity)
    Hcodomain = Hcodomain0+Hcodomain1

    # Image-reconstruction using the IRGNM method
    setting = HilbertSpaceSetting(op=op, Hdomain=Hdomain, Hcodomain=Hcodomain)
    if complex_g:
        init_vec = grid.complex_space().ones()
    else:
        init_vec = op.domain.join(grid.ones(), grid.zeros())

    solver = IrgnmCG(
        setting, data, init=init_vec,
        regpar=5e-4, regpar_step=2/3,
        inner_it_logging_level=logging.INFO)
    stoprule = (
        rules.CountIterations(max_iterations=100) +
        rules.Discrepancy(
            setting.Hcodomain.norm,
            data,
            noiselevel=setting.Hcodomain.norm(np.sqrt(data/intensity)),
            tau=1.0
        )
    )

    if not complex_g:  # g_map is in logarithmic form where real is log(abs) and imag is phase
        g_map_exact = log2complex(log_g)
    else:
        g_map_exact = log_g

    # plot exact solution and data
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    axs[0, 0].set_title('Exact |g|')
    im = axs[0, 0].imshow(np.abs(g_map_exact))
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 1].set_title('Exact  arg(g)')
    im = axs[0, 1].imshow(np.angle(g_map_exact))
    fig.colorbar(im, ax=axs[0, 1])

    data_comp = op.codomain.split(data)
    fig2, axs2 = plt.subplots(2, len(data_comp), sharex=True, sharey=True)
    for j in range(len(data_comp)):
        im = axs2[0, j].imshow(data_comp[j])
        fig2.colorbar(im, ax=axs2[0, j])
        axs2[0, j].set_title('Simulated data')

    #reco, reco_data = solver.run(stoprule)
    for reco, reco_data in solver.until(stoprule):
        Newton_step = solver.iteration_step_nr
        # Print reconstruction error
        if complex_g:
            reco_error1 = reco.real-exact_solution.real
            reco_error2 = reco.imag-exact_solution.imag
        else:
            reco_error1, reco_error2 = op.domain.split(reco-exact_solution)
        print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
            Newton_step,
            np.linalg.norm(reco_error1)/np.linalg.norm(np.exp(log_g.real)),
            np.linalg.norm(reco_error2)/np.linalg.norm(log_g.imag)))
        # Plot reults
        if Newton_step % 2 == 0 or stoprule.triggered:
            if complex_g:
                reco_amp = np.abs(reco)
                reco_phase = np.angle(reco)
            else:
                reco_amp, reco_phase = op.domain.split(reco)
            reco_data_comp = op.codomain.split(reco_data)

            axs[1, 0].set_title('Reco |g|, step {}'.format(Newton_step))
            im = axs[1, 0].imshow(reco_amp)
            if Newton_step == 2:
                fig.colorbar(im, ax=axs[1, 0])
            axs[1, 1].set_title('Reco arg(g), step {}'.format(Newton_step))
            im = axs[1, 1].imshow(reco_phase)
            if Newton_step == 2:
                fig.colorbar(im, ax=axs[1, 1])

            for j in range(len(reco_data_comp)):
                im = axs2[1, j].imshow(reco_data_comp[j])
                if Newton_step == 2:
                    fig2.colorbar(im, ax=axs2[1, j])
                axs2[1, j].set_title('recon. data step {}'.format(Newton_step))
            plt.show(block=False)
            plt.pause(0.1)
    plt.show(block=True)


if __name__ == '__main__':
    main()
