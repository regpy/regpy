import logging
import sys

import numpy as np
from scipy.datasets import ascent

from regpy.vecsps import UniformGridFcts
from regpy.operators import CoordinateProjection
from regpy.hilbert import L2, HmDomain
from regpy.solvers import Setting
from regpy.solvers.nonlinear.irgnm import IrgnmCG
import regpy.stoprules as rules
import regpy.util as util

util.set_rng_seed(15873098306879350073259142812684978477)

from . import import_example_package

import_example_package("./examples/pinem/")

from operators import get_wave_field_reco


def test_wave_field_inversion():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
    )

    r"""Shows inversion of imaging system with modeled by Fresnel propagator. The (complex) image :math:`x` is reconstructed from
    :math:`|\mathcal{D}_{+N}(x)|^{2}) and \(|\mathcal{D}_{-N}(x)|^{2}) where \(\mathcal{D}` is the Fresnel propagator.
    The reconstruction is done using the iteratively regularized Gauss-Newton method.
    """


    # Example parameters
    fresnel_number = 5e2   # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale

    # in grid (i.e. with the size of one pixel for the above choice of grid)
    noise_level = 0.001       # Noise level in the simulated data
    intensity = 1e3
    # sol_type = 'phase'
    # sol_type = 'modulus'
    sol_type = None

    # define grid
    Xdim = 256
    Y_dim = 256
    grid = UniformGridFcts(np.linspace(0, 1, Xdim, endpoint=False),
                        np.linspace(0, 1, Y_dim, endpoint=False)).real_space()
    cgrid = grid.complex_space()
    [Xco, Yco] = np.meshgrid(np.arange(-1, 1, 2/Xdim), np.arange(-1, 1, 2/Y_dim))
    mask = (abs(Xco+0.2) <= 0.2) & (abs(Yco) <= 0.4)
    mask = mask | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35)) <= 0.01)

    # Forward operator and its domain
    op = get_wave_field_reco(cgrid, fresnel_number, mask.astype(float), sol_type,parallel=False)  

    if sol_type == None:
        h_domain =  HmDomain(cgrid,mask,dtype=complex,index=1)
    else:
        h_domain = HmDomain(grid,mask,index=1)
    op = op

    # Create phantom image (= padded example-image)
    picture = ascent()
    exact_solution = picture[-Xdim//2:, -Y_dim//2:].astype(np.float64)/255
    if sol_type is None:
        exact_solution = exact_solution + 0.3j*2*np.pi * \
            picture[:Xdim//2, :Y_dim//2].astype(np.float64)/255
    pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2,
                    (grid.shape[1] - exact_solution.shape[1])//2])
    exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)
    exact_solution = exact_solution * mask  # - 4*(1-mask)

    # Create exact data and Poisson data
    exact_data = op(exact_solution)
    data = op.codomain.poisson(intensity * exact_data)/intensity

    # define codomain Gram matrix based on observed data to approximate log-likelihood
    h_codomain0 = L2(grid, weights=(1+intensity*data[0])/intensity)
    h_codomain1 = L2(grid, weights=(1+intensity*data[1])/intensity)  
    h_codomain2 = L2(grid, weights=(1+intensity*data[2])/intensity)
    h_codomain = h_codomain0+h_codomain1+h_codomain2

    # Image reconstruction using the IRGNM method
    setting = Setting(op=op,penalty=h_domain,data_fid=h_codomain)

    init_vec = np.zeros_like(exact_solution)

    solver = IrgnmCG(
        setting, data, regpar=0.1, regpar_step=2/3, init=init_vec,
        inner_it_logging_level=logging.INFO
    )
    stoprule = (
        rules.CountIterations(max_iterations=100) +
        rules.Discrepancy(
            setting.h_codomain.norm,
            data,
            noiselevel=setting.h_codomain.norm((data/intensity).component_wise(np.sqrt)),
            tau=1.01
        )
    )

    # plot data
    data_comp = op.codomain.split(data)

    # perform reconstruction    
    for reco, reco_data in solver.until(stoprule):
        newton_step = solver.iteration_step_nr
        reco_error = reco-exact_solution
        print('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}'.format(
            newton_step,
            np.linalg.norm(reco_error.real)/np.linalg.norm(exact_solution.real),
            np.linalg.norm(reco_error.imag)/np.linalg.norm(exact_solution.imag)))


sys.path.pop(0)