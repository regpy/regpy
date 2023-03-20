import logging
from multiprocessing.spawn import get_command_line
from operator import ge

from scipy.sparse import csc_matrix 
from scipy.sparse.linalg import spsolve
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from copy import deepcopy
import regpy.stoprules as rules
from numpy.core.numeric import ones_like
from regpy.discrs import UniformGrid, DirectSum
from regpy.discrs.tensor_bases import ChebyshevBasis, LegendreBasis
from regpy.hilbert import L2, Sobolev, Hm0_domain
import regpy.hilbert as hilbert
from regpy.operators import Operator, SquaredModulus, Ptw_Multiplication, Vector_of_operators
from regpy.operators import CoordinateProjection, Zero, InnerShift, OuterShift
from regpy.operators import DirectSum as opDirectSum
from regpy.operators.PINEM import PINEM_g_to_data, complex_PINEM_g_to_data
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
from regpy.solvers.newton import NewtonCG
from scipy.io import loadmat, savemat
from scipy.misc import ascent
from numpy.linalg import norm
from regpy.util.imshow_fig import imshow_fig
from math import isfinite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

####################### conversion routines for plotting complex-valued fields

def complex_to_rgb_log(z):
    logdat = np.log(np.abs(z))
    minlog = np.min(logdat)
    maxlog = np.max(logdat)
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), (logdat-minlog)/(maxlog-minlog) ))
    return hsv_to_rgb(HSV)

def complex_to_rgb(z):
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), np.abs(z)/np.max((np.abs(z[:]))), ))
    return hsv_to_rgb(HSV)

##################### operator needed for fixing g on parts of the grid where its values are known

class ForgetSecond(Operator):
    def __init__(self,domain1,domain2):
        self.domain2 = domain2
        super().__init__(DirectSum(domain1, domain2),domain2,linear=True)

    def _eval(self,x):
        x1,x2 = self.domain.split(x)
        return x1
   
    def _adjoint(self,y):
        return self.domain.join(y,self.domain2.zeros())
        
""" harmonic extension of a function on part of a 2D regular grid  
input: 
        mask: binary mask with True at point where values are given.
            At the boundary mask must have values True.
    values: array of function values: only values at points where mask=True are relevant
    damping: The extension satisfies (-Delta + damping)u =0, so only for the default 
                damping = 0 the extension is harmonic. 
                damping >0 may be used to create initial guesses for synthetic data generated with damping=0.       
output:
        An array u which coincides with values at {mask=True}, and an approximation of a 
        harmonic function on {mask=False} which is globally continuous
"""
def harmonic_extension(mask, values, damping = 0):
    u = values.flatten()
    G = np.where(mask,0,1) # boolean to integer
    k_int = np.nonzero(G)  # integer coordinates of interior points
    k_ext = np.nonzero(1-G) # integer coordinates of exterior points
    G[k_int] = 1+np.arange(len(k_int[0]))

    G1 = G.flatten()
    [m,n] = G.shape
    # Indices of interior points
    p = np.where(G1)[0] # list of numbers of interior points in flattened array
    N = len(p)
    f = np.zeros((N,),dtype=u.dtype) # right hand side of matrix equation 

    # Connect interior points to themselves with 4's.
    i = G1[p]-1
    j = G1[p]-1
    s = (damping/(m*n) + 4.)*np.ones(p.shape)

    # for k = north, east, south, west
    for k in [-1, n, 1, -n]:
        # Possible neighbors in k-th direction
        Q = G1[p+k]
        # Index of points with interior neighbors in k-th direction
        q = np.where(Q)[0]
        q_ext = np.where(Q==0)[0]
        # Connect interior points to neighbors with -1's.
        i = np.concatenate([i, G1[p[q]]-1])
        j = np.concatenate([j,Q[q]-1])
        s = np.concatenate([s,-np.ones(q.shape)])
        i_ext = G1[p[q_ext]]-1
        f[i_ext] = f[i_ext]+u[p[q_ext]+k]
    # sparse matrix with 5 diagonals
    negLap= csc_matrix((s, (i,j)),(N,N))
    u = values.copy()
    u[k_int]=spsolve(negLap,f)
    return u

############################# simulated solutions and setups for testing inversion methods

def load_simulated_g(filename):
    mat = loadmat(filename)
    g_map = mat['pm']['g_map'][0][0]
    mask = mat['pm']['mask'][0][0]
    mask_binary = mat['pm']['mask_binary'][0][0].astype(dtype=bool)
    px_size = mat['pm']['px_sizes'][0][0]
    px_size = px_size * 1e-9 #convert nm to m
    return g_map, mask, mask_binary, px_size

def setup_simulated_g(g_is_complex=True,using_g_squared_measurement=True, parallel=True,list_of_filters=None,N=7):
    filename = r"./data/FresnelPinemMap_obj_javier_2.mat"
    g_map, mask, mask_binary, px_size = load_simulated_g(filename)
    fov = tuple(x*px_size for x in mask.shape)
    lambda_electron = 2.51e-12
    defocus = 900e-6
    fresnelNumber = np.prod(fov)/(defocus * lambda_electron)
    # Uniform grid
    N1,N2 = mask.shape
    A_Psi0_Multiplier = mask.astype(complex)
    boundary_mask = np.zeros_like(mask_binary)
    boundary_mask[0,:]=True; boundary_mask[-1,:]=True
    boundary_mask[:,0]=True; boundary_mask[:,-1]=True
    mask_binary_bd = mask_binary & ~boundary_mask

    grid = UniformGrid(np.linspace(0, 1, N1, endpoint=False),
                           np.linspace(0, 1, N2, endpoint=False))
    if g_is_complex:
        op = complex_PINEM_g_to_data(grid, fresnelNumber,mask_binary_bd,A_Psi0_Multiplier, 
            list_of_filters = list_of_filters,
            N=N, 
            parallel=parallel
        )
        if using_g_squared_measurement:
            op2 = Ptw_Multiplication(grid,1.0-mask_binary) * SquaredModulus(grid.complex_space())
            op = Vector_of_operators([op2, op])
        return op, grid, g_map, g_map, mask_binary_bd, ~boundary_mask
    else:
        op = PINEM_g_to_data(grid, fresnelNumber, mask_binary_bd, A_Psi0_Multiplier, 
                    list_of_filters = list_of_filters,
                    N=N, 
                    parallel=parallel
                    )
        exact_solution = op.domain.join(np.abs(g_map),
                                        np.unwrap(np.angle(g_map.T)).T)
        if using_g_squared_measurement:
            op2 = Ptw_Multiplication(grid,1.0-mask_binary) * SquaredModulus(grid.real_space()) * ForgetSecond(grid,grid)
            op = Vector_of_operators([op2, op])

        return op, grid, exact_solution, g_map, mask_binary_bd, ~boundary_mask

def setup_synthetic_g(g_is_complex=True, using_g_squared_measurement=True, parallel=True):
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
    mask_p = (abs(Xco+0.2) <= 0.2) & (abs(Yco) <= 0.4)
    mask_p = mask_p | (abs((Xco-0.35)*(Xco-0.35)+(Yco-0.35)*(Yco-0.35)) <= 0.01)

    mask = (abs(Xco)<=0.8) & (abs(Yco)<=0.8)
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
    g_map_phase = harmonic_extension(~mask | mask_p, g_map_phase*mask)
    g_map_amp = harmonic_extension(~mask | mask_p, g_map_amp*mask)
    g_map = g_map_amp * np.exp(1j * g_map_phase)
    # g_map /= 10*abs(g_map).max()
    # g_map += ones_like(g_map)
    g_map = g_map.astype(complex)*mask

    if g_is_complex:
        op = complex_PINEM_g_to_data(grid, fresnelNumber, mask,
                                     A_Psi0_Multiplier, N=N, parallel=parallel)
        if using_g_squared_measurement:
            op2 = SquaredModulus(grid.complex_space())
            op = Vector_of_operators([op2, op])
    else:
        op = PINEM_g_to_data(grid, fresnelNumber, mask, A_Psi0_Multiplier, N=N, parallel=True)
        if using_g_squared_measurement:
            op2 = Ptw_Multiplication(grid,1.0-mask_p) * SquaredModulus(grid.real_space()) * ForgetSecond(grid,grid)
            op = Vector_of_operators([op2, op])

    if g_is_complex:
        exact_solution = g_map
    else:
        exact_solution = op.domain.join(np.abs(g_map), np.angle(g_map))
    return op, grid, exact_solution, g_map, mask & ~mask_p, mask_p

def main():
    ################################ set parameters 

    # intermediate results will be written to file names starting with output_filename
    output_filename = r'./PINEM_tests/test'
    # If none, the initial guess is either zero (for synthetic g) or constructed from initial 
    # partial knowledge of |g| and phase(g)
    # Otherwise, if it is an output of a previous inversion method, this reconstruction is used as initial guess
    # init_guess_filename = r'./PINEM_tests/reco_phase_param_full17.mat'
    init_guess_filename = None #r'./PINEM_tests/reco_phase_param19.mat'
    # Name of file containing experimental data. 
    # If this is not None, data (i.e. the right hand side of the operator equation) will be loaded from this file rather 
    # then generated by simulation, i.e. application of the forward operator to g and adding noise.
    experimental_data_filename = None 
    # If true, a true solution g from simulations will be used to test inversion methods, otherwise a picture
    g_is_from_simulation = True                        
    # If true, the unknown g will be parameterized as a usual complex-valued grid function, 
    # otherwise by its modulus and phase: (ex_amp,ex_phase)|-> ex_amp * exp(1j*ex_phase) = g
    g_is_complex = False
    # use additional measurement of |g|^2 from EELS as part of the forward operator
    using_g_squared_measurement = False
    # If true, abs(g) is assumed to be known on the nanotip.
    # If g_is_complex==False, this is imposed as contraint, otherwise it is used as initial guess. 
    abs_g_is_partly_known = True
    assert g_is_complex or (not abs_g_is_partly_known or not using_g_squared_measurement)
    # If using_polynomial_basis_for_phase==True, the phase will be parameterized by 
    # a tensor-product polynomial basis of degrees pol_degrees in x and y directions. 
    # Starting with small polynomial degrees helps to avoid getting trapped in local minimima.
    # Only relevant if g_is_complex==False.
    using_polynomial_basis_for_phase = False
    pol_degrees = (20,7)
    # total number of counts for gain and loss data
    total_nr_counts = 1e10
    # total number of counts for EELS data
    total_nr_counts_EELS = 1e12
    # use log(|g|) instead of |g| for darkness in phase plots of g and g_rec. Makes phase visible everywhere
    plot_log_g = True

    if g_is_from_simulation:
        op, grid, exact_solution, g_map, mask_a, mask_p \
            = setup_simulated_g(g_is_complex=g_is_complex, 
                                using_g_squared_measurement = using_g_squared_measurement,
                                list_of_filters= [ [1,3,5,7],[-1,-3,-5,-7]],
                                #N=8,
                                parallel=True
                                )
        # uncomment this if amplitude of g assumed to be known everywhere
        # mask_a = np.full(mask_a.shape,False,dtype=bool)
        sobolev_index = 1
        if g_is_complex:
            regpar = 1e-4; regpar_step = 0.8
        else:
            sobolev_index_phase = 3; sobolev_index_ampl = 1; regpar = 1e-8;
            #sobolev_index_phase = 1; regpar = 1e-3;
            regpar_step = 0.8
    else:
        op, grid, exact_solution, g_map, mask_a, mask_p \
            = setup_synthetic_g(g_is_complex=g_is_complex, 
                                using_g_squared_measurement = using_g_squared_measurement)
        if abs_g_is_partly_known:
            regpar = 1e-6 
        else:
            regpar = 5e-4
        sobolev_index = 1
        regpar_step = 1/3

    ################################################   initialize forward operator
    if g_is_complex:
        # Hdomain = Sobolev(grid.complex_space(), index=sobolev_index)
        Hdomain = Hm0_domain(mask_p, dtype=complex, index = sobolev_index)
        g0 = harmonic_extension(1-mask_p,g_map)
        proj = CoordinateProjection(grid.complex_space(),mask_p)
        projection = InnerShift(proj,g0)
        extension = OuterShift(proj.adjoint,g0)
    else:
        if abs_g_is_partly_known:
            if mask_a.any():
                prior_ampl = np.abs(g_map) # harmonic_extension(~mask_a,np.abs(g_map),damping =400)
                ampl_proj = 1.*CoordinateProjection(grid, mask_a)
                ampl_domain = Hm0_domain(mask_a, index = sobolev_index_ampl)
            else: # amplitude is known everywhere
                prior_ampl = np.abs(g_map)
                ampl_proj = Zero(grid)
                ampl_domain = L2(grid) # irrelevant, only needed formally
        else:
            prior_ampl = harmonic_extension(~mask_p,np.abs(g_map),damping =0)
            ampl_proj = CoordinateProjection(grid, mask_p)
            ampl_domain = Hm0_domain(mask_p, index = sobolev_index_ampl)
        ampl_projection = InnerShift(ampl_proj,prior_ampl)
        ampl_extension = OuterShift(ampl_proj.adjoint,prior_ampl)

        if using_polynomial_basis_for_phase:
            coeff_grid = UniformGrid(np.arange(pol_degrees[0]),np.arange(pol_degrees[1]))
            phase_domain = L2(coeff_grid)         
            phase_extension =  LegendreBasis(coeff_grid,grid)
            phase_projection = phase_extension.adjoint
        else:
            # outer boundary values of phase must also be fixed for use of Sobolev norm
            prior_phase = harmonic_extension(~mask_p,np.unwrap(np.angle(g_map.T)).T,damping =0)
            phase_proj = CoordinateProjection(grid, mask_p)
            phase_projection = InnerShift(phase_proj,prior_phase)
            phase_extension = OuterShift(phase_proj.adjoint,prior_phase)
            weight = (prior_ampl/np.max(prior_ampl))*g_map.shape[0]**2
            #weight = np.ones_like(g_map.real)*g_map.shape[0]**2
            #weight[mask_p] = 0  # impose Neumann conditions at outer boundaries
            phase_domain = Hm0_domain(mask_p, index = sobolev_index_phase)# weight = weight)

        Hdomain = ampl_domain + phase_domain
        projection =  opDirectSum(ampl_projection, phase_projection)
        extension = opDirectSum(ampl_extension,phase_extension)

    # as extension is also needed for plotting, a copy is required to avoid errors 
    # on the use of revoked copies 
    extension2 = deepcopy(extension)
    op_ext = op * extension2
 
    ############################### compute synthetic data or load experimental data from file
    flat_codomain = DirectSum(*op.codomain.summands, flatten=True)
    if experimental_data_filename:
        raise NotImplementedError
    else:
        if using_g_squared_measurement:
            exact_data = op(exact_solution)
            #exact_data = op_ext(projection(exact_solution))
            exact_data_comp = op.codomain.split(exact_data)
            if isfinite(total_nr_counts_EELS):
                scal_EELS = np.sum(exact_data_comp[0])/total_nr_counts_EELS
                noisy_data0 = scal_EELS * np.random.poisson(exact_data_comp[0]/scal_EELS)
            else:
                noisy_data0 = exact_data_comp[0] 
                scal_EELS = 1
            if isfinite(total_nr_counts):
                scal = np.sum(exact_data_comp[1])/total_nr_counts
                noisy_data1 = scal*np.random.poisson(exact_data_comp[1]/scal)
            else:
                noisy_data1 = exact_data_comp[1]
                scal = 1
            data = op.codomain.join(noisy_data0, noisy_data1)
        else:
            exact_data = op(exact_solution)
            if isfinite(total_nr_counts):
                scal = np.sum(exact_data)/total_nr_counts
                data = scal*np.random.poisson(exact_data/scal)
            else:
                data = exact_data
                scal =1
            data_comp = op.codomain.split(data)

    ################################## define Hilbert space setting
    if using_g_squared_measurement:
        Hcodomain0 = L2(grid, weights = 1/(scal_EELS**2 + scal_EELS*data_comp[0]))
        data_comp = flat_codomain.split(data)
        Hcodomain1 = L2(grid, weights=1/(scal**2 + scal*data_comp[1]))
        for j in range(2, len(flat_codomain)):
            Hcodomain1 = Hcodomain1 + L2(grid, weights=1/(scal**2 +scal*data_comp[j]))
        Hcodomain = hilbert.DirectSum(Hcodomain0, Hcodomain1)
    else:
        # define codomain Gram matrix based on observed data to approximate log-likelihood
        Hcodomain = L2(grid, weights=1/(scal**2+scal*data_comp[0])) 
        for j in range(1, len(data_comp)):
            Hcodomain = Hcodomain + L2(grid, weights=1/(scal**2 + scal*data_comp[j]))
    setting = HilbertSpaceSetting(op=op_ext, Hdomain=Hdomain, Hcodomain=Hcodomain)

    ##################### define initial guess
    if init_guess_filename:
        assert not using_polynomial_basis_for_phase
        mat = loadmat(init_guess_filename)
        init_ampl = mat['reco_amp']
        init_phase = mat['reco_phase']
        if g_is_complex:
            init_vec = init_ampl * np.exp(1j * init_phase)
        else:
            init_vec = op.domain.join(init_ampl, init_phase)
    else:
        if g_is_from_simulation:
            angle = np.deg2rad(10)
            X, Y = np.meshgrid(np.linspace(0, 1, np.size(mask_a, 1)), np.linspace(0, 1, np.size(mask_a, 0)))
            init_phase = -1 * (np.sin(angle)*X + np.cos(angle)*Y) * 2*np.pi * 3 + 2.5
        else: 
            init_phase = np.zeros_like(g_map,dtype=float)
        if g_is_complex:
            if abs_g_is_partly_known:
                g0 = abs(g_map) * np.exp(1j*init_phase)
                init_vec = harmonic_extension(~mask_a,(1-mask_a) * g0)
            else:
                init_vec = grid.complex_space().ones() * mask_a
        else:
            if using_polynomial_basis_for_phase:
                T = phase_extension.asLinearOperator() # Legendre basis
                init_phase_coeff_flat = lsq_linear(T,grid.flatten(init_phase)).x
                init_phase_coeff = phase_extension.domain.fromflat(init_phase_coeff_flat)
                init_vec_proj = op_ext.domain.join(ampl_projection(prior_ampl),init_phase_coeff)
            else:
                init_vec = op.domain.join(prior_ampl,init_phase)
    if not using_polynomial_basis_for_phase:
        init_vec_proj = projection(init_vec)

    ############################### initialize regularization method and stopping rule
   
    if using_g_squared_measurement:
        data_comp = flat_codomain.split(data)
        nr_data = len(data_comp)
        sqrtdata = flat_codomain.join(*[np.sqrt(scal_EELS*data_comp[0]),
            *[np.sqrt(scal*data_comp[k]) for k in range(1,nr_data)]])
    else:
        sqrtdata = np.sqrt(scal*data) 
    stoprule = (
        rules.Discrepancy(
            setting.Hcodomain.norm,
            data,
            noiselevel= setting.Hcodomain.norm(sqrtdata),
            tau=1
        ) +
        rules.CountIterations(max_iterations=1,while_type=True) 
    )
    solver = NewtonCG(
        setting, data, init=init_vec_proj,
        cgmaxit=10, rho=0.95
    )
#    solver = IrgnmCG(
#        setting, data, init=init_vec_proj,
#        regpar=regpar, regpar_step=regpar_step,
#        inner_it_logging_level=logging.INFO
#        )
        
    ############################## routines for reconstruction error evaluation and for plotting 
    def reconstruction_error(_exact, _reconstruction):
        def fnorm(arr):
            return norm(arr[:])

        nonlocal g_is_complex
        nonlocal mask_a
        nonlocal mask_p
        if g_is_complex:
            reco_error1 = fnorm((np.abs(_reconstruction)-np.abs(_exact)))/fnorm(_exact)
            ex_phase = np.unwrap(np.angle(_exact.T)).T
            reco_phase = np.unwrap(np.angle(_reconstruction.T)).T
            reco_error2 = fnorm(np.exp(1j*reco_phase)-np.exp(1j*ex_phase))/np.sqrt(np.prod(_exact.shape))
            reco_error3 = fnorm(_reconstruction-_exact)/fnorm(_exact)
        else:
            ex_amp, ex_phase = op.domain.split(_exact)
            reco_amp, reco_phase = op.domain.split(_reconstruction)
            reco_error1 = fnorm(reco_amp-ex_amp)/fnorm(ex_amp)
            #reco_error2 = norm(mask_p*(reco_phase-ex_phase))/fnorm(mask_p*ex_phase)
            reco_error2 = fnorm(np.exp(1j*reco_phase)-np.exp(1j*ex_phase))/np.sqrt(np.prod(ex_phase.shape))
            # fnorm((reco_phase-ex_phase))/fnorm(ex_phase)
            reco_error3 = fnorm(reco_amp*np.exp(1j*reco_phase) -ex_amp*np.exp(1j*ex_phase))/fnorm(ex_amp)
        return reco_error1, reco_error2, reco_error3

    def plot_reco(fig1,fig2,reco_amp,reco_phase,reco_data_comp,g_map,ex_data_comp,Newton_step):
        plotdata = []
        plotdata.append({'pos': (1, 0), 'data': reco_amp.T,
                        'title': '|g_rec| it.{}'.format(Newton_step)})
        plotdata.append({'pos': (2, 0), 'data': reco_amp.T-np.abs(g_map.T),
                            'title': 'Error ||g|-|g_rec||  it.{}'.format(Newton_step)})
        if plot_log_g:
            plotdata.append({'pos': (1, 1), 'data': complex_to_rgb_log(reco_amp.T*np.exp(1j*reco_phase.T)),
                            'title': 'log(g_rec) it.{}'.format(Newton_step)})
        else:
            plotdata.append({'pos': (1, 1), 'data': complex_to_rgb(reco_amp.T*np.exp(1j*reco_phase.T)),
                            'title': 'g_rec it.{}'.format(Newton_step)})

        plotdata.append({'pos': (2, 1), 'data': np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                            'title': 'error |g_rec-g| it.{}'.format(Newton_step)})
        #plotdata.append({'pos': (2, 1), 'data': np.abs(np.exp(1j*reco_phase.T)-(g_map/(np.abs(g_map)+1e-16)).T),
        #                    'title': '|g_rec/|g_rec|-g/|g|| it.{}'.format(Newton_step)})
        fig1.plot(plotdata)

        plotdata = [{'pos': (1, j), 'data': reco_data_comp[j].T,
                        'title':'rec. data it.{}'.format(Newton_step)}
                    for j in range(nr_data)]
        for j in range(nr_data):
            plotdata.append({'pos': (2, j), 'data': reco_data_comp[j].T-ex_data_comp[j].T,
                            'title': 'diff'})
        fig2.plot(plotdata)

    def plot_write_safe(reco,reco_data,fig1,fig2,axs3, Newton_step,
        do_plottings=True, stats =None,output_filename = 'test'
        ):

        ereco = extension(reco)

        # fix unidentified constant global phase 
        if g_is_complex:
            ex_phase = np.unwrap(np.angle(exact_solution.T)).T
            reco_phase = np.unwrap(np.angle(ereco.T)).T
            reco_amp = np.abs(ereco)
        else:
            _, ex_phase = op.domain.split(exact_solution)
            reco_amp, reco_phase = op.domain.split(ereco)
        phase_correction = np.median(ex_phase[~mask_a])-np.median(reco_phase[~mask_a]) 
        reco_phase += phase_correction
        if g_is_complex:
            ereco = reco_amp * np.exp(1j*reco_phase)
        else:
            ereco = op.domain.join(reco_amp,reco_phase)
            
        reco_error1, reco_error2, reco_error3 = reconstruction_error(exact_solution, ereco)
        logging.info('rel. reconstruction errors step {}: modulus: {:1.4f}, phase: {:1.4f}, norm: {:1.4f}'.format(
            Newton_step, reco_error1, reco_error2, reco_error3))
        stats['ampl_err'].append(reco_error1)
        stats['phase_err'].append(reco_error2)
        stats['complex_err'].append(reco_error3)
        stats['residuals'].append(setting.Hcodomain.norm(reco_data-exact_data))
        if hasattr(solver, "nr_inner_its") and callable(solver.nr_inner_its):
            stats['nr_inner_steps'].append(solver.nr_inner_its())

        savemat(output_filename+'{}.mat'.format(Newton_step),
            {'reco_amp':reco_amp, 'reco_phase': reco_phase, 'stats': stats}
        )

        reco_data_comp = flat_codomain.split(reco_data)
        ex_data_comp = flat_codomain.split(data)

        if do_plottings:
            plot_reco(fig1,fig2,reco_amp,reco_phase,reco_data_comp,g_map,ex_data_comp,Newton_step)
            axs3[0].cla()
            
            axs3[0].plot(stats['ampl_err']/stats['ampl_err'][0], label='amplitude error')
            axs3[0].plot(stats['phase_err']/stats['phase_err'][0], label='phase error')
            axs3[0].plot(stats['complex_err']/stats['complex_err'][0], label='complex error')
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

    ################ plot and evaluate exact solution and exact data and initial error
    if not g_is_complex:
        ex_abs, ex_phase = op.domain.split(exact_solution)
    else:
        ex_abs = np.abs(g_map)
        ex_phase = np.unwrap(np.angle(g_map.T)).T
    fig1 = imshow_fig(3, 2)
    if plot_log_g:
        plotdata1 = [{'pos': (0, 0), 'data': np.abs(g_map.T), 'title': 'exact |g|'},
                    {'pos': (0, 1), 'data': complex_to_rgb_log(g_map.T), 'title': 'log(g) with phase'}]
    else:
        plotdata1 = [{'pos': (0, 0), 'data': np.abs(g_map.T), 'title': 'exact |g|'},
                    {'pos': (0, 1), 'data': complex_to_rgb(g_map.T), 'title': 'g with phase'}]
    fig1.plot(plotdata1)

    data_comp = flat_codomain.split(data)
    nr_data = len(data_comp)
    fig2 = imshow_fig(3, nr_data)
    plot_data2 = [{'pos': (0, j), 'data': data_comp[j].T, 'title':'sim. data'}
                    for j in range(nr_data)]
    if using_g_squared_measurement and nr_data ==3:
        plot_data2[0]['title'] = 'sim. ampl'
        plot_data2[1]['title'] = 'sim. gain'
        plot_data2[2]['title'] = 'sim. loss'    
    if not using_g_squared_measurement and nr_data ==2:
        plot_data2[0]['title'] = 'sim. gain'
        plot_data2[1]['title'] = 'sim. loss'  
    fig2.plot(plot_data2)

    if hasattr(solver, "nr_inner_its"):
        fig3, axs3 = plt.subplots(3, 1, sharex=False, sharey=False)
    else:
        fig3, axs3 = plt.subplots(2, 1, sharex=False, sharey=False)

    stats = {'ampl_err': [], 'phase_err': [], 'complex_err': [], 'residuals': [], 'nr_inner_steps': []}
    plot_write_safe(solver.x,solver.y,fig1,fig2,axs3,0,
        do_plottings=True,stats=stats,output_filename= output_filename)

    ########################################## perform inversion
    for Newton_step, [reco, reco_data] in enumerate(solver.while_(stoprule),1):
        plot_write_safe(solver.x,solver.y,fig1,fig2,axs3,Newton_step,
            do_plottings=~stoprule.triggered,stats=stats,output_filename= output_filename)
 
    plt.show(block=True)
    reco = stoprule.x
    reco_data = stoprule.y
    # np.save('PINEM_tests/reco',reco)


if __name__ == '__main__':
    main()
