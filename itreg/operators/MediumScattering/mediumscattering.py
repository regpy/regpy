from itreg.operators import LinearOperator, NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate
from itreg.spaces import L2
from itreg.grids import Square_1D

import numpy as np

import scipy.sparse.linalg as scsla
from scipy import *

from .Scattering2D import Scattering2D
from .Scattering3D import Scattering3D
from .MediumScatteringBase import MediumScatteringBase
 

class MediumScattering(NonlinearOperator):
    """
    Implements acoustic scattering problems for an inhomogeneous medium.
    The forward problem is solved by Vainikko's method by Vainikko's fast
    solver of the Lippmann Schwinger equation.
    References:
        T. Hohage: On the numerical solution of a 3D inverse medium scattering
        problem. Inverse Problems, 17:1743-1763, 2001.
    
        G. Vainikko: Fast solvers of the Lippmann-Schwinger equation
        in: Direct and inverse problems of mathematical physics
           edited by R.P.Gilbert, J.Kajiwara, and S.Xu, Kluwer, 2000

    Parameters
    ----------
    NrTwoGridIterations: Number of iterations of two grid solver
    kappa: 
    rho: radius of supported ball
    gmres: properties of gmres solver
    
    The following need to be specified in main:
    coords: coordinates of underlying grid
    sobo_index: index of sobolev space
    
    The following need to be specified in Scattering2D:
    N_coarse: Size of coarser grid
    N_inc: number of incident waves
    N_meas: number of measured waves
    inc_directions: directions of incident plane waves
    meas_directions: directions of measured plane waves
    """

    def __init__(self, domain, range=None):
        gmres=gmres_prop
        gmres.gmres_restart=10
        gmres.gmres_tol=1e-14
        gmres.gmres_maxit=100
        
#        plotting=plotting_prop
#        plotting.xplot_ind=np.linspace(0, len(domain.parameters_domain.x_coo)-1, len(domain.parameters_domain.x_coo), dtype=int)
#        plotting.yplot_ind=np.linspace(0, len(domain.parameters_domain.y_coo)-1, len(domain.parameters_domain.y_coo), dtype=int)
#        range=domain or range
        
        syntheticdata_flag=True
        noiselevel=0

        NrTwoGridIterations=3
        kappa=3
        #rho=1
        #sobo_index=domain.parameters_domain.sobo_index
        xdag_rcoeff=1
        xdag_icoeff=0.5
        #init_guess
        #initguess_fct='zerofct'

        amplitude=amplitude_prop
        amplitude.amplitude_data=False
        amplitude.intensity=1
        amplitude.ampl_vector_length=1

        printing=printing_prop
        printing.verbose=1
        
        
        if domain.dim==3:
            Scattering_prop=Scattering3D(domain, amplitude, kappa)            
        if domain.dim==2: 
            Scattering_prop=Scattering2D(domain, amplitude, kappa)
        range=L2(Square_1D((1, 2*Scattering_prop.Ninc*Scattering_prop.Nmeas), 3, 1))
        super().__init__(Params(domain, range, syntheticdata_flag=syntheticdata_flag, noiselevel=noiselevel, 
            NrTwoGridIterations=NrTwoGridIterations, kappa=kappa,
            xdag_rcoeff=xdag_rcoeff,
            xdag_icoeff=xdag_icoeff, amplitude=amplitude, gmres_prop=gmres, printing=printing,
            scattering=Scattering_prop))







    @instantiate
    class  operator(OperatorImplementation):
        def eval(self, params, x, data, differentiate, **kwargs):
            data.contrast=1j*np.zeros((np.prod(params.scattering.N)))
            np.put(data.contrast, params.domain.parameters_domain.ind_support, x)
            if params.scattering.N_coarse:
                Nfac = np.prod(params.scattering.N_coarse)/np.prod(params.scattering.N)
                contrast_hat = np.fft.fftn(np.reshape(data.contrast,params.scattering.N, order='F'))
                if params.domain.dim==2:
                    data.contrast_coarse = Nfac*np.fft.ifftn(contrast_hat[params.scattering.prec.dual_x_coarse.astype(int),:][:,params.scattering.prec.dual_y_coarse.astype(int)])
                if params.domain.dim==3:
                    data.contrast_coarse = Nfac*np.fft.ifftn(contrast_hat[params.scattering.prec.dual_x_coarse.astype(int),:, :][:,params.scattering.prec.dual_y_coarse.astype(int),:][:, :, params.scattering.prec.dual_z_coarse.astype(int)])
            u_total=1j*np.zeros((params.scattering.Xdim, params.scattering.Ninc))
            u_inf=1j*np.zeros((params.scattering.Nmeas, params.scattering.Ninc))
            #The following needs to be in parallel code
            for j in range(0, params.scattering.Ninc): #loop over incident waves
                #solve Lippmann-Schwinger-equation v+a(k*v)=a*u_inc
                #for the unknown v = a u_total. The Fourier coefficients of the periodic
                #convolution kernel k are precomputed
                rhs=complex(0,1)*np.zeros(np.prod(params.scattering.N))
                np.put(rhs, params.domain.parameters_domain.ind_support, x*params.scattering.prec.incMatrix[:, j])
                rhs=np.reshape(rhs, params.scattering.N, order='F')
            

                if not params.scattering.N_coarse:
                    LippmannSchwingerOperator=scsla.LinearOperator((np.prod(params.scattering.N), np.prod(params.scattering.N)), matvec=(lambda x: MediumScatteringBase.LippmannSchwingerOp(params, data, x)))
                    [v,flag] = scsla.gmres(LippmannSchwingerOperator, rhs.reshape(np.prod(params.scattering.N), order='F'), restart=params.gmres.gmres_restart, tol=params.gmres.gmres_tol, maxiter=params.gmres.gmres_maxit)
                    if not flag==0:
                        print('Convergence problem')
                    else:
                        print('Gmres converged')
                else:
                    rhs=np.fft.fftn(rhs)
                    v=MediumScatteringBase.SolveTwoGrid(params, data, rhs)
                #compute far field pattern
                u_inf[:,j]=np.dot(params.scattering.prec.farfieldMatrix, v.reshape(np.prod(params.scattering.N), order='F')[params.domain.parameters_domain.ind_support])

                #The total field can be recovered from v in a stable manner by the formula
                #u_total=ui-k*v
                if differentiate:
                    aux=np.fft.ifftn(params.scattering.prec.K_hat*np.fft.fftn(np.reshape(v, params.scattering.N, order='F')))
                    u_total[:,j]=params.scattering.prec.incMatrix[:, j]-np.take(aux.T, params.domain.parameters_domain.ind_support)

            if differentiate:
                data.u_total=u_total.copy()
            return MediumScatteringBase.ComplexDataToData(params, data, u_inf.reshape(params.scattering.Nmeas*params.scattering.Ninc, order='F'))



    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            d_u_inf=1j*np.zeros((params.scattering.Nmeas, params.scattering.Ninc))
            #need to be computed parallel
            for j in range(0, params.scattering.Ninc):
                rhs=complex(0,1)*np.zeros(np.prod(params.scattering.N))
                np.put(rhs, params.domain.parameters_domain.ind_support, x*data.u_total[:, j])
                rhs=rhs.reshape(params.scattering.N, order='F')

                if not params.scattering.N_coarse:
                    LippmannSchwingerOperator=scsla.LinearOperator((np.prod(params.scattering.N), np.prod(params.scattering.N)), matvec=(lambda x: MediumScatteringBase.LippmannSchwingerOp(params, data, x)))
                    [v,flag] = scsla.gmres(LippmannSchwingerOperator, rhs.reshape(np.prod(params.scattering.N), order='F'), restart=params.gmres.gmres_restart, tol=params.gmres.gmres_tol, maxiter=params.gmres.gmres_maxit)
                    if not flag==0:
                        print('Convergence problem')
                else:
                    rhs=np.fft.fftn(rhs)
                    v=MediumScatteringBase.SolveTwoGrid(params, data, rhs)
                d_u_inf[:,j]=np.dot(params.scattering.prec.farfieldMatrix, v.reshape(np.prod(params.scattering.N), order='F')[params.domain.parameters_domain.ind_support])
            return MediumScatteringBase.ComplexDataToData_derivative(params, data, d_u_inf.reshape(params.scattering.Nmeas*params.scattering.Ninc, order='F'))
        
        def adjoint(self, params, x, data, **kwargs):
            d_contrast=np.zeros(params.scattering.Xdim)
            d_u_inf_mat=np.reshape(MediumScatteringBase.ComplexDataToData_adjoint(params, data, x), (params.scattering.Nmeas, params.scattering.Ninc), order='F')
            for j in range(0, params.scattering.Ninc):
                rhs=complex(0,1)*np.zeros(np.prod(params.scattering.N))
                np.put(rhs, params.domain.parameters_domain.ind_support, np.dot(params.scattering.prec.farfieldMatrix.conj().T, d_u_inf_mat[:, j]))
                rhs=np.reshape(rhs, params.scattering.N, order='F')
                
                if not params.scattering.N_coarse:
                    AdjointLippmannSchwingerOperator=scsla.LinearOperator((np.prod(params.scattering.N), np.prod(params.scattering.N)), matvec=(lambda x: MediumScatteringBase.AdjointLippmannSchwingerOp(params, data, x)))
                    [v,flag] = scsla.gmres(AdjointLippmannSchwingerOperator, rhs.reshape(np.prod(params.scattering.N), order='F'), restart=params.gmres.gmres_restart, tol=params.gmres.gmres_tol, maxiter=params.gmres.gmres_maxit)
                    if not flag==0:
                        print('Convergence problem')
                else:
                    v=MediumScatteringBase.AdjointSolveTwoGrid(params, data, rhs)
                    v=np.fft.ifftn(v)

                d_contrast=d_contrast + np.conj(data.u_total[:,j]) * np.take(v.reshape(np.prod(params.scattering.N), order='F'), params.domain.parameters_domain.ind_support)
            return d_contrast
        
class gmres_prop:
    def __init__(self):
        return
    
class printing_prop:
    def __init__(self):
        return
    
class amplitude_prop:
    def __init__(self):
        return


            








    