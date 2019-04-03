from . import LinearOperator, NonlinearOperator, OperatorImplementation, Params
from itreg.util import instantiate
from itreg.spaces import L2


import numpy as np

import scipy.sparse.linalg as scsla
from scipy import *

from . import Scattering2D
from . import Scattering3D
from . import MediumScatteringBase
#from Scattering2D import Scattering2D
#from MediumScatteringBase import ComplexDataToData, ComplexDataToData_derivative, ComplexDataToData_adjoint, SolveTwoGrid, AdjointSolveTwoGrid, LippmannSchwingerOp, AdjointLippmannSchwingerOp

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

    def __init__(self, domain, dim, range=None):
#        range=domain or range
        op_name='acousticMedBase'
        syntheticdata_flag=True
        noiselevel=0

        NrTwoGridIterations=3
        kappa=3
        rho=1
        #sobo_index=domain.parameters_domain.sobo_index
        xdag_rcoeff=1
        xdag_icoeff=0.5
        #init_guess
        initguess_fct='zerofct'

        amplitude_data=False
        intensity=1
        ampl_vector_length=1

        gmres_restart=10
        gmres_tol=1e-14
        gmres_maxit=100
        verbose=1
        sobo_index=0
        #N=domain.parameters_domain.N
            
        #ind_support=domain.parameters_domain.ind_support

        #Fourierweights=domain.parameters_domain.Fourierweights
        
        #(Xdim, Ydim, inc_directions, meas_directions, xdag, init_guess, N, N_coarse, Ninc, Nmeas, K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse)=Scattering2D.Scattering2D(domain, amplitude_data, rho, kappa, ampl_vector_length)    
        if dim==3:
            (Xdim, Ydim, inc_directions, meas_directions, xdag, init_guess, N, N_coarse, Ninc, Nmeas, K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse)=Scattering3D.Scattering3D(domain, amplitude_data, rho, kappa, ampl_vector_length)
        if dim==2: 
            (Xdim, Ydim, inc_directions, meas_directions, xdag, init_guess, N, N_coarse, Ninc, Nmeas, K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse)=Scattering2D.Scattering2D(domain, amplitude_data, rho, kappa, ampl_vector_length)
        range=L2(np.linspace(0, 2*Ninc*Nmeas, 2*Ninc*Nmeas))
        super().__init__(Params(domain, range,
            dim=dim, op_name=op_name, syntheticdata_flag=syntheticdata_flag, noiselevel=noiselevel, Xdim=Xdim, Ydim=Ydim,
            NrTwoGridIterations=NrTwoGridIterations, kappa=kappa, rho=rho, inc_directions=inc_directions,
            meas_directions=meas_directions, sobo_index=domain.parameters_domain.sobo_index, xdag=xdag, xdag_rcoeff=xdag_rcoeff,
            xdag_icoeff=xdag_icoeff, init_guess=init_guess, initguess_fct=initguess_fct, amplitude_data=amplitude_data,
            intensity=intensity, ampl_vector_length=ampl_vector_length, gmres_restart=gmres_restart,
            gmres_tol=gmres_tol, gmres_maxit=gmres_maxit, verbose=verbose, N=domain.parameters_domain.N, N_coarse=N_coarse, Ninc=Ninc, 
            Nmeas=Nmeas, ind_support=domain.parameters_domain.ind_support, K_hat=K_hat, K_hat_coarse=K_hat_coarse, farfieldMatrix=farfieldMatrix,
            incMatrix=incMatrix, Fourierweights=domain.parameters_domain.Fourierweights, dual_x_coarse=dual_x_coarse, dual_y_coarse=dual_y_coarse,
            dual_z_coarse=dual_z_coarse))







    @instantiate
    class  operator(OperatorImplementation):
        def eval(self, params, x, data, differentiate, **kwargs):
            data.contrast=1j*np.zeros((np.prod(params.N)))
            np.put(data.contrast, params.ind_support, x)
            if params.N_coarse:
                Nfac = np.prod(params.N_coarse)/np.prod(params.N)
                contrast_hat = np.fft.fftn(np.reshape(data.contrast,params.N, order='F'))
                if params.dim==2:
                    data.contrast_coarse = Nfac*np.fft.ifftn(contrast_hat[params.dual_x_coarse.astype(int),:][:,params.dual_y_coarse.astype(int)])
                if params.dim==3:
                    data.contrast_coarse = Nfac*np.fft.ifftn(contrast_hat[params.dual_x_coarse.astype(int),:, :][:,params.dual_y_coarse.astype(int),:][:, :, params.dual_z_coarse.astype(int)])
            u_total=1j*np.zeros((params.Xdim, params.Ninc))
            u_inf=1j*np.zeros((params.Nmeas, params.Ninc))
            #The following needs to be in parallel code
            for j in range(0, params.Ninc): #loop over incident waves
                #solve Lippmann-Schwinger-equation v+a(k*v)=a*u_inc
                #for the unknown v = a u_total. The Fourier coefficients of the periodic
                #convolution kernel k are precomputed
                rhs=complex(0,1)*np.zeros(np.prod(params.N))
                np.put(rhs, params.ind_support, x*params.incMatrix[:, j])
                rhs=np.reshape(rhs, params.N, order='F')
            

                if not params.N_coarse:
                    LippmannSchwingerOperator=scsla.LinearOperator((np.prod(params.N), np.prod(params.N)), matvec=(lambda x: MediumScatteringBase.LippmannSchwingerOp(params, data, x)))
                    [v,flag] = scsla.gmres(LippmannSchwingerOperator, rhs.reshape(np.prod(params.N), order='F'), restart=params.gmres_restart, tol=params.gmres_tol, maxiter=params.gmres_maxit)
                    if not flag==0:
                        print('Convergence problem')
                    else:
                        print('Gmres converged')
                else:
                    rhs=np.fft.fftn(rhs)
                    v=MediumScatteringBase.SolveTwoGrid(params, data, rhs)
                #compute far field pattern
                u_inf[:,j]=np.dot(params.farfieldMatrix, v.reshape(np.prod(params.N), order='F')[params.ind_support])

                #The total field can be recovered from v in a stable manner by the formula
                #u_total=ui-k*v
                if differentiate:
                    aux=np.fft.ifftn(params.K_hat*np.fft.fftn(np.reshape(v, params.N, order='F')))
                    u_total[:,j]=params.incMatrix[:, j]-np.take(aux.T, params.ind_support)

            if differentiate:
                data.u_total=u_total.copy()
            return MediumScatteringBase.ComplexDataToData(params, data, u_inf.reshape(params.Nmeas*params.Ninc, order='F'))



    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, x, data, **kwargs):
            d_u_inf=1j*np.zeros((params.Nmeas, params.Ninc))
            #need to be computed parallel
            for j in range(0, params.Ninc):
                rhs=complex(0,1)*np.zeros(np.prod(params.N))
                np.put(rhs, params.ind_support, x*data.u_total[:, j])
                rhs=rhs.reshape(params.N, order='F')

                if not params.N_coarse:
                    LippmannSchwingerOperator=scsla.LinearOperator((np.prod(params.N), np.prod(params.N)), matvec=(lambda x: MediumScatteringBase.LippmannSchwingerOp(params, data, x)))
                    [v,flag] = scsla.gmres(LippmannSchwingerOperator, rhs.reshape(np.prod(params.N), order='F'), restart=params.gmres_restart, tol=params.gmres_tol, maxiter=params.gmres_maxit)
                    if not flag==0:
                        print('Convergence problem')
                else:
                    rhs=np.fft.fftn(rhs)
                    v=MediumScatteringBase.SolveTwoGrid(params, data, rhs)
                d_u_inf[:,j]=np.dot(params.farfieldMatrix, v.reshape(np.prod(params.N), order='F')[params.ind_support])
            return MediumScatteringBase.ComplexDataToData_derivative(params, data, d_u_inf.reshape(params.Nmeas*params.Ninc, order='F'))
        
        def adjoint(self, params, x, data, **kwargs):
            d_contrast=np.zeros(params.Xdim)
            d_u_inf_mat=np.reshape(MediumScatteringBase.ComplexDataToData_adjoint(params, data, x), (params.Nmeas, params.Ninc), order='F')
            for j in range(0, params.Ninc):
                rhs=complex(0,1)*np.zeros(np.prod(params.N))
                np.put(rhs, params.ind_support, np.dot(params.farfieldMatrix.transpose(), d_u_inf_mat[:, j]))
                rhs=np.reshape(rhs, params.N, order='F')
                
                if not params.N_coarse:
                    AdjointLippmannSchwingerOperator=scsla.LinearOperator((np.prod(params.N), np.prod(params.N)), matvec=(lambda x: MediumScatteringBase.AdjointLippmannSchwingerOp(params, data, x)))
                    [v,flag] = scsla.gmres(AdjointLippmannSchwingerOperator, rhs.reshape(np.prod(params.N), order='F'), restart=params.gmres_restart, tol=params.gmres_tol, maxiter=params.gmres_maxit)
                    if not flag==0:
                        print('Convergence problem')
                else:
                    v=MediumScatteringBase.AdjointSolveTwoGrid(params, data, rhs)
                    v=np.fft.ifftn(v)

                d_contrast=d_contrast + np.conj(data.u_total[:,j]) * np.take(v.reshape(np.prod(params.N), order='F'), params.ind_support)
            return d_contrast



            








    