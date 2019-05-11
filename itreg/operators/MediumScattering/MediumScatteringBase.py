import numpy as np
import scipy.sparse.linalg as scsla
from functools import partial


class MediumScatteringBase:
    def SolveTwoGrid(params, data, rhs):
        xhat_coarse=params.scattering.prec.dual_x_coarse.copy()
        yhat_coarse=params.scattering.prec.dual_y_coarse.copy()
        zhat_coarse=params.scattering.prec.dual_z_coarse
        v=1j*np.zeros(rhs.shape)
        if params.printing.verbose>=3: print('residuals two-grid its: ')
        for counter in range(0, params.NrTwoGridIterations):
            if counter>0:
                if params.domain.dim==2:
                    rhs_coarse = np.fft.fftn(data.contrast_coarse*np.fft.ifftn(params.scattering.prec.K_hat_coarse*v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]))
                    if params.printing.verbose>=3:
                        vold=v
                    v = rhs- np.fft.fftn(np.reshape(data.contrast,params.scattering.N, order='F')*np.fft.ifftn(params.scattering.prec.K_hat*v))
                    if params.printing.verbose>=3:
                        print(np.linalg.norm(v[:]-vold[:]))
                    rhs_coarse = v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)] + rhs_coarse
                if params.domain.dim==3:
                    rhs_coarse = np.fft.fftn(data.contrast_coarse*np.fft.ifftn(params.scattering.prec.K_hat_coarse*v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]))
                    if params.printing.verbose>=3:
                        vold=v
                    v = rhs- np.fft.fftn(np.reshape(data.contrast,params.scattering.N, order='F')*np.fft.ifftn(params.scattering.prec.K_hat*v))
                    if params.printing.verbose>=3:
                        print(np.linalg.norm(v[:]-vold[:]))
                    rhs_coarse = v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)] + rhs_coarse

            else:
                if params.domain.dim==2:
                    rhs_coarse=rhs[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]
                if params.domain.dim==3:
                    rhs_coarse=rhs[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]


            LippmannSchwingerOperatorCoarse=scsla.LinearOperator((np.prod(params.scattering.N_coarse), np.prod(params.scattering.N_coarse)), matvec=(partial(MediumScatteringBase.LippmannSchwingerOpCoarse, params=params, data=data)))
            [v_coarse,flag] = scsla.gmres(LippmannSchwingerOperatorCoarse, np.reshape(rhs_coarse, np.prod(params.scattering.N_coarse), order='F'), restart=params.gmres_prop.gmres_restart, tol=params.gmres_prop.gmres_tol, maxiter=params.gmres_prop.gmres_maxit)
            if not flag==0:
                print('warning! Convergence problem in GMRES on coarse grid: Flag ', flag)

            if params.domain.dim==2:
                for x in range(0, np.size(xhat_coarse)):
                    for y in range(0, np.size(yhat_coarse)):
                        v[int(xhat_coarse[x]), int(yhat_coarse[y])]=np.reshape(v_coarse, params.scattering.N_coarse, order='F')[int(x), int(y)]
            if params.domain.dim==3:
                for x in range(0, np.size(xhat_coarse)):
                    for y in range(0, np.size(yhat_coarse)):
                        for z in range(0, np.size(zhat_coarse)):
                            v[int(xhat_coarse[x]), int(yhat_coarse[y]), int(zhat_coarse[z])]=np.reshape(v_coarse, params.scattering.N_coarse, order='F')[int(x), int(y), int(z)]



        if params.printing.verbose>=3: print('\n')
        v=np.fft.ifftn(v)
        return v


    def AdjointSolveTwoGrid(params, data, rhs):
        xhat_coarse=params.scattering.prec.dual_x_coarse
        yhat_coarse=params.scattering.prec.dual_y_coarse
        zhat_coarse=params.scattering.prec.dual_z_coarse
        rhs=np.fft.fftn(rhs)
        v=1j*np.zeros(rhs.shape)
        for counter in range(0, params.NrTwoGridIterations):
            if counter>0:
                if params.domain.dim==2:
                    rhs_coarse = params.scattering.prec.K_hat_coarse*np.fft.fftn(data.contrast_coarse*np.fft.ifftn(v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]))
                    v=rhs-params.scattering.prec.K_hat*np.fft.fftn(np.reshape(data.contrast, params.scattering.N, order='F')*np.fft.ifftn(v))
                    rhs_coarse = v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)] + rhs_coarse
                if params.domain.dim==3:
                    rhs_coarse = params.scattering.prec.K_hat_coarse*np.fft.fftn(data.contrast_coarse*np.fft.ifftn(v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]))
                    v=rhs-params.scattering.prec.K_hat*np.fft.fftn(np.reshape(data.contrast, params.scattering.N, order='F')*np.fft.ifftn(v))
                    rhs_coarse = v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)] + rhs_coarse
            else:
                if params.domain.dim==2:
                    rhs_coarse = rhs[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]
                if params.domain.dim==3:
                    rhs_coarse = rhs[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]
            AdjointLippmannSchwingerOperatorCoarse=scsla.LinearOperator((np.prod(params.scattering.N_coarse), np.prod(params.scattering.N_coarse)), matvec=(lambda x: MediumScatteringBase.AdjointLippmannSchwingerOpCoarse(params, data, x)))
            [v_coarse,flag] =scsla.gmres(AdjointLippmannSchwingerOperatorCoarse, rhs_coarse.reshape(np.prod(params.scattering.N_coarse), order='F'), restart=params.gmres_prop.gmres_restart, tol=params.gmres_prop.gmres_tol, maxiter=params.gmres_prop.gmres_maxit)
            if not flag==0:
                print('warning! Convergence problem in GMRES on coarse grid: Flag ',flag)
            if params.domain.dim==2:
                for x in range(0, np.size(xhat_coarse)):
                    for y in range(0, np.size(yhat_coarse)):
                        v[int(xhat_coarse[x]), int(yhat_coarse[y])]=np.reshape(v_coarse, params.scattering.N_coarse, order='F')[x, y]
            if params.domain.dim==3:
                for x in range(0, np.size(xhat_coarse)):
                    for y in range(0, np.size(yhat_coarse)):
                        for z in range(0, np.size(zhat_coarse)):
                            v[int(xhat_coarse[x]), int(yhat_coarse[y]), int(zhat_coarse[z])]=np.reshape(v_coarse, params.scattering.N_coarse, order='F')[x, y, z]
        return v
