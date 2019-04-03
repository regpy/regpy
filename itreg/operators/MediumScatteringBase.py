#This file contains base functions


import numpy as np
import scipy.sparse.linalg as scsla
from functools import partial
    

def SolveTwoGrid(params, data, rhs):
    xhat_coarse=params.dual_x_coarse.copy()
    yhat_coarse=params.dual_y_coarse.copy()
    zhat_coarse=params.dual_z_coarse
    v=1j*np.zeros(rhs.shape)
    if params.verbose>=3: print('residuals two-grid its: ')
    for counter in range(0, params.NrTwoGridIterations):
        if counter>0:
            if params.dim==2:
                rhs_coarse = np.fft.fftn(data.contrast_coarse*np.fft.ifftn(params.K_hat_coarse*v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]))
                if params.verbose>=3: 
                    vold=v
                v = rhs- np.fft.fftn(np.reshape(data.contrast,params.N, order='F')*np.fft.ifftn(params.K_hat*v))
                if params.verbose>=3:
                    print(np.linalg.norm(v[:]-vold[:]))
                rhs_coarse = v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)] + rhs_coarse
            if params.dim==3:
                rhs_coarse = np.fft.fftn(data.contrast_coarse*np.fft.ifftn(params.K_hat_coarse*v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]))
                if params.verbose>=3: 
                    vold=v
                v = rhs- np.fft.fftn(np.reshape(data.contrast,params.N, order='F')*np.fft.ifftn(params.K_hat*v))
                if params.verbose>=3:
                    print(np.linalg.norm(v[:]-vold[:]))
                rhs_coarse = v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)] + rhs_coarse
            
        else:
            if params.dim==2:
                rhs_coarse=rhs[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]
            if params.dim==3:
                rhs_coarse=rhs[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]

        
        LippmannSchwingerOperatorCoarse=scsla.LinearOperator((np.prod(params.N_coarse), np.prod(params.N_coarse)), matvec=(partial(LippmannSchwingerOpCoarse, params=params, data=data)))
        [v_coarse,flag] = scsla.gmres(LippmannSchwingerOperatorCoarse, np.reshape(rhs_coarse, np.prod(params.N_coarse), order='F'), restart=params.gmres_restart, tol=params.gmres_tol, maxiter=params.gmres_maxit)
        if not flag==0:
            print('warning! Convergence problem in GMRES on coarse grid: Flag ', flag)

        if params.dim==2:
            for x in range(0, np.size(xhat_coarse)):
                for y in range(0, np.size(yhat_coarse)):
                    v[int(xhat_coarse[x]), int(yhat_coarse[y])]=np.reshape(v_coarse, params.N_coarse, order='F')[int(x), int(y)]
        if params.dim==3:
            for x in range(0, np.size(xhat_coarse)):
                for y in range(0, np.size(yhat_coarse)):
                    for z in range(0, np.size(zhat_coarse)):
                        v[int(xhat_coarse[x]), int(yhat_coarse[y]), int(zhat_coarse[z])]=np.reshape(v_coarse, params.N_coarse, order='F')[int(x), int(y), int(z)]
     


    if params.verbose>=3: print('\n')
    v=np.fft.ifftn(v)
    return v


def AdjointSolveTwoGrid(params, data, rhs):
    xhat_coarse=params.dual_x_coarse
    yhat_coarse=params.dual_y_coarse
    zhat_coarse=params.dual_z_coarse
    rhs=np.fft.fftn(rhs)
    v=1j*np.zeros(rhs.shape)
    for counter in range(0, params.NrTwoGridIterations):
        if counter>0:
            if params.dim==2:
                rhs_coarse = params.K_hat_coarse*np.fft.fftn(data.contrast_coarse*np.fft.ifftn(v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]))
                v=rhs-params.K_hat*np.fft.fftn(np.reshape(data.contrast, params.N, order='F')*np.fft.ifftn(v))
                rhs_coarse = v[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)] + rhs_coarse
            if params.dim==3:
                rhs_coarse = params.K_hat_coarse*np.fft.fftn(data.contrast_coarse*np.fft.ifftn(v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]))
                v=rhs-params.K_hat*np.fft.fftn(np.reshape(data.contrast, params.N, order='F')*np.fft.ifftn(v))
                rhs_coarse = v[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)] + rhs_coarse
        else:
            if params.dim==2:
                rhs_coarse = rhs[xhat_coarse.astype(int),:][:,yhat_coarse.astype(int)]
            if params.dim==3:
                rhs_coarse = rhs[xhat_coarse.astype(int),:, :][:,yhat_coarse.astype(int), :][:, :, zhat_coarse.astype(int)]
        AdjointLippmannSchwingerOperatorCoarse=scsla.LinearOperator((np.prod(params.N_coarse), np.prod(params.N_coarse)), matvec=(lambda x: AdjointLippmannSchwingerOpCoarse(params, data, x)))
        [v_coarse,flag] =scsla.gmres(AdjointLippmannSchwingerOperatorCoarse, rhs_coarse.reshape(np.prod(params.N_coarse), order='F'), restart=params.gmres_restart, tol=params.gmres_tol, maxiter=params.gmres_maxit)
        if not flag==0:
            print('warning! Convergence problem in GMRES on coarse grid: Flag ',flag)
        if params.dim==2:
            for x in range(0, np.size(xhat_coarse)):
                for y in range(0, np.size(yhat_coarse)):
                    v[int(xhat_coarse[x]), int(yhat_coarse[y])]=np.reshape(v_coarse, params.N_coarse, order='F')[x, y]
        if params.dim==3:
             for x in range(0, np.size(xhat_coarse)):
                for y in range(0, np.size(yhat_coarse)):
                    for z in range(0, np.size(zhat_coarse)):
                        v[int(xhat_coarse[x]), int(yhat_coarse[y]), int(zhat_coarse[z])]=np.reshape(v_coarse, params.N_coarse, order='F')[x, y, z]
    return v
        

def ComplexDataToData(params, data, complexdata):
    n = np.size(complexdata)
    res = np.zeros(2*n,)
    np.put(res, np.arange(0, 2*n, 2), np.real(complexdata))
    np.put(res, np.arange(1, 2*n, 2), np.imag(complexdata))
    if params.ampl_vector_length>1:
        data.real_data = np.reshape(res, (int(params.ampl_vector_length),int(2*n/params.ampl_vector_length)), order='F')
        res = np.sum(data.real_data*data.real_data,1)
    return res
        

def ComplexDataToData_derivative(params, data, complex_h):
    n = np.size(complex_h)
    res = np.zeros(2*n,)
    np.put(res, np.arange(0, 2*n, 2), np.real(complex_h))
    np.put(res, np.arange(1, 2*n, 2), np.imag(complex_h))
    if params.ampl_vector_length>1:
        data.real_data=np.reshape(res,(int(params.ampl_vector_length),int(2*n/params.ampl_vector_length)), order='F')
        res = 2*np.sum(data.real_data*np.reshape(res, (int(params.ampl_vector_length),int(2*n/params.ampl_vector_length)), order='F'),1)
    return res     
        

def ComplexDataToData_adjoint(params, data, g):
    m=np.size(g)
    d=params.ampl_vector_length
    if params.ampl_vector_length>1:
        g = np.reshape(2*np.tile(g,(d,1))*data.real_data, (m*d,1), order='F')
    return g[np.arange(0, m*d, 2)]+1j*g[np.arange(1, m*d, 2)]


def LippmannSchwingerOp(v, params, data):
    #operator v|-> v+ a.*(K*v) of the Lippmann-Schwinger equation
    # in space domain on fine grid
    v_mat=np.zeros(params.N)
    v_mat = np.reshape(v,params.N, order='F')
    v_mat = np.fft.ifftn(params.K_hat * np.fft.fftn(v_mat))
    return v + data.contrast * v_mat.reshape(np.prod(params.N), order='F')

def AdjointLippmannSchwingerOp(params, data, v):
    #adjoint of LippmannSchwingerOp
    aux=np.zeros(params.N)
    aux = np.reshape(np.conj(data.contrast)*v,params.N, order='F')
    aux = np.fft.ifftn(np.conj(params.K_hat) * np.fft.fftn(aux))
    return v + aux.reshape(np.prod(params.N), order='F')


def LippmannSchwingerOpCoarse(v, params, data):
    v_mat=np.zeros(params.N_coarse)
    v_mat = np.reshape(v,params.N_coarse, order='F')
    v_mat = np.fft.fftn(data.contrast_coarse* np.fft.ifftn(params.K_hat_coarse*v_mat))
    return v+ np.reshape(v_mat, np.prod(params.N_coarse), order='F') 


def AdjointLippmannSchwingerOpCoarse(params, data, v):
    v_mat=np.zeros(params.N_coarse)
    v_mat = np.reshape(v, params.N_coarse, order='F')
    v_mat = np.conj(params.K_hat_coarse)* np.fft.fftn(np.conj(data.contrast_coarse)*np.fft.ifftn(v_mat))
    return v +  np.reshape(v_mat, np.prod(params.N_coarse), order='F')
        