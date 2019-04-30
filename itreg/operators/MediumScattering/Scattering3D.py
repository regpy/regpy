# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:15:01 2019

@author: Hendrik Müller
"""

import numpy as np  
import scipy.special as scsp
import numpy.matlib
import matplotlib.pyplot as plt

class Scattering3D:
    def __init__(self, domain, amplitude, kappa):        
        #define default values and merge with parameters given
        self.N=domain.parameters_domain.N
        self.N_coarse=(4, 4, 4)
    
        m_i = 1 #Ninc = 2*m_i^2;
    
        m_m = 1 # Nmeas = 2*m_m^2;
        phi = np.pi*np.ones((2*m_i, 1))*np.reshape(np.linspace(1, m_i, num=m_i)/m_i, (1, m_i))
        phi = np.transpose(phi)
        theta = numpy.matlib.repmat(np.pi*np.linspace(1, 2*m_i, 2*m_i)/m_i, 1, m_i)
        self.inc_directions = np.append(np.append(np.cos(phi).reshape(2*m_i**2)*np.cos(theta), np.cos(phi).reshape(2*m_i**2)*np.sin(theta)), np.sin(phi).reshape(2*m_i**2)).reshape((3,2*m_i**2))
        phi = np.pi*np.ones((2*m_m,1)).dot(np.reshape(np.linspace(1, m_m, m_m)/m_m, (1, m_m)))
        phi = np.transpose(phi)
        theta = numpy.matlib.repmat(np.pi*np.linspace(1, 2*m_m, 2*m_m)/m_m, 1,  m_m)
        self.meas_directions = np.append(np.append(np.cos(phi).reshape(2*m_m**2)*np.cos(theta), np.cos(phi).reshape(2*m_m**2)*np.sin(theta)), np.sin(phi).reshape(2*m_m**2)).reshape((3, 2*m_m**2))
    
      
    # parameters determining the simulated true contrast. The refractive index
    # is n = 1 + xdag_rcoeff*xdag_real + i*xdag_icoeff*xdag_imag.
    #xdag_rcoeff = 1
    #xdag_icoeff = 0.5
    #xdag_real ='cross' 
    #xdag_imag = 'shiftedGaussian'
    # initial guess
    #initguess_fct = 'zerofct'
            
    # parameters for plotting
    # in what the letters encode the following
    #    'r' real part
    #    'i' imaginary part
    #    'a' absolute value
    # where encodes the corresponding figure numbers
    #plotWhat.contrast = 'riRI';
    #plotWhat.where_contrast = (1);
    #plotWhat.contrast_n1 = 2;
    #plotWhat.contrast_n2 = 2;
    #plotWhat.farfield = 'r';
    #plotWhat.where_farfield = (2);
    #def.plotWhat = plotWhat;
    #F = set_parameters(F,prbl,def);
    
    ## compute precomputable quantities

        if self.N_coarse:
            if np.max(np.asarray(self.N_coarse)/np.asarray(self.N))>1:
                print('Error: Coarse Grid not coarser than fine grid')

        if np.mod(self.N[0],2)==1 or np.mod(self.N[1],2)==1 or np.mod(self.N[2], 2)==1:
            print('error: Nx, Ny and  must be even!')

            
    #number of incident directions
        self.Ninc = np.size(self.inc_directions[1, :])
    # number of measurement points
        self.Nmeas = np.size(self.meas_directions[1, :])
    # x, y and z coordinates of grid points in computational domain
        x_coo = (4*domain.parameters_domain.rho/self.N[0]*np.arange(-self.N[0]/2, (self.N[0]-1)/2, step=1))
        y_coo = (4*domain.parameters_domain.rho/self.N[1]*np.arange(-self.N[1]/2, (self.N[1]-1)/2, step=1))
        z_coo = (4*domain.parameters_domain.rho/self.N[2]*np.arange(-self.N[2]/2, (self.N[1]-1)/2, step=1))
        [X, Y, Z] = np.meshgrid(x_coo, y_coo, z_coo)
    #ind_support=np.asarray(np.reshape(X, np.prod(N), order='F')**2+np.reshape(Y, np.prod(N), order='F')**2+np.reshape(Z, np.prod(N))<=rho**2).nonzero()
    
    # compute default value for true contrast
    #if F.syntheticdata_flag == true
    #xdag = -xdag_rcoeff*feval(F.xdag_real, X(F.ind_support), Y(F.ind_support), Z(F.ind_support)) ...
    #    - 1i*F.xdag_icoeff*feval(F.xdag_imag, X(F.ind_support), Y(F.ind_support), Z(F.ind_support));

    # initial guess
    # F.init_guess = feval(F.initguess_fct, X(F.ind_support), Y(F.ind_support), Z(F.ind_support));
        self.xdag=0
        self.init_guess=0
    
        K_hat = self.ComputeFKConvolutionKernel(self.N, domain.parameters_domain.rho, kappa)
    
    
        if self.N_coarse:
            K_hat_coarse = self.ComputeFKConvolutionKernel(self.N_coarse, domain.parameters_domain.rho, kappa)
            x_coo_coarse = (4*domain.parameters_domain.rho/self.N[0])*np.arange(-self.N_coarse[0]/2, (self.N_coarse[0]-1)/2, step=1)
            y_coo_coarse = (4*domain.parameters_domain.rho/self.N[1])*np.arange(-self.N_coarse[1]/2, (self.N_coarse[1]-1)/2, step=1)
            z_coo_coarse = (4*domain.parameters_domain.rho/self.N[2])*np.arange(-self.N_coarse[2]/2, (self.N_coarse[2]-1)/2, step=1)
            [X_coarse,Y_coarse,Z_coarse] = np.meshgrid(x_coo_coarse,y_coo_coarse,z_coo_coarse)
            #ind_support_coarse = np.asarray(np.reshape(X_coarse, np.prod(self.N_coarse), order='F')**2+np.reshape(Y_coarse, np.prod(self.N_coarse), order='F')**2+np.reshape(Z_coarse, np.prod(self.N_coarse))<=domain.parameters_domain.rho**2).nonzero()
            dual_x_coarse=np.append(np.linspace(0, int(self.N_coarse[0]/2-1), num=int(self.N_coarse[0]/2)), np.linspace(int(self.N[0]-self.N_coarse[0]/2), int(self.N[0]-1), num=int(self.N_coarse[0]/2)))
            dual_y_coarse=np.append(np.linspace(0, int(self.N_coarse[1]/2-1), num=int(self.N_coarse[1]/2)), np.linspace(int(self.N[1]-self.N_coarse[1]/2), int(self.N[1]-1), num=int(self.N_coarse[1]/2)))
            dual_z_coarse=np.append(np.linspace(0, int(self.N_coarse[2]/2-1), num=int(self.N_coarse[2]/2)), np.linspace(int(self.N[2]-self.N_coarse[2]/2), int(self.N[2]-1), num=int(self.N_coarse[2]/2)))
            
    #Fourier weights defining inner product in X for applyGramX and applyGramX_inv
    #Fourierweights = np.fft.fftshift((1+ (4*rho/N[0])**2*X**2 + (4*rho/N[1])**2*Y**2 + (4*rho/N[2])**2*Z**2)**sobo_index)
    # set up far field matrix
        farfieldMatrix = 1j*np.zeros((self.Nmeas, np.size(domain.parameters_domain.ind_support)))
        for j in range(0, self.Nmeas):
            aux=(kappa**2*(4*domain.parameters_domain.rho)**3/np.prod(self.N))/(4* np.pi)*np.exp(-np.complex(0,1)*kappa*self.meas_directions[0, j]*X-np.complex(0,1)*kappa*self.meas_directions[1, j] * Y, np.complex(0,1)*kappa*self.meas_directions[2, j] * Z)
            farfieldMatrix[j,:]=aux.reshape(np.prod(self.N), order='F')[domain.parameters_domain.ind_support]
        
       
        
        incMatrix = 1j*np.zeros((np.size(domain.parameters_domain.ind_support), self.Ninc))
        for j in range(0, self.Ninc):
            aux=np.exp(-np.complex(0,1)*kappa*self.inc_directions[0, j]*X-np.complex(0,1)*kappa*self.inc_directions[1, j]*Y-np.complex(0,1)*kappa*self.inc_directions[2, j]*Z)
            incMatrix[:,j]=aux.reshape(np.prod(self.N), order='F')[domain.parameters_domain.ind_support]
    
#Check here whether we have to transpose the incMatrix
        
#    xplot_ind=np.where(np.abs(x_coo)<=rho)[0]
#    yplot_ind=np.where(np.abs(y_coo)<=rho)[0]
#    zplot_ind=np.where(np.abs(z_coo)<=rho)[0]
#    xplot_slice = np.asarray([0, rho])
#    yplot_slice = np.asarray([0, rho])
#    zplot_slice = np.asarray([-rho, 0])
            
##setup interface
        self.Xdim = np.size(domain.parameters_domain.ind_support)
        self.Ydim = 2*self.Ninc*self.Nmeas/amplitude.ampl_vector_length
        self.prec=Scattering_prec(K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse)
        self.plotting=plotting_prop(domain)        
        return         
    
   
    
    def ComputeFKConvolutionKernel(self, N, rho, kappa):
        # compute Fourier coefficients of convolution kernel
        # compute Fourier coefficients of convolution kernel, see [Vainikko2000]
        jval1=np.arange(-N[0]/2, N[0]/2)
        jval2=np.arange(-N[1]/2, N[1]/2)
        jval3=np.arange(-N[2]/2, N[2]/2)
        [J1, J2, J3] = np.meshgrid(jval1, jval2, jval3)
        piabsJ = np.pi*np.sqrt(J1**2+J2**2+J3**2)
        R = 2*rho*kappa
        aux=piabsJ*scsp.jv(1, piabsJ)*scsp.hankel1(0, R)-(R*scsp.hankel1(1, R))*scsp.jv(0, piabsJ)
        K_hat=(0.5/R)*(R**2/(piabsJ*piabsJ-R**2))*(1+0.5*np.pi*aux*np.complex(0,1))
        K_hat[int(N[0]/2), int(N[1]/2), int(N[2]/2)]=-1/(2*R)+0.25*np.pi*np.complex(0,1)*scsp.hankel1(1, R)
        special_ind=np.where(piabsJ==R)[0]
        np.put(K_hat, special_ind, 0.125*np.pi*R*np.complex(0,1)*(scsp.jv(1, R)*scsp.hankel1(0,R)+scsp.jv(1, R)*scsp.hankel1(1,R)))
        return 2*R*np.fft.fftshift(K_hat)
    
    def plotX(self, grid, contrast):
        return
    
class Scattering_prec:
    def __init__(self, K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse):
        self.K_hat=K_hat
        self.K_hat_coarse=K_hat_coarse
        self.farfieldMatrix=farfieldMatrix
        self.incMatrix=incMatrix
        self.dual_x_coarse=dual_x_coarse
        self.dual_y_coarse=dual_y_coarse
        self.dual_z_coarse=dual_z_coarse
        return
    
class plotting_prop:
    def __init__(self, domain):
        self.xplot_ind=np.where(np.abs(domain.coords[0,:])<=domain.parameters_domain.rho)[0]
        self.yplot_ind=np.where(np.abs(domain.coords[1,:])<=domain.parameters_domain.rho)[0]
        self.zplot_ind=np.where(np.abs(domain.coords[2,:])<=domain.parameters_domain.rho)[0]        
        return
    
   