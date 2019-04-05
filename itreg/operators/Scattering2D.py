import numpy as np  
import scipy.special as scsp
import matplotlib.pyplot as plt

class Scattering2D:
    def __init__(self, domain, amplitude_data, rho, kappa, ampl_vector_length):        
        #define default values and merge with parameters given
        self.N=domain.parameters_domain.N
        self.N_coarse=(32, 32)
        self.Ninc=16
        self.Nmeas=16
        self.inc_directions=np.array([np.cos(2*np.pi*np.linspace(1, self.Ninc, self.Ninc)/self.Ninc), np.sin(2*np.pi*np.linspace(1, self.Ninc, self.Ninc)/self.Ninc)])
        self.meas_directions=np.array([np.cos(2*np.pi*np.linspace(1, self.Nmeas, self.Nmeas)/self.Nmeas), np.sin(2*np.pi*np.linspace(1, self.Nmeas, self.Nmeas)/self.Nmeas)])
        
        #parameter for plotting

        #check input parameter for consistency
        if self.N_coarse:
            if np.max(np.asarray(self.N_coarse)/np.asarray(self.N))>1:
                print('Error: Coarse Grid not coarser than fine grid')
        if np.mod(self.N[0], 2)==1 or np.mod(self.N[1], 2)==1:
            print('Error: N must be even')
        if amplitude_data:  
            ampl_vector_length=2
            Y_weight=Y_weight**2
            
        Y_weight=(2*np.pi)**2/(self.Ninc*self.Nmeas)
        #x_coo=(4*rho/N[0])*np.arange(-N[0]/2, (N[0]-1)/2, 1)
        #y_coo=(4*rho/N[1])*np.arange(-N[1]/2, (N[1]-1)/2, 1)
        #Y, X=np.meshgrid(y_coo, x_coo)
        X=domain.parameters_domain.X
        Y=domain.parameters_domain.Y
        #ind_support=np.asarray(np.reshape(X, np.prod(N), order='F')**2+np.reshape(Y, np.prod(N), order='F')**2<=rho**2).nonzero()

        #compute default value for true contrast
        #if syntheticdata_flag ==True:
        #feval lines
        #for first:
        self.init_guess=0
        self.xdag=0

        K_hat=self.ComputeFKConvolutionKernel(self.N, rho, kappa)
        if self.N_coarse:
            K_hat_coarse=self.ComputeFKConvolutionKernel(self.N_coarse, rho, kappa)
            x_coo_coarse=(4*rho/self.N[0])*np.arange(self.N_coarse[0]/2, (self.N_coarse[0]-1)/2, 1)
            y_coo_coarse=(4*rho/self.N[1])*np.arange(self.N_coarse[1]/2, (self.N_coarse[1]-1)/2, 1)
            Y_coarse, X_coarse=np.meshgrid(y_coo_coarse, x_coo_coarse)
            dual_x_coarse=np.append(np.linspace(0, int(self.N_coarse[0]/2-1), num=int(self.N_coarse[0]/2)), np.linspace(int(self.N[0]-self.N_coarse[0]/2), int(self.N[0]-1), num=int(self.N_coarse[0]/2)))
            dual_y_coarse=np.append(np.linspace(0, int(self.N_coarse[1]/2-1), num=int(self.N_coarse[1]/2)), np.linspace(int(self.N[1]-self.N_coarse[1]/2), int(self.N[1]-1), num=int(self.N_coarse[1]/2)))
            dual_z_coarse=1
            
        #Fourier weights defining product in X for applyGramX and applyGramX_inv
        #Fourierweights=np.fft.fftshift((1+(4*rho/N[0])**2*X*X+(4*rho/N[1])**2*Y*Y)**(sobo_index))
        #set up far field matrix
        farfieldMatrix=1j*np.zeros((self.Nmeas, np.size(domain.parameters_domain.ind_support)))
        for j in range(0, self.Nmeas):
            aux=(kappa**2*(4*rho)**2/np.prod(self.N))*np.exp(-np.complex(0,1)*kappa*self.meas_directions[0, j]*X-np.complex(0,1)*kappa*self.meas_directions[1, j] * Y)
            farfieldMatrix[j,:]=aux.reshape(np.prod(self.N), order='F')[domain.parameters_domain.ind_support]
        incMatrix=1j*np.zeros((np.size(domain.parameters_domain.ind_support), self.Ninc))
        for j in range(0, self.Ninc):
            aux=np.exp(-np.complex(0,1)*kappa*self.inc_directions[0, j]*X-np.complex(0,1)*kappa*self.inc_directions[1, j]*Y)
            incMatrix[:,j]=aux.reshape(np.prod(self.N), order='F')[domain.parameters_domain.ind_support]

        #setup interface
        self.Xdim=np.size(domain.parameters_domain.ind_support)
        self.Ydim=2*self.Ninc*self.Nmeas/ampl_vector_length
        self.prec=Scattering_prec(K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse)
        self.plotting=plotting_prop(domain, rho)
        return

    def ComputeFKConvolutionKernel(self, N, rho, kappa):
        #compute Fourier coefficients of convolution kernel
        jval1=np.arange(-N[0]/2, N[0]/2)
        jval2=np.arange(-N[1]/2, N[1]/2)
        J2, J1=np.meshgrid(jval2, jval1)
        piabsJ=np.pi*np.sqrt(J1*J1+J2*J2)
        R=2*rho*kappa
        aux=piabsJ*scsp.jv(1, piabsJ)*scsp.hankel1(0, R)-(R*scsp.hankel1(1, R))*scsp.jv(0, piabsJ)
        K_hat=(0.5/R)*(R**2/(piabsJ*piabsJ-R**2))*(1+0.5*np.pi*aux*np.complex(0,1))
        K_hat[int(N[0]/2), int(N[1]/2)]=-1/(2*R)+0.25*np.pi*np.complex(0,1)*scsp.hankel1(1, R)
        special_ind=np.where(piabsJ==R)[0]
        np.put(K_hat, special_ind, 0.125*np.pi*R*np.complex(0,1)*(scsp.jv(1, R)*scsp.hankel1(0,R)+scsp.jv(1, R)*scsp.hankel1(1,R)))
        return 2*R*np.fft.fftshift(K_hat)


    def plotX(self, grid, contrast):
        aux=np.ones(self.N)
        np.put(aux, grid.ind_support, 1-contrast)
        x_coo=grid.coords[0,:]
        y_coo=grid.coords[1,:]
        plt.contourf(x_coo[self.plotting.xplot_ind], y_coo[self.plotting.yplot_ind], np.real(aux[self.plotting.xplot_ind,:][:,self.plotting.yplot_ind]))
#set colorbar axis
#set title     
        plt.show()

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
    def __init__(self, grid, rho):
        self.xplot_ind=np.where(np.abs(grid.coords[0,:])<=rho)[0]
        self.yplot_ind=np.where(np.abs(grid.coords[1,:])<=rho)[0]
        return
            

