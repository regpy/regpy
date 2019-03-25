import numpy as np  
import scipy.special as scsp


def Scattering2D(domain, amplitude_data, rho, kappa, ampl_vector_length):        
    #define default values and merge with parameters given
    op_name='acousticMed2D'
    N=domain.parameters_domain.N
    N_coarse=(32, 32)
    Ninc=1
    Nmeas=2
    inc_directions=np.array([np.cos(2*np.pi*np.linspace(1, Ninc, Ninc)/Ninc), np.sin(2*np.pi*np.linspace(1, Ninc, Ninc)/Ninc)])
    meas_directions=np.array([np.cos(2*np.pi*np.linspace(1, Nmeas, Nmeas)/Nmeas), np.sin(2*np.pi*np.linspace(1, Nmeas, Nmeas)/Nmeas)])
            
    #parameter for plotting

    #check input parameter for consistency
    if N_coarse:
        if np.max(np.asarray(N_coarse)/np.asarray(N))>1:
            print('Error: Coarse Grid not coarser than fine grid')
    if np.mod(N[0], 2)==1 or np.mod(N[1], 2)==1:
        print('Error: N must be even')
    if amplitude_data:  
        ampl_vector_length=2
        Y_weight=Y_weight**2
            
    Y_weight=(2*np.pi)**2/(Ninc*Nmeas)
    #x_coo=(4*rho/N[0])*np.arange(-N[0]/2, (N[0]-1)/2, 1)
    #y_coo=(4*rho/N[1])*np.arange(-N[1]/2, (N[1]-1)/2, 1)
    #Y, X=np.meshgrid(y_coo, x_coo)
    x_coo=domain.parameters_domain.x_coo
    y_coo=domain.parameters_domain.y_coo
    X=domain.parameters_domain.X
    Y=domain.parameters_domain.Y
    #ind_support=np.asarray(np.reshape(X, np.prod(N), order='F')**2+np.reshape(Y, np.prod(N), order='F')**2<=rho**2).nonzero()

    #compute default value for true contrast
    #if syntheticdata_flag ==True:
        #feval lines
    #for first:
    init_guess=0
    xdag=0

    K_hat=ComputeFKConvolutionKernel(N, rho, kappa)
    if N_coarse:
        K_hat_coarse=ComputeFKConvolutionKernel(N_coarse, rho, kappa)
        x_coo_coarse=(4*rho/N[0])*np.arange(N_coarse[0]/2, (N_coarse[0]-1)/2, 1)
        y_coo_coarse=(4*rho/N[1])*np.arange(N_coarse[1]/2, (N_coarse[1]-1)/2, 1)
        Y_coarse, X_coarse=np.meshgrid(y_coo_coarse, x_coo_coarse)
        dual_x_coarse=np.append(np.linspace(0, int(N_coarse[0]/2-1), num=int(N_coarse[0]/2)), np.linspace(int(N[0]-N_coarse[0]/2), int(N[0]-1), num=int(N_coarse[0]/2)))
        dual_y_coarse=np.append(np.linspace(0, int(N_coarse[1]/2-1), num=int(N_coarse[1]/2)), np.linspace(int(N[1]-N_coarse[1]/2), int(N[1]-1), num=int(N_coarse[1]/2)))
        dual_z_coarse=1
            
    #Fourier weights defining product in X for applyGramX and applyGramX_inv
    #Fourierweights=np.fft.fftshift((1+(4*rho/N[0])**2*X*X+(4*rho/N[1])**2*Y*Y)**(sobo_index))
    #set up far field matrix
    farfieldMatrix=1j*np.zeros((Nmeas, np.size(domain.parameters_domain.ind_support)))
    for j in range(0, Nmeas):
        aux=(kappa**2*(4*rho)**2/np.prod(N))*np.exp(-np.complex(0,1)*kappa*meas_directions[0, j]*X-np.complex(0,1)*kappa*meas_directions[1, j] * Y)
        farfieldMatrix[j,:]=aux.reshape(np.prod(N), order='F')[domain.parameters_domain.ind_support]
    incMatrix=1j*np.zeros((np.size(domain.parameters_domain.ind_support), Ninc))
    for j in range(0, Ninc):
        aux=np.exp(-np.complex(0,1)*kappa*inc_directions[0, j]*X-np.complex(0,1)*kappa*inc_directions[1, j]*Y)
        incMatrix[:,j]=aux.reshape(np.prod(N), order='F')[domain.parameters_domain.ind_support]
    xplot_ind=np.where(np.abs(x_coo)<=rho)[0]
    yplot_ind=np.where(np.abs(y_coo)<=rho)[0]

    #setup interface
    Xdim=np.size(domain.parameters_domain.ind_support)
    Ydim=2*Ninc*Nmeas/ampl_vector_length
    return (Xdim, Ydim, inc_directions, meas_directions, xdag, init_guess, N, N_coarse, Ninc, Nmeas, K_hat, K_hat_coarse, farfieldMatrix, incMatrix, dual_x_coarse, dual_y_coarse, dual_z_coarse)

def ComputeFKConvolutionKernel(N, rho, kappa):
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

