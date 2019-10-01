from itreg.operators import NonlinearOperator, Params

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

class ParallelMri(NonlinearOperator):
    def __init__(self,
                 domain,
                 codomain=None,
                nr_coils=20,
                Nx=200,
                Ny=20,
                samplingIndx=None,
                Fourier_weights=None,
                ):

        if samplingIndx is None:
            undersampling = 2
            nrcenterlines =  8
                #create sampling pattern P
            P = np.zeros((Nx,Ny))
            P[:,np.arange(0, Ny, undersampling)] = 1
            P[:,int((Ny-nrcenterlines)/2):int((Ny+nrcenterlines)/2)] = 1
            samplingIndx = np.nonzero(np.reshape(P, (Nx*Ny, 1), order='F'))[0]

        if codomain is None:
            coords_range=np.ones(nr_coils*len(samplingIndx))
            grid_range=User_Defined(coords_range, coords_range.shape)
            codomain=L2(grid_range)


        if Fourier_weights is None:
            Fourier_weights = np.zeros((Nx,Ny))
            #for k in range(0, Nx):
                #for j in range(0, Ny):
                    #d = (k  / Nx - 0.5)**2 + (j  / Ny - 0.5)**2
                    #Fourier_weights[k, j] = (1. + 220. * d)**32

        super().__init__(
            Params(domain,
                codomain,
                samplingIndx=samplingIndx,
                Fourier_weights=Fourier_weights,
                nr_coils=nr_coils,
                Nx=Nx,
                Ny=Ny))

    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, rho_and_coils, data, **kwargs):
            rho_and_coils=rho_and_coils+1j*np.zeros(rho_and_coils.shape)
            nr_coils = params.nr_coils
            samplingIndx = params.samplingIndx
            N = params.Nx*params.Ny

            # current spin density and coil sensitivities are stored for
            # later evaluations of F.derivative and F.adjoint:
            # The derivative will be taken at the point (rho,coils)
            data.rho = np.reshape(rho_and_coils[0:N],(params.Nx,params.Ny), order='F')+1j*np.zeros((params.Nx,params.Ny))
            data.coils = np.reshape(rho_and_coils[N:],(params.Nx,params.Ny,nr_coils), order='F')+1j*np.zeros((params.Nx,params.Ny,nr_coils))

            data_mes = 1j*np.zeros((len(samplingIndx),nr_coils))
            aux=1j*np.zeros((params.Nx, params.Ny))
            #data_mes=1j*np.zeros((N, nr_coils))

            for j in range(0, nr_coils):
                data.coils[:,:,j] = myifft(data.coils[:,:,j])
                aux = myfft(data.rho * data.coils[:,:,j])
                data_mes[:,j] = np.reshape(aux, N, order='F')[samplingIndx]
            return data_mes.reshape(len(samplingIndx)*nr_coils, order='F')



    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, h, data, **kwargs):
            nr_coils = params.nr_coils
            samplingIndx = params.samplingIndx
            N = params.Nx*params.Ny

            d_K = 1j*np.zeros((len(samplingIndx),nr_coils))
            aux=1j*np.zeros((params.Nx, params.Ny))
            d_rho = np.reshape(h[0:N],(params.Nx,params.Ny), order='F')+1j*np.zeros((params.Nx, params.Ny))
            d_coils = np.reshape(h[N:],(params.Nx,params.Ny,nr_coils), order='F')+1j*np.zeros((params.Nx,params.Ny,nr_coils))

            for j in range(0, nr_coils):
                aux = myfft(data.rho * myifft(d_coils[:,:,j]) + d_rho * data.coils[:,:,j])
                d_K[:,j]= np.reshape(aux, N, order='F')[samplingIndx]
            return np.reshape(d_K, len(samplingIndx)*nr_coils, order='F')

        def adjoint(self, params, d_K, data, **kwargs):
            nr_coils = params.nr_coils
            samplingIndx = params.samplingIndx
            Nx = params.Nx
            Ny = params.Ny
            M = len(samplingIndx)

            aux = 1j*np.zeros(Nx*Ny)
            d_rho = 1j*np.zeros((Nx,Ny))
            d_coils = 1j*np.zeros((Nx,Ny,nr_coils))
            aux2=1j*np.zeros((Nx, Ny))
            #print(d_K.shape)
            for j in range(0, nr_coils):
                aux[samplingIndx] =  d_K[j*M:(j+1)*M]
                aux2 = myifft(np.reshape(aux, (Nx, Ny), order='F'))
                d_rho = d_rho + aux2 * np.conjugate(data.coils[:,:,j])
                d_coils[:,:,j] = myfft(aux2 * np.conjugate(data.rho))
            d_rho_and_coils = np.append(np.reshape(d_rho, Nx*Ny, order='F'),np.reshape(d_coils, Nx*Ny*nr_coils, order='F'))
            return d_rho_and_coils

def myfft(rho):
    Nx, Ny=rho.shape
    data=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rho)))/np.sqrt(Nx*Ny)
    return data

def myifft(data):
    Nx, Ny=data.shape
    rho=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data)))*scp.sqrt(Nx*Ny)
    return rho

class plots():
    def __init__(self,
                 op,
                 reco,
                 reco_data,
                 data,
                 exact_solution,
                 Nx=None,
                 Ny=None,
                 nr_coils=None,
                 samplingIndx=None
                 ):

        self.op=op
        self.Nx=Nx or self.op.params.Nx
        self.Ny=Ny or self.op.params.Ny
        self.nr_coils=nr_coils or self.op.params.nr_coils
        self.reco=reco
        self.reco_data=reco_data
        self.data=data
        self.samplingIndx=samplingIndx or self.op.params.samplingIndx
        self.exact_solution=exact_solution



    def plot_samplingindex(self):
        P=np.zeros((self.Nx*self.Ny))
        P[self.samplingIndx]=1
        P=P.reshape(self.Nx, self.Ny, order='F')

        fig = plt.figure(figsize=(8, 8))
        vmax = abs(P).max()
        X, Y=np.meshgrid(np.arange(0, 1, 1/self.Ny), np.arange(0, 1, 1/self.Nx))
        cs=plt.pcolormesh(X, Y, P, cmap=plt.cm.seismic)
        plt.title('sampling Index')
        plt.colorbar()
        plt.show()

    def plot_data(self, coil=None):
        coil=coil or 0
        P_data=np.zeros((self.Nx*self.Ny, self.nr_coils))
        P_data2=np.zeros((self.Nx, self.Ny, self.nr_coils))
        P_data[self.samplingIndx, coil]=self.reco_data[int(coil)*self.samplingIndx.shape[0]:int(coil+1)*self.samplingIndx.shape[0]]
        for i in range(0, self.nr_coils):
            P_data2[:, i]=P_data[:, i].reshape((self.Nx, self.Ny), order='F')

        data_print=np.zeros((self.Nx*self.Ny, self.nr_coils))
        data_print2=np.zeros((self.Nx, self.Ny, self.nr_coils))
        data_print[self.samplingIndx, coil]=self.data[int(coil)*self.samplingIndx.shape[0]:int(coil+1)*self.samplingIndx.shape[0]]
        for i in range(0, self.nr_coils):
            data_print2[:, i]=data_print[:, i].reshape((self.Nx, self.Ny), order='F')

        fig, axs = plt.subplots(1, 2,sharey=True,figsize=(8, 10))
        X, Y=np.meshgrid(np.arange(0, 1, 1/self.Ny), np.arange(0, 1, 1/self.Nx))
        cs=axs[0].contourf(X,Y,P_data2[:, 0],cmap=plt.cm.seismic)
        axs[1].contourf(X,Y,data_print2[:, 0].reshape((self.Nx, self.Ny), order='F'),cmap=plt.cm.seismic)
        fig.colorbar(cs)
        axs[0].set_title('reco_data')
        axs[1].set_title('data')
        plt.show()


    def plot_rho(self):
        N=self.Nx*self.Ny
        rho = np.reshape(self.reco[0:N],(self.Nx,self.Ny), order='F')

        fig, axs = plt.subplots(1, 2,sharey=True,figsize=(8, 10))
        X, Y=np.meshgrid(np.arange(0, 1, 1/self.Ny), np.arange(0, 1, 1/self.Nx))
        cs=axs[0].contourf(X,Y,rho,cmap=plt.cm.seismic)
        axs[1].contourf(X,Y,self.exact_solution[0:N].reshape((self.Nx, self.Ny), order='F'),cmap=plt.cm.seismic)
        fig.colorbar(cs)
        axs[0].set_title('reco')
        axs[1].set_title('exact solution')
        plt.show()
