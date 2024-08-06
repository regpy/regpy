import numpy as np
#from x_ray_phase_contrast import Corr, _build_fresnel_2, fresnel_prop, Ptw_Multiplication, Mat, Theta, Tau, Proj
from regpy.operators import FourierTransform

#defines the cut-off function, e.g., the bump function
def cutoff_function_2d(x, y):
    xx,yy = np.meshgrid(x,y)
    mask=((1-xx**2-yy**2)>0)
    func=np.zeros(xx.shape)
    func[mask]=(np.exp(1)*np.exp(-1/(1-xx**2-yy**2)))[mask]
    return func

def _create_Vcov(N, N_b, create_type='fourier_domain', sigma=None, xsample=None, ysample=None, coherence_len=None, grid=None):    
    # create random vector V in spatial and frequency domain
    if create_type=='spatial':
        # create random vector V ( we suppose that K=VV* and V has small rank N_b)
        vec=np.zeros((N_b, N, N), dtype=complex)
        for i in range(0, N_b):
            vec[i, :, :]=1/np.sqrt(2)*(np.random.randn(N**2).reshape(N, N)+complex(0,1)*np.random.randn(N**2).reshape(N, N))
        #Multiply with a rapid decaying function
        vec=vec*np.exp(-(xsample)**2/(2*sigma**2)).reshape(N, 1)*np.exp(-(ysample)**2/(2*sigma**2)).reshape(1, N)
        #perform singular value decomposition ( a process to find PCA)
        U, S, V=np.linalg.svd(vec.reshape(N_b, N**2), full_matrices=False)
        # Vcov defines the orthogonal eigenvector (i.e., the principal components)))
        Vcov=V.T.conj()*S
    elif create_type=='Fresnelprop':
        col=fresnel_prop(grid, coherence_len**2)
        conv=np.zeros((N_b, N, N), dtype=complex)
        for i in range(0, N_b):    
            fjsq=np.random.randn(N**2).reshape(N, N)
            conv[i, :, :]=col(fjsq)
            # The eigenvectors are being orthonormalized    
        U, S, V=np.linalg.svd(conv.reshape(N_b, N**2), full_matrices=False)
        Vcov=V.T.conj()*S
        
    elif create_type=='fourier_random':
        
        fourier=FourierTransform(grid, centered=True)
        #Multiply with the Gaussian function in Fourier domain,
        func_freq=np.exp(-np.linalg.norm(fourier.codomain.coords, axis=0)**2/sigma**2)
    
        # perform the cut-off function  
        vec_cut=cutoff_function_2d(xsample,ysample)
        # defines the vector V in the spatial domain after performing inverse Fourier transform
        vec_cutted=np.zeros((N_b, N, N), dtype=complex)
        for i in range(0, N_b):
            vec_cutted[i, :, :]=fourier.adjoint(fourier.domain.randn()*func_freq)*vec_cut
        # perform SVD procedure    
        U, S, V=np.linalg.svd(vec_cutted.reshape(N_b, N**2), full_matrices=False)
        Vcov=V.T.conj()*S
        
    else:
        raise ValueError('No method specified')
        
    return Vcov