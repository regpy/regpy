import numpy as np
import scipy.linalg as scla
from scipy.special import j0, j1, y0, y1

def setup_iop_data(bd, kappa):
    #Computes data needed to set up the boundary integral matrices to avoid repeated computations.

    dimension = len(bd.z.shape)
    dim=np.max([np.size(bd.z, l) for l in range(0, dimension)])

    #Compute matrix of distances of grid points.
    VEC_1, VEC_2=np.meshgrid(bd.z[0, :], bd.z[0, :], indexing='ij')
    t1=(VEC_1-VEC_2).reshape(dim**2)
    
    VEC_3, VEC_4=np.meshgrid(bd.z[1, :], bd.z[1, :], indexing='ij')
    t2=(VEC_3-VEC_4).reshape(dim**2)
    kdist = kappa*np.sqrt(t1**2 + t2**2)
    kdist =kdist.reshape(dim, dim)
    
    kdist_2=kdist+np.diag(np.ones(dim))
    bess_H0 = j0(kdist_2) + complex(0, 1)*y0(kdist_2)
    #bess_H0 = besselh(0,1,dat.kdist)
    for j in range(0, dim):
        bess_H0[j,j]=1

    #bess_H1_quot = (j1(kdist) + complex(0,1)*y1(kdist))/(kdist)
    bess_H1_quot = (j1(kdist_2) + complex(0,1)*y1(kdist_2))/(kdist_2)
    
    #Set up prototyp of the singularity of boundary integral operators.
    t=2*np.pi*np.arange(1, dim)/dim
    logsin = scla.toeplitz(np.append(np.asarray([1]), np.log(4*np.sin(t/2)**2)))

    #Quadrature weight for weight function log(4*(sin(t-tau))**2).
    sign=np.ones(dim)
    sign[np.arange(1, dim, 2)]=-1
    t = 2*np.pi*np.arange(0, dim)/dim
    s=0
    for m in range(0, int(dim/2)-1):
        s=s+np.cos((m+1)*t)/(m+1)
    logsin_weights = scla.toeplitz(-2*(s+sign/dim)/dim)

    return DatObject(kappa, logsin_weights, logsin, bess_H0, bess_H1_quot, \
                      kdist)

class DatObject(object):
    def __init__(self, kappa, logsin_weights, logsin, bess_H0, bess_H1_quot, \
                      kdist):
        self.kappa=kappa
        self.logsin_weights=logsin_weights
        self.logsin=logsin
        self.bess_H0=bess_H0
        self.bess_H1_quot=bess_H1_quot
        self.kdist=kdist
