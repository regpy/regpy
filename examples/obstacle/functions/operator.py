import numpy as np
import scipy.linalg as scla
import scipy.sparse as scsp

def op_S(bd, dat):
    r""" Set up matrix representing the single layer potential operator

    \[
        (S\phi)(z(t)):=2|z'(t)|\int_0^{2\pi}\Phi(z(t),z(s))|z'(s)|\phi(s) ds.
    \]
    
    References
    ----------
    - R. Kress & D. Colton "Inverse Acoustic
    and Electomagnetic Scattering Theory", 2019.

    As opposed to this reference we multiplied by |z'(t)| to obtain a
    complex symmetric matrix."""

    dim = np.size(bd.z,1)
    M1 = -1/(2*np.pi)*dat.bess_H0.real
    M = complex(0,1)/2*dat.bess_H0
    M1_logsin = M1* dat.logsin
    M2 = M - M1_logsin
    
    for j  in range(0, dim):
        M2[j, j] = (complex(0, 1)/2 - np.euler_gamma/np.pi - 1/np.pi*np.log(dat.kappa/2*bd.zpabs[j]))
    S = 2*np.pi*(M1*dat.logsin_weights + M2/dim)*(np.outer(bd.zpabs,bd.zpabs))

    return S

def op_K(bd, dat):
    r""" Set up matrix representing the double layer potential operator

    \[
        (K\phi)(z(t)):=2|z'(t)|\int_0^{2\pi}{\frac{\partial\Phi(z(t),z(s))}{\partial\nu(z(s))}
        |z'(s)|\phi(z(s)) ds
    \]

    References
    ----------
    - R. Kress & D. Colton "Inverse Acoustic
    and Electomagnetic Scattering Theory", 2019."""

    dim = np.size(bd.z,1)
    kappa = dat.kappa

    aux = np.dot(bd.z.T, bd.normal)-np.dot(np.ones((dim, 2)),(bd.normal*bd.z))
    H = 0.5*complex(0, 1)*kappa**2*aux*dat.bess_H1_quot
    H1 = -kappa**2/(2*np.pi)*aux*dat.bess_H1_quot.real
    H2 = H - H1*dat.logsin
    for j in range(0, dim):
        H1[j, j] = 0
        H2[j, j] = 1/(2*np.pi)*(np.dot(bd.normal[:,j].T, bd.zpp[:,j]))/bd.zpabs[j]**2

    K = (2*np.pi)*scsp.spdiags(bd.zpabs.T, 0, dim, dim)*(H1*dat.logsin_weights + H2/dim)
    return K

