import numpy as np

def farfield_matrix(bd, dire, kappa, weight_sl, weight_dl):
    FFmat=np.zeros((len(dire), np.size(bd.z,1)), dtype=complex)
    for l, meas in enumerate(dire):
         FFmat[l,:] = np.pi/(np.size(bd.z,1)*np.sqrt(8*np.pi*kappa))*np.exp(-complex(0,1)*np.pi/4)\
         *(weight_dl*kappa*meas.dot(bd.normal)+complex(0,1)*weight_sl*bd.zpabs)*np.exp(-complex(0,1)*kappa*(meas.dot(bd.z)))

    return FFmat

def farfield_matrix_trans(bd, dire, kappa_ex, weight_sl_ex, weight_dl_ex):
    FFmat=np.zeros((len(dire), 2*np.size(bd.z,1)), dtype=complex)
    FFmat_a=np.zeros((len(dire), np.size(bd.z,1)), dtype=complex)
    FFmat_b=np.zeros((len(dire), np.size(bd.z,1)), dtype=complex)

    for l, meas in enumerate(dire):
         FFmat_a[l,:] = 2*np.pi/(np.size(bd.z,1)*np.sqrt(8*np.pi*kappa_ex))*np.exp(complex(0,1)*np.pi/4)\
         *(-complex(0,1)*weight_dl_ex*kappa_ex*meas.dot(bd.normal))*np.exp(-complex(0,1)*kappa_ex*(meas.dot(bd.z)))
    for l, meas in enumerate(dire):
         FFmat_b[l,:] = 2*np.pi/(np.size(bd.z,1)*np.sqrt(8*np.pi*kappa_ex))*np.exp(complex(0,1)*np.pi/4)\
         *(weight_sl_ex*bd.zpabs)*np.exp(-complex(0,1)*kappa_ex*(meas.dot(bd.z)))
    
    FFmat = np.hstack((FFmat_a, FFmat_b))
    return FFmat