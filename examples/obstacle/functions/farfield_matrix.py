import numpy as np

def farfield_matrix(bd,dire,kappa,weight_sl,weight_dl):
    #Set up matrix corresponding to the far field evaluation of a combined single and double layer potential with weigths weight_sl and weight_dl.

    FFmat =  np.pi / (np.size(bd.z,1)*np.sqrt(8*np.pi*kappa)) * np.exp(-complex(0,1)*np.pi/4) \
         * (weight_dl*kappa*dire.T.dot(bd.normal) +complex(0,1)*weight_sl*np.matlib.repmat(bd.zpabs,np.size(dire,1),1)) \
         * np.exp(-complex(0,1)*kappa* (dire.T.dot(bd.z)))
    return FFmat

def farfield_matrix_trans(bd,dire,kappa,weight_sl,weight_dl):

    FFmat_a = 2*np.pi / (np.size(bd.z,1)*np.sqrt(8*np.pi*kappa)) * np.exp(complex(0,1)*np.pi/4) \
            * (-complex(0,1)*weight_dl*kappa*dire.T*bd.normal) \
            * np.exp(-complex(0,1)*kappa* (dire.T * bd.z))

    FFmat_b = 2*np.pi / (np.size(bd.z,1)*np.sqrt(8*np.pi*kappa)) * np.exp(complex(0,1)*np.pi/4) \
            * (weight_sl*np.matlib.repmat(bd.zpabs,np.size(dire,1),1)) \
            * np.exp(-complex(0,1)*kappa* (dire.T * bd.z))

    FFmat = [FFmat_a, FFmat_b]
    return FFmat
