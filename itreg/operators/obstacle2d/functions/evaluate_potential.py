# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:04:49 2019

@author: Björn Müller
"""
import numpy as np
import mpmath


def evaluate_potential(bd,phi,zz,kappa,weightSL, weightDL):
    """ Evaluates the combined layer potential with density phi at points zz and
    % wave length kappa.
    % The weights of the single and double layer potential are given by
    % the input parameters weightSL and weightDL, respectively."""
    Nbd =np.size(bd.z,1)
    zz_len = np.size(zz,1)
    field = np.zeros(zz_len)
    for j in range(1, zz_len):
        kdist = kappa * np.sqrt((np.sum( (np.matlib.repmat(zz[:,j],1,Nbd)-bd.z)**2,1)))
        fieldSL = (0.25*complex(0,1)*2*np.pi/Nbd)*(mpmath.besselj(0,kdist)*(bd.zpabs.T*phi))
        fieldDL = (0.25*complex(0,1)*kappa**2*2*np.pi/Nbd) \
             * ((zz[:,j].T * bd.normal - np.sum(bd.normal*bd.z,1)) \
            *  (mpmath.besselj(1,kdist)/kdist)) * phi
        field[j]  = weightSL*fieldSL + weightDL*fieldDL
    return field
