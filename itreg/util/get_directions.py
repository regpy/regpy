#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:10:20 2019

@author: agaltsov
"""

import numpy as np

def get_directions(p):
    # Computes the measuring directions for the experiment around the
    # incident direction

    phi = np.arange(p['N_INC'])/p['N_INC']  # range [0,1)
    dphi = (np.arange(p['N_MEAS'])/(p['N_MEAS']-1)-0.5)

    # Measurement directions as points in the complex plane
    z_meas = np.exp(2j*np.pi*(np.tile(phi[:,np.newaxis],(1,p['N_MEAS']))  \
        + np.tile(dphi[np.newaxis,:],(p['N_INC'],1))))

    # Realification
    inc_dir = np.asarray([[np.real(w),np.imag(w)] for w in np.exp(2j*np.pi*phi)])
    meas_dir = np.asarray([[([np.real(w),np.imag(w)]) for w in L] for L in z_meas])

    return (inc_dir,meas_dir)
