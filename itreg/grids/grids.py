# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:22:54 2019

@author: hendr
"""

import numpy as np

from . import Grid

class Square_1D(Grid):
    def __init__(self, shape, center, spacing):
        x_coo=spacing*np.arange(-shape[0]/2+center, (shape[0]-1)/2+center, 1)
        super().__init__(np.array([x_coo]), shape)

class Square_2D(Grid):
    def __init__(self, shape, center, spacing):
        x_coo=spacing*np.arange(-shape[0]/2+center, (shape[0]-1)/2+center, 1)
        y_coo=spacing*np.arange(-shape[1]/2+center, (shape[1]-1)/2+center, 1)
        super().__init__(np.array([x_coo, y_coo]), shape)
        
class Square_3D(Grid):
    def __init__(self, shape, center, spacing):
        x_coo=spacing*np.arange(-shape[0]/2+center, (shape[0]-1)/2+center, 1)
        y_coo=spacing*np.arange(-shape[1]/2+center, (shape[1]-1)/2+center, 1)
        z_coo=spacing*np.arange(-shape[2]/2+center, (shape[2]-1)/2+center, 1)
        super().__init__(np.array([x_coo, y_coo, z_coo]), shape) 

        