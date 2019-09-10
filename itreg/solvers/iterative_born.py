#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:09:18 2019

@author: agaltsov
"""

import numpy as np
from . import Solver
from itreg.util.nfft_ewald import NFFT, Rep
import matplotlib.pyplot as plt

import logging

class IterativeBorn(Solver):
    """Solver based on Born approximation with non-zero background potential

     Literature
     ----------
      [1] R.G. Novikov, Sbornik Mathematics 206, 120-34, 2015
      [2] <Add article where [1] is implemented>"""


    def __init__(self,op,proj,data,inc_directions,farfield_directions,p,m):

        self.op = op

        # Current approximation to the solution
        self.x = np.zeros(proj.codomain.shape,dtype=complex)

        # Initializing the NFFT framework
        self.NFFT = NFFT(inc_directions,farfield_directions,proj,p)
        self.x_hat = self.NFFT.zeros(dtype=complex)
        self.y = self.NFFT.zeros(dtype=complex)

        # Initializing inversion data
        self.rhs = self.NFFT.convert(data,Rep.PairsOfDirections,Rep.EwaldSphere)

        self.iteration = 0
        self.radius = m['RADIUS']
        self.maxiter = m['MAXITER']

    def __call__(self,x,x_hat,y,r=0.5,Born=False):
        """ Evaluates the new approximation to the solution

            Parameters:
            -----------
            x: Current approximation
            x_hat: Fourier transform of the current approximation
            y: farfield data of the current approximation

            Output:
            -------
            Rewrites x, x_hat and y
            """


        I_S = self.NFFT.submanifold_indicator(r)

        if not Born:
            x_hat[:] = ( x_hat-y + self.rhs) * I_S
        else:
            # Compute the Born approximation to the solution
            x_hat[:] = self.rhs * I_S

        # Switch to the coordinate domain
        x[:]= self.NFFT.convert(x_hat,Rep.EwaldSphere,Rep.CoordinateDomain)

        # Evaluate operator at current approximation
        y[:] = self.NFFT.convert(self.op(x),Rep.PairsOfDirections,Rep.EwaldSphere)

    def __iter__(self):
        return self

    def __next__(self):

        if self.iteration >= self.maxiter:
            raise StopIteration
        self.iteration += 1

        # Cutoff radius
        self.r = self.radius[self.iteration if self.iteration<len(self.radius) else -1]

        # Evaluate the next approximation
        self(self.x,self.x_hat,self.y,self.r,self.iteration==1)

        return self.x, self.y


    def display(self,f):
            self.NFFT.display(f)

    def derivative(self,dx,r=0.5):
        """ Derivative of the map F: x[j]->x[j+1] """

        I_S = self.NFFT.submanifold_indicator(r)

        dx_hat = self.NFFT.convert(dx,Rep.CoordinateDomain,Rep.EwaldSphere)
        _, dF = self.op.linearize(self.x)
        dy = self.NFFT.convert(dF(dx),Rep.PairsOfDirections,Rep.EwaldSphere)

        dw_hat = (dx_hat - dy) * I_S
        dw = self.NFFT.convert(dw_hat,Rep.EwaldSphere,Rep.CoordinateDomain)

        return dw
