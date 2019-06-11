# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 00:18:35 2019

@author: Bjoern Mueller
"""

from .GenCurve import GenCurve
import numpy as np
import numpy.matlib

"""soll von GenCurve erben"""
class GenTrig:
    """ The class GenTrig describes boundaries of domains in R^2 which are
     parameterized by
          z(t) = [z_1(t),z_2(t)]      0<=t<=2pi
     where z_1 and z_2 are trigonometric polynomials with N coefficient.
     Here N must be even, so the highest order monomial is cos(t*N/2),
     but sin(t*N/2) does not occur.
     z and its derivatives are sampled at n equidistant points.
     Application of the Gramian matrix and its inverse w.r.t. the
     Sobolev norm ||z||_{H^s} are implemented."""
    
    def __init__(self, N_fk, sobo_Index, type=None, coeff=None, **kwargs):
    
        self.N_fk=N_fk
        self.sobo_index=sobo_Index
        self.type=None
        self.coeff=None  # coefficients of the trigonometric polynomials

    

    def GenTrig(self, N, s):
        
        self.GenCurve('GenTrig')
        if N%2==1:
            ValueError('N should be even')
        self.type = 'GenTrig'
        self.coeff = np.zeros(2*N)
        self.sobo_Index = s

    def compute_FK(self, val, n):    
        
        # computes n Fourier coeffients to the point values given by by val.
        # such that ifft(fftshift(coeffhat)) is an interpolation of val
        
        if n%2==1:
            ValueError('length of t should be even')

        N = len(val)
        coeffhat = np.fft.fft(val)
        coeffhat2 = np.zeros(n)
        if (n>=N):
            coeffhat2[0:N/2]= coeffhat[0:N/2]
            coeffhat2[n-N/2+1:n] = coeffhat[N/2+1:N]
            if (n>N):
                coeffhat2[N/2] = 0.5*coeffhat[N/2]
                coeffhat2[n-N/2] = 0.5*coeffhat[N/2]
            else: #n==N
                coeffhat2[N/2] = coeffhat[N/2]

        else:
            coeffhat2[0:int(n/2)] = coeffhat[0:int(n/2)]
            coeffhat2[int(n/2)+1:n] = coeffhat[N-int(n/2)+1:N]
            coeffhat2[int(n/2)] = 0.5*(coeffhat[int(n/2)]+coeffhat[N-int(n/2)+1])

        coeffhat2 = n/N*np.fft.fftshift(coeffhat2)
        return coeffhat2

    def bd_eval(self, n, der):
        
        # evaluates the first der derivatives of the parametrization of
        # the curve on n equidistant time points
        N = int(len(self.coeff)/2)

        
        """transpose ?"""
        coeffhat = np.append(self.compute_FK(self.coeff[0:N],n), \
            self.compute_FK(self.coeff[N:2*N],n)).reshape(2, n)
        self.z = np.append(np.real(np.fft.ifft(np.fft.fftshift( coeffhat[0,:]))), \
            np.real(np.fft.ifft(np.fft.fftshift( coeffhat[1,:])))).reshape(2, coeffhat[0,:].shape[0])
        if der>=1:
            """array indices"""
            self.zp = np.append(np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2-1, n)) * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2-1, n)) * coeffhat[1,:])))).reshape(2, coeffhat[0,:].shape[0])
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            #outer normal vector
            self.normal = np.append(self.zp[1,:], -self.zp[0,:]).reshape(2, self.zp[0, :].shape[0])
   

        if der>=2:
            """array indices"""
            self.zpp = np.append(np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2-1, n))**2 * coeffhat[1,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2-1, n))**2 * coeffhat[2,:])))).reshape(2, coeffhat[1, :].shape[0])

        if der>=3:
            self.zppp = np.append(np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2, n))**3 * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2, n))**3 * coeffhat[1,:])))).reshape(2, coeffhat[1, :].shape[0])
        if der>3:
            raise ValueError('only derivatives up to order 3 implemented')

    def der_normal(self, h):        
        
        #computes the normal part of the perturbation of the curve caused by
        #perturbing the coefficient vector curve.coeff in direction h
        N= len(h)/2
        n = np.size(self.z)
        if N==n:
            """transpose ?"""
            hn= [h[0:n].transpose(), h[n:2*n].transpose()].reshape(2, n)
        else:
            h_hat = [self.compute_FK(h[0:N],n), \
                self.compute_FK(h[N:2*N],n)].transpose()
            hn = [np.real(np.fft.ifft(np.fft.fftshift(h_hat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift(h_hat[1,:])))].reshape(2, h_hat[0, :].shape[0])
        """transpose ?"""   
        der = (np.sum(hn*self.normal,1)/self.zpabs).tranpose()
        return der
        
    
    def adjoint_der_normal(self, g):
        
        #applies the adjoint of the linear mapping h->der_normal(curve,h) to g
        N = len(self.coeff)/2
        n= len(g)
        """repmat ?, transpose?"""
        adj_n = numpy.matlib.repmat(g.transpose/self.zpabs,2,1) * self.normal
        if N==n:
            adj = [adj_n[0,:].transpoe(), adj_n[1,:].transpose()].reshape(2, adj_n[0, :].shape[0])
        else:
            adj_hat = [self.compute_FK(adj_n[0,:],N), \
                self.compute_FK(adj_n[1,:],N)]*n/N.reshape(2, adj_n[0, :].shape[0])
            adj = [np.fft.ifft(np.fft.fftshift(adj_hat[:,0])), \
                np.fft.ifft(np.fft.fftshift(adj_hat[:,1]))].reshape(2, adj_hat[:, 0].shape[0])
        return adj

    def arc_length_der(self, h):    
        
            #computes the derivative of h with respect to arclength
            n=len(self.zpabs)
            """transpsoe ?"""
            dhds = np.fft.ifft(np.fft.fftshift((1j*np.linspace(-n/2, n/2-1, n)).transpose()*self.compute_FK(h,n)))/self.zpabs.transpose()
            return dhds
        
    def L2err(self, q1, q2):
        res=self.params.domain.norm(q1-q2)/np.sqrt(len(q1))
        return res
        
    def coeff2Curve(self, coeff, n):   
        
        
        N = len(coeff)/2
        """transpose ?"""
        coeffhat = [self.compute_FK(coeff[0:N],N), \
            self.compute_FK(coeff[N:2*N],N)].transpose()
        pts = [np.real(np.fft.ifft(np.fft.fftshift( coeffhat[0,:]))), \
            np.real(np.fft.ifft(np.fft.fftshift( coeffhat[1,:])))].reshape(2, coeffhat[0, :].shape[0])
        return pts
