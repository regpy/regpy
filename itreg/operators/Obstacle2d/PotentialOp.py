# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 22:28:46 2019

@author: Bjoern Mueller
"""

from itreg.operators import NonlinearOperator, OperatorImplementation, Params
from .Obstacle2dBaseOp import Obstacle2dBaseOp
from .Obstacle2dBaseOp import bd_params
from itreg.util import instantiate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class PotentialOp(NonlinearOperator):
    """ identification of the shape of a heat source from measurements of the
     heat flux.
     see F. Hettlich & W. Rundell "Iterative methods for the reconstruction
     of an inverse potential problem", Inverse Problems 12 (1996) 251–266
     or
     sec. 3 in T. Hohage "Logarithmic convergence rates of the iteratively
     regularized Gauss–Newton method for an inverse potential
     and an inverse scattering problem" Inverse Problems 13 (1997) 1279–1299
    """
     
    def __init__(self, domain, range=None, **kwargs):

            range = range or domain
            radius = 1.5            # radius of outer circle
            Nfwd = 64               # nr. of discretization points for forward solver
            Nfwd_synth = 256        # nr of discretization points for computation of synthetic data
            N_meas = 64             # number of measurement points
            """transpose"""
            meas_angles=2*np.pi*np.linspace(0, N_meas-1, N_meas).transpose()/N_meas             # angles of measure points
            
            N=Nfwd
            cosin=np.zeros((N, N))
            sinus=np.zeros((N, N))
            sin_fl=np.zeros((N, N))
            cos_fl=np.zeros((N, N))
            obstacle=Obstacle2dBaseOp()
            obstacle.Obstacle2dBasefunc()
            bd=obstacle.bd
            super().__init__(Params(domain, range, radius=radius, Nfwd=Nfwd, 
                 Nfwd_synth=Nfwd_synth, N_meas=N_meas, meas_angles=meas_angles,
                 cosin=cosin, sinus=sinus, sin_fl=sin_fl, cos_fl=cos_fl, bd=bd,
                 obstacle=obstacle))

    @instantiate
    class operator(OperatorImplementation):
        def eval(self, params, coeff, **kwargs):
            """self.bd.coeff"""
            params.bd.coeff = coeff
            N = params.Nfwd
            t= 2*np.pi*np.linspace(0, N-1, N)/N
            t_fl = params.meas_angles
            
            for j in range(0, N):
                params.cosin[j,:] = np.cos((j+1)*t)
                params.sinus[j,:] = np.sin((j+1)*t)
                params.sin_fl[:,j] = np.sin((j+1)*t_fl)
                params.cos_fl[:,j] = np.cos((j+1)*t_fl)
            """F.bd has to be specified"""
            params.bd.bd_eval(N,1)
#            params.bd.radial(N,1)
            q=params.bd.q[0, :]
#            q = params.bd.q[0,:]
            if q.max() >= params.radius:
                raise ValueError('reconstructed object penetrates measurement circle')
            
            if q.min()<=0:
                raise ValueError('reconstructed radial function negative')
                
            """exact meaning of q.?""" 
            qq = q**2
            
            flux = 1/(2*params.radius*N) * np.sum(qq)*np.ones(len(params.meas_angles))
            fac = 2/(N*params.radius)
            for j in range(0, int((N-1)/2)):
                fac= fac/params.radius
                qq = qq * q
                flux = flux + (fac/(j+3)) * params.cos_fl[:,j] * np.sum(qq*params.cosin[j,:]) \
                    + (fac/(j+3)) * params.sin_fl[:,j] * np.sum(qq*params.sinus[j,:])
           
            if (N % 2==0):
                fac = fac/params.radius
                qq = qq * q
                flux = flux + fac * params.cos_fl[:,int(N/2)] * np.sum(qq*params.cosin[int(N/2),:])
            return flux
                
    @instantiate
    class derivative(OperatorImplementation):
        def eval(self, params, h_coeff, data, **kwargs):
            
            N = params.Nfwd
            """transpose ?"""
            h = params.bd.der_normal(h_coeff).transpose()
            q = params.bd.q[0,:]
            qq = params.bd.zpabs
            
            der = 1/(params.radius*N) * np.sum(qq*h)*np.ones(len(params.meas_angles))
            fac = 2/(N*params.radius)
            for j in range(0, int((N-1)/2)):
                fac= fac/params.radius
                qq = qq*q
                der = der + fac * params.cos_fl[:,j] * np.sum(qq*h*params.cosin[j,:]) \
                    + fac * params.sin_fl[:,j] * np.sum(qq*h*params.sinus[j,:])
            
            if (N % 2==0):
                fac = fac/params.radius
                qq = qq*q
                der = der + fac * params.cos_fl[:,N/2] * np.sum(qq*h*params.cosin[N/2,:])
            return der
        
        
        def adjoint(self, params, g, data, **kwargs):
            N = params.Nfwd
            q = params.bd.q[0,:]
            qq = params.bd.zpabs
            
            """transpose?"""
            adj = 1/(params.radius*N) *np.sum(g) * qq.transpose()
            fac = 2/(N*params.radius)
            for j in range(0, int((N-1)/2)):
                fac= fac/params.radius
                qq = qq*q
                """transpose?"""
                adj = adj + fac * np.sum(g*params.cos_fl[:,j]) * (params.cosin[j,:]*qq).transpose() \
                    + fac * np.sum(g*params.sin_fl[:,j]) * (params.sinus[j,:]*qq).transpose()
            
            if (N % 2==0):
                fac = fac/params.radius
                qq = qq*q
                """transpose?"""
                adj = adj + fac * np.sum(g*params.cos_fl[:,N/2]) * (params.cosin[N/2,:]*qq).transpose()
            
            adj = params.bd.adjoint_der_normal(adj)
            return adj


class plots():
    def __init__(self,
                 op,
                 reco,
                 reco_data,
                 data,
                 exact_solution,
#                 nr_plots,
#                 fig_rec=None,
                 figsize=(8, 8),
#                 Nx=None,
#                 Ny=None, 
                 ):
        
        self.op=op
#        self.Nx=Nx or self.op.params.Nx
#        self.Ny=Ny or self.op.params.Ny
        self.reco=reco
        self.reco_data=reco_data
        self.data=data
        self.exact_solution=exact_solution 
#        self.nr_plots=nr_plots
#        self.fig_rec=fig_rec
        self.figsize=figsize
    
    def plotting(self):    
#    function F = plot(F,x_k,x_start,y_k,y_obs,k)
#            nr_plots = self.nr_plots
            
            fig, axs = plt.subplots(1, 2,sharey=True,figsize=self.figsize)
            fig.title('Potential Problem')
            axs[0].title('Domain')
            axs[1].title('Heat source')
            axs[1].plot(self.exact_data)
            axs[1].plot(self.reco_data)
            bd=self.op.params.bd
            pts=bd.coeff2Curve(self.reco)
            pts_2=bd.coeff2Curve(self.exact_solution)
            poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), animated=True, fill=False)
            poly_2=Polygon(np.column_stack([pts_2[0, :], pts_2[1, :]]), animated=True, fill=False)
            axs[0].add_patch(poly)
            axs[0].add_patch(poly_2)
            xmin=1.5*min(pts[0, :].min(), pts_2[0, :].min())
            xmax=1.5*max(pts[0, :].max(), pts_2[0, :].max())
            ymin=1.5*min(pts[1, :].min(), pts_2[1, :].min())
            ymax=1.5*max(pts[1, :].max(), pts_2[1, :].max())
            axs[0].set_xlim((xmin, xmax))
            axs[0].set_ylim((ymin, ymax))
            plt.show()
                
            
        