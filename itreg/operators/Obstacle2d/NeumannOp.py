# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:52:57 2019

@author: Björn Müller
"""


from .Obstacle2dBaseOp import Obstacle2dBaseOp
from .functions.operator import op_S
from .functions.operator import op_T
from .functions.operator import op_K
from .functions.farfieldmatrix import farfield_matrix
from .functions.farfieldmatrix import farfield_matrix_trans
from .set_up_iopdata import setup_iop_data

from .. import NonlinearOperator

import numpy as np


class NeumannOp(NonlinearOperator):
    """ 2 dimensional obstacle scattering problem with Neumann boundary condition
     see T. Hohage "Convergence rates of a regularized Newton method
     in sound-hard inverse scattering" SIAM J. Numer. Anal. 36:125--142, 1999"""
    
    def __init__(self, domain, codomain=None, **kwargs):
        codomain = codomain or domain
        super().__init__(domain, codomain)
        self.kappa = 3            # wave number
        self.N_ieq = 32           # 2*F_ieq is the number of discretization points
        self.N_ieq_synth = 32     # 2*N_ieq is the number of discretization points for
        """the boundary integral equation when computing synthetic data (choose different
         to N_ieq to avoid invere crime)"""
        self.meas_directions = 64 # measurement directions
        self.inc_directions = np.asarry([1,0]).reshape((2,1))
        self.obstacle=Obstacle2dBaseOp()
        self.obstacle.Obstacle2dBasefunc()
        self.bd=self.obstacle.bd
        
        self.u=None  # values of total field at boundary
        """ weights of single and double layer potentials"""
        self.wSL = -complex(0,1)*self.kappa
        self.wDL = 1
        # LU factors + permuation for integral equation matrix
        self.L=None
        self.U=None
        self.perm=None
        self.FF_combined=None
        self.op_name = 'NeumannOp'
        """ use a mixed single and double layer potential ansatz with
         weights wSL and wDL"""

        self.Ydim = 2* np.size(self.meas_directions,1) * np.size(self.inc_directions,1)

    
    
        
    def _eval(self, coeff, differentiate=False):
        
        
        """ solve the forward scattering problem for the obstacle parameterized by
        % coeff. Quantities needed again for the computation of derivatives and
        % adjoints are stored as members of F."""
        
        self.bd.coeff = coeff
        """compute the grid points of the boundary parameterized by coeff and derivatives
        of the parametrization and save these quantities as members of F.bd"""
        self.bd.bd_eval(2*self.N_ieq,3)
        Iop_data = setup_iop_data(self.bd,self.kappa)
        #Iop = op_T(F.bd,Iop_data) - i*F.eta*op_K(F.bd,Iop_data).' + i*F.eta*diag(F.bd.zpabs);
        if self.wDL!=0:
            Iop = self.wDL*op_T(self.bd,Iop_data)
        else:
            Iop = np.zeros(np.size(F.bd.z,2),np.size(F.bd.z,2));
        if self.wSL!=0:
            Iop = Iop + self.wSL*(op_K(self.bd,Iop_data).T - np.diag(self.bd.zpabs))
        #F.Iop=Iop;
        self.u = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        FF_DL = farfield_matrix(self.bd,self.meas_directions,self.kappa,0,1)
#What is lu
        self.L, self.U,self.perm = lu(Iop,'vector')
        self.FF_combined = farfield_matrix(self.bd,self.meas_directions,self.kappa,self.wSL,self.wDL)
#farfield has to be introduced in another way
        farfield = []
        
        for l in range (0,np.size(self.inc_directions,1)):
            rhs = -2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].T*self.bd.z)* \
                (self.wDL*complex(0,1)*self.kappa*self.inc_directions[:,l].T*self.bd.normal + self.wSL*self.bd.zpabs)
            self.u[:,l] = (self.L.T) / ((self.U.T) / rhs[self.perm.astype(int)].T)
            complex_farfield = FF_DL * self.u[:,l]
            farfield=np.append(farfield, np.append(complex_farfield.real, comlex_farfield.imag)).reshape((3*complex_farfield.shape[0], complex_farfield.shape[1]))
        return farfield
        
        def _derivative(F,h):
#define der in another form
            der = []
            n=np.size(self.u,0)
            for l in range(0, np.size(self.inc_directions,1)):
                duds = self.bd.arc_length_der(self.u[:,l])
                hn = self.bd.der_normal(h)
                rhs = self.bd.arc_length_der(hn*duds) + self.kappa**2* hn * self.u[:,l]
                rhs = 2*rhs * self.bd.zpabs.T
                phi = self.U / (self.L / rhs[self.perm.astype(int)])
                complex_farfield = self.FF_combined * phi
                der=np.append(der, np.append(complex_farfield.real, complex_farfield.imag)).reshape((3*complex_farfield.shape[0], complex_farfield.shape[1]))


        
        def _adjoint(F,g):
            res = np.zeros(2*self.N_ieq)
            v = np.zeros(2*self.N_ieq)
            N_FF = np.size(self.meas_directions,1)
            n=np.size(self.u,0)
            for l in range(0, np.size(self.inc_directions,1)):
                g_complex = g(2*(l)*N_FF+np.arange(1, N_FF+1) )+ complex(0,1)*g(2*(l)*N_FF+np.arange(N_FF+1, 2*N_FF+1))
                phi = self.FF_combined.T*g_complex
                v[self.perm.astype(int)] = (self.L.T / (self.U.T / phi))
                dvds=  self.bd.arc_length_der(v)
                duds =  self.bd.arc_length_der(self.u[:,l])
                res = res -2*(np.conjugate(dvds)*duds - self.kappa**2*np.conjugate(v)*self.u[:,l]).real
            adj = self.bd.adjoint_der_normal(res * self.bd.zpabs.T)
            return adj
        
        def other_X_err(self,h):
            res = np.sqrt(((h-self.xdag).T*(h-self.xdag)).real)
            return res
        
        
def create_synthetic_data(self):
        bd = self.bd_ex.bd_eval(2*self.N_ieq_synth,3)
        """compute the grid points of the exact boundary and derivatives of the
        %parametrization and save these quantities as members of bd_ex"""
        
        #set up the boudary integral operator
        Iop_data = setup_iop_data(bd,self.kappa)
        
        if self.wDL!=0:
            Iop = self.wDL*op_T(bd,Iop_data)
        else:
            Iop = np.zeros(np.size(bd.z,1),np.size(bd.z,1))

        if self.wSL!=0:
            Iop = Iop + self.wSL*(op_K(bd,Iop_data).T - np.diag(bd.zpabs))

        """F.bd_ex = bd;
        %set up the matrix mapping the density to the far field pattern
        %FF_combined = farfield_matrix(bd,F.meas_directions,F.kappa,-i*F.eta,1.);"""
        FF_combined = farfield_matrix(bd,self.meas_directions,self.kappa,self.wSL,self.wDL)
        farfield = []
        for l in range(0, np.size(self.inc_directions,1)):
            rhs = - 2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].T*bd.z)* \
                (complex(0,1)*self.kappa*self.inc_directions[:,l].T*bd.normal)
            phi = Iop/rhs.T
            complex_farfield = FF_combined * phi
            farfield = np.append(farfield, np.append(complex_farfield.real, complex_farfield.imag)).reshape((3*farfield.shape[0], farfield.shape[1]))

        noise = np.random.randn(np.size(farfield))
        data = farfield + self.noiselevel * noise/np.sqrt(noise.T*self.codomain.gram(noise))
        
        return data
