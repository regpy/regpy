# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:52:57 2019

@author: Björn Müller
"""

from .Obstacle2dBaseOp import Obstacle2dBaseOp
#from .Obstacle2dBaseOp import Obstacle2dBaseOp
from .functions.operator import op_S
from .functions.operator import op_T
from .functions.operator import op_K
from .functions.farfieldmatrix import farfield_matrix
from .functions.farfieldmatrix import farfield_matrix_trans
from .set_up_iopdata import setup_iop_data

from .. import Operator

import numpy as np
import scipy.linalg as scla


class NeumannOp(Operator):
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
        self.inc_directions = np.asarray([1,0]).reshape((2,1))
        self.obstacle=Obstacle2dBaseOp()
        self.bd=self.obstacle.bd



        self.op_name = 'NeumannOp'
        self.syntheticdata_flag = True
        self.kappa = 3    # wave number
        # directions of incident waves
        N_inc = 1
        #t=2*pi*[0:N_inc-1]/N_inc;
        t = 0.5
        self.inc_directions = np.append(np.cos(t), np.sin(t)).reshape((2, N_inc))

        N_meas = 64
        t= 2*np.pi*np.arange(0, N_meas)/N_meas
        self.meas_directions = np.append(np.cos(t), np.sin(t)).reshape((2, N_meas))
        self.noiselevel = 0.01
        self.N_ieq = 64

        self.true_curve = 'nonsym_shape'
        #'peanut','round_rect', 'apple',
        #'three_lobes','pinched_ellipse','smoothed_rectangle','nonsym_shape'
        self.bd_type = 'GenTrig'




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
        % adjoints are stored as members of self."""

        self.bd.coeff = coeff
        """compute the grid points of the boundary parameterized by coeff and derivatives
        of the parametrization and save these quantities as members of F.bd"""
        self.bd.bd_eval(2*self.N_ieq,3)
        Iop_data = setup_iop_data(self.bd,self.kappa)
        #Iop = op_T(F.bd,Iop_data) - i*F.eta*op_K(F.bd,Iop_data).' + i*F.eta*diag(F.bd.zpabs);
        if self.wDL!=0:
            Iop = self.wDL*op_T(self.bd,Iop_data)
        else:
            Iop = np.zeros(np.size(self.bd.z,2),np.size(self.bd.z,2));
        if self.wSL!=0:
            Iop = Iop + self.wSL*(op_K(self.bd,Iop_data).T - np.diag(self.bd.zpabs))
            self.Iop=Iop
        #F.Iop=Iop;
        self.u = complex(0,1)*np.zeros((2*self.N_ieq,np.size(self.inc_directions,1)))
        FF_DL = farfield_matrix(self.bd,self.meas_directions,self.kappa,0,1)

        self.perm_mat, self.L, self.U =scla.lu(Iop)
        self.perm=self.perm_mat.dot(np.arange(0, np.size(self.bd.z,1)))
        self.FF_combined = farfield_matrix(self.bd,self.meas_directions,self.kappa,  \
                                           self.wSL,self.wDL)
        farfield = []

        for l in range (0,np.size(self.inc_directions,1)):
            rhs = -2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].T.dot(self.bd.z))* \
                (self.wDL*complex(0,1)*self.kappa*self.inc_directions[:,l].T.dot(self.bd.normal) \
                                     + self.wSL*self.bd.zpabs)
            self.u[:, l]=np.linalg.solve(self.L.T, \
                          np.linalg.solve(self.U.T, rhs[self.perm.astype(int)]))
            complex_farfield = FF_DL.dot(self.u[:,l])
            farfield=np.append(farfield, complex_farfield)
        return farfield

    def _derivative(self,h):

            der = []
            for l in range(0, np.size(self.inc_directions,1)):
                duds = self.bd.arc_length_der(self.u[:,l])
                hn = self.bd.der_normal(h)
                rhs = self.bd.arc_length_der(hn*duds) + self.kappa**2* hn*(self.u[:,l])
                rhs = 2*rhs*(self.bd.zpabs.T)
                phi=np.linalg.solve(self.U, np.linalg.solve(self.L, rhs[self.perm.astype(int)]))
                complex_farfield = self.FF_combined.dot(phi)
                der=np.append(der, complex_farfield)
            return der


    def _adjoint(self,g):
            res = complex(0,1)*np.zeros(2*self.N_ieq)
            v = complex(0,1)*np.zeros(2*self.N_ieq)
            N_FF = np.size(self.meas_directions,1)
            for l in range(0, np.size(self.inc_directions,1)):
                g_complex=g[2*(l)*N_FF+np.arange(0, N_FF)]
                phi = self.FF_combined.T.dot(g_complex)
                v[self.perm.astype(int)] = np.linalg.solve(self.L.T, \
                                          np.linalg.solve(self.U.T, phi))
                dvds=  self.bd.arc_length_der(v)
                duds =  self.bd.arc_length_der(self.u[:,l])
                res = res -2*(np.conjugate(dvds)*duds - self.kappa**2*np.conjugate(v)*  \
                              self.u[:,l]).real
            adj = self.bd.adjoint_der_normal(res * self.bd.zpabs.T)
            return adj

    def other_X_err(self,h):
            res = np.sqrt(((h-self.xdag).T*(h-self.xdag)).real)
            return res

    def accept_proposed(self, x):
        return True


def create_synthetic_data(self):
        self.op.obstacle.bd_ex.bd_eval(self.op.obstacle.bd_ex, 2*self.op.N_ieq_synth,3)
        """compute the grid points of the exact boundary and derivatives of the
        %parametrization and save these quantities as members of bd_ex"""

        #set up the boudary integral operator
        bd=self.op.obstacle.bd_ex
        Iop_data = setup_iop_data(bd,self.op.kappa)

        if self.op.wDL!=0:
            Iop = self.op.wDL*op_T(bd,Iop_data)
        else:
            Iop = np.zeros(np.size(bd.z,1),np.size(bd.z,1))

        if self.op.wSL!=0:
            Iop = Iop + self.op.wSL*(op_K(bd,Iop_data).T - np.diag(bd.zpabs))

        """F.bd_ex = bd;
        %set up the matrix mapping the density to the far field pattern
        %FF_combined = farfield_matrix(bd,F.meas_directions,F.kappa,-i*F.eta,1.);"""
        FF_combined = farfield_matrix(bd,self.op.meas_directions,self.op.kappa,  \
                                      self.op.wSL,self.op.wDL)
        farfield = []
        for l in range(0, np.size(self.op.inc_directions,1)):
            rhs = - 2*np.exp(complex(0,1)*self.op.kappa*self.op.inc_directions[:,l].T.dot(bd.z))* \
                (complex(0,1)*self.op.kappa*self.op.inc_directions[:,l].T.dot(bd.normal))
            phi = np.linalg.solve(Iop, rhs)
            complex_farfield = FF_combined.dot(phi)
            farfield = np.append(farfield, np.append(complex_farfield.real, complex_farfield.imag))

        farfield=farfield.reshape((2*np.size(self.op.inc_directions, 1), bd.zpabs.shape[0]))
        noise = np.random.randn(np.size(farfield))
        noise=noise.reshape((2*np.size(self.op.inc_directions, 1), bd.zpabs.shape[0]))
        complex_noise=noise[0, :]+complex(0,1)*noise[1, :]
        data = farfield + self.op.noiselevel * complex_noise/ \
                            np.sqrt(complex_noise*self.Hcodomain.gram(complex_noise))

        return data[0, :].real+complex(0, 1)*data[1, :].real
