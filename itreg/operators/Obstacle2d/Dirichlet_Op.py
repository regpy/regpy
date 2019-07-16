# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 20:12:14 2019

@author: Björn Müller
"""
from .Obstacle2dBaseOp import Obstacle2dBaseOp
from .functions.operator import op_S
from .functions.operator import op_T
from .functions.operator import op_K
from .functions.farfield import farfield_matrix
from .functions.farfield import farfield_matrix_trans

from . import NonlinearOperator


class ScatOp(NonlinearOperator):
    
    def init(self, domain, codomain=None, **kwargs):
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

    

        """ 2 dimensional obstacle scattering problem with Dirichlet boundary condition
         see sec. 4 in T. Hohage "Logarithmic convergence rates of the iteratively
         regularized Gauss-Newton method for an inverse potential
         and an inverse scattering problem" Inverse Problems 13 (1997) 1279�1299"""
    
    
        self.dudn=None  # normal derivative of total field at boundary
        """ weights of single and double layer potentials"""
        self.wSL=-1*complex(0,1)*self.kappa 
        self.wDL=1
        """ LU factors + permuation for integral equation matrix"""
        self.L=None
        self.U=None
        self.perm=None
        self.FF_combined=None
        self.op_name='DirichletOp'
        """ use a mixed single and double layer potential ansatz with
             weights wSL and wDL"""
        self.Ydim = 2* np.size(self.meas_directions) * np.size(self.inc_directions, 1) 
        
    def _eval(self, coeff, **kwargs):
        
    
        """ solve the forward Dirichlet problem for the obstacle parameterized by
         coeff. Quantities needed again for the computation of derivatives and
         adjoints are stored as members of F."""
        
        self.bd.coeff = coeff
        """compute the grid points of the boundary parameterized by coeff and derivatives
        of the parametrization and save these quantities as members of F.bd"""
        self.bd.bd_eval(2*self.N_ieq,2)
        Iop_data = setup_iop_data(self.bd,self.kappa)
        #Iop = diag(F.bd.zpabs)+ op_K(F.bd,Iop_data) - i*F.eta*op_S(F.bd,Iop_data)
        if self.wSL!=0:
            Iop = self.wSL*op_S(self.bd,Iop_data)
        else:
            Iop = np.zeros(np.size(self.bd.z,1),np.size(self.bd.z,1))
        if self.wDL!=0:
            Iop = Iop + self.wDL*(np.diag(self.bd.zpabs)+ op_K(self.bd,Iop_data))
        self.dudn = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        FF_SL = farfield_matrix(self.bd,self.meas_directions,self.kappa,-1.,0.)
#What is lu
        self.L, self.U,self.perm = lu(Iop,'vector')
        #F.Iop=Iop;
        self.FF_combined = farfield_matrix(self.bd,self.meas_directions,self.kappa,self.wSL,self.wDL);
        farfield = []
        
        for l in range(0, np.size(self.inc_directions, 1)):
            rhs = 2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].T*self.bd.z)*  \
                (self.wDL*complex(0,1)*self.kappa*self.inc_directions[:,l].T*self.bd.normal +self.wSL*self.bd.zpabs)
            #same as
            #F.dudn(:,l) = F.Iop.' \ rhs.';
#self.perm is not defined up to now
            self.dudn[:,l] = (self.L.T) \ ((self.U.T) \ rhs[self.perm.astype(int)].T)
            complex_farfield = FF_SL * self.dudn[:,l]
            farfield = np.append([farfield, complex_farfield.real, complex_farfield.imag].reshape((3*farfield.shape[0], farfield.shape[1]]))
        return farfield
        
        def _derivative(self, h):
            der = []
            for l in range(0, np.size(self.inc_directiosn,1 ))
                rhs = - 2*self.dudn[:,l] * self.bd.der_normal(h) * self.bd.zpabs.T
                #same as
                #phi = F.Iop \ rhs;
                phi = self.U \ (self.L \ rhs(self.perm))
                complex_farfield = self.FF_combined * phi
                der=np.append(der, np.append(complex_farfield.real, complex_farfield.imag)).reshape((3*der.shape[0], der.shape[1]))
            return der
        
        def _adjoint(self, g):
            res = np.zeros(2*self.N_ieq)
            rhs = np.zeros(2*self.N_ieq)
            N_FF = np.size(F.meas_directions,1)
            for  l in range(0, np.size(self.inc_directions,1)):
                g_complex = g(2*(l)*N_FF+np.arange(1, N_FF+1)) + comlpex(0,1)*g(2*(l)*N_FF+np.arange(N_FF+1, 2*N_FF+1))
                phi = F.FF_combined.T*g_complex
                #rhs = F.Iop' \ phi;
                #rhs(F.perm) = F.L' \ (F.U' \ phi);
                rhs[self.perm.astype(int)] = ((phi.T/self.U)/self.L).T
                res = res -2*rhs*np.conjugate(self.dudn[:,l]).real
            adj = self.bd.adjoint_der_normal(res * self.bd.zpabs.T)
            return adj
