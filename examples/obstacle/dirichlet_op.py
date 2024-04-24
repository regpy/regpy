import numpy as np
import scipy.linalg as scla

from  functions.operator import op_S
from  functions.operator import op_K
from  functions.farfield_matrix import farfield_matrix
from  functions.setup_iop_data import setup_iop_data
from  regpy.operators import Operator
from  regpy.vecsps.curve import StarCurveDiscr
from  regpy.vecsps import UniformGridFcts
from regpy.vecsps.curve import GenTrigDiscr

class DirichletOp(Operator):
    r"""Operator that maps the shape of a sound-soft obstacle to the far-field measurements. 
    The scattering problem is described by

    \[
        \begin{cases}
            \Delta u +\kappa^2 u = 0 & \text{ in } \mathbb{R}^2\backslash\overline{D}\\
             u = 0  & \text{ on } \partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u^s}{\partial r}-i\kappa u^s)=0 & \text{ for } r=|x|,
        \end{cases}
    \]

    where \(u=u^s+u^i)\ is the total field and \(D)\ is a bounded obstacle in \(\mathbb{R}^2)\ with \(\partial D\in\mathcal{C}^2)\.
    
    Attributes
    ----------
    domain : StarTrigDiscr or GenTrigDiscr
        The domain that represents the boundary curves. Actually, any star shaped curve
        vector space that can compute derivatives along the curve and derivatives wrt. coefficient
        perturbations works.
    kappa : complex
        Wave number.
    N_ieq : int
        Number of discrete boundary points.
    N_ieq_synth : int
        Number of discretization points for the boundary integral equation when computing synthetic data. 
    N_inc : int
        Number of incident direction.
    N_meas : int
        Number of measurement direction.
    N_FK : int
        Number of Fourier coefficients.

    References
    ----------
    - T. Hohage "Logarithmic convergence rates of the iteratively regularized
      Gauss–Newton method for an inverse potential and an inverse scattering problem", Inverse
      Problems, 13 (1997) 1279–1299.
    """

    def __init__(self, kappa, true_curve, N_ieq_synth, N_ieq, N_inc, N_meas, N_FK, **kwargs):
        self.bd_ex = StarCurveDiscr(2*N_ieq_synth)
        """Exact curve class. 2*N_ieq_synth is the number of discretization points for the boundary integral 
        equation when computing synthetic data (choose different to N_ieq to avoid inverse crime)."""
        self.bd_ex_curve=self.bd_ex.bd_eval(true_curve, 2*N_ieq_synth, 3)
        """Compute the grid points of the exact boundary and derivatives of the parametrization
            and save these quantities as members of bd_ex set up the boudary integral operator."""
        self.kappa = kappa 
        """Wave number."""          
        self.N_ieq = N_ieq
        """(2*self.N_ieq) is the number of discrete boundary points."""
        self.N_inc = N_inc
        """Number of incident direction."""
        t=2*np.pi*np.arange(0, self.N_inc)/self.N_inc
        self.inc_directions = np.append(np.cos(t), np.sin(t)).reshape((2, self.N_inc))
        """Incident direction."""
        self.N_meas = N_meas
        """Number of measurement direction."""
        t= 2*np.pi*np.arange(0, self.N_meas)/self.N_meas
        self.meas_directions = np.append(np.cos(t), np.sin(t)).reshape((2, self.N_meas))
        """Measurement direction."""
        self.N_FK = N_FK
        """Number of Fourier coefficients."""
        self.domain_curve = None
        self.dudn=None  
        """Normal derivative of total field at boundary.""" 
        self.w_sl=-1*complex(0,1)*self.kappa
        self.w_dl=1
        """Weights of single and double layer potentials. Use a mixed single and double layer potential ansatz with
        weights w_sl and w_dl."""
        self.L=None
        self.U=None
        self.perm=None
        """LU factors + permuation for integral equation matrix."""
        self.FF_combined=None
        self.Y_dim=codomain.size
        assert self.Y_dim == np.size(self.meas_directions, 1)*np.size(self.inc_directions, 1)
        
        super().__init__(
            domain=GenTrigDiscr(64),
            codomain=UniformGridFcts(np.linspace(0, 2*np.pi, self.Y_dim, endpoint=False), dtype=complex),
            linear=False
        )

    def _create_synthetic_data(self, **kwargs):
        
        wdlTmp=self.w_dl
        self.w_dl=0
        
        Iop_data = setup_iop_data(self.bd_ex, self.kappa)
       
        if self.w_sl!=0:
            Iop = self.w_sl*op_S(self.bd_ex, Iop_data)

        else:
            Iop = np.zeros(np.size(self.bd_ex_curve.z, 1), np.size(self.bd_ex_curve.z, 1))
        if self.w_dl!=0:
            Iop = Iop + self.w_dl*(np.diag(self.bd_ex_curve.zpabs)+ op_K(self.bd_ex, Iop_data))
            
        FF_combined = farfield_matrix(self.bd_ex, self.meas_directions, self.kappa, self.w_sl, self.w_dl)

        farfield = []

        for l in range(0, np.size(self.inc_directions, 1)):
            rhs = -2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].reshape((1,2)).dot(self.bd_ex_curve.z))*self.bd_ex_curve.zpabs
            rhs=rhs.flatten()
            phi = scla.solve(Iop, rhs)
            complex_farfield=FF_combined.dot(phi)
            farfield=np.append(farfield, complex_farfield)

        self.w_dl=wdlTmp
        return farfield
    
    def _eval(self, coeff, **kwargs):
        self.domain_curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)
        Iop_data = setup_iop_data(self.domain_curve, self.kappa)

        if self.w_sl!=0:
            Iop = self.w_sl*op_S(self.domain_curve, Iop_data)
        else:
            Iop = np.zeros(np.size(self.domain.curve,1),np.size(self.domain.curve,1))
        if self.w_dl!=0:
            Iop = Iop + self.w_dl*(np.diag(self.domain_curve.zpabs)+op_K(self.domain_curve,Iop_data))

        self.dudn = np.zeros((2*self.N_ieq, np.size(self.inc_directions,1)), dtype=complex)
        FF_SL = farfield_matrix(self.domain_curve,self.meas_directions,self.kappa,-1.,0.)

        self.perm_mat, self.L, self.U = scla.lu(Iop)
        self.perm = self.perm_mat.dot(np.arange(0, np.size(self.domain_curve.z,1)))
        self.FF_combined = farfield_matrix(self.domain_curve,self.meas_directions,self.kappa, \
                                           self.w_sl,self.w_dl)
        farfield = []

        for l in range(0, np.size(self.inc_directions, 1)):
            rhs = 2*np.exp(complex(0,1)*self.kappa*self.inc_directions[:,l].T.dot(self.domain_curve.z))*  \
                (self.w_dl*complex(0,1)*self.kappa*self.inc_directions[:,l].T.dot(self.domain_curve.normal) \
                                         +self.w_sl*self.domain_curve.zpabs)

            self.dudn[:, l] = np.linalg.solve(self.L.T, \
                     np.linalg.solve(self.U.T, rhs[self.perm.astype(int)]))
            complex_farfield = np.dot(FF_SL, self.dudn[:,l])
            farfield = np.append(farfield, complex_farfield)

        return farfield

    def _derivative(self, h):
            der = []
            for l in range(0, np.size(self.inc_directions, 1)):
                rhs = - 2*self.dudn[:,l]*(self.domain_curve.der_normal(h))*(self.domain_curve.zpabs.T)
                phi = np.linalg.solve(self.U, np.linalg.solve(self.L, rhs[self.perm.astype(int)]))
                complex_farfield = self.FF_combined.dot(phi)
              
                der = np.append(der, complex_farfield)
            return der

    def _adjoint(self, g):
             
            res = np.zeros(2*self.N_ieq)
            rhs = np.zeros(2*self.N_ieq, dtype=complex)
            N_FF = np.size(self.meas_directions,1)

            for  l in range(0, np.size(self.inc_directions,1)):
                g_complex = g[(l)*N_FF+np.arange(0, N_FF)]
                phi = self.FF_combined.T.conjugate().dot(g_complex)

                rhs[self.perm.astype(int)] = np.linalg.solve(self.L.T.conjugate(), \
                np.linalg.solve(self.U.T.conjugate(), phi))
                
                res = res-2*(rhs*np.conjugate(self.dudn[:,l])).real

            adj = self.domain_curve.adjoint_der_normal(res*self.domain_curve.zpabs)

            return adj
