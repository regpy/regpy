import numpy as np
import scipy.linalg as scla
import os
import sys
sys.path.append(os.path.dirname(__file__))
from functions.operator import op_S
from functions.operator import op_K
from functions.farfield_matrix import farfield_matrix
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator
from regpy.vecsps import GridFcts
from regpy.vecsps.curve import GenTrigDiscr





class DirichletOp(Operator):
    r"""Operator that maps the shape of a sound-soft obstacle to the far-field measurements. 
    The scattering problem is described by

	.. math::
        \begin{cases}
            \Delta u +\kappa^2 u = 0 & \text{ in } \mathbb{R}^2\backslash\overline{D}\\
             u = 0  & \text{ on } \partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u^s}{\partial r}-i\kappa u^s)=0 & \text{ for } r=|x|,
        \end{cases}


    where \(u=u^s+u^i)\ is the total field and \(D)\ is a bounded obstacle in \(\mathbb{R}^2)\ with \(\partial D\in\mathcal{C}^2)\.
    
    Attributes
    ----------
    kappa : complex
        Wave number.
    N_ieq : int
        Number of discrete boundary points.
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

    def __init__(self, kappa, N_ieq=128, N_inc=4, N_meas=64, N_FK=64, **kwargs):   
        self.kappa = kappa 
        """Wave number."""          
        self.N_ieq = N_ieq
        """(2*self.N_ieq) is the number of discrete boundary points."""
        if isinstance(N_inc, int) and N_inc > 0:
            self.N_inc = N_inc
            """Number of incident direction."""
            t=2*np.pi*np.arange(0, self.N_inc)/self.N_inc
            self.inc_directions=[np.array([np.cos(s), np.sin(s)]) for s in t]
            """Incident direction."""
        elif isinstance(N_inc, list) and all([dir.shape == (2,) for dir in N_inc]):
            self.N_inc = len(N_inc)
            """Number of incident direction."""
            self.inc_directions = N_inc 
            """Incident direction."""
        else: 
            raise ValueError("Incident direction neither an arry of direction nor an positiv integer")

        if isinstance(N_meas, int) and N_meas > 0:
            self.N_meas = N_meas
            """Number of measurement direction."""
            t=2*np.pi*np.arange(0, self.N_meas)/self.N_meas
            self.meas_directions=[np.array([np.cos(s), np.sin(s)]) for s in t]
            """Measurement direction."""
        elif isinstance(N_meas, list) and all([meas.shape == (2,) for meas in N_meas]):
            self.N_meas = len(N_meas)
            """Number of Measurement direction."""
            self.meas_directions = N_meas 
            """Measurement direction."""
        else: 
            raise ValueError("Measurement direction neither an arry of direction nor an positiv integer")

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

        meas_dir=np.linspace(0, 2*np.pi, self.N_meas, endpoint=False)
        inc_dir=np.linspace(0, 2*np.pi, self.N_inc, endpoint=False)
        codomain=GridFcts(meas_dir, inc_dir, dtype=complex)

        super().__init__(
            domain=GenTrigDiscr(2*self.N_FK),
            codomain=codomain,
            linear=False
        )
    
    def _eval(self, coeff, **kwargs):

        self.domain_curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)
        Iop_data = setup_iop_data(self.domain_curve, self.kappa)

        if self.w_sl!=0:
            Iop = self.w_sl*op_S(self.domain_curve, Iop_data)
        else:
            Iop = np.zeros(np.size(self.domain.curve,1),np.size(self.domain.curve,1))
        if self.w_dl!=0:
            Iop = Iop + self.w_dl*(np.diag(self.domain_curve.zpabs)+op_K(self.domain_curve,Iop_data))

        self.dudn = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        FF_SL = farfield_matrix(self.domain_curve,self.meas_directions,self.kappa,-1.,0.)

        self.perm_mat, self.L, self.U = scla.lu(Iop)
        self.perm = self.perm_mat.dot(np.arange(0, np.size(self.domain_curve.z,1)))
        self.FF_combined = farfield_matrix(self.domain_curve,self.meas_directions,self.kappa, \
                                           self.w_sl,self.w_dl)
        
        farfield = np.zeros((self.N_meas,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs = 2*np.exp(complex(0,1)*self.kappa*dir.dot(self.domain_curve.z))*  \
                (self.w_dl*complex(0,1)*self.kappa*dir.dot(self.domain_curve.normal) \
                                         +self.w_sl*self.domain_curve.zpabs)
            self.dudn[:, l] = np.linalg.solve(self.L.T, \
                     np.linalg.solve(self.U.T, rhs[self.perm.astype(int)]))
            farfield[:,l] = np.dot(FF_SL, self.dudn[:,l])
        return farfield

    def _derivative(self, h):
            der = np.zeros((self.N_meas,self.N_inc),dtype=complex)
            for l in range(0, self.N_inc):
                rhs = - 2*self.dudn[:,l]*(self.domain_curve.der_normal(h))*(self.domain_curve.zpabs.T)
                phi = np.linalg.solve(self.U, np.linalg.solve(self.L, rhs[self.perm.astype(int)]))
                der[:,l] = self.FF_combined.dot(phi)
            return der

    def _adjoint(self, g):
            res = np.zeros(2*self.N_ieq, dtype=float)
            rhs = np.zeros(2*self.N_ieq, dtype=complex)

            for  l in range(0, self.N_inc):
                phi = self.FF_combined.T.conjugate().dot(g[:,l])

                rhs[self.perm.astype(int)] = np.linalg.solve(self.L.T.conjugate(), \
                np.linalg.solve(self.U.T.conjugate(), phi))
                
                res += -2*(rhs*np.conjugate(self.dudn[:,l])).real

            adj = self.domain_curve.adjoint_der_normal(res*self.domain_curve.zpabs)

            return adj


def create_synthetic_data(Dir_op, true_curve, **kwargs):
    wdlTmp=1*Dir_op.w_dl
    Dir_op.w_dl=0

    Iop_data = setup_iop_data(true_curve, Dir_op.kappa)
    if Dir_op.w_sl!=0:
        Iop = Dir_op.w_sl*op_S(true_curve, Iop_data)
    else:
        Iop = np.zeros(np.size(true_curve.z, 1), np.size(true_curve.z, 1))
    if Dir_op.w_dl!=0:
        Iop = Iop + Dir_op.w_dl*(np.diag(true_curve.zpabs) + op_K(true_curve, Iop_data))
        
    FF_combined = farfield_matrix(true_curve, Dir_op.meas_directions, Dir_op.kappa, Dir_op.w_sl, Dir_op.w_dl)

    farfield = np.zeros((Dir_op.N_meas, Dir_op.N_inc),dtype = complex)
    for l, dir in enumerate(Dir_op.inc_directions):
        rhs = -2*np.exp(complex(0,1)*Dir_op.kappa*dir.dot(true_curve.z))*true_curve.zpabs
        rhs=rhs.flatten()
        phi = scla.solve(Iop, rhs)
        farfield[:,l]=FF_combined.dot(phi)

    Dir_op.w_dl=wdlTmp
    return farfield, true_curve