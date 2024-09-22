import numpy as np
import scipy.linalg as scla

from functions.operator import op_T
from functions.operator import op_K

from functions.farfield_matrix import farfield_matrix
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator

from regpy.vecsps.curve import StarCurveDiscr
from regpy.vecsps import GridFcts
from regpy.vecsps.curve import GenTrigDiscr

class NeumannOp(Operator):
    r"""Operator that maps the shape of a sound-hard obstacle to the far-field measurements. 
    The scattering problem is modeled by

	.. math::
        \begin{cases}
            \Delta u +\kappa^2 u = 0 & \text{ in }\mathbb{R}^2\backslash\overline{D}\\
            \frac{\partial u}{\partial\nu}=0  & \text{ on }\partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u^s}{\partial r}-i\kappa u^s)=0 &\text{ for } r=|x|.
        \end{cases}


    where \(u=u^s+u^i\) is the total field and \(D\) is the bounded obstacle in \mathbb{R}^2 with \(\partial D\in\mathcal{C}^2\).
    
    References
    ----------
    - T. Hohage. "Convergence rates of a regularized Newton method in sound-hard inverse scattering", 
    SIAM journal on numerical analysis, 36 (1998): 125-142."""

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
            raise ValueError("Incident direction neither an arry of direction nor a positive integer")

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
            raise ValueError("Measurement direction neither an arry of direction nor a positive integer")

        self.N_FK = N_FK
        """Number of Fourier coefficients."""
        self.domain_curve = None
        self.u=None  
        """Values of total field at boundary."""
        self.w_sl = -complex(0,1)*self.kappa
        self.w_dl = 1
        """Weights of single and double layer potentials."""
        self.L = None
        self.U = None
        self.perm = None
        """LU factors + permuation for integral equation matrix."""
        self.FF_combined=None

        meas_dir = np.linspace(0, 2*np.pi, self.N_meas, endpoint=False)
        inc_dir = np.linspace(0, 2*np.pi, self.N_inc, endpoint=False)
        codomain = GridFcts(meas_dir, inc_dir, dtype=complex)

        super().__init__(
            domain=GenTrigDiscr(2*self.N_FK),
            codomain=codomain,
            linear=False
        )


    def _eval(self, coeff, differentiate=False): 

        self.domain_curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)
        Iop_data = setup_iop_data(self.domain_curve, self.kappa)

        if self.w_dl!=0:
            Iop = self.w_dl*op_T(self.domain_curve, Iop_data)
          
        else:
            Iop = np.zeros(np.size(self.domain_curve, 1), np.size(self.domain_curve, 1))
        
        if self.w_sl!=0:
            Iop = Iop + self.w_sl*(op_K(self.domain_curve, Iop_data).T - np.diag(self.domain_curve.zpabs))
        

        self.u = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        FF_DL = farfield_matrix(self.domain_curve, self.meas_directions, self.kappa, 0, 1)
        

        self.perm_mat, self.L, self.U =scla.lu(Iop)
        self.perm=self.perm_mat.dot(np.arange(0, np.size(self.domain_curve.z,1)))
        self.FF_combined = farfield_matrix(self.domain_curve, self.meas_directions, self.kappa,\
                                           self.w_sl, self.w_dl)
        
        farfield = np.zeros((self.N_meas, self.N_inc), dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs = -2*np.exp(complex(0,1)*self.kappa*dir.dot(self.domain_curve.z))*\
                (self.w_dl*complex(0,1)*self.kappa*dir.dot(self.domain_curve.normal)\
                                         +self.w_sl*self.domain_curve.zpabs)
            self.u[:, l] = np.linalg.solve(self.L.T,\
                     np.linalg.solve(self.U.T, rhs[self.perm.astype(int)]))
            
            farfield[:,l] = np.dot(FF_DL, self.u[:,l])
   
        return farfield
    

    def _derivative(self, h):
            der = np.zeros((self.N_meas, self.N_inc),dtype=complex)
            for l in range(0, self.N_inc):
                duds = self.domain_curve.arc_length_der(self.u[:,l])
                hn = self.domain_curve.der_normal(h)
                rhs = self.domain_curve.arc_length_der(hn*duds) + self.kappa**2* hn*(self.u[:,l])
                rhs = 2*rhs*(self.domain_curve.zpabs.T)
                phi=np.linalg.solve(self.U, np.linalg.solve(self.L, rhs[self.perm.astype(int)]))
                der[:,l] = self.FF_combined.dot(phi)
            return der


    def _adjoint(self, g):
            res = complex(0,1)*np.zeros(2*self.N_ieq)
            v = complex(0,1)*np.zeros(2*self.N_ieq)

            for l in range(0, self.N_inc):
                phi = self.FF_combined.T.conjugate().dot(g[:,l])
                v[self.perm.astype(int)] = np.linalg.solve(self.L.T.conjugate(),\
                np.linalg.solve(self.U.T.conjugate(), phi))
                
                dvds = self.domain_curve.arc_length_der(v)
                duds = self.domain_curve.arc_length_der(self.u[:,l])
                res = res -2*(np.conjugate(dvds)*duds - self.kappa**2*np.conjugate(v)*\
                              self.u[:,l]).real
            adj = self.domain_curve.adjoint_der_normal(res*self.domain_curve.zpabs.T)
            return adj


def create_synthetic_data(Neu_op, true_curve, N_ieq_synth=64, **kwargs):
    bd_ex = StarCurveDiscr(2*N_ieq_synth)
    bd_ex_curve=bd_ex.bd_eval(true_curve, 3)
    
    Iop_data = setup_iop_data(bd_ex, Neu_op.kappa)

    if Neu_op.w_dl!=0:
        Iop = Neu_op.w_dl*(op_T(bd_ex, Iop_data))
    else:
        Iop = np.zeros(np.size(bd_ex_curve.z, 1), np.size(bd_ex_curve.z, 1))

    if Neu_op.w_sl!=0:
        Iop = Iop + Neu_op.w_sl*(op_K(bd_ex, Iop_data).T-np.diag(bd_ex_curve.zpabs))
   
    FF_combined = farfield_matrix(bd_ex, Neu_op.meas_directions, Neu_op.kappa, Neu_op.w_sl, Neu_op.w_dl)
    farfield = np.zeros((Neu_op.N_meas, Neu_op.N_inc), dtype = complex)

    for l, dir in enumerate(Neu_op.inc_directions):
        rhs = -2*np.exp(complex(0,1)*Neu_op.kappa*dir.dot(bd_ex_curve.z))*(complex(0,1)*Neu_op.kappa*dir.dot(bd_ex_curve.normal))
        rhs = rhs.flatten()
        phi = scla.solve(Iop, rhs)
        farfield[:,l]=FF_combined.dot(phi)

    return farfield, bd_ex_curve