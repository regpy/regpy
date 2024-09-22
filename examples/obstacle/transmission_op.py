import numpy as np
import scipy.linalg as scla

from functions.operator import op_S
from functions.operator import op_T
from functions.operator import op_K

from functions.farfield_matrix import farfield_matrix_trans
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator

from regpy.vecsps.curve import StarCurveDiscr
from regpy.vecsps import GridFcts
from regpy.vecsps.curve import GenTrigDiscr


class TransmissionOp(Operator):
    r"""Operator that maps an admissible boundary \partial D onto the corresponding far field pattern. 
    The related transmission problem for the Helmholtz equation is modeled by

	.. math::
        \begin{cases}
            \Delta u_i +\kappa_i^2 u_i = 0 & \text{ in } D \\
            \Delta u_e +\kappa_e^2 u_e = 0 & \text{ in } \mathbb{R}^2\backslash\overline{D}\\
             u_i=u & \text{ on } \partial D\\
            \frac{\partial u_i}{\partial\nu}=\rho\frac{\partial u}{\partial\nu} & \text{ on }\partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u_e}{\partial r}-i\kappa u_e)=0 &\text{ for } r=|x|.
        \end{cases}

    where \rho\in\mathbb{C}\backslash 0, \(u=u_e+u^{inc}\) is the total field in \mathbb{R}^2\backslash\overline{D}.

    References
    ----------
    see T. Hohage & C. Schormann. "A Newton-type method for a transmission
    problem in inverse scattering", Inverse Problems, 14 (1998), 1207-1227."""
    
    def __init__(self, kappa_in, kappa_ex, rho=4.3-6*complex(0,1), N_ieq=128, N_inc=4, N_meas=64, N_FK=64, **kwargs):
        self.N_ieq = N_ieq          
        """(2*self.N_ieq) is the number of discrete boundary points."""
        self.kappa_in = kappa_in         
        """Interior wave number."""
        self.kappa_ex = kappa_ex          
        """Exterior wave number."""
        self.rho = rho
        """Density ratio."""

        self.w_sl_ex = -1
        self.w_dl_ex = 1
        self.w_sl_in = rho 
        self.w_dl_in = -1
        """Weights of single and double layer potentials."""

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
            raise ValueError("Measurement direction neither an arry of direction nor an positiv integer")
        
        self.N_FK = N_FK
        """Number of Fourier coefficients."""

        self.domain_curve = None
        self.FF_combined = None
        
        self.Iop = None
        self.dudn = None
        self.ue       = None
        self.duednu   = None
        self.ui       = None
        self.duidnu   = None
        self.uinc     = None
        self.duincdnu = None
        self.u        = None

        meas_dir = np.linspace(0, 2*np.pi, self.N_meas, endpoint=False)
        inc_dir = np.linspace(0, 2*np.pi, self.N_inc, endpoint=False)
        codomain = GridFcts(meas_dir, inc_dir, dtype=complex)

        super().__init__(
            domain = GenTrigDiscr(2*self.N_FK),
            codomain = codomain,
            linear = False
        )


    def _eval(self, coeff, differentiate=False):
        self.domain_curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)

        Iop_data_ex = setup_iop_data(self.domain_curve, self.kappa_ex)
        Iop_data_in = setup_iop_data(self.domain_curve, self.kappa_in)

        Iop1 = self.w_dl_ex*op_K(self.domain_curve, Iop_data_ex)+\
                        self.w_dl_in*op_K(self.domain_curve, Iop_data_in)+(self.w_dl_ex-self.w_dl_in-4)*np.diag(self.domain_curve.zpabs)  
        Iop2 = self.w_sl_ex*op_S(self.domain_curve, Iop_data_ex)+self.w_sl_in*op_S(self.domain_curve, Iop_data_in)
        Iop3 = self.w_dl_ex*op_T(self.domain_curve, Iop_data_ex)+self.w_dl_in*op_T(self.domain_curve, Iop_data_in)
        Iop4 = self.w_sl_ex*op_K(self.domain_curve, Iop_data_ex).T+self.w_sl_in*op_K(self.domain_curve, Iop_data_in).T+\
                        (self.w_sl_in-2*self.rho-self.w_sl_ex-2)*np.diag(self.domain_curve.zpabs)
    
        upper_Iop = np.hstack((Iop1, Iop2))
        lower_Iop = np.hstack((Iop3, Iop4))
        Iop = np.vstack((upper_Iop, lower_Iop))

        R1  = -self.w_dl_in*op_K(self.domain_curve, Iop_data_in)+(self.w_dl_in+2)*np.diag(self.domain_curve.zpabs) 
        R2  = -self.w_sl_in*op_S(self.domain_curve, Iop_data_in) 
        R3  = -self.w_dl_in*op_T(self.domain_curve, Iop_data_in) 
        R4  = -self.w_sl_in*op_K(self.domain_curve, Iop_data_in).T+(2*self.rho-self.w_sl_in)*np.diag(self.domain_curve.zpabs)
       
        upper_R = np.hstack((R1, R2))
        lower_R = np.hstack((R3, R4))
        R = np.vstack((upper_R, lower_R))

        self.Iop = np.linalg.inv(Iop).dot(R)

        self.FF_combined = farfield_matrix_trans(self.domain_curve, self.meas_directions,\
                                    self.kappa_ex, self.w_sl_ex, self.w_dl_ex)

        farfield = np.zeros((self.N_meas, self.N_inc), dtype=complex)
        self.dudn = np.zeros((4*self.N_ieq, self.N_inc), dtype=complex)

        self.ue       = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        self.duednu   = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        self.ui       = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        self.duidnu   = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        self.uinc     = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        self.duincdnu = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        self.u        = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)

        for l, dir in enumerate(self.inc_directions):
            self.uinc[:, l] = (np.exp(1*complex(0,1)*self.kappa_ex*dir.dot(self.domain_curve.z))).T

            self.duincdnu[:, l] = np.exp(1*complex(0,1)*self.kappa_ex*dir.dot(self.domain_curve.z))*\
                     (complex(0,1)*self.kappa_ex*dir.dot(self.domain_curve.normal))/self.domain_curve.zpabs
            
            rhs = np.vstack((self.uinc[:, l], self.duincdnu[:, l]))
            rhs = rhs.flatten() 

            self.dudn[:,l] = self.Iop.dot(rhs)
            farfield[:,l] = self.FF_combined.dot(self.dudn[:, l])
      
            self.ue[:,l] = self.dudn[0:2*self.N_ieq, l]
            self.duednu[:,l] = self.dudn[2*self.N_ieq:4*self.N_ieq, l]
            self.ui[:,l] = self.ue[:,l]+self.uinc[:, l] 
            self.duidnu[:,l] = self.rho*(self.duednu[:, l]+self.duincdnu[:, l])
            self.u[:,l] = self.ui[:,l]-self.rho*(self.ue[:, l]+self.uinc[:, l])
            

    def _derivative(self, h): 
            der = np.zeros((self.N_meas, self.N_inc), dtype=complex)
            hn  = self.domain_curve.der_normal(h)

            for l in range(0, self.N_inc):
                duds  = self.domain_curve.arc_length_der(self.u[:,l])
                rhs_a = hn*(self.duednu[:,l]+self.duincdnu[:,l]-self.duidnu[:,l])
                rhs_b = self.domain_curve.arc_length_der(hn*duds)+self.kappa_in**2*hn\
                        *self.ui[:,l]-self.rho*self.kappa_ex**2*hn*(self.ue[:,l]+self.uinc[:,l])
                rhs_b = rhs_b/self.rho
                rhs = np.vstack((rhs_a, rhs_b))
                rhs = rhs.flatten() 
                phi = self.Iop.dot(rhs)
                der[:,l] = self.FF_combined.dot(phi)
              
            return der


    def _adjoint(self, g):
            res = complex(0,1)*np.zeros(2*self.N_ieq)
            for l in range(0, self.N_inc):
                phi = self.FF_combined.T.conjugate().dot(g[:,l]) 
                rhs   = self.Iop.T.conjugate().dot(phi)

                rhs_a = rhs[0:2*self.N_ieq]
                rhs_b = rhs[2*self.N_ieq:4*self.N_ieq]
        
                res = res +np.real(np.conjugate(self.duednu[:,l]+self.duincdnu[:,l]-\
                                                self.duidnu[:,l])*rhs_a \
                    -np.conjugate(self.domain_curve.arc_length_der(self.u[:,l])/self.rho)*\
                    self.domain_curve.arc_length_der(rhs_b/self.domain_curve.zpabs.T)*self.domain_curve.zpabs.T \
                    +np.conjugate(self.kappa_in**2*self.ui[:,l]/self.rho-\
                                  self.kappa_ex**2*(self.ue[:,l]+self.uinc[:,l]))*rhs_b)
            adj = self.domain_curve.adjoint_der_normal(res)

            print(adj)
            return adj
    

def create_synthetic_data(Trans_op, true_curve, N_ieq_synth=64, **kwargs):

    bd_ex = StarCurveDiscr(2*N_ieq_synth)
    bd_ex_curve=bd_ex.bd_eval(true_curve, 3)

    Iop_data_ex = setup_iop_data(bd_ex, Trans_op.kappa_ex)
    Iop_data_in = setup_iop_data(bd_ex, Trans_op.kappa_in)
     
    Iop1 = Trans_op.w_dl_ex*op_K(bd_ex, Iop_data_ex)+Trans_op.w_dl_in*op_K(bd_ex, Iop_data_in)+(Trans_op.w_dl_ex-Trans_op.w_dl_in-4)*np.diag(bd_ex_curve.zpabs)
    Iop2 = Trans_op.w_sl_ex*op_S(bd_ex, Iop_data_ex)+Trans_op.w_sl_in*op_S(bd_ex, Iop_data_in)
    Iop3 = Trans_op.w_dl_ex*op_T(bd_ex, Iop_data_ex)+Trans_op.w_dl_in*op_T(bd_ex, Iop_data_in)
    Iop4 = Trans_op.w_sl_ex*op_K(bd_ex, Iop_data_ex).T+Trans_op.w_sl_in*op_K(bd_ex, Iop_data_in).T+(Trans_op.w_sl_in-2*Trans_op.rho-Trans_op.w_sl_ex-2)*np.diag(bd_ex_curve.zpabs)
    
    upper_Iop = np.hstack((Iop1, Iop2))
    lower_Iop = np.hstack((Iop3, Iop4))
    Iop = np.vstack((upper_Iop, lower_Iop))
  
    R1 = -Trans_op.w_dl_in*op_K(bd_ex, Iop_data_in)+(Trans_op.w_dl_in+2)*np.diag(bd_ex_curve.zpabs)
    R2 = -Trans_op.w_sl_in*op_S(bd_ex, Iop_data_in)
    R3 = -Trans_op.w_dl_in*op_T(bd_ex, Iop_data_in)
    R4 = -Trans_op.w_sl_in*op_K(bd_ex, Iop_data_in).T+(2*Trans_op.rho-Trans_op.w_sl_in)*np.diag(bd_ex_curve.zpabs)

    upper_R = np.hstack((R1, R2))
    lower_R = np.hstack((R3, R4))
    R = np.vstack((upper_R, lower_R))

    Iop = np.linalg.inv(Iop).dot(R)

    FF_combined = farfield_matrix_trans(bd_ex, Trans_op.meas_directions, Trans_op.kappa_ex, Trans_op.w_sl_ex, Trans_op.w_dl_ex)
    farfield = np.zeros((Trans_op.N_meas, Trans_op.N_inc), dtype = complex)
    for l, dir in enumerate(Trans_op.inc_directions):

        rhs_a = np.exp(complex(0,1)*Trans_op.kappa_ex*dir.dot(bd_ex_curve.z))
        rhs_b = np.exp(complex(0,1)*Trans_op.kappa_ex*dir.dot(bd_ex_curve.z))\
            *(complex(0,1)*Trans_op.kappa_ex*(dir.dot(bd_ex_curve.normal)))/bd_ex_curve.zpabs
        rhs = np.vstack((rhs_a, rhs_b))
        rhs = rhs.flatten() 
        phi = Iop.dot(rhs)
        farfield[:,l] = FF_combined.dot(phi)
    return farfield, bd_ex_curve