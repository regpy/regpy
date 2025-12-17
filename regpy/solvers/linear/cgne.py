from regpy.util import Errors

from ..general import RegSolver

__all__ = ["CGNE"]

class CGNE(RegSolver):
    r"""
    The conjugate gradient method applied to the normal equation :math:`T^*T=T^*g` for solving linear inverse problems :math:`Tf=g`.
    Regularization is achieved by early stopping, typically using the discrepancy principle. 

    Parameters
    ----------
    setting: regpy.solvers.Setting
       Regularization setting involving Hilbert space norms
    data: array-like default: None
        Right hand side g. If it is None the data is taken from the setting.
    x0: array-like, default:None
        First iteration. zero() if None
    logging_level: default: logging.INFO
        Controls amount of output
    """
    def __init__(self, setting, data=None, x0 =None, logging_level = "INFO"):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="CGNE requires the operator to be linear!"))
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if data not in self.op.codomain:
            raise ValueError(Errors.not_in_vecsp(data,self.op.codomain,vec_name="data",space_name="codomain"))
        if x0 is not None and x0 not in self.op.domain:
            raise ValueError(Errors.not_in_vecsp(x0,self.op.domain,vec_name="first iteration",space_name="domain"))
        
        self.log.setLevel(logging_level)
        self.x0 = x0
        r"""The zero-th CG iterate. x0=Null corresponds to xref=zeros()"""

        if x0 is not None:
            self.x = x0.copy()
            """The current iterate."""
            self.y = self.op(self.x)
            """The image of the current iterate under the operator."""
        else:
            self.x = self.op.domain.zeros()
            self.y = self.op.codomain.zeros()

        self.g_res = self.op.adjoint(self.h_codomain.gram(data-self.y)) 
        r"""The gram matrix applied to the residual of the normal equation. 
        :math:`g_res = T^* G_Y (data-T self.x)`  in each iteration with operator T and Gram matrices G_x, G_Y.
        """
        res = self.h_domain.gram_inv(self.g_res)
        """The residual of the normal equation."""
        self.sq_norm_res = (self.op.domain.vdot(self.g_res, res)).real
        """The squared norm of the residual."""
        self.dir = res
        """The direction of descent."""
        self.g_dir = self.g_res.copy()
        """The Gram matrix applied to the direction of descent."""

    def _next(self):
        Tdir = self.op(self.dir)
        g_Tdir = self.h_codomain.gram(Tdir)
        alpha = self.sq_norm_res / (self.op.codomain.vdot(g_Tdir, Tdir)).real

        self.x += alpha * self.dir

        self.y += alpha * Tdir

        self.g_res -= alpha * (self.op.adjoint(g_Tdir) )
        res = self.h_domain.gram_inv(self.g_res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = (self.op.codomain.vdot(self.g_res, res)).real
        beta = self.sq_norm_res / sq_norm_res_old

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res