import logging
from regpy.solvers import RegSolver


class CGNE(RegSolver):
    """
    The conjugate gradient method applied to the normal equation \(T^*T=T^*g\) for solving linear inverse problems \(Tf=g\).
    Regularization is achieved by early stopping, typically using the discrepancy principle. 

    Parameters: 
    setting: RegularizationSetting
       Regularization setting involving Hilbert space norms

    data: array-like
        Right hand side g

    x0: array-like, default:None
        First iteration. zero() if None

    logging_level: default: logggin.INFO
        Controls amount of output
    """
    def __init__(self, setting, data, x0 =None, logging_level = logging.INFO):
        assert setting.op.linear

        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.x0 = x0
        """The zero-th CG iterate. x0=Null corresponds to xref=zeros()"""

        if x0 is not None:
            self.x = x0.copy()
            """The current iterate."""
            self.y = self.op(self.x)
            """The image of the current iterate under the operator."""
        else:
            self.x = self.op.domain.zeros()
            self.y = self.op.codomain.zeros()

        self.g_res = self.op.adjoint(self.h_codomain.gram(data-self.y)) 
        """The gram matrix applied to the residual of the normal equation. 
        g_res = T^* G_Y (data-T self.x)  in each iteration with operator T and Gram matrices G_x, G_Y.
        """
        res = self.h_domain.gram_inv(self.g_res)
        """The residual of the normal equation."""
        self.sq_norm_res = (self.op.domain.vec_type.vdot(self.g_res, res)).real
        """The squared norm of the residual."""
        self.dir = res
        """The direction of descent."""
        self.g_dir = self.g_res.copy()
        """The Gram matrix applied to the direction of descent."""

    def _next(self):
        Tdir = self.op(self.dir)
        g_Tdir = self.h_codomain.gram(Tdir)
        alpha = self.sq_norm_res / (self.op.codomain.vec_type.vdot(g_Tdir, Tdir)).real

        self.x += alpha * self.dir

        self.y += alpha * Tdir

        self.g_res -= alpha * (self.op.adjoint(g_Tdir) )
        res = self.h_domain.gram_inv(self.g_res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = (self.op.codomain.vec_type.vdot(self.g_res, res)).real
        beta = self.sq_norm_res / sq_norm_res_old

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res