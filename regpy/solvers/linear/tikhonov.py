from math import sqrt,inf

from regpy.util import Errors
from regpy.functionals.base import SquaredNorm
from regpy.operators import Identity,Operator
from regpy.stoprules import CountIterations

from ..general import RegSolver, Setting

__all__ = ["TikhonovCG","TikhonovAlphaGrid","NonstationaryIteratedTikhonov"]

class TikhonovCG(RegSolver):
    r"""The Tikhonov method for linear inverse problems. Minimizes
    
    .. math::
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2

    using a conjugate gradient method. 
    To determine a stopping index yielding guaranteed error bounds, a partial embedded minimal residual method (MR) is 
    used, which can be implemented by updating a scalar parameter in each iteration. 
    For details on the use of the embedded MR method, as proposed by H. Egger in 
    "Numerical realization of Tikhonov regularization: appropriate norms, implementable stopping criteria, and optimal algorithms" 
    in Oberwolfach Reports 9/4, page 3009-3010, 2013;
    see also the Master thesis by Andrea Dietrich 
    "Analytische und numerische Untersuchung eines Abbruchkriteriums für das CG-Verfahren zur Minimierung 
    von Tikhonov Funktionalen", Univ. Göttingen, 2017 

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    data : setting.op.codomain [default: None]
        The measured data. 
        If None, then setting must have SquaredNorm as data fidelity and penalty term. In this case xref is ignored, 
        and if setting is a Setting, then also regpar is ignored.
        If not None, then setting.penalty and setting.data_fid are ignored except for their Hilbert space structures. 
    regpar : float [default:None]
        The regularization parameter. Must be positive. If None, then setting must be a Tikhonov. 
    xref: setting.op.domain [default: None]
        Reference value in the Tikhonov functional. The default is equivalent to xref = setting.op.domain.zeros().
    x0: setting.op.domain  [default: None]
        Starting value of the CG iteration. If None, setting.op.domain.zeros() is used as starting value. 
    tol : float, default: None
        The absolute tolerance - it guarantees that difference of the final CG iterate to the exact minimizer of the Tikhonov functional  
        in setting.h_domain.norm is smaller than tol. If None, this criterion is not active (analogously for reltolx and reltoly).   
        If the noise level is given, it is reasonable value to choose tol in the order of the propagated data noise level, 
        which is noiselevel/2*sqrt(regpar)
    reltolx: float, default: 10/sqrt(regpar)
        Relative tolerance in domain. Guarantees that the relative error w.r.t. setting.h_domain.norm is smaller than reltolx.
        The motivation for the default value is similar to that given for tol, assuming a reasonable 
        signal-to-noise ratio for the Tikhonov minimizer. 
    reltoly: float, default: None
        Relative tolerance in codomain.
    all_tol_criteria: bool (default: True)
        If True, the iteration is stopped if all specified tolerance criteria are satisfied. 
        If False, the iteration is stopped if one criterion is satisfied.
    krylov_basis : Compute orthonormal basis vectors of the Krylov subspaces while running CG solver
    preconditioner : Preconditioner such that the iteration is done on 
        :math:`\Vert TP x - data\Vert^2 + regpar * \Vert Px - xref\Vert^2`
        The iterates (self.x) still solve the original equation without preconditioner.
    """

    def __init__(
        self, setting, data=None, regpar=None, xref=None, 
        x0 =None, 
        tol=None, reltolx=None, reltoly=None, 
        all_tol_criteria = True,
        krylov_basis=None, preconditioner=None,
        logging_level = "INFO"
        ):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="TikhonovCG in as a linear solver requires the operator to be linear!"))
        self.log.setLevel(logging_level)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        if xref is None and setting.penalty_shift is not None:
            xref=setting.penalty_shift
            
        if(setting.is_tikhonov):
            if regpar is not None:
                self.log.warning('Ignoring given value of regularization parameter')
            regpar = (self.penalty.a/self.data_fid.a) * self.regpar
        else:
            if regpar is None:
                raise ValueError(Errors.value_error("If the setting is not a Tikhonov setting the Regularization parameter needs to be specified in TIkhonovCG!"))
            regpar *= self.penalty.a/self.data_fid.a

        self.regpar = regpar
        """The regularization parameter."""
        #self.log.debug('rel. tolerances: {} in domain, {} in codomain, {} reduction residual'.format(reltolx,reltoly,tol))
        self.x0 = x0
        """The zero-th CG iterate. x0=Null corresponds to xref=zeros()"""

        self.tol = tol
        """The absolute tolerance in the domain."""
        self.reltolx = reltolx
        """The relative tolerance in the domain."""
        self.reltoly = reltoly
        """The relative tolerance in the codomain."""
        if tol is None  and reltolx is None and reltoly is None:
            self.reltolx = 10./sqrt(regpar)

        if x0 is not None:
            self.x = x0.copy()
            """The current iterate."""
            self.y = self.op(self.x)
            """The image of the current iterate under the operator."""
        else:
            self.x = self.op.domain.zeros()
            self.y = self.op.codomain.zeros()

        if self.reltolx is not None:
            self.sq_norm_x = 0
        if self.reltoly is not None:
            self.g_y = self.h_codomain.gram(self.y)
            self.norm_y = self.op.codomain.vdot(self.y,self.g_y)
            if self.x0 is not None:
                self.y0 = self.y
                self.g_y0 = self.g_y

        if preconditioner is None:
            self.preconditioner = Identity (self.h_domain.vecsp)
        else: 
            self.preconditioner = preconditioner

        self.g_res = self.op.adjoint(self.h_codomain.gram(data-self.y))
        """The gram matrix applied to the residual of the normal equation. 
        g_res = T^* G_Y (data-T self.x) + regpar G_X(xref-self.x) in each iteration with operator T and Gram matrices G_x, G_Y.
        """
        if xref is not None:
            self.g_res += self.regpar *self.h_domain.gram(xref-self.x)
        elif x0 is not None:
            self.g_res -= self.regpar * self.h_domain.gram(self.x)
        self.g_res=self.preconditioner.adjoint(self.g_res)
        res = self.h_domain.gram_inv(self.g_res)
        """The residual of the normal equation."""
        self.sq_norm_res = self.op.domain.vdot(self.g_res, res).real
        """The squared norm of the residual."""
        self.dir = self.preconditioner(res)
        """The direction of descent."""
        if(isinstance(self.preconditioner,Identity)):
            self.g_dir = self.g_res.copy()
        else:
            self.g_dir = self.h_domain.gram(self.dir)
        """The Gram matrix applied to the direction of descent."""
        self.kappa = 1
        """ratio of the squared norms of the residuals of the CG method and the MR-method.
        Used for error estimation."""
        self.all_tol_criteria = all_tol_criteria
        if self.all_tol_criteria:
            self.isconverged = {'tol': self.tol is None, 'reltolx': self.reltolx is None, 'reltoly': self.reltoly is None}
        else:
            self.isconverged = {'tol': self.tol is not None, 'reltolx': self.reltolx is not None, 'reltoly': self.reltoly is not None}

        self.krylov_basis=krylov_basis
        if self.krylov_basis is not None: 
            self.iteration_number=0
            self.krylov_basis[self.iteration_number, :] = res / sqrt(self.sq_norm_res)
        """In every iteration step of the Tikhonov solver a new orthonormal vector is computed"""


    def _next(self):
        Tdir = self.op(self.dir)
        g_Tdir = self.h_codomain.gram(Tdir)
        alpha_pre = (self.op.codomain.vdot(g_Tdir, Tdir) + self.regpar * self.op.domain.vdot(self.g_dir, self.dir)).real
        if alpha_pre == 0:
            raise RuntimeError(f"The update scaling failed it would be nan in iteration {self.iteration_step_nr}.")
        stepsize = self.sq_norm_res / alpha_pre  # This parameter is often called alpha. We do not use this name to avoid confusion with the regularization parameter.

        self.x += stepsize * self.dir
        if self.reltolx is not None:
            if self.x0 is None:
                self.sq_norm_x = self.h_domain.inner(self.x,self.x)
            else:
                self.sq_norm_x = self.h_domain.inner(self.x-self.x0,self.x-self.x0)

        self.y += stepsize * Tdir
        if self.reltoly is not None:
            self.g_y += stepsize * g_Tdir
            if self.x0 is None:
                self.norm_y = self.op.codomain.vdot(self.g_y, self.y).real
            else: 
                self.norm_y = self.op.codomain.vdot(self.g_y-self.g_y0, self.y-self.y0).real

        self.g_res -= stepsize * self.preconditioner.adjoint(self.op.adjoint(g_Tdir)+self.regpar*self.g_dir)
        res = self.h_domain.gram_inv(self.g_res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = self.op.domain.vdot(self.g_res, res).real
        beta = self.sq_norm_res / sq_norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number+=1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = res / sqrt(self.sq_norm_res)

        self.kappa = 1 + beta * self.kappa

        if self.krylov_basis is None or self.iteration_number > self.krylov_basis.shape[0]:
            """If Krylov subspace basis is computed, then stop the iteration only if the number of iterations exceeds the order of the Krylov space"""
            
            tol_report = 'it.{} kappa={} err/Tol '.format(self.iteration_step_nr,self.kappa)
            if self.reltolx is not None:
                valx = sqrt(self.sq_norm_res / self.sq_norm_x / self.kappa) / self.regpar
                tol_report = tol_report+'rel X:{:1.1e}/{:1.1e} '.format(valx,self.reltolx / (1 + self.reltolx))
                if valx < self.reltolx / (1 + self.reltolx):
                    self.isconverged['reltolx'] = True
                else:
                    self.isconverged['reltolx'] = False

            if self.reltoly is not None:
                valy = sqrt(self.sq_norm_res / self.norm_y / self.kappa / self.regpar)
                tol_report = tol_report+"rel Y:{:1.1e}/{:1.1e} ".format(valy,self.reltoly / (1 + self.reltoly))
                if valy < self.reltoly / (1 + self.reltoly):
                    self.isconverged['reltoly'] = True
                else:
                    self.isconverged['reltoly'] = False    

            if self.tol is not None:
                val = sqrt(self.sq_norm_res / self.kappa)/ self.regpar  
                tol_report = tol_report+"abs X: {:1.1e}/{:1.1e}".format(val,self.tol)
                if val < self.tol:
                   self.isconverged['tol'] = True
                else:
                    self.isconverged['tol'] = False

            if self.all_tol_criteria:
                converged = self.isconverged['tol'] and self.isconverged['reltolx'] and self.isconverged['reltoly']
            else:
                converged = self.isconverged['tol'] or self.isconverged['reltolx'] or self.isconverged['reltoly']
            if converged:
                self.log.info(tol_report)
                return self.converge()
            else:
                self.log.debug(tol_report)

        self.dir *= beta
        self.dir += self.preconditioner(res)
        if(isinstance(self.preconditioner,Identity)):
            self.g_dir *= beta
            self.g_dir += self.g_res
        else:
            self.g_dir=self.h_domain.gram(self.dir)
        


class GeometricSequence:
    r"""Iterator generating a geometric sequence
    
    Parameters
    ----------
    alpha0 : float
        :math:`\alpha_0` the initial regularization parameter 
    q : float
        Rate of the geometric sequence

    Notes
    ----- 
    Sequence defined recursively by
    
    .. math::
        \alpha_0 &= \alpha_0 \\
        \alpha_{n+1} &= q*\alpha_n
    """    
    def __init__(self, alpha0,q):
        self.alpha = alpha0
        self.alpha0 = alpha0
        self.q = q

    def __iter__(self):
        self.alpha = self.alpha0
        return self

    def __next__(self):
        result = self.alpha
        self.alpha = self.alpha*self.q
        return result

class TikhonovAlphaGrid(RegSolver):
    r"""Class runnning Tikhonov regularization on a grid of different regularization parameters.
    This allows to choose the regularization parameter by some stopping rule. 
    Tikhonov functionals are minimized by an inner CG iteration.

    Parameters
    ----------
    setting:  regpy.solvers.Setting
        The setting of the forward problem.
    data: array-like
        The right hand side.
    alphas: Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the seuqence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    max_CG_iter: integer, default 1000.
        maximum number of CG iterations. 
    xref: array-like, default None
        initial guess in Tikhonov functional. Default corresponds to zeros()
    delta = float, default None
        data noise level
    tol_fac: float, default 0.5
        absolute tolerance for CG iterations is tol_fac*delta/sqrt(alpha)

    Notes
    -----
    Further keyword arguments for TikhonovCG can be given. 
    """
    def __init__(self,setting, data, alphas, xref=None,max_CG_iter=1000,
                 delta=None,tol_fac=0.5, logging_level= "INFO"):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="TikhonovAlphaGrid in as a linear solver requires the operator to be linear!"))
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = alphas
        self.data = data
        """Right hand side of the operator equation."""
        self.xref = xref
        """initial guess in Tikhonov functional."""
        if self.xref is not None:
            self.x = self.xref
            self.y = self.op(self.xref)
        else:
            self.x = self.op.domain.zeros()
            self.y = self.op.codomain.zeros()
        self.max_CG_iter = max_CG_iter
        """maximum number of CG iterations."""    
        self.delta = delta
        """data noise level."""
        self.tol_fac = tol_fac
        """ absolute tolerance for CG iterations is tol_fac*delta/sqrt(alpha) if delta is specfied, 
        otherwise relative tolerance in domain is tol_fac/sqrt(alpha)"""
        self.logging_level = logging_level
        """logging level for CG iteration."""

    def _next(self):
        try:
            alpha = next(self._alphas)
        except StopIteration:
            return self.converge()
        inner_stoprule = CountIterations(max_iterations=self.max_CG_iter)
        inner_stoprule.log = self.log.getChild('CountIterations')
        inner_stoprule.log.setLevel("WARNING")
        if self.delta is None:
            tikhcg =TikhonovCG(self.setting,data=self.data,regpar=alpha,xref=self.xref,x0=self.xref,
                               reltolx = self.tol_fac / sqrt(alpha),
                               logging_level=self.logging_level
                               )
        else:
            tikhcg =TikhonovCG(self.setting,data = self.data,regpar = alpha,xref=self.xref,x0=self.xref,
                               tol= self.tol_fac * self.delta / sqrt(alpha),
                                logging_level=self.logging_level
                               )
        self.x, self.y = tikhcg.run(inner_stoprule)
        self.log.info('alpha = {}, inner CG its = {}'.format(alpha,inner_stoprule.iteration))

class NonstationaryIteratedTikhonov(RegSolver):
    r"""Iterated Tikhonov regularization with a given (fixed) sequence of regularization parameters.
       Tikhonov functionals are minimized by an inner CG iteration.

    Parameters
    ----------
    setting:  regpy.solvers.Setting
        The setting of the forward problem.
    data: array-like
        The right hand side.
    alphas: Either an iterable giving the grid of alphas or a tuple (alpha0,q)
        In the latter case the seuqence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    xref: array-like, default None
        initial guess in Tikhonov functional. Default corresponds to zeros()
    delta = float, default None
        data noise level
    tol_fac: float, default 0.5
        absolute tolerance for CG iterations is tol_fac*delta/sqrt(alpha)
    """
    def __init__(self,setting, data, alphas, xref=None, max_CG_iter=1000,
                 delta=None,tol_fac=0.5, logging_level= "INFO"):
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="TikhonovAlphaGrid in as a linear solver requires the operator to be linear!"))
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = alphas
        self.data = data
        """Right hand side of the operator equation."""
        self.xref = xref
        """initial guess in Tikhonov functional."""
        if self.xref is not None:
            self.x = self.xref
            self.y = self.op(self.xref)
        else:
            self.x = self.op.domain.zeros()
            self.y = self.op.codomain.zeros()
        self.max_CG_iter = max_CG_iter
        """maximum number of CG iterations."""    
        self.delta = delta
        """data noise level."""
        self.tol_fac = tol_fac
        """ absolute tolerance for CG iterations is tol_fac*delta/sqrt(alpha) if delta is specfied, 
        otherwise relative tolerance in domain is tol_fac/sqrt(alpha)"""
        self.logging_level = logging_level
        """logging level for CG iteration."""
        self.alpha_eff = inf
        r"""effective regularization parameter. 1/alpha_eff is the sum of the reciprocals of the previous alpha's"""

    def _next(self):
        try:
            alpha = next(self._alphas)
        except StopIteration:
            return self.converge()
        self.alpha_eff = 1./(1./alpha + 1./self.alpha_eff)
        inner_stoprule = CountIterations(max_iterations=self.max_CG_iter)
        inner_stoprule.log = self.log.getChild('CountIterations')
        inner_stoprule.log.setLevel("WARNING")
        if self.delta is None:
            tikhcg =TikhonovCG(self.setting,data = self.data,regpar=alpha,xref=self.x,x0=self.x,
                               reltolx = self.tol_fac / sqrt(self.alpha_eff),
                               logging_level=self.logging_level
                               )
        else:
            tikhcg =TikhonovCG(self.setting,data = self.data,regpar=alpha,xref=self.x,x0=self.x,
                               tol= self.tol_fac * self.delta / sqrt(self.alpha_eff),
                                logging_level=self.logging_level
                               )
        self.x, self.y = tikhcg.run(inner_stoprule)
        self.log.info('alpha_eff = {}, inner CG its = {}'.format(self.alpha_eff,inner_stoprule.iteration))

class TikhonovCGOnlyDomain(RegSolver):
    r"""The Tikhonov method for linear inverse problems. Minimizes
    
    .. math::
        \Vert T x - data\Vert^2 + regpar * \Vert x - xref\Vert^2

    using a conjugate gradient method. 
    The method here is a modification of TikhonovCG, where we use the strategy to only use :math:`T^\ast T` to prevent 
    ever computing an object in the codomain. Thus the linear operator `T` has to implement `_adjoint_eval` so that we
    can use this app. Note that the evaluation of :math:`T^\ast T x` is done by `setting.op._adjoint_eval()` and it has
    to incorporate the Gram matrix in the codomain! that is we assume that :math:`T^\ast T x = T^T G_Y T x` with the 
    Gram matrix :math:`G_Y` in the codomain. However, the Gram matrix in the domain is not part of this application and
    is taken from the penalty functional.

    To determine a stopping index yielding guaranteed error bounds, a partial embedded minimal residual method (MR) is 
    used, which can be implemented by updating a scalar parameter in each iteration. 
    For details on the use of the embedded MR method, as proposed by H. Egger in 
    "Numerical realization of Tikhonov regularization: appropriate norms, implementable stopping criteria, and optimal algorithms" 
    in Oberwolfach Reports 9/4, page 3009-3010, 2013;
    see also the Master thesis by Andrea Dietrich 
    "Analytische und numerische Untersuchung eines Abbruchkriteriums für das CG-Verfahren zur Minimierung 
    von Tikhonov Funktionalen", Univ. Göttingen, 2017 

    Parameters
    ----------
    setting : regpy.solvers.Setting
        The setting of the forward problem.
    backprop_data : setting.op.domain [default: None]
        The back propagated measured data given by :math:`T^\ast g^\delta`. Note that you have to incorporate the 
        appropriate Gram matrix of the codomain in this back propagation!
    regpar : float [default:None]
        The regularization parameter. Must be positive. If None, then setting must not contain it. 
    xref: setting.op.domain [default: None]
        Reference value in the Tikhonov functional. The default is equivalent to xref = setting.op.domain.zeros().
    x0: setting.op.domain  [default: None]
        Starting value of the CG iteration. If None, setting.op.domain.zeros() is used as starting value. 
    tol : float, default: None
        The absoluted tolerance - it guarantees that difference of the final CG iterate to the exact minimizer of the Tikhonov functional  
        in setting.h_domain.norm is smaller than tol. If None, this criterion is not active (analogously for reltolx and reltoly).   
        If the noise level is given, it is reasonable value to choose tol in the order of the propagated data noise level, 
        which is noiselevel/2*sqrt(regpar)
    reltolx: float, default: 10/sqrt(regpar)
        Relative tolerance in domain. Guarantees that the relative error w.r.t. setting.h_domain.norm is smaller than reltolx.
        The motivation for the default value is similar to that given for tol, assuming a resonable 
        signal-to-noise ratio for the Tikhonov minimizer. 
    all_tol_criteria: bool (default: True)
        If True, the iteration is stopped if all specified tolerance criteria are satisfied. 
        If False, the iteration is stopped if one criterion is satisfied.
    krylov_basis : list or None
        Compute orthonormal basis vectors of the Krylov subspaces while running CG solver
    preconditioner : setting.op.domain -> setting.op.domain, default: Identity
        A preconditioner for the CG method. The preconditioner should be an approximation of the inverse of the operator in the normal equation.
        If None, the identity is used as preconditioner.
    logging_level : str, default: "INFO"
        The logging level of this class. Possible values are "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL".    
    """

    def __init__(
        self, setting, backprop_data, regpar=None, xref=None, 
        x0 =None, 
        tol=None, reltolx=None, reltoly=None, 
        all_tol_criteria = True,
        krylov_basis=None, 
        preconditioner=None,
        logging_level = "INFO"
        ):
        try:
            self.log.setLevel(logging_level)        
        except Exception as e:
            self.log.setLevel("INFO")
            self.log.warning(f"Could not set logging level to {logging_level}, using INFO. Error: {e}")

        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="TikhonovAlphaGrid in as a linear solver requires the operator to be linear!"))

        if backprop_data not in setting.op.domain:
            raise ValueError(Errors.value_error("The back propagated data backprop_data must be an element of setting.op.domain"))
        self.backprop_data = backprop_data
        """The back propagated data :math:`T^\ast g^\delta`."""

        if regpar is None:
            if not setting.is_tikhonov:
                raise ValueError(Errors.value_error("If regpar is None, setting must contain the regularization parameter."))
            self.regpar = setting.regpar
        elif isinstance(regpar, (int, float)) and regpar > 0:
            self.regpar = regpar
        else:
            raise ValueError(Errors.value_error("regpar must be a positive float or None"))
        
        if x0 is not None:
            if x0 in setting.op.domain:
                self.x = x0.copy()
                """The current iterate."""
                self.x0 = x0
                """The zero-th CG iterate. x0=Null corresponds to xref=zeros()"""
            else:
                raise ValueError(Errors.value_error("The starting value x0 must be an element of setting.op.domain"))
        else:
            self.x = self.op.domain.zeros()
            self.x0 = self.op.domain.zeros()

        self.y = None
        """The image of the current iterate under the operator. Is always None, since we never compute it."""

        self.TastT = self.op.adjoint_eval
        r"""The operator T^* T."""

        self.TastT_x0 = self.TastT(self.x0)
        r""" The application of T^* T to the zero-th CG iterate value x0."""

        if preconditioner is None:
            self.preconditioner = self.h_domain.vecsp.identity
            self.penalty = self.h_domain.vecsp.identity
        elif isinstance(preconditioner, Operator) and preconditioner.domain == self.h_domain.vecsp and preconditioner.codomain == self.h_domain.vecsp: 
            self.preconditioner = preconditioner
            self.penalty = self.preconditioner * self.h_domain.gram * self.preconditioner * self.h_domain.gram_inv
        else:
            raise TypeError("preconditioner must be an Operator from setting.h_domain.vecsp to setting.h_domain.vecsp")

        self.g_res = self.preconditioner( self.backprop_data - self.TastT_x0)
        r"""The gram matrix applied to the residual of the normal equation. 
        g_res = T^* G_Y (data-T self.x) + regpar G_X(xref-self.x) in each iteration with operator T and Gram matrices G_x, G_Y.
        """
        if xref is not None:
            self.g_res += self.regpar *self.preconditioner( self.h_domain.gram(xref-self.x) )
        elif x0 is not None:
            self.g_res -= self.regpar *self.preconditioner( self.h_domain.gram(self.x) )

        res = self.h_domain.gram_inv(self.g_res)
        """The residual of the normal equation."""
        self.sq_norm_res = self.op.domain.vdot(self.g_res, res).real
        """The squared norm of the residual."""
        self.dir = res
        """The direction of descent."""
        self.g_dir = self.g_res.copy()
        """The Gram matrix applied to the direction of descent."""
        self.kappa = 1
        """ratio of the squared norms of the residuals of the CG method and the MR-method.
        Used for error estimation."""

        self.krylov_basis=krylov_basis
        if self.krylov_basis is not None: 
            self.iteration_number=0
            self.krylov_basis[self.iteration_number, :] = res / self.op.domain.norm(res)
        """In every iteration step of the Tikhonov solver a new orthonormal vector is computed"""

        self.tol = tol
        """The absolute tolerance in the domain."""
        self.reltolx = reltolx
        """The relative tolerance in the domain."""

        if tol is None  and reltolx is None:
            self.reltolx = 10./sqrt(regpar)

        if self.reltolx is not None:
            self.sq_norm_x = 0

        self.all_tol_criteria = all_tol_criteria
        if self.all_tol_criteria:
            self.isconverged = {'tol': self.tol is None, 'reltolx': self.reltolx is None}
        else:
            self.isconverged = {'tol': self.tol is not None, 'reltolx': self.reltolx is not None}


    def _next(self):
        TastGTdir = self.TastT(self.preconditioner(self.dir))
        alpha_pre = (self.op.domain.vdot(TastGTdir, self.dir) + self.regpar * self.op.domain.vdot(self.penalty (self.g_dir), self.dir)).real
        if alpha_pre == 0:
            raise ZeroDivisionError(f"The update scaling failed in iteration {self.iteration_step_nr}! Would lead to division by zero.")
        stepsize = self.sq_norm_res / alpha_pre  # This parameter is often called alpha. We do not use this name to avoid confusion with the regularization parameter.

        self.x += stepsize * self.dir
        if self.reltolx is not None:
            if self.x0 is None:
                self.sq_norm_x = self.h_domain.inner(self.x,self.x)
            else:
                self.sq_norm_x = self.h_domain.inner(self.x-self.x0,self.x-self.x0)

        self.g_res -= stepsize * (self.preconditioner( TastGTdir )+ self.regpar * self.penalty (self.g_dir) )
        res = self.h_domain.gram_inv(self.g_res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = self.op.domain.vdot(self.g_res, res).real
        beta = self.sq_norm_res / sq_norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number+=1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = res / self.op.domain.norm(res)

        self.kappa = 1 + beta * self.kappa

        if self.krylov_basis is None or self.iteration_number > self.krylov_basis.shape[0]:
            """If Krylov subspace basis is computed, then stop the iteration only if the number of iterations exceeds the order of the Krylov space"""
            
            tol_report = 'it.{} kappa={} err/Tol '.format(self.iteration_step_nr,self.kappa)
            if self.reltolx is not None:
                valx = sqrt(self.sq_norm_res / self.sq_norm_x / self.kappa) / self.regpar
                tol_report = tol_report+'rel X:{:1.1e}/{:1.1e} '.format(valx,self.reltolx / (1 + self.reltolx))
                if valx < self.reltolx / (1 + self.reltolx):
                    self.isconverged['reltolx'] = True
                else:
                    self.isconverged['reltolx'] = False

            if self.tol is not None:
                val = sqrt(self.sq_norm_res / self.kappa)/ self.regpar  
                tol_report = tol_report+"abs X: {:1.1e}/{:1.1e}".format(val,self.tol)
                if val < self.tol:
                   self.isconverged['tol'] = True
                else:
                    self.isconverged['tol'] = False

            if self.all_tol_criteria:
                converged = self.isconverged['tol'] and self.isconverged['reltolx']
            else:
                converged = self.isconverged['tol'] or self.isconverged['reltolx']
            if converged:
                self.log.info(tol_report)
                return self.converge()
            else:
                self.log.debug(tol_report)

        self.dir *= beta
        self.dir += res
        self.g_dir *= beta
        self.g_dir += self.g_res
