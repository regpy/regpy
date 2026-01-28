from math import sqrt,inf
import numpy as np

from regpy.util import Errors
from regpy.functionals.base import SquaredNorm
from regpy.operators import Identity,Operator
from regpy.stoprules import CountIterations
import numpy as np

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
        The setting of the forward problem. Must have quadratic penalty and data fidelity terms. 
    data : setting.op.codomain, optional
        The measured data. Default None means that the data from setting is used.
    regpar : float|None, optional
        The regularization parameter. Must a positive of None. 
        If None, then the regularization parameter is taken from the setting, which must be Tikhonov. 
    xref: setting.op.domain  or None, optional
        Reference value in the Tikhonov functional. If None, it is taken from the setting. 
    update_setting: bool, optional
        Flag wheather or not values for data, regpar, and xref that are provided explicitly should be updated in the setting.
        If the setting does not have a regularization parameter, no regularization parameter will be set in the setting. 
        Defaults to True.
    x0: setting.op.domain or None, optional
        Starting value of the CG iteration. If None (default), setting.op.domain.zeros() is used as starting value. 
    tol : float or None, default: None
        The absolute tolerance - it guarantees that difference of the final CG iterate to the exact minimizer of the Tikhonov functional  
        in setting.h_domain.norm is smaller than tol. If None, this criterion is not active (analogously for reltolx and reltoly).   
        If the noise level is given, it is reasonable value to choose tol in the order of the propagated data noise level, 
        which is noiselevel/2*sqrt(regpar)
    reltolx: float or None, default: 1e-2
        Relative tolerance in domain. Guarantees that the relative error w.r.t. setting.h_domain.norm is smaller than reltolx.
        The motivation for the default value is similar to that given for tol, assuming a reasonable 
        signal-to-noise ratio for the Tikhonov minimizer. 
    reltoly: float or None, default: None
        Relative tolerance in codomain.
    all_tol_criteria: bool (default: True)
        If True, the iteration is stopped if all specified tolerance criteria are satisfied. 
        If False, the iteration is stopped if one criterion is satisfied.
    krylov_basis : list or None, optional 
        Defaults to None. Otherwise, an orthonormal basies of the Krylov subspace is computed while running CG solver
    preconditioner : Preconditioner such that the iteration is done on 
        :math:`\Vert TP x - data\Vert^2 + regpar * \Vert Px - xref\Vert^2`
        The iterates (self.x) still solve the original equation without preconditioner.
    x_exact: setting.op.domain or None, optional
    y_exact: setting.op.codomain or None, optional
        These parameters are intended for monitoring convergence of the CG iteration and the performance of the stopping rules. 
        If not None (default), they should be (an approximation of) the exact minimizer of the Tikhonov functional and its image 
        under setting.op. In this case, the internal stopping rules track the errors in the domain and/or codomain.
    """
    from regpy.stoprules import StopRule
    class KappaTrackAndMinIt(StopRule):
        """Stopping rule which tracks the value of kappa and ensures a minimum number of iterations 
         if a Krylov basis must be computed."""
        def __init__(self,min_it=0,logging_level="INFO"):
            super().__init__(logging_level)        
            self.history_dict["kappa"] = []
            self.min_it = min_it
            self.it = 0

        def __repr__(self):
            return 'KappaTrackAndMinIt({})'.format(self.min_it)

        def _stop(self):
            kappa = self.solver.kappa
            self.history_dict["kappa"].append(kappa)
            self.log_info = 'kappa:{:1.1e}'.format(kappa)
            self.it+=1
            return (self.it>=self.min_it)

    class RelTolXStop(StopRule):
        """Stopping rule based on relative error in the domain of the operator."""
        def __init__(self, tol:float = 0.,logging_level="INFO",x_exact=None):
            super().__init__(logging_level)
            self.x_exact= x_exact
            self.tol, self.tolexpr = tol,  tol/(1.+tol)
            self.history_dict["rel_err_x"] = []
            if x_exact is not None:
                self.history_dict["rel_err_x_true"] = []

        def __repr__(self):
            return 'RelTolXStop({})'.format(self.tol)
        
        def _complete_init_with_solver(self, solver):
            super()._complete_init_with_solver(solver)
            self.sq_norm_x = self.solver.h_domain.norm(self.solver.x)**2
            self.x0 = None if self.sq_norm_x == 0. else self.solver.x.copy()

        def _stop(self):     
            if self.x0 is None:
                self.sq_norm_x = self.solver.h_domain.norm(self.solver.x)**2
            else:
                self.sq_norm_x = self.solver.h_domain.norm(self.solver.x-self.x0)**2

            if self.sq_norm_x == 0.:
                valx = 2*self.tolexpr
                self.log_info = 'rel X:--(x=0)!'
            else:
                valx = sqrt(self.solver.sq_norm_res / self.sq_norm_x / self.solver.kappa) / self.solver.regpar
                self.log_info = 'rel X:{:1.1e}>={:1.1e} '.format(valx,self.tolexpr)
            self.history_dict["rel_err_x"].append(valx)
            if self.x_exact is not None:
                self.history_dict["rel_err_x_true"].append(self.solver.setting.h_domain.norm(self.x_exact-self.solver.x)/np.sqrt(self.sq_norm_x))
            return valx < self.tolexpr
                         
    class RelTolYStop(StopRule):
        """Stopping rule based on relative error in the domain of the operator.
        """        
        def __init__(self, tol:float = 0.,logging_level="INFO",y_exact=None):
            super().__init__(logging_level)
            self.tol, self.tolexpr = tol,  tol/(1.+tol)
            self.y_exact = y_exact
            self.history_dict["rel_err_y"] = []
            if y_exact is not None:
                self.history_dict["rel_err_y_true"] = []
            self.first_step=True

        def __repr__(self):
            return 'RelTolYStop({})'.format(self.tol)

        def _complete_init_with_solver(self, solver):
            super()._complete_init_with_solver(solver)
            self.g_y = self.solver.hessS(self.solver.y)
            self.norm_y = self.solver.op.codomain.vdot(self.solver.y,self.g_y)
            if self.norm_y!=0:
                self.y0 = self.solver.y
                self.g_y0 = self.g_y
            else:
                self.g_y0 = None
        
        def _stop(self):
            if self.first_step:
                self.first_step = False
            else:
                self.g_y += self.solver.stepsize * self.solver.g_Tdir
                if self.g_y0 is None:
                    self.norm_y = self.solver.op.codomain.vdot(self.g_y, self.solver.y).real
                else: 
                    self.norm_y = self.solver.op.codomain.vdot(self.g_y-self.g_y0, self.solver.y-self.y0).real
            if self.norm_y==0:
                valy = 2*self.tolexpr
                self.log_info="rel Y:--(y=0) "    
            else:
                valy = sqrt(self.solver.sq_norm_res / self.norm_y / self.solver.kappa / self.solver.regpar)
                self.log_info="rel Y:{:1.1e}>={:1.1e} ".format(valy,self.tolexpr)                
            self.history_dict["rel_err_y"].append(valy)
            if self.y_exact is not None:
                self.history_dict["rel_err_y_true"].append(self.solver.setting.h_codomain.norm(self.y_exact-self.solver.y)/self.norm_y)
            return  valy < self.tolexpr     

    class TolStop(StopRule):
        """Stopping rule based on absolute tolerance
        """
        def __init__(self, tol:float = 0.,logging_level="INFO",x_exact=None):
            super().__init__(logging_level)
            self.tol=tol
            self.x_exact = x_exact
            self.history_dict["abs_err"] = []
            if self.x_exact is not None:
                self.history_dict["abs_err_true"] = []

        def __repr__(self):
            return 'TolStop({})'.format(self.tol)

        def _stop(self):
            val = sqrt(self.solver.sq_norm_res / self.solver.kappa)/ self.solver.regpar  
            self.history_dict["abs_err"].append(val)
            if self.x_exact is not None:
                self.history_dict["abs_err_true"].append(self.solver.setting.h_domain.norm(self.x_exact-self.solver.x))
            self.log_info="abs X:{:1.1e}>={:1.1e}".format(val,self.tol)
            return val<self.tol

    def __init__(
                self, setting:Setting, 
                data=None, 
                regpar:float|None=None, 
                xref=None, 
                x0 =None, 
                tol:float|None=None, reltolx:float|None=1e-2, reltoly:float|None=None, 
                all_tol_criteria:bool = True,
                max_it:int = 1000,
                krylov_basis:list|None=None, 
                preconditioner:Operator|None=None,
                logging_level:str = "INFO",
                update_setting:bool = True,
                x_exact= None, 
                y_exact= None
                ):
        super().__init__(setting)
        self.tol, self.reltolx, self.reltoly = tol, reltolx, reltoly
        self.all_tol_criteria, self.max_it, self.logging_level = all_tol_criteria, max_it, logging_level
        out,par = self.check_applicability(setting)
        if not out['applicable']:
            raise ValueError(Errors.not_applicable_solver("TikhonovCG",out['info']))
        self.log.setLevel(logging_level)
        self.x =   setting.get_or_update_initial_guess(x0, update_setting)
        """The zero-th CG iterate."""
        setting.get_or_update_data(data, update_setting)
   
        if regpar is not None and not (isinstance(regpar,(float,int)) and regpar >0):
            raise ValueError(Errors.value_error("The regularization parameter must be None or positive!",obj=regpar))
        if regpar is not None:  
            self.regpar = regpar 
            if update_setting:
                if setting.is_tikhonov and regpar !=  par['regpar']:
                    self.log.warning(f"Changing setting.regpar from {setting.regpar:.2e} to {regpar:.2e}!")
                    setting.regpar = regpar   
        else:
            if setting.is_tikhonov:
                regpar = par['regpar']
            else:
                raise ValueError(Errors.value_error("If the setting is not a Tikhonov setting the regularization parameter needs to be specified in TIkhonovCG!"))
        self.regpar = regpar
        """The regularization parameter."""

        self.hessR = self.penalty.hessian(self.x)        
        if xref is not None:
            self.log.warning("Providing xref as an argument to the CG solver is deprecated. Please provide it via the setting. Results may not be consistent!")
        else:
            xref = -self.hessR.inverse(self.penalty.subgradient(self.op.domain.zeros()))
        self.y = self.op(self.x)
        """The image of the current iterate under the operator."""
        self.hessS = self.data_fid.hessian(self.y)
        data = -self.hessS.inverse(self.data_fid.subgradient(self.op.codomain.zeros()))

        if preconditioner is None:
            self.preconditioner = Identity (self.h_domain.vecsp)
        else: 
            self.preconditioner = preconditioner

        self.g_res = self.op.adjoint(self.hessS(data-self.y))
        """The gram matrix applied to the residual of the normal equation. 
        g_res = T^* G_Y (data-T self.x) + regpar G_X(xref-self.x) in each iteration with operator T and Gram matrices G_x, G_Y.
        """
        if xref is not None:
            self.g_res += self.regpar *self.hessR(xref-self.x)
        elif x0 is not None:
            self.g_res -= self.regpar * self.hessR(self.x)
        self.g_res=self.preconditioner.adjoint(self.g_res)
        res = self.hessR.inverse(self.g_res)
        """The residual of the normal equation."""
        self.sq_norm_res = self.op.domain.vdot(self.g_res, res).real
        """The squared norm of the residual."""
        self.dir = self.preconditioner(res)
        """The direction of descent."""
        if(isinstance(self.preconditioner,Identity)):
            self.g_dir = self.g_res.copy()
        else:
            self.g_dir = self.hessR(self.dir)
        """The Gram matrix applied to the direction of descent."""
        self.kappa = 1
        """ratio of the squared norms of the residuals of the CG method and the MR-method.
        Used for error estimation."""

        if x_exact is not None and x_exact not in setting.op.domain:
            raise TypeError(Errors.type_error("x_exact must be None or in op.domain.",x_exact))
        self.x_exact = x_exact    
        if y_exact is not None and y_exact not in setting.op.codomain:
            raise TypeError(Errors.type_error("y_exact must be None or in op.codomain.",y_exact))
        self.y_exact = y_exact    

        self.krylov_basis=krylov_basis
        if self.krylov_basis is not None: 
            self.iteration_number=0
            self.krylov_basis[self.iteration_number, :] = res / sqrt(self.sq_norm_res)
        """In every iteration step of the Tikhonov solver a new orthonormal vector is computed"""

    def _next(self):
        Tdir = self.op(self.dir)
        self.g_Tdir = self.hessS(Tdir)
        alpha_pre = (self.op.codomain.vdot(self.g_Tdir, Tdir) + self.regpar * self.op.domain.vdot(self.g_dir, self.dir)).real
        if alpha_pre == 0:
            raise RuntimeError(f"The update scaling failed it would be nan in iteration {self.iteration_step_nr}.")
        self.stepsize = self.sq_norm_res / alpha_pre  # This parameter is often called alpha. We do not use this name to avoid confusion with the regularization parameter.

        self.x += self.stepsize * self.dir
        self.y += self.stepsize * Tdir

        self.g_res -= self.stepsize * self.preconditioner.adjoint(self.op.adjoint(self.g_Tdir)+self.regpar*self.g_dir)
        res = self.hessR.inverse(self.g_res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = self.op.domain.vdot(self.g_res, res).real
        beta = self.sq_norm_res / sq_norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number+=1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = res / sqrt(self.sq_norm_res)

        self.kappa = 1 + beta * self.kappa

        self.dir *= beta
        self.dir += self.preconditioner(res)
        if(isinstance(self.preconditioner,Identity)):
            self.g_dir *= beta
            self.g_dir += self.g_res
        else:
            self.g_dir=self.hessR(self.dir)

    def get_stoprule(self,other_stuprule=None):
        """ Constructs the internal StopRule defined by the parameters of the constructor
        """
        from regpy.stoprules import AndCombineRules,CombineRules, CountIterations             
        stoprule_list = []
        if self.tol is not None:
            self.tol_stop = TikhonovCG.TolStop(self.tol,logging_level=self.logging_level,x_exact= self.x_exact)
            stoprule_list.append(self.tol_stop)
        if self.reltolx is not None:
            self.reltolx_stop = TikhonovCG.RelTolXStop(self.reltolx,logging_level=self.logging_level,x_exact= self.x_exact)
            stoprule_list.append(self.reltolx_stop)
        if self.reltoly is not None:
            self.reltoly_stop = TikhonovCG.RelTolYStop(self.reltoly,logging_level=self.logging_level,y_exact= self.y_exact)
            stoprule_list.append(self.reltoly_stop)
        if self.tol is None  and self.reltolx is None and self.reltoly is None:
            self.reltolx_stop = TikhonovCG.RelTolXStop(10./sqrt(self.regpar),logging_level=self.logging_level)
            stoprule_list.append(self.reltolx_stop)
        combined_rule = AndCombineRules(stoprule_list) if self.all_tol_criteria else CombineRules(stoprule_list)
        self.kappa_track = TikhonovCG.KappaTrackAndMinIt(self.krylov_basis.shape[0] if self.krylov_basis else 0, 
                                                    logging_level= self.logging_level)
        return CountIterations(self.max_it) + (combined_rule & self.kappa_track) + other_stuprule if other_stuprule else \
               CountIterations(self.max_it) + (combined_rule & self.kappa_track) 

    def run(self, stoprule=None):
        """ Runs the method with the stoprule specified by the parameters of the constructor.

        Parameters:
        stoprule: StopRule, optional
            Defaults to None. Otherwise, the given StopRule is (or-) combined with the internal StopRule. 
        """
        self.stoprule = self.get_stoprule(other_stuprule=stoprule)
        return super().run(stoprule=self.stoprule)
        
    def get_convergence_histories(self):
        """ Yields the stoprule specified by the parameters of the constructor
        """
        hist = self.kappa_track.history_dict.copy()
        if hasattr(self,'tol_stop'):
            hist.update(self.tol_stop.history_dict)
        if hasattr(self,'reltolx_stop'):
            hist.update(self.reltolx_stop.history_dict)
        if hasattr(self,'reltoly_stop'):
            hist.update(self.reltoly_stop.history_dict)
        return hist


    def primal(self):
        return (self.x,self.y)

    @staticmethod
    def check_applicability(setting,op_norm=None)->tuple[dict,dict]:
        out = {'info':''}; par = {}
        if not  setting.penalty.is_quadratic and setting.penalty.is_convex:
            out['info'] += 'Penalty term is not convex quadratic.'
        if not setting.data_fid.is_quadratic and setting.data_fid.is_convex:
            out['info'] += 'Data fidelity term is not convex quadratic. '
        if not setting.op.linear:
            out['info'] += 'Operator is not linear.'
        out['applicable'] = out['info'] == ''
        if out['applicable'] and setting.is_tikhonov:
            if hasattr(setting.penalty,'a') and hasattr(setting.data_fid,'a'):
                par['regpar'] = setting.regpar*setting.penalty.a / setting.data_fid.a
            else:
                par['regpar'] = setting.regpar
            if op_norm is not None:
                conv_param = par['regpar']*setting.penalty.convexity_param
                cond = (op_norm**2*setting.data_fid.Lipschitz + conv_param)/conv_param
                out['rate'] = (sqrt(cond)-1) / (sqrt(cond)+1)
            else:
                out['rate'] = np.nan
        else:
            out['rate'] = np.nan
        return out,par

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
    def __init__(self,setting:Setting, data, alphas, xref=None,max_CG_iter=1000,
                 delta=None,tol_fac:float=0.5, logging_level    :str= "INFO"):
        from regpy.solvers.nonlinear.gen_tikhonov import GeometricSequence        
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
        self.alpha = alpha
        self.error_prop = 1/(2*np.sqrt(alpha))
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
        from regpy.solvers.nonlinear.gen_tikhonov import GeometricSequence
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
        The back propagated measured data given by :math:`T^{\ast} g^{\delta}`. Note that you have to incorporate the 
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
        self, setting:Setting, backprop_data, 
        regpar=None, xref=None, x0 =None, 
        tol:float|None=None, reltolx:float|None=None, reltoly:float|None=None, 
        all_tol_criteria = True,
        krylov_basis: list|None =None, 
        preconditioner:Operator|None=None,
        logging_level:bool = "INFO",
        update_setting:bool=True
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
        """The back propagated data :math:`T^{\ast} g^{obs}`."""

        if regpar is None:
            if not setting.is_tikhonov:
                raise ValueError(Errors.value_error("If regpar is None, setting must contain the regularization parameter."))
            self.regpar = setting.regpar
        elif isinstance(regpar, (int, float)) and regpar > 0:
            self.regpar = regpar
        else:
            raise ValueError(Errors.value_error("regpar must be a positive float or None"))

        self.x0 = setting.get_or_update_initial_guess(x0, update_setting)
        """The zero-th CG iterate."""
        self.x = self.x0.copy()
        """The current iterate."""
        self.hessR = self.penalty.hessian(self.x)

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
            self.penalty = self.preconditioner * self.hessR * self.preconditioner * self.hessR.inverse
        else:
            raise TypeError("preconditioner must be an Operator from setting.h_domain.vecsp to setting.h_domain.vecsp")

        self.g_res = self.preconditioner( self.backprop_data - self.TastT_x0)
        r"""The gram matrix applied to the residual of the normal equation. 
        g_res = T^* G_Y (data-T self.x) + regpar G_X(xref-self.x) in each iteration with operator T and Gram matrices G_x, G_Y.
        """
        if xref is not None:
            self.g_res += self.regpar *self.preconditioner( self.hessR(xref-self.x) )
        elif x0 is not None:
            self.g_res -= self.regpar *self.preconditioner( self.hessR(self.x) )

        res = self.hessR.inverse(self.g_res)
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

    def _next(self):
        TastGTdir = self.TastT(self.preconditioner(self.dir))
        alpha_pre = (self.op.domain.vdot(TastGTdir, self.dir) + self.regpar * self.op.domain.vdot(self.penalty (self.g_dir), self.dir)).real
        if alpha_pre == 0:
            raise ZeroDivisionError(f"The update scaling failed in iteration {self.iteration_step_nr}! Would lead to division by zero.")
        stepsize = self.sq_norm_res / alpha_pre  # This parameter is often called alpha. We do not use this name to avoid confusion with the regularization parameter.

        self.x += stepsize * self.dir

        self.g_res -= stepsize * (self.preconditioner( TastGTdir )+ self.regpar * self.penalty (self.g_dir) )
        res = self.hessR.inverse(self.g_res)

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
