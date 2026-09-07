from collections.abc import Iterable
from math import sqrt,inf
import numpy as np
import logging

from regpy.util import Errors
from regpy.vecsps import Vector, ReadOnlyVector
from regpy.functionals.base import SquaredNorm
from regpy.operators import Identity, Operator, ObsDataT
from regpy.stoprules import CountIterations, StopRule

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability_linear

__all__ = [
    "TikhonovCG",
    "LavrentievCG",
    "TikhonovAlphaGrid",
    "NonstationaryIteratedTikhonov",
]


def _axpy(space, target, scalar, direction, work):
    """Compute ``target += scalar * direction`` using a reusable buffer."""
    space.copyto(work, direction)
    work *= scalar
    target += work
    return target


def _contains_count_iterations(stoprule):
    """Return whether a possibly combined rule contains an iteration cap."""
    if stoprule is None:
        return False
    if isinstance(stoprule, CountIterations):
        return True
    return any(
        _contains_count_iterations(rule)
        for rule in getattr(stoprule, 'rules', ())
    )


def _regularized_inner_setting(
    setting: Setting,
    regpar: float,
    logging_level: int | str,
) -> Setting:
    """Create an independent regularized setting for an inner CG solve."""
    if setting.codomain_free:
        return Setting(
            setting.op.adjoint_eval,
            setting.penalty,
            setting.penalty,
            regpar=regpar,
            codomain_free=True,
            explicitDataWarning=False,
            logging_level=logging_level,
        )
    return Setting(
        setting.op,
        setting.penalty,
        setting.data_fid,
        regpar=regpar,
        explicitDataWarning=False,
        logging_level=logging_level,
    )

class TikhonovCG(RegSolver):
    r"""The Tikhonov method for linear inverse problems. Minimizes
    
    .. math::
        \Vert T x - data\Vert^2 + \alpha \Vert x - init\Vert^2

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
        The setting of the forward problem. It must contain a positive
        regularization parameter and have quadratic penalty and data-fidelity
        terms.
    data : Vector or ObsDataT or None [default: None]
        The measured data, included for backward compatibility. 
        The preferred way is to specify data in the setting.
    init: Vector or None [default: None]
        Reference value in the Tikhonov functional. If omitted, the penalty
        functional's reference vector is used.
    xref : Vector or None [default: None]
        Deprecated alias for ``init``.
    x0: Vector or None [default: None]
        Starting value of the CG iteration. If omitted, ``setting.init`` is
        used. This can differ from the penalty reference, in particular for
        automatically constructed dual settings.
    tol : float or None [default: None]
        The absolute tolerance - it guarantees that difference of the final CG iterate to the exact minimizer of the Tikhonov functional  
        in setting.h_domain.norm is smaller than tol. If None, this criterion is not active (analogously for reltolx and reltoly).   
        If the noise level is given, it is reasonable value to choose tol in the order of the propagated data noise level, 
        which is ``noiselevel / (2 * sqrt(setting.regpar))``.
    max_it: int [default: 1000]
        maximal number of CG iterations
    reltolx: float or None [default: 1e-2]
        Relative tolerance in domain. Guarantees that the relative error w.r.t. setting.h_domain.norm is smaller than reltolx.
        The motivation for the default value is similar to that given for tol, assuming a reasonable 
        signal-to-noise ratio for the Tikhonov minimizer. 
    reltoly: float or None [default: None]
        Relative tolerance in codomain.
    all_tol_criteria: bool [default: True]
        If True, the iteration is stopped if all specified tolerance criteria are satisfied. 
        If False, the iteration is stopped if one criterion is satisfied.
    krylov_basis : list[Vector] or None [default: None] 
        If not None, an orthonormal basis of the Krylov subspace is computed while running CG solver
    preconditioner: regpy.operators.operator
        Preconditioner such that the iteration is done on 
        :math:`\Vert TP x-data\Vert^2+\alpha\Vert Px-init\Vert^2`.
        The iterates (self.x) still solve the original equation without preconditioner.
    x_exact: Vector or None [default: None]
    y_exact: Vector or None [default: None]
        These parameters are intended for monitoring convergence of the CG iteration and the performance of the stopping rules. 
        If not None (default), they should be (an approximation of) the exact minimizer of the Tikhonov functional and its image 
        under setting.op. In this case, the internal stopping rules track the errors in the domain and/or codomain.

    Notes
    -----
    For a codomain-free setting, the method backpropagates ``data`` once and
    delegates the solution of

    .. math::
        \left(T^*T + \alpha H\right)x
        = T^*data + \alpha H init

    to :class:`LavrentievCG`, using ``setting.op.adjoint_eval`` and
    ``setting.op.adjoint_data``. Consequently no codomain vector is created.
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
            domain = self.solver.op.domain
            self._g_delta_x = domain.empty()
            self.solver.hessR(
                self.solver.x, out=self._g_delta_x
            )
            self.sq_norm_x = max(
                domain.vdot(self._g_delta_x, self.solver.x).real,
                0.,
            )
            if self.sq_norm_x == 0.:
                self.x0 = None
            else:
                self.x0 = domain.empty()
                domain.copyto(self.x0, self.solver.x)
            self._delta_x = domain.empty()

        def _stop(self):
            if self.x0 is None:
                delta_x = self.solver.x
            else:
                self.solver.op.domain.copyto(self._delta_x, self.solver.x)
                self._delta_x -= self.x0
                delta_x = self._delta_x
            self.solver.hessR(delta_x, out=self._g_delta_x)
            self.sq_norm_x = max(
                self.solver.op.domain.vdot(
                    self._g_delta_x, delta_x
                ).real,
                0.,
            )

            if self.sq_norm_x == 0.:
                valx = 2*self.tolexpr
                self.log_info = 'rel X:--(x=0)!'
            else:
                valx = sqrt(
                    self.solver.sq_norm_res
                    / self.sq_norm_x
                    / self.solver.kappa
                ) / self.solver.regpar
                self.log_info = 'rel X:{:1.1e}>={:1.1e} '.format(
                    valx, self.tolexpr
                )
            self.history_dict["rel_err_x"].append(valx)
            if self.x_exact is not None:
                self.solver.op.domain.copyto(self._delta_x, self.x_exact)
                self._delta_x -= self.solver.x
                self.solver.hessR(
                    self._delta_x, out=self._g_delta_x
                )
                true_error = sqrt(max(
                    self.solver.op.domain.vdot(
                        self._g_delta_x, self._delta_x
                    ).real,
                    0.,
                ))
                self.history_dict["rel_err_x_true"].append(
                    np.inf if self.sq_norm_x == 0 else true_error / np.sqrt(self.sq_norm_x)
                )
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
            codomain = self.solver.op.codomain
            self.g_y = codomain.empty()
            self._work = codomain.empty()
            self._delta_g = codomain.empty()
            self._delta_y = codomain.empty()
            self._g_delta_y = codomain.empty()
            self.solver.hessS(self.solver.y, out=self.g_y)
            self.norm_y = codomain.vdot(self.solver.y, self.g_y)
            if self.norm_y != 0:
                # These must be copies: solver.y and self.g_y are updated in-place.
                self.y0 = codomain.empty()
                codomain.copyto(self.y0, self.solver.y)
                self.g_y0 = codomain.empty()
                codomain.copyto(self.g_y0, self.g_y)
            else:
                self.g_y0 = None

        def _stop(self):
            codomain = self.solver.op.codomain
            if self.first_step:
                self.first_step = False
            else:
                _axpy(
                    codomain,
                    self.g_y,
                    self.solver.stepsize,
                    self.solver.g_Tdir,
                    self._work,
                )
                if self.g_y0 is None:
                    self.norm_y = codomain.vdot(
                        self.g_y, self.solver.y
                    ).real
                else:
                    codomain.copyto(self._delta_g, self.g_y)
                    self._delta_g -= self.g_y0
                    codomain.copyto(self._delta_y, self.solver.y)
                    self._delta_y -= self.y0
                    self.norm_y = codomain.vdot(
                        self._delta_g, self._delta_y
                    ).real

            if self.norm_y == 0:
                valy = 2*self.tolexpr
                self.log_info = "rel Y:--(y=0) "
            else:
                valy = sqrt(
                    self.solver.sq_norm_res
                    / self.norm_y
                    / self.solver.kappa
                    / self.solver.regpar
                )
                self.log_info = "rel Y:{:1.1e}>={:1.1e} ".format(
                    valy, self.tolexpr
                )
            self.history_dict["rel_err_y"].append(valy)
            if self.y_exact is not None:
                codomain.copyto(self._delta_y, self.y_exact)
                self._delta_y -= self.solver.y
                self.solver.hessS(
                    self._delta_y, out=self._g_delta_y
                )
                true_error = sqrt(max(
                    codomain.vdot(
                        self._g_delta_y, self._delta_y
                    ).real,
                    0.,
                ))
                self.history_dict["rel_err_y_true"].append(
                    np.inf
                    if self.norm_y == 0
                    else true_error / sqrt(self.norm_y)
                )
            return valy < self.tolexpr

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

        def _complete_init_with_solver(self, solver):
            super()._complete_init_with_solver(solver)
            self._delta_x = self.solver.op.domain.empty()
            self._g_delta_x = self.solver.op.domain.empty()

        def _stop(self):
            val = (
                sqrt(self.solver.sq_norm_res / self.solver.kappa)
                / self.solver.regpar
            )
            self.history_dict["abs_err"].append(val)
            if self.x_exact is not None:
                self.solver.op.domain.copyto(self._delta_x, self.x_exact)
                self._delta_x -= self.solver.x
                self.solver.hessR(
                    self._delta_x, out=self._g_delta_x
                )
                self.history_dict["abs_err_true"].append(
                    sqrt(max(
                        self.solver.op.domain.vdot(
                            self._g_delta_x, self._delta_x
                        ).real,
                        0.,
                    ))
                )
            self.log_info = "abs X:{:1.1e}>={:1.1e}".format(val, self.tol)
            return val < self.tol

    def __init__(
                self, setting:Setting, 
                data : ReadOnlyVector | ObsDataT | None = None, 
                init: ReadOnlyVector|None =None, 
                x0: Vector|None = None, 
                tol: float|None = None, 
                reltolx: float|None = 1e-2, 
                reltoly: float|None = None, 
                all_tol_criteria: bool = True,
                max_it: int = 1000,
                krylov_basis: list[Vector] | None=None, 
                preconditioner: Operator|None = None,
                logging_level: int | str = logging.INFO,
                x_exact: ReadOnlyVector | None = None, 
                y_exact: ReadOnlyVector | None = None,
                xref: ReadOnlyVector | None = None,
                ):
        super().__init__(setting)
        self.tol, self.reltolx, self.reltoly = tol, reltolx, reltoly
        self.all_tol_criteria, self.max_it, self.logging_level = all_tol_criteria, max_it, logging_level

        if xref is not None:
            if init is not None:
                raise ValueError(Errors.value_error(
                    "Specify only one of init and the deprecated xref."
                ))
            self.log.warning("xref is deprecated; use init instead.")
            init = xref

        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError(
                'TikhonovCG not applicable to this setting. '
                + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        """Right-hand side of the operator equation."""
        self.init = self.op.domain.empty()
        if init is None:
            self.penalty.reference_vector(out=self.init)
        else:
            self.op.domain.copyto(self.init, par['init'])
        """Reference value in the Tikhonov functional."""
        self.regpar = par['regpar']
        """The regularization parameter."""

        self.log.setLevel(logging_level)
        if x0 is not None:
            if x0 not in self.op.domain:
                raise TypeError(Errors.not_in_vecsp(
                    x0, self.op.domain, vec_name='x0', space_name='domain'
                ))
            self.x = self.op.domain.empty()
            self.op.domain.copyto(self.x, x0)
            """The current iterate."""
        else:
            self.x = self.op.domain.empty()
            initial_iterate = setting.init if init is None else self.init
            self.op.domain.copyto(self.x, initial_iterate)

        if setting.codomain_free:
            if reltoly is not None or y_exact is not None:
                raise ValueError(Errors.value_error(
                    "reltoly and y_exact are unavailable in the "
                    "codomain-free TikhonovCG method."
                ))
            self.backprop_data = self.op.domain.empty()
            self.op.adjoint_data(self.data, out=self.backprop_data)
            self._normal_setting = _regularized_inner_setting(
                setting,
                self.regpar,
                logging_level,
            )
            self._lavrentiev = LavrentievCG(
                self._normal_setting,
                data=self.backprop_data,
                init=self.init,
                x0=self.x,
                tol=tol,
                reltolx=reltolx,
                all_tol_criteria=all_tol_criteria,
                max_it=max_it,
                krylov_basis=krylov_basis,
                preconditioner=preconditioner,
                logging_level=logging_level,
                x_exact=x_exact,
            )
            self._sync_lavrentiev_state()
            return

   
        self.hessR = self.penalty.hessian(self.x)
        domain = self.op.domain
        codomain = self.op.codomain
        self.y = codomain.empty()
        """The image of the current iterate under the operator."""
        self.op(self.x, out=self.y)
        self.hessS = self.data_fid.hessian(self.y)

        if preconditioner is None:
            self.preconditioner = Identity(self.h_domain.vecsp)
        else:
            self.preconditioner = preconditioner

        self.dir = domain.empty()            # the direction of descent
        self.g_dir = domain.empty()          # hessR applied to self.dir
        self._Tdir = codomain.empty()        # T applied to self.dir
        self.g_Tdir = codomain.empty()       # hessS applied to self.g_Tdir
        self._res = domain.empty()           # the residual of the normal equation. 
        self.g_res = domain.empty()          # The gram matrix applied to the residual of the normal equation:
        # g_res = T^* hessS (data-T self.x) + regpar hessR(init-self.x) in each iteration
        # with operator T and hessians hessR, hessS of the penalty and data fidelity functionals
        self._domain_work = domain.empty()   # buffer
        self._domain_work2 = domain.empty()  # buffer

        codomain.copyto(self._Tdir, self.data)
        self._Tdir -= self.y
        self.hessS(self._Tdir, out=self.g_Tdir)
        self.op.adjoint(self.g_Tdir, out=self._domain_work)
        domain.copyto(self._domain_work2, self.init)
        self._domain_work2 -= self.x
        self.hessR(self._domain_work2, out=self._res)
        _axpy(
            domain,
            self._domain_work,
            self.regpar,
            self._res,
            self._domain_work2,
        )

        self.preconditioner.adjoint(self._domain_work, out=self.g_res)       
        self.hessR.inverse(self.g_res, out=self._res)
        self.sq_norm_res = domain.vdot(self.g_res, self._res).real
        # self.sq_norm_res is the squared norm of the residual.
        self.preconditioner(self._res, out=self.dir)
        if isinstance(self.preconditioner, Identity):
            domain.copyto(self.g_dir, self.g_res)
        else:
            self.hessR(self.dir, out=self.g_dir)

        self.kappa = 1
        """ratio of the squared norms of the residuals of the CG method and the MR-method.
        kappa is used for error estimation."""

        if x_exact is not None and x_exact not in setting.op.domain:
            raise TypeError(Errors.type_error("x_exact must be None or in op.domain.",x_exact))
        self.x_exact = x_exact    
        if y_exact is not None and y_exact not in setting.op.codomain:
            raise TypeError(Errors.type_error("y_exact must be None or in op.codomain.",y_exact))
        self.y_exact = y_exact    

        self.krylov_basis=krylov_basis
        if self.krylov_basis is not None:
            self.iteration_number = 0
            self.krylov_basis[self.iteration_number, :] = self._res
            self.krylov_basis[self.iteration_number, :] /= sqrt(
                self.sq_norm_res
            )
        if np.sqrt(self.sq_norm_res) <=1e-15:
            self.log.info('TikhonovCG: stopped at iteration 0 as residual is 0 to machine precision.')
            self.converge()            

    def _sync_lavrentiev_state(self) -> None:
        """Mirror the public state of the delegated codomain-free solver."""
        inner = self._lavrentiev
        self.x = inner.x
        self.y = None
        self.sq_norm_res = inner.sq_norm_res
        self.kappa = inner.kappa
        self.stepsize = getattr(inner, 'stepsize', None)

    def _next(self):
        if self.setting.codomain_free:
            if not self._lavrentiev.next():
                self.converge()
                return
            self._sync_lavrentiev_state()
            return

        domain = self.op.domain
        codomain = self.op.codomain

        self.op(self.dir, out=self._Tdir)
        self.hessS(self._Tdir, out=self.g_Tdir)
        alpha_pre = (
            codomain.vdot(self.g_Tdir, self._Tdir)
            + self.regpar * domain.vdot(self.g_dir, self.dir)
        ).real
        if alpha_pre == 0:
            raise RuntimeError(
                f"The update scaling failed it would be nan in iteration "
                f"{self.iteration_step_nr}."
            )
        # This parameter is often called alpha. We avoid that name because
        # alpha denotes the regularization parameter elsewhere.
        self.stepsize = self.sq_norm_res / alpha_pre

        _axpy(
            domain,
            self.x,
            self.stepsize,
            self.dir,
            self._domain_work,
        )
        self._Tdir *= self.stepsize
        self.y += self._Tdir

        self.op.adjoint(self.g_Tdir, out=self._domain_work)
        _axpy(
            domain,
            self._domain_work,
            self.regpar,
            self.g_dir,
            self._domain_work2,
        )
        self.preconditioner.adjoint(
            self._domain_work, out=self._domain_work2
        )
        self._domain_work2 *= self.stepsize
        self.g_res -= self._domain_work2
        self.hessR.inverse(self.g_res, out=self._res)

        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = domain.vdot(self.g_res, self._res).real
        beta = self.sq_norm_res / sq_norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number += 1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = self._res
                self.krylov_basis[self.iteration_number, :] /= sqrt(
                    self.sq_norm_res
                )

        self.kappa = 1 + beta * self.kappa

        self.preconditioner(self._res, out=self._domain_work)
        self.dir *= beta
        self.dir += self._domain_work
        if isinstance(self.preconditioner, Identity):
            self.g_dir *= beta
            self.g_dir += self.g_res
        else:
            self.hessR(self.dir, out=self.g_dir)

        if np.sqrt(self.sq_norm_res) <=1e-15:
            self.log.info(f'TikhonovCG: stopped at iteration {self.iteration_step_nr} as residual is 0 to machine precision.')
            self.converge()

    def primal(self): # for duality-based stopping rules
        return (self.x,self.y)

    def get_stoprule(self,other_stoprule=None):
        """ Constructs the internal StopRule defined by the parameters of the constructor
        """
        if self.setting.codomain_free:
            return self._lavrentiev.get_stoprule(other_stoprule)
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
        self.kappa_track = TikhonovCG.KappaTrackAndMinIt(self.krylov_basis.shape[0] if self.krylov_basis is not None else 0, 
                                                    logging_level= self.logging_level)
        internal_rule = combined_rule & self.kappa_track
        if not _contains_count_iterations(other_stoprule):
            internal_rule = (
                CountIterations(
                    self.max_it, logging_level=self.logging_level
                )
                + internal_rule
            )
        return (
            internal_rule + other_stoprule
            if other_stoprule is not None
            else internal_rule
        )

    def run(self, 
            stoprule: StopRule | None = None, 
            max_it: int | None = None):
        """ Runs the method with the stoprule specified by the parameters of the constructor.

        Parameters:
        stoprule: StopRule, optional
            Defaults to None. Otherwise, the given StopRule is (or-) combined with the internal StopRule. 
        max_it: int, optional
            Defaults to None. Otherwise, the given maximal number of
            iterations is added to the internal stopping rule for this run.
        """
        if self.setting.codomain_free:
            result = self._lavrentiev.run(
                stoprule=stoprule,
                max_it=max_it,
            )
            self.iteration_step_nr = self._lavrentiev.iteration_step_nr
            self._sync_lavrentiev_state()
            return result[0], None
        if max_it is not None:
            iteration_limit = CountIterations(
                max_it, logging_level=self.logging_level
            )
            stoprule = (
                iteration_limit
                if stoprule is None
                else iteration_limit + stoprule
            )
        self.stoprule = self.get_stoprule(other_stoprule=stoprule)
        return super().run(stoprule=self.stoprule)
        
    def get_convergence_histories(self):
        """ Yields the stoprule specified by the parameters of the constructor
        """
        if self.setting.codomain_free:
            return self._lavrentiev.get_convergence_histories()
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
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: Vector | None = None,
        user_init: Vector | None = None,
    ) -> tuple[dict, dict]:
        """Check whether TikhonovCG can be used for ``setting``."""
        out, par = check_Hilbert_applicability_linear(
            setting,
            regularization_parameter_required = True,
            check_is_norm_squared_type= False,
            user_data = user_data,
            user_init = user_init
        )
        out['applicable'] = out['info'] == ''

        if op_norm is not None and not setting.codomain_free:
            conv_param = par['regpar']*setting.penalty.convexity_param
            cond = (op_norm**2*setting.data_fid.Lipschitz + conv_param)/conv_param
            out['rate'] = (sqrt(cond)-1) / (sqrt(cond)+1)
        else:
            out['rate'] = np.nan

        return out,par

class TikhonovAlphaGrid(RegSolver):
    r"""Class running Tikhonov regularization on a grid of different regularization parameters.
    This allows to choose the regularization parameter by some stoprule. 
    Tikhonov functionals are minimized by an inner CG iteration.

    Parameters
    ----------
    setting:  regpy.solvers.Setting
        The setting of the forward problem.
    data : Vector or ObsDataT or None [default: None]
        Right-hand side in ``setting.op.codomain``. In the default case,
        ``setting.data`` is used. For a codomain-free setting, it may be any
        object accepted by ``setting.op.adjoint_data``.
    alphas : iterable or tuple [default: (1.0, 2 / 3)]
        Either an iterable giving the grid of alphas or a tuple ``(alpha0, q)``.
        In the latter case the sequence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    init : Vector or None [default: None]
        Initial guess and reference value in the Tikhonov functional. In the
        default case, it is taken from the setting.
    max_CG_iter : int [default: 1000]
        Maximum number of CG iterations.
    delta : float or None [default: None]
        Data noise level.
    tol_fac : float [default: 0.5]
        Absolute tolerance for CG iterations is
        ``tol_fac * delta / sqrt(alpha)``.
    logging_level : int or str [default: "INFO"]
        Logging level for the inner CG iteration.

    Notes
    -----
    Further keyword arguments for TikhonovCG can be given. 
    """
    def __init__(
        self,
        setting: Setting,
        data: Vector | ObsDataT | None = None,
        alphas: Iterable[float] | tuple[float, float] = (1.0, 2 / 3),
        init: Vector | None = None,
        max_CG_iter: int = 1000,
        delta: float | None = None,
        tol_fac: float = 0.5,
        logging_level: int | str = "INFO",
    ) -> None:
        from ..nonlinear.gen_tikhonov import GeometricSequence
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError(
                'TikhonovAlphaGrid not applicable to this setting. '
                + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = iter(alphas)
        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        """Right-hand side of the operator equation."""
        self.init = par['init']
        """Initial guess and reference value in the Tikhonov functional."""
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        if setting.codomain_free:
            self.y = None
            self.backprop_data = self.op.domain.empty()
            self.op.adjoint_data(self.data, out=self.backprop_data)
        else:
            self.y = self.op.codomain.empty()
            self.op(self.x, out=self.y)
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
        common = {
            'setting': _regularized_inner_setting(
                self.setting, alpha, self.logging_level
            ),
            'init': self.init,
            'x0': self.init,
            'logging_level': self.logging_level,
        }
        if self.setting.codomain_free:
            common['data'] = self.backprop_data
            solver_class = LavrentievCG
        else:
            common['data'] = self.data
            solver_class = TikhonovCG
        if self.delta is None:
            common['reltolx'] = self.tol_fac / sqrt(alpha)
        else:
            common['tol'] = self.tol_fac * self.delta / sqrt(alpha)
        tikhcg = solver_class(**common)
        self.x, self.y = tikhcg.run(inner_stoprule)
        if self.setting.codomain_free:
            self.y = None
        self.log.info('alpha = {}, inner CG its = {}'.format(alpha,inner_stoprule.iteration))

    @staticmethod
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: Vector | ObsDataT | None = None,
        user_init: Vector | None = None,
    ) -> tuple[dict, dict]:
        """Check whether the parameter-grid method can be used."""
        out, par = check_Hilbert_applicability_linear(
            setting,
            op_norm=op_norm,
            user_data=user_data,
            user_init=user_init,
        )
        out['applicable'] = out['info'] == ''
        return out, par

class NonstationaryIteratedTikhonov(RegSolver):
    r"""Iterated Tikhonov regularization with a given (fixed) sequence of regularization parameters.
       Tikhonov functionals are minimized by an inner CG iteration.

    Parameters
    ----------
    setting:  regpy.solvers.Setting
        The setting of the forward problem.
    data : Vector or ObsDataT or None [default: None]
        Right-hand side in ``setting.op.codomain``. In the default case,
        ``setting.data`` is used. For a codomain-free setting, it may be any
        object accepted by ``setting.op.adjoint_data``.
    alphas : iterable or tuple [default: (1.0, 2 / 3)]
        Either an iterable giving the grid of alphas or a tuple ``(alpha0, q)``.
        In the latter case the sequence :math:`(alpha0*q^n)_{n=0,1,2,...}` is generated.
    init : Vector or None [default: None]
        Initial guess. In the default case, it is taken from the setting.
    max_CG_iter : int [default: 1000]
        Maximum number of CG iterations.
    delta : float or None [default: None]
        Data noise level.
    tol_fac : float [default: 0.5]
        Absolute tolerance for CG iterations is
        ``tol_fac * delta / sqrt(alpha)``.
    logging_level : int or str [default: "INFO"]
        Logging level for the inner CG iteration.
    """
    def __init__(
        self,
        setting: Setting,
        data: Vector | ObsDataT | None = None,
        alphas: Iterable[float] | tuple[float, float] = (1.0, 2 / 3),
        init: Vector | None = None,
        max_CG_iter: int = 1000,
        delta: float | None = None,
        tol_fac: float = 0.5,
        logging_level: int | str = "INFO",
    ) -> None:
        from ..nonlinear.gen_tikhonov import GeometricSequence
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError(
                'NonstationaryIteratedTikhonov not applicable to this setting. '
                + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)
        if isinstance(alphas,tuple) and len(alphas)==2:
            self._alphas = GeometricSequence(alphas[0],alphas[1])
        else:
            self._alphas = iter(alphas)
        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        """Right-hand side of the operator equation."""
        self.init = par['init']
        """Initial guess."""
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        if setting.codomain_free:
            self.y = None
            self.backprop_data = self.op.domain.empty()
            self.op.adjoint_data(self.data, out=self.backprop_data)
        else:
            self.y = self.op.codomain.empty()
            self.op(self.x, out=self.y)
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
        common = {
            'setting': _regularized_inner_setting(
                self.setting, alpha, self.logging_level
            ),
            'init': self.x,
            'x0': self.x,
            'logging_level': self.logging_level,
        }
        if self.setting.codomain_free:
            common['data'] = self.backprop_data
            solver_class = LavrentievCG
        else:
            common['data'] = self.data
            solver_class = TikhonovCG
        if self.delta is None:
            common['reltolx'] = self.tol_fac / sqrt(self.alpha_eff)
        else:
            common['tol'] = self.tol_fac * self.delta / sqrt(self.alpha_eff)
        tikhcg = solver_class(**common)
        self.x, self.y = tikhcg.run(inner_stoprule)
        if self.setting.codomain_free:
            self.y = None
        self.log.info('alpha_eff = {}, inner CG its = {}'.format(self.alpha_eff,inner_stoprule.iteration))

    check_applicability = staticmethod(
        TikhonovAlphaGrid.check_applicability
    )

class LavrentievCG(RegSolver):
    r"""Lavrentiev regularization for a self-adjoint linear problem.

    The method computes the solution of

    .. math::
        \left(A + \alpha H\right)x
        = data + \alpha H x_{\rm init},

    by conjugate gradients. Here :math:`A` is the self-adjoint forward
    operator, :math:`H` is the Hessian of the quadratic penalty, and
    :math:`\alpha` is ``setting.regpar``. Thus ``init`` is both the reference
    vector of the Lavrentiev regularization and, unless ``x0`` is supplied,
    the initial CG iterate.

    The embedded minimal-residual quantity ``kappa`` and the absolute and
    relative domain stopping rules are the same as in :class:`TikhonovCG`.
    No codomain residual is required.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Hilbert setting with a linear self-adjoint forward operator and a
        positive regularization parameter.
    data : ReadOnlyVector or None [default: None]
        Right-hand side in ``setting.op.codomain``. If omitted, it is resolved
        from the setting.
    init : ReadOnlyVector or None [default: None]
        Reference vector in the Lavrentiev functional. If omitted, the
        penalty functional's reference vector is used.
    x0 : ReadOnlyVector or None [default: None]
        Initial CG iterate. If omitted, ``setting.init`` is used independently
        of the functional's reference vector.
    tol, reltolx : float or None
        Absolute and relative domain tolerances for the internal stopping
        rule. If both are ``None``, the same fallback as in
        :class:`TikhonovCG` is used.
    all_tol_criteria : bool [default: True]
        Require all active tolerances instead of any one of them.
    max_it : int [default: 1000]
        Maximum number of inner CG iterations.
    krylov_basis : array-like or None [default: None]
        Optional storage for an orthonormal Krylov basis.
    preconditioner : Operator or None [default: None]
        Optional right preconditioner.
    """

    def __init__(
        self,
        setting: Setting,
        data: ReadOnlyVector | None = None,
        init: ReadOnlyVector | None = None,
        x0: ReadOnlyVector | None = None,
        tol: float | None = None,
        reltolx: float | None = 1e-2,
        all_tol_criteria: bool = True,
        max_it: int = 1000,
        krylov_basis: list[Vector] | None = None,
        preconditioner: Operator | None = None,
        logging_level: int | str = logging.INFO,
        x_exact: ReadOnlyVector | None = None,
    ) -> None:
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError(
                'LavrentievCG not applicable to this setting. ' + out['info']
            )

        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.tol = tol
        self.reltolx = reltolx
        self.reltoly = None
        self.all_tol_criteria = all_tol_criteria
        self.max_it = max_it
        self.logging_level = logging_level
        self.regpar = par['regpar']

        domain = self.op.domain
        self.data = par['data']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "LavrentievCG requires data either in the setting or as an "
                "argument."
            ))
        if self.data not in self.op.codomain:
            raise TypeError(Errors.not_in_vecsp(
                self.data,
                self.op.codomain,
                vec_name='data',
                space_name='codomain',
            ))

        self.init = domain.empty()
        if init is None:
            self.penalty.reference_vector(out=self.init)
        else:
            domain.copyto(self.init, par['init'])
        self.x = domain.empty()
        if x0 is None:
            initial_iterate = setting.init if init is None else self.init
            domain.copyto(self.x, initial_iterate)
        else:
            if x0 not in domain:
                raise TypeError(Errors.not_in_vecsp(
                    x0, domain, vec_name='x0', space_name='domain'
                ))
            domain.copyto(self.x, x0)

        self.hessR = self.penalty.hessian(self.x)
        if preconditioner is None:
            self.preconditioner = Identity(domain)
        elif (
            isinstance(preconditioner, Operator)
            and preconditioner.domain == domain
            and preconditioner.codomain == domain
        ):
            self.preconditioner = preconditioner
        else:
            raise TypeError(
                "preconditioner must be an Operator on setting.op.domain"
            )

        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)

        self._domain_work = domain.empty()
        self._domain_work2 = domain.empty()
        self._res = domain.empty()
        self._A_dir = domain.empty()
        self.g_res = domain.empty()
        self.dir = domain.empty()
        self.g_dir = domain.empty()

        # Dual residual: data - A*x + alpha*H*(init-x).
        domain.copyto(self._domain_work, self.data)
        self._domain_work -= self.y
        domain.copyto(self._domain_work2, self.init)
        self._domain_work2 -= self.x
        self.hessR(self._domain_work2, out=self._res)
        _axpy(
            domain,
            self._domain_work,
            self.regpar,
            self._res,
            self._domain_work2,
        )
        self.preconditioner.adjoint(
            self._domain_work, out=self.g_res
        )
        self.hessR.inverse(self.g_res, out=self._res)
        self.sq_norm_res = domain.vdot(self.g_res, self._res).real
        self.preconditioner(self._res, out=self.dir)
        self.hessR(self.dir, out=self.g_dir)
        self.kappa = 1
        """ratio of the squared norms of the residuals of the CG method and the MR-method.
        Used for error estimation."""

        if x_exact is not None and x_exact not in domain:
            raise TypeError(Errors.not_in_vecsp(
                x_exact, domain, vec_name='x_exact', space_name='domain'
            ))
        self.x_exact = x_exact
        self.y_exact = None

        self.krylov_basis = krylov_basis
        if self.krylov_basis is not None:
            self.iteration_number = 0
            self.krylov_basis[self.iteration_number, :] = self._res
            if self.sq_norm_res > 0:
                self.krylov_basis[self.iteration_number, :] /= sqrt(
                    self.sq_norm_res
                )
        if np.sqrt(self.sq_norm_res) < 1e-15:
            self.log.info(f'LavrentievCG: stopped at iteration 0 as residual is 0 to machine precision.')
            self.converge()

    def _next(self):
        domain = self.op.domain

        self.op(self.dir, out=self._A_dir)
        alpha_pre = (
            domain.vdot(self._A_dir, self.dir)
            + self.regpar * domain.vdot(self.g_dir, self.dir)
        ).real
        if alpha_pre <= 0:
            raise RuntimeError(
                f"The regularized self-adjoint system is not positive "
                f"definite in iteration {self.iteration_step_nr}."
            )
        # This parameter is often called alpha. We avoid that name because
        # alpha denotes the regularization parameter elsewhere.
        self.stepsize = self.sq_norm_res / alpha_pre

        _axpy(
            domain,
            self.x,
            self.stepsize,
            self.dir,
            self._domain_work,
        )
        _axpy(
            domain,
            self.y,
            self.stepsize,
            self._A_dir,
            self._domain_work,
        )

        domain.copyto(self._domain_work, self._A_dir)
        _axpy(
            domain,
            self._domain_work,
            self.regpar,
            self.g_dir,
            self._domain_work2,
        )
        self.preconditioner.adjoint(
            self._domain_work, out=self._domain_work2
        )
        self._domain_work2 *= self.stepsize
        self.g_res -= self._domain_work2
        self.hessR.inverse(self.g_res, out=self._res)
       
        sq_norm_res_old = self.sq_norm_res
        self.sq_norm_res = domain.vdot(self.g_res, self._res).real
        beta = self.sq_norm_res / sq_norm_res_old

        if self.krylov_basis is not None:
            self.iteration_number += 1
            if self.iteration_number < self.krylov_basis.shape[0]:
                self.krylov_basis[self.iteration_number, :] = self._res
                if self.sq_norm_res > 0:
                    self.krylov_basis[self.iteration_number, :] /= sqrt(
                        self.sq_norm_res
                    )

        self.kappa = 1 + beta * self.kappa
        self.preconditioner(self._res, out=self._domain_work)
        self.dir *= beta
        self.dir += self._domain_work
        self.hessR(self.dir, out=self.g_dir)

        if np.sqrt(self.sq_norm_res) <=1e-15:
            self.log.info(f'LavrentievCG: stopped at iteration {self.iteration_step_nr} as residual is 0 to machine precision.')
            self.converge()         

    def get_stoprule(self, other_stoprule=None):
        """Construct the MR-based internal stopping rule."""
        from regpy.stoprules import AndCombineRules, CombineRules

        stoprule_list = []
        if self.tol is not None:
            self.tol_stop = TikhonovCG.TolStop(
                self.tol,
                logging_level=self.logging_level,
                x_exact=self.x_exact,
            )
            stoprule_list.append(self.tol_stop)
        if self.reltolx is not None:
            self.reltolx_stop = TikhonovCG.RelTolXStop(
                self.reltolx,
                logging_level=self.logging_level,
                x_exact=self.x_exact,
            )
            stoprule_list.append(self.reltolx_stop)
        if not stoprule_list:
            self.reltolx_stop = TikhonovCG.RelTolXStop(
                10 / sqrt(self.regpar),
                logging_level=self.logging_level,
                x_exact=self.x_exact,
            )
            stoprule_list.append(self.reltolx_stop)

        combined_rule = (
            AndCombineRules(stoprule_list)
            if self.all_tol_criteria
            else CombineRules(stoprule_list)
        )
        min_it = (
            self.krylov_basis.shape[0]
            if self.krylov_basis is not None
            else 0
        )
        self.kappa_track = TikhonovCG.KappaTrackAndMinIt(
            min_it, logging_level=self.logging_level
        )
        rule = combined_rule & self.kappa_track
        if not _contains_count_iterations(other_stoprule):
            rule = (
                CountIterations(
                    self.max_it, logging_level=self.logging_level
                )
                + rule
            )
        return rule + other_stoprule if other_stoprule is not None else rule

    def run(
        self,
        stoprule: StopRule | None = None,
        max_it: int | None = None,
    ):
        """Run CG with its internal rule and optional external limits.

        Parameters
        ----------
        stoprule : StopRule or None, optional
            An additional stopping rule combined with the internal rule.
        max_it : int or None, optional
            An additional maximum number of iterations for this run.
        """
        if max_it is not None:
            iteration_limit = CountIterations(
                max_it, logging_level=self.logging_level
            )
            stoprule = (
                iteration_limit
                if stoprule is None
                else iteration_limit + stoprule
            )
        self.stoprule = self.get_stoprule(other_stoprule=stoprule)
        return super().run(stoprule=self.stoprule)

    def get_convergence_histories(self):
        """Return histories recorded by the internal stopping rules."""
        hist = self.kappa_track.history_dict.copy()
        if hasattr(self, 'tol_stop'):
            hist.update(self.tol_stop.history_dict)
        if hasattr(self, 'reltolx_stop'):
            hist.update(self.reltolx_stop.history_dict)
        return hist

    def primal(self):
        return self.x, self.y

    @staticmethod
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: ReadOnlyVector | None = None,
        user_init: ReadOnlyVector | None = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Check Hilbert applicability and self-adjointness."""
        out, par = check_Hilbert_applicability_linear(
            setting,
            regularization_parameter_required=True,
            check_is_norm_squared_type=False,
            user_data=user_data,
            user_init=user_init,
            op_norm=op_norm,
            **kwargs,
        )
        if not getattr(setting.op, 'self_adjoint', False):
            out['info'] += 'Operator is not self-adjoint. '
        data = par.get('data')
        if data is not None and data not in setting.op.codomain:
            out['info'] += 'Data do not belong to the operator codomain. '
        out['applicable'] = out['info'] == ''
        out['rate'] = np.nan
        return out, par
