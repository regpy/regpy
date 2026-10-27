from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from regpy.vecsps.base import ReadOnlyVector, Vector
import logging
import numpy as np

from regpy.operators import Operator, PtwMultiplication
from regpy.util import Errors

from ..solver import RegSolver
from ..setting import Setting
from .tikhonov import TikhonovCG

__all__ = ["ADMM"]

class ADMM(RegSolver):
    r"""The ADMM method for minimizing :math:`\frac{1}{\alpha}S(Tf) + R(f)`. 
    ADMM solves the problem :math:`\min_{u,v}[F(u)+G(v)]` under the constraint that :math:`Au+Bv=b`. Choosing 
    
    .. math::
        A&:=\begin{pmatrix} T \\ I \end{pmatrix} ,\; \\
        B&:=\begin{pmatrix} -I & 0 \\ 0 & -I \end{pmatrix}, \; \\
        b&:=\begin{pmatrix} 0 \\ 0 \end{pmatrix} ,\; \\
        F(f)&:= 0,\; \\
        G\begin{pmatrix} v_1 \\ v_2 \end{pmatrix}&:=\frac{1}{\alpha}S(v_1)+R(v_2) ,\; \\
    
    leads to a nice splitting of the operator :math:`T` and the functional :math:`R` seen in the Lagrangian
    
    .. math::
        L_\gamma(f,v_1,v_2,p_1,p_2):=& \\
        &\frac{1}{\alpha}S(v_1) + R(v_2) \\
        &- \langle\gamma p_1,Tf-v_1\rangle \\
        &- \langle\gamma p_2,f-v_2\rangle \\
        &+ \frac{\gamma}{2} \Vert Tf - v_1 \Vert^2 \\
        &+ \frac{\gamma}{2} \Vert f - v_2 \Vert^2.
    
    The minimization for :math:`f` reduces to the minimization of a quadratic Tikhonov functional.  
    This can be achieved by the CG method, but ADMM is usually only competitive if 
    a closed-form expression for the Tikhonov regularizer is available. 
    If the operator is marked as an SVD decomposition (as it is the case, e.g., for certain 
    convolution operators), a Tikhonov regularizer as computed by SVD. 
    A regularized inverse in form of a `regpy.operators.operator` can also be passed as argument. 
    Splitting up the minimization for :math:`v_1` and :math:`v_2` one gets the algorithm below requiring the proximal 
    operators for the penalty and data fidelity functional. 
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem. Includes the penalty and data fidelity functionals.
    init: ReadOnlyVector | None [default: None]
        The initial guess. Relevant keys are v1, v2, p1 and p2. If a key does not exist or if the value in None, 
        the corresponding variable is initialized by zero.
    gamma: float [default: 1]
        Augmentation to the Lagrangian. Must be strictly greater than zero.
    proximal_pars_data_fidelity: dict[str, Any] | None [default: None]
        Parameter dictionary passed to the computation of the prox-operator for the data fidelity term
    proximal_pars_penalty: dict[str, Any] | None [default: None]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term
    regularizedInverse: Operator | None [default: None]
        The operator (T^dagger T+I)^-1 on the domain Hilbert space, where
        T^dagger=G_X^-1 T* G_Y. A supplied operator takes precedence.
        Otherwise a labelled SVD is used to precompute the inverse. Without
        either option, applicability checking reports a caveat and CG is used. The
        regularization parameter of this inner inverse is 1, independent
        of the outer Tikhonov parameter and gamma.
    cg_pars: dict[str, Any] | None [default: None]
        Parameter dictionary passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    logging_level: int | str [default: 'INFO']
        logging level
    is_primal: bool [default: True]
        Whether iterates represent the original primal problem.
        Pass False when solving its dual. This labels coordinates and does not
        change the iteration formulas or construct a dual setting.
    """

    def __init__(self,  setting: Setting, init: ReadOnlyVector | None=None, gamma: float = 1, proximal_pars_data_fidelity: dict[str, Any] | None = None, proximal_pars_penalty: dict[str, Any] | None = None, 
                 regularizedInverse: Operator | None=None, cg_pars: dict[str, Any] | None = None,logging_level: int | str = "INFO",
                 is_primal: bool = True,
                 ) -> None:
        init = {} if init is None else init
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("ADMM requires the setting to contain a regularization parameter!"))       
        super().__init__(setting, is_primal=is_primal)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ADMM requires the operator to be linear!"))
        if regularizedInverse is not None and not isinstance(regularizedInverse,Operator):
            raise TypeError(Errors.not_instance(regularizedInverse,Operator,add_info="ADMM requires the the regularized inverse to be either not given and None or a proper Operator!"))
        
        self.log.setLevel(logging_level)

        out, _ = ADMM.check_applicability(setting, regularizedInverse=regularizedInverse)
        if not out['applicable']:
            raise RuntimeError('ADMM not applicable in this setting. '+out['info'])

        self.setting = setting

        self.v1 = init['v1'] if 'v1' in init and init['v1'] is not None else self.op.codomain.zeros()
        self.v2 = init['v2'] if 'v2' in init and init['v2'] is not None else self.op.domain.zeros()
        self.p1 = init['p1'] if 'p1' in init and init['p1'] is not None else self.op.codomain.zeros()
        self.p2 = init['p2'] if 'p2' in init and init['p2'] is not None else self.op.domain.zeros()

        self.gamma = gamma
        """ Augmentation parameter to Lagrangian. """
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        """ Prox parameters of data fidelity."""
        self.proximal_pars_penalty = proximal_pars_penalty
        """ Prox parameters of penalty."""
        self.gramXinv = self.h_domain.gram_inv
        self.gramY = self.h_codomain.gram
        self.regularizedInverse = regularizedInverse
        if regularizedInverse is None and getattr(self.op, 'is_svd', False):
            from .spectral import SVD
            hilbert_setting = Setting(self.op, self.h_domain, self.h_codomain)
            decomposition = SVD(hilbert_setting)
            # U maps spectral coordinates into X*, with U* G_X^-1 U = I.
            # I - G_X^-1 U diag(sigma^2/(1+sigma^2)) U* also acts
            # as identity on the null space of T, for rectangular SVDs.
            sigma = decomposition.singular_values
            weights = np.empty_like(sigma, dtype=float)
            small = sigma <= 1.
            weights[small] = sigma[small]**2 / (1. + sigma[small]**2)
            weights[~small] = 1. / (1. + (1. / sigma[~small])**2)
            self.regularizedInverse = (
                self.op.domain.identity
                - self.gramXinv * decomposition.U
                * PtwMultiplication(decomposition.U.domain, weights)
                * decomposition.U.adjoint
            )

        if cg_pars is None:
            cg_pars = {}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        if self.regularizedInverse is None:
            self.x, self.y = TikhonovCG(
                setting=Setting(
                    self.op, self.h_domain, self.h_codomain,
                    data=self.v1+self.p1, init=self.v2+self.p2, regpar=1.,
                ),
                **self.cg_pars
            ).run()
        else:
            self.x = self.regularizedInverse(self.v2+self.p2 + self.gramXinv(self.op.adjoint(self.gramY(self.v1+self.p1))))
            self.y = self.op(self.x)

    @staticmethod
    def check_applicability(setting: Setting, regularizedInverse: Operator | None = None,op_norm: float | None=None) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Check the requirements of this solver for the supplied setting.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        regularizedInverse: Operator | None [default: None]
            Regularizedinverse.
        op_norm: float | None [default: None]
            Operator-norm estimate; None requests numerical estimation.
        
        
        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            The first dictionary contains information on applicability, the second is empty here. 
        """
        out = {'info': ''}; par = {}
        if not setting.is_tikhonov:
            out['info'] += 'Setting has no regularization parameter. '
        if not setting.is_convex:
            out['info'] += 'Setting is not convex. '
        if setting.codomain_free:
            out['info'] += 'ADMM requires codomain vectors. '
        if not 'proximal' in setting.penalty.methods:
            out['info'] += 'Missing prox in penalty. '
        if (not setting.codomain_free
                and not 'proximal' in setting.data_fid.methods):
            out['info'] += 'Missing prox in data fidelity functional. '
        if not setting.op.linear:
            out['info'] += 'ADMM requires a linear operator. '
        if regularizedInverse is not None:
            if not isinstance(regularizedInverse, Operator):
                out['info'] += 'regularizedInverse must be an Operator. '
            elif (regularizedInverse.domain != setting.op.domain
                  or regularizedInverse.codomain != setting.op.domain):
                out['info'] += 'regularizedInverse must map the operator domain to itself. '
        elif not getattr(setting.op, 'is_svd', False):
            out['caveats'] = (
                'No supplied regularizedInverse and operator is not marked as SVD; '
                'ADMM will use inner TikhonovCG solves.'
            )
        out['applicable'] = out['info']==''
        if out['applicable']:
            out['info'] += 'ADMM applied to ' + ('primal' if setting._primal_setting is None else 'dual') + ' problem.  '
            out['rate'] = -1.
        return out, par

    def primal(self) -> tuple[Vector | None, Vector | None]:
        r"""Return the current primal iterate and operator image, when available.
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        return (self.x, self.y)

    def dual(self) -> tuple[Vector | None, Vector | None]:
        r"""Use p=0 at iteration zero; reconstruct the usual dual pair later.
        
        
        Returns
        -------
        tuple[Vector | None, Vector | None]
            Result described above.
        """
        if self.iteration_step_nr == 0:
            return self.op.codomain.zeros(), self.op.domain.zeros()
        _, dual = self.setting._complete_primal_dual_tuples(
            primal=self.primal(), dual=None,
        )
        return dual

    def _next(self) -> None:
        self.v1 = self.data_fid.proximal(self.y-self.p1, 1/(self.gamma*self.setting.regpar), self.proximal_pars_data_fidelity)
        self.v2 = self.penalty.proximal(self.x-self.p2, 1/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.y-self.v1)
        self.p2 -= self.gamma*(self.x-self.v2)

        if self.regularizedInverse is None:
            self.x, self.y = TikhonovCG(
                setting=Setting(
                    self.op, self.h_domain, self.h_codomain,
                    data=self.v1+self.p1, init=self.v2+self.p2, regpar=1.,
                ),
                **self.cg_pars
            ).run()
        else:
            self.x = self.regularizedInverse(self.v2+self.p2 + self.gramXinv(self.op.adjoint(self.gramY(self.v1+self.p1))))
            self.y = self.op(self.x)


