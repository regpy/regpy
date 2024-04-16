import logging
import numpy as np

from regpy.solvers import Solver
from regpy import util
from regpy.functionals import Functional
from regpy.solvers import RegularizationSetting

from regpy.solvers.linear.tikhonov import TikhonovCG

class ADMM(Solver):
    r"""The ADMM method for minimizing \(S(Tf) + \alpha * R(f))\. 
    ADMM solves the problem \(\min_{u,v}[F(u)+G(v)])\ under the constraint that \(Au+Bv=b)\. Choosing 
    \[
        A:=\begin{pmatrix} T \\ I \end{pmatrix} ,\;
        B:=\begin{pmatrix} -I & 0 \\ 0 & -I \end{pmatrix}, \;
        b:=\begin{pmatrix} 0 \\ 0 \end{pmatrix} ,\;
        F(f):= 0,\;
        G\begin{pmatrix} v_1 \\ v_2 \end{pmatrix}:=R(v_1)+S(v_2) ,\;
    \]
    leads to a nice splitting of the operator \(T)\ and the functional \(R)\ seen in the Lagrangian
    \[
        L_\gamma(f,v_1,v_2,p_1,p_2):= 
        S(v_1) + R(v_2) 
        - \langle\gamma p_1,Tf-v_1\rangle 
        - \langle\gamma p_2,f-v_2\rangle
        + \frac{\gamma}{2} \Vert Tf - v_1 \Vert^2
        + \frac{\gamma}{2} \Vert f - v_2 \Vert^2.
    \]
    The minimization for \(f)\ simply reduces to a Tikhonov functional and is treated as such in the Algorithm. 
    Splitting up the minimization for \(v_1)\ and \(v_2)\ one gets the algorithm below requiring the proximal 
    operators for the penalty and data fidelity functional. 

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals.
    init : dict
        The initial guess. Must contain v1, v2, p1 and p2 keys. 
    gamma : float, optional
        Augmentation to the Lagrangian. Must be strictly greater than zero. 
    regpar : float
        The regularization parameter for the prox of penalty. Must be positive.
    proximal_pars_data_fidelity : dict, optional
        Parameter dictionary passed to the computation of the prox-operator for the data fidelity term
    proximal_pars_penalty : dict, optional
        Parameter dictionary passed to the computation of the prox-operator for the penalty term
    cg_pars : dict, optional
        Parameter dictionary passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    """
    def __init__(self,  setting, init, gamma = 1, regpar = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, cg_pars = None):
        assert isinstance(setting,RegularizationSetting)
        super().__init__()
        self.setting = setting
        """Regularization Setting."""
        assert self.setting.op.linear

        self.v1 = init['v1']
        self.v2 = init['v2']
        self.p1 = init['p1']
        self.p2 = init['p2']

        self.gamma = gamma
        """ Augmentation parameter to Lagrangian. """
        self.regpar = regpar
        """ Regularization parameter for inner Tikhonov."""
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        """ Prox parameters of data fidelity."""
        self.proximal_pars_penalty = proximal_pars_penalty
        """ Prox parameters of penalty."""

        if cg_pars is None:
            cg_pars = {}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""

        self.x, self.y = TikhonovCG(
            setting=RegularizationSetting(self.setting.op, self.setting.h_domain, self.setting.h_codomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2,
            regpar=self.regpar,
            **self.cg_pars
        ).run()

    def _next(self):
        self.v1 = self.setting.data_fid.proximal(self.setting.op(self.x)-self.p1, 1/self.gamma, self.proximal_pars_data_fidelity)
        self.v2 = self.setting.penalty.proximal(self.x-self.p2, 1/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.setting.op(self.x)-self.v1)
        self.p2 -= self.gamma*(self.x-self.v2)

        self.x, self.y = TikhonovCG(
            setting=RegularizationSetting(self.setting.op, self.setting.h_domain, self.setting.h_codomain),
            data=self.v1+self.p1,
            xref=self.v2+self.p2,
            regpar=self.regpar,
            **self.cg_pars
        ).run()
