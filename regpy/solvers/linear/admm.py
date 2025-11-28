from regpy.operators import Operator, ConvolutionOperator
from regpy.util import Errors

from ..general import RegSolver, RegularizationSetting, TikhonovRegularizationSetting
from .tikhonov import TikhonovCG

__all__ = ["ADMM","AMA"]

class ADMM(RegSolver):
    r"""The ADMM method for minimizing \(\frac{1}{\alpha}S(Tf) + R(f))\. 
    ADMM solves the problem \(\min_{u,v}[F(u)+G(v)])\ under the constraint that \(Au+Bv=b)\. Choosing 

    .. math::
        A&:=\begin{pmatrix} T \\ I \end{pmatrix} ,\; \\
        B&:=\begin{pmatrix} -I & 0 \\ 0 & -I \end{pmatrix}, \; \\
        b&:=\begin{pmatrix} 0 \\ 0 \end{pmatrix} ,\; \\
        F(f)&:= 0,\; \\
        G\begin{pmatrix} v_1 \\ v_2 \end{pmatrix}&:=\frac{1}{\alpha}S(v_1)+R(v_2) ,\; \\

    leads to a nice splitting of the operator \(T)\ and the functional \(R)\ seen in the Lagrangian

    .. math::
        L_\gamma(f,v_1,v_2,p_1,p_2):=& \\
        &\frac{1}{\alpha}S(v_1) + R(v_2) \\
        &- \langle\gamma p_1,Tf-v_1\rangle \\
        &- \langle\gamma p_2,f-v_2\rangle \\
        &+ \frac{\gamma}{2} \Vert Tf - v_1 \Vert^2 \\
        &+ \frac{\gamma}{2} \Vert f - v_2 \Vert^2.

    The minimization for \(f)\ simply reduces to the minimization of a quadratic Tikhonov functional.  This can 
    be achieved by the CG method, but ADMM is particularly efficient if a closed form expression is available for the 
    Tikhonov regularizer as for convolution operators or a matrix factorization. A corresponding `regpy.operators.operator` 
    can be passed as argument. 
    Splitting up the minimization for \(v_1)\ and \(v_2)\ one gets the algorithm below requiring the proximal 
    operators for the penalty and data fidelity functional. 

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals.
    init : dict [default: {}]
        The initial guess. Relevant keys are v1, v2, p1 and p2. If a key does not exist or if the value in None, 
        the corresponding variable is initialized by zero. 
    gamma : float [default: 1]
        Augmentation to the Lagrangian. Must be strictly greater than zero. 
    proximal_pars_data_fidelity : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the data fidelity term
    proximal_pars_penalty : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the penalty term
    regularizedInverse: `regpy.operators.Operator` [default: None]
        The operator \( (T^*T+\I)^{-1})\. If None, this operator is computed if T is a regpy.operators.Convolution. 
        Otherwise, the application of this inverse operator is implemented by CG.
    cg_pars : dict [default: {}]
        Parameter dictionary passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    logging_level: [default: logging.INFO]
        logging level
    """

    def __init__(self,  setting, init={}, gamma = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, 
                 regularizedInverse=None, cg_pars = None,logging_level = "INFO"):
        if not isinstance(setting,TikhonovRegularizationSetting):
            raise TypeError(Errors.not_instance(setting,TikhonovRegularizationSetting,add_info="ADMM requires the Setting to be a Tikhonov setting!"))
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ADMM requires the operator to be linear!"))
        if regularizedInverse is not None and not isinstance(regularizedInverse,Operator):
            raise TypeError(Errors.not_instance(regularizedInverse,Operator,add_info="ADMM requires the the regularized inverse to be either not given and None or a proper Operator!"))
        
        self.log.setLevel(logging_level)

        out, _ = ADMM.check_applicability(setting, regularizaedInverse=regularizedInverse)
        if out['applicable']==False and not out['info'] == 'No efficient regularized inverse seems to be available. ':
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
        if regularizedInverse is None and isinstance(setting.op, ConvolutionOperator) and setting.op.domain.shape_codomain==():
            adj = setting.op.conv_adjoint()
            regularizedInverse = adj.composition(setting.op)
            self.regularizedInverse = regularizedInverse.functional_calculus(lambda t: 1./(1.+t))
        else:
            self.regularizedInverse = regularizedInverse
        """ operator (T^*T+I)^{-1}"""

        if cg_pars is None:
            cg_pars = {}
        self.cg_pars = cg_pars
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.gramXinv = self.h_codomain.gram.inverse
        """ The inverse of the Gram matrix of the domain of the forward operator"""
        self.gramY = self.h_codomain.gram
        """ The Gram matrix of the image space of the forward operator"""        

        if self.regularizedInverse is None:
            self.x, self.y = TikhonovCG(
                setting=RegularizationSetting(self.op, self.h_domain, self.h_codomain),
                data=self.v1+self.p1,
                xref=self.v2+self.p2,
                regpar=1.,
                **self.cg_pars
            ).run()
        else:
            self.x = self.regularizedInverse(self.v2+self.p2 + self.op.adjoint(self.v1+self.p1))
            self.y = self.op(self.x)

    def check_applicability(setting, regularizaedInverse = None,op_norm=None):
        out = {'info': ''}; par = {}
        if not 'proximal' in setting.penalty.methods:
            out['info'] += 'Missing prox in penalty. '
        if not 'proximal' in setting.data_fid.methods:
            out['info'] += 'Missing prox in data fidelity functional. '
        if regularizaedInverse is None and not \
            (isinstance(setting.op, ConvolutionOperator) and setting.op.domain.shape_codomain==()):
            out['info'] += 'No efficient regularized inverse seems to be available. '
        out['applicable'] = out['info']==''
        if out['applicable']:
            out['info'] += 'Ergodic rate O(1/n).'
            out['rate'] = -1
        return out, par

    def _next(self):
        self.v1 = self.data_fid.proximal(self.y-self.p1, 1/(self.gamma*self.setting.regpar), self.proximal_pars_data_fidelity)
        self.v2 = self.penalty.proximal(self.x-self.p2, 1/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.y-self.v1)
        self.p2 -= self.gamma*(self.x-self.v2)

        if self.regularizedInverse is None:
            self.x, self.y = TikhonovCG(
                setting=RegularizationSetting(self.op, self.h_domain, self.h_codomain),
                data=self.v1+self.p1,
                xref=self.v2+self.p2,
                regpar=1.,
                **self.cg_pars
            ).run()
        else:
            self.x = self.regularizedInverse(self.v2+self.p2 + self.gramXinv(self.op.adjoint(self.gramY(self.v1+self.p1))))
            self.y = self.op(self.x)


class AMA(RegSolver):
    r"""The alternating minimization algorithm (AMA) for minimizing \(\frac{1}{\alpha}S(Tf) + R(f))\ with \(R)\ strongly convex.
    AMA solves the problem \(\min_{u,v}[F(u)+G(v)])\ under the constraint that \(Au+Bv=b)\. We choose

    .. math::
        T=A, B=-I, b=0, f=u, F=R and G=R 

    In contrast to standard ADMM we neglected the quadratic term in the update formula for :math:`f=u` leading to the iteration
    
    .. math::
       f^{l+1} &:= \argmin_f[R(f)-\langle T^*p^l,f\rangle]\; \\
       g^{l+1} &:= \mathrm{prox}_{\gamma^{-1}S}(Tf^{l+1)-\gamma^{-1}p^l) \\
       p^{l+1} &:= p^l + \gamma(T f^{l+1}-g^{l+1})

    
    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem. Includes the penalty and data fidelity functionals.
    init : dict [default: {}]
        The initial guess. Relevant keys are g and p. If a key does not exist or if the value in None, 
        the corresponding variable is initialized by zero. 
    gamma : float [default: 1]
        Augmentation to the Lagrangian. Must be strictly greater than zero. 
    proximal_pars_data_fidelity : dict [default: {}]
        Parameter dictionary passed to the computation of the prox-operator for the data fidelity term
    logging_level: [default: logging.INFO]
        logging level
    compute_dual: boolean [False]
        sets if dual is computed, it is not directly necessary for the algorithm. The default is False
    """

    def __init__(self,  setting, init={}, gamma = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, 
                 cg_pars = None,logging_level = "INFO",compute_dual = False):
        if not isinstance(setting,TikhonovRegularizationSetting):
            raise TypeError(Errors.not_instance(setting,TikhonovRegularizationSetting,add_info="AMA requires the Setting to be a Tikhonov setting!"))
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="AMA requires the operator to be linear!"))
        
        self.log.setLevel(logging_level)

        out, _ = AMA.check_applicability(setting)
        if out['applicable']==False:
            raise RuntimeError('AMA not applicable in this setting. '+out['info'])

        self.setting = setting

        self.g = init['g'] if 'g' in init and init['g'] is not None else self.op.codomain.zeros()
        self.p = init['p'] if 'p' in init and init['p'] is not None else self.op.codomain.zeros()
 
        self.gamma = gamma
        """ Augmentation parameter to Lagrangian. """
        self.proximal_pars_data_fidelity = proximal_pars_data_fidelity
        """ Prox parameters of data fidelity."""
        self.proximal_pars_penalty = proximal_pars_penalty
        """ Prox parameters of penalty."""
        self.gramY = self.h_codomain.gram
        """ The gram matrix of the image space"""

        if not hasattr(self,"compute_dual") or not self.compute_dual:
            self.compute_dual = compute_dual
        if self.compute_dual:
            self._compute_dual()

    def check_applicability(setting,op_norm=None):
        out = {'info': ''}; par = {}
        if not 'proximal' in setting.penalty.methods:
            out['info'] += 'Missing prox in penalty. '
        if not setting.penalty.convexity_param>0:
            out['info'] += 'Penalty functional not strongly convex. '
        if not 'proximal' in setting.data_fid.methods:
            out['info'] += 'Missing prox in data functional. '
        out['applicable'] = out['info']==''
        if out['applicable']:
            out['info'] += 'Ergodic rate O(1/n).'
            out['rate'] = -1
        return out, par

    def _compute_dual(self):
        self.dual = self.gramY(self.p)

    def _next(self):
        Tstar_p = self.op.adjoint(self.gramY(self.p))
        self.f = self.penalty.conj.subgradient(Tstar_p)
        if not self.penalty.is_subgradient(Tstar_p,self.f):
            raise Warning('update f may not be correct')
        Tf = self.op(self.f)
        self.g = self.data_fid.proximal(Tf-(1./self.gamma)*self.p,1./self.gamma)
        self.p += self.gamma*(self.g - Tf) 

        if self.compute_dual:
            self._compute_dual()
