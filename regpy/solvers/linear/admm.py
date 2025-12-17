from regpy.operators import Operator, ConvolutionOperator
from regpy.util import Errors

from ..general import RegSolver, Setting
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

    The minimization for :math:`f` simply reduces to the minimization of a quadratic Tikhonov functional.  This can 
    be achieved by the CG method, but ADMM is particularly efficient if a closed form expression is available for the 
    Tikhonov regularizer as for convolution operators or a matrix factorization. A corresponding `regpy.operators.operator` 
    can be passed as argument. 
    Splitting up the minimization for :math:`v_1` and :math:`v_2` one gets the algorithm below requiring the proximal 
    operators for the penalty and data fidelity functional. 

    Parameters
    ----------
    setting : regpy.solvers.Setting
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
        The operator :math:` (T^*T+\I)^{-1}`. If None, this operator is computed if T is a regpy.operators.Convolution. 
        Otherwise, the application of this inverse operator is implemented by CG.
    cg_pars : dict [default: {}]
        Parameter dictionary passed to the inner `regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    logging_level: [default: logging.INFO]
        logging level
    """

    def __init__(self,  setting, init={}, gamma = 1, proximal_pars_data_fidelity = None, proximal_pars_penalty = None, 
                 regularizedInverse=None, cg_pars = None,logging_level = "INFO"):
        if not setting.is_tikhonov:
            raise ValueError(Errors.value_error("ADMM requires the setting to contain a regularization parameter!"))       
        super().__init__(setting)
        if not self.op.linear:
            raise ValueError(Errors.not_linear_op(self.op,add_info="ADMM requires the operator to be linear!"))
        if regularizedInverse is not None and not isinstance(regularizedInverse,Operator):
            raise TypeError(Errors.not_instance(regularizedInverse,Operator,add_info="ADMM requires the the regularized inverse to be either not given and None or a proper Operator!"))
        
        self.log.setLevel(logging_level)

        out, _ = ADMM.check_applicability(setting, regularizedInverse=regularizedInverse)
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
                setting=Setting(self.op, self.h_domain, self.h_codomain),
                data=self.v1+self.p1,
                xref=self.v2+self.p2,
                regpar=1.,
                **self.cg_pars
            ).run()
        else:
            self.x = self.regularizedInverse(self.v2+self.p2 + self.op.adjoint(self.v1+self.p1))
            self.y = self.op(self.x)

    def check_applicability(setting, regularizedInverse = None,op_norm=None):
        out = {'info': ''}; par = {}
        if not 'proximal' in setting.penalty.methods:
            out['info'] += 'Missing prox in penalty. '
        if not 'proximal' in setting.data_fid.methods:
            out['info'] += 'Missing prox in data fidelity functional. '
        if regularizedInverse is None and not \
            (isinstance(setting.op, ConvolutionOperator) and setting.op.domain.shape_codomain==()):
            out['info'] += 'No efficient regularized inverse seems to be available. '
        out['applicable'] = out['info']==''
        if out['applicable']:
            out['info'] += 'Ergodic rate O(1/n).'
            out['rate'] = -1
        return out, par

    def primal(self):
        return (self.x, self.y)

#    def dual(self):
#        p = self.setting.primal_to_dual(self.primal)
#        return p,self.op.adjoint(p)

    def _next(self):
        self.v1 = self.data_fid.proximal(self.y-self.p1, 1/(self.gamma*self.setting.regpar), self.proximal_pars_data_fidelity)
        self.v2 = self.penalty.proximal(self.x-self.p2, 1/self.gamma, self.proximal_pars_penalty)
        self.p1 -= self.gamma*(self.y-self.v1)
        self.p2 -= self.gamma*(self.x-self.v2)

        if self.regularizedInverse is None:
            self.x, self.y = TikhonovCG(
                setting=Setting(self.op, self.h_domain, self.h_codomain),
                data=self.v1+self.p1,
                xref=self.v2+self.p2,
                regpar=1.,
                **self.cg_pars
            ).run()
        else:
            self.x = self.regularizedInverse(self.v2+self.p2 + self.gramXinv(self.op.adjoint(self.gramY(self.v1+self.p1))))
            self.y = self.op(self.x)


