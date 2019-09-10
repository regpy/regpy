"""SQP inner solver"""


import logging
import numpy as np
import scipy

import setpath #noqa
from itreg.util import CGNE_reg
from itreg.operators import Weighted
from . import Inner_Solver

__all__ = ['SQP']


class SQP(Inner_Solver):

    """ The SQP method.

    Solves the inner problem

    .. math ::  \operatorname{argmin} ~ \operatorname{S}(F(x_in) +
                F'(x_in)(h); y_{obs}) + \mbox{regpar}~ R(x_in + h - x_0)

    iteratively.

    S is replaced by the second order Taylor-Approximation and then the
    corresponding quadratic problem is solved. This is repeated iteratively
    until the update is small enough.

    The following parameters are fixed:
         - maximum number of CG iterations:
             N_CG = 50
         - replace KL(a,b) by KL(a+_offset, b+_offset):
             offset0 =2e-6
         - offset is reduced in each Newton step by a factor offset_step:
             offset_step = 0.8
         - relative tolerance value for termination of inner iteration:
             update_ratio = 0.01
         - max number of inner iterations to minimize the KL-functional:
             inner_kl_it = 10

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    x_input : array
    y_input : array
    alpha : float
        Parameter for the CG method.
    it : integer
        Number of outer iterations.
    intensity : float
        Intensity of the operator.

    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    x : array
        The current point.
    y : array
        The value at the current point.
    alpha : float
        Parameter for the CG method.
    it : integer
        Number of outer iterations.
    intensity : float
        Intensity of the operator.

    """

    def __init__(self, op, data, init, x_input, y_input,
                 alpha, it, intensity = 1):
        """Initialize parameters """

        super().__init__(logging.getLogger(__name__))

        self.op = op
        self.data = data
        self.init = init
        self.x = x_input + 0j
        self.y = y_input
        self.alpha = alpha
        self.it = it
        self.intensity = intensity


        # maximum number of CG iterations
        self._N_CG = 50
        # replace KL(a,b) by KL(a+_offset, b+_offset)
        self._offset0 =2e-6
        # offset is reduced in each Newton step by a factor _offset_step
        self._offset_step = 0.8
        # relative tolerance value for termination of inner iteration
        self._update_ratio = 0.01
        # max number of inner iterations to minimize the KL-functional
        self._inner_kl_it = 10

        self.preparations()

    def preparations(self):
        """Define some more parameters."""

        self._offset = (self._offset0 * self.intensity
                        * self._offset_step**(self.it - 1))
        self._chi = self.data != -self._offset
        self._smb = scipy.logical_not(self._chi)
        self._h = np.zeros(len(self.x)) + 0j
        self._y_kl = self.y +0j
        self._first = 1
        self._norm_update = self._update_ratio + 1
        self._l = 1
        # step size
        self._mu = 1

    def next(self):
        """Run a single SQP iteration.

        Returns
        -------

        bool
            Since next() may return False, this algorithm can stop on its own
            without using a stoprule.

        """
        self._cont = (self._norm_update > self._update_ratio * self._first and
                      self._l <= self._inner_kl_it and
                      self._mu > 0)
        self._til_y_kl = self._y_kl
        self._weight = (self._chi * (self.data + self._offset + 0j)**0.5 /
                        (np.sqrt(2.)*(self._til_y_kl + self._offset)))
        self._b = ((1/np.sqrt(2)) * self._chi
                   / np.sqrt(self.data + self._offset + 0j)
                   * (self.data - self._til_y_kl) - 0.5*self._smb)
        self._opw = Weighted(self.op, self._weight)

        self._hl = CGNE_reg(op=self._opw, y=self._b,
                            xref=self.init - self.x - self._h,
                            regpar=self.alpha, cgmaxit=self._N_CG)

        self._y_kl_update = self.op.derivative.eval(self.op.params, self._hl)
        self._mask = self._y_kl_update < 0
        self._tmp1 = -0.9 * self._offset - self._y_kl
        self._tmp2 = self._tmp1[self._mask]
        self._tmp3 = self._y_kl_update[self._mask]

        #stepsize control
        if not np.any(self._mask):
            self._mu = 1
        else:
            self._mu = min(np.min(self._tmp2/self._tmp3),1)
        self._h += self._mu * self._hl
        self._y_kl += self._mu * self._y_kl_update
        self._norm_update = self._mu * self.op.domain.norm(self._hl)
        if self._l == 1:
            self._first = self._norm_update
        self._l += 1

        if not self._cont:
            self.x += self._h
            self.it += 1
        return self._cont
