import numpy as np

from . import Solver


class IrgnmCG(Solver):
    """The IRGNM_CG method.

    Solves the potentially non-linear, ill-posed equation:

       .. math:: T(x) = y,

    where   :math:`T` is a Frechet-differentiable operator. The number of
    iterations is effectively the regularization parameter and needs to be
    picked carefully.

    IRGNM stands for Iteratively Regularized Gauss Newton Method. CG stands for
    the Conjugate Gradient method. The regularized Newton equations are solved
    by the conjugate gradient method applied to the normal equation. The "outer
    iteration" and the "inner iteration" are referred to as the Newton
    iteration and the CG iteration, respectively. The CG method with all its
    iterations is run in each Newton iteration.

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    cgtol : list of float, optional
        Contains three tolerances:
        The first entry controls the relative accuracy of the Newton update in
        preimage (space of "x").
        The second entry controls the relative accuracy of the Newton update in
        data space.
        The third entry controls the reduction of the residual.
    regpar0 : float, optional
    regpar_step : float, optional
        With these (regpar0, regpar_step) we compute the regulization parameter
        for the k-th Newton step by regpar0*regpar_step^k.

    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    regpar0 : float
    regpar_step : float
        Needed for the computation of the regulization parameter for the k-th
        Newton step.
    k : int
        Is the k-th Newton step.
    cgtol : list of float
        Contains three tolerances.
    x : array
        The current point.
    y : array
        The value at the current point.
    """

    def __init__(self, setting, data, init, cgmaxit=50, regpar0=1, regpar_step=2/3,
                 cgtol=[0.3, 0.3, 1e-6]):
        super().__init__()
        self.setting = setting
        self.data = data
        self.init = init
        self.x = self.init
        self.cgmaxit = cgmaxit
        self.regpar = regpar0
        self.regpar_step = regpar_step
        self.cgtol = cgtol
        self.y, self.deriv = setting.op.linearize(self.x)

    def _next(self):
        """Run a single IRGNM_CG iteration.

        The while loop is the CG method, it has four conditions to stop. The
        first three work with the tolerances given in ``self.cgtol``. The last
        condition checks if the maximum number of CG iterations
        (``self.cgmaxit``) is reached.

        The CG method solves by CGNE


        .. math:: A h = b,

        with

        .. math:: A := G_X^{-1} F^{' *} G_Y F' + regpar*I
        .. math:: b := G_X^{-1} F^{' *} G_Y y + regpar*ref

        where

        +--------------------+-------------------------------------+
        | :math:`F`          | self.op                             |
        +--------------------+-------------------------------------+
        | :math:`G_X,~ G_Y`  | self.op.domain.gram, self.op.codomain.gram |
        +--------------------+-------------------------------------+
        | :math:`G_X^{-1}`   | self.op.domain.gram_inv               |
        +--------------------+-------------------------------------+
        | :math:`F'`         | self.op.derivative()                |
        +--------------------+-------------------------------------+
        | :math:`F'*`        | self.op.derivative().adjoint        |
        +--------------------+-------------------------------------+


        Returns
        -------
        bool
            Always True, as the IRGNM_CG method never stops on its own.

        """

        # Preparations for the CG method
        residual = self.data - self.y
        ztilde = self.setting.codomain.gram(residual)
        stilde = self.deriv.adjoint(ztilde) + self.regpar*self.setting.domain.gram(self.init - self.x)
        s = self.setting.domain.gram_inv(stilde)
        d = s
        dtilde = stilde
        norm_s = np.real(inner(stilde, s))
        norm_s0 = norm_s
        norm_h = 0
        h = np.zeros(np.shape(s))
        Th = np.zeros(np.shape(residual))
        Thtilde = Th

        cgstep = 0
        kappa = 1

        while (
              # First condition
              np.sqrt(norm_s/norm_h/kappa) / self.regpar > self.cgtol[0] / (1+self.cgtol[0]) and
              # Second condition
              np.sqrt(norm_s
                      / np.real(inner(Thtilde, Th))
                      / kappa/self.regpar)
              > self.cgtol[1] / (1+self.cgtol[1]) and
              # Third condition
              np.sqrt(np.float64(norm_s)/norm_s0/kappa) > self.cgtol[2] and
              # Fourth condition
              cgstep <= self.cgmaxit):

            z = self.deriv(d)
            ztilde = self.setting.codomain.gram(z)
            gamma = norm_s / np.real(self.regpar * inner(dtilde, d) + inner(ztilde, z))
            h = h + gamma*d
            Th = Th + gamma*z
            Thtilde = Thtilde + gamma*ztilde
            stilde -= (gamma*(self.deriv(ztilde) + self.regpar*dtilde)).real
            s = self.setting.domain.gram_inv(stilde)
            norm_s_old = norm_s
            norm_s = np.real(inner(stilde, s))
            beta = norm_s / norm_s_old
            d = s + beta*d
            dtilde = stilde + beta*dtilde
            norm_h = inner(h, self.setting.domain.gram(h))
            kappa = 1 + beta*kappa
            cgstep += 1

        self.x += h
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step


def inner(a, b):
    return np.real(np.vdot(a, b))
