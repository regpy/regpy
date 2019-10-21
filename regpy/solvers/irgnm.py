import numpy as np

from regpy.solvers import HilbertSpaceSetting, Solver
from regpy.solvers.tikhonov import TikhonovCG


class IrgnmCG(Solver):
    """The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

        ||T(x_n) + T'[x_n] h - data||**2 + regpar_n * ||x_n + h - init||**2

    where `T` is a Frechet-differentiable operator, using `regpy.solvers.tikhonov.TikhonovCG`.
    `regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: `2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cgpars : dict
        Parameter dictionary passed to the inner `regpy.solvers.tikhonov.TikhonovCG` solver.
    """

    def __init__(self, setting, data, regpar, regpar_step=2 / 3, init=None, cgpars=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""

    def _next(self):
        self.log.info('Running Tikhonov solver.')
        step, _ = TikhonovCG(
            setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.data - self.y,
            regpar=self.regpar,
            xref=self.init - self.x,
            **self.cgpars
        ).run()
        self.x += step
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step


class IrgnmCGLanczos(Solver):
    r"""The IRGNM_CG method.

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
    op : :class:`Operator <regpy.operators.Operator>`
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
    alpha0 : float, optional
    alpha_step : float, optional
        With these (alpha0, alpha_step) we compute the regulization parameter
        for the k-th Newton step by alpha0*alpha_step^k.

    Attributes
    ----------
    op : :class:`Operator <regpy.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    alpha0 : float
    alpha_step : float
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

    def __init__(self, setting, data, init, cgmaxit=50, alpha0=1, alpha_step=2 / 3.,
                 cgtol=[0.3, 0.3, 1e-6]):
        """Initialization of parameters."""

        # super().__init__(logging.getLogger(__name__))
        super().__init__()
        self.setting = setting
        self.data = data
        self.init = init
        self.x = self.init

        # Parameter for the outer iteration (Newton method)
        self.k = 0

        # Parameters for the inner iteration (CG method)
        self.cgmaxit = cgmaxit
        self.alpha0 = alpha0
        self.alpha_step = alpha_step
        self.cgtol = cgtol

        self.eigval_num = 3
        # orthonormalization computed in which krylov space
        self.krylov_num = 10
        self.orthonormal = np.zeros((self.krylov_num, self.data.shape[0]))
        self.need_prec_update = True

        # Update of the variables in the Newton iteration and preparation of
        # the first CG step.
        self.outer_update()

        # Initialize Lanczos method
        self._v = np.random.randn(np.size(self.data))
        self._v /= np.linalg.norm(self._v)

    def outer_update(self):
        """Updates and computes variables for the Newton iteration.

        In this function all variables of the current Newton iteration are
        updated, after the CG method is used. Furthermore some variables for
        the next time the CG method is used (in the next Newton iteration) are
        prepared.
        """
        self.y, deriv = self.setting.op.linearize(self.x)
        self._residual = self.data - self.y
        if self.need_prec_update == False:
            self._residual = self.data - self.setting.op(self.M_right @ self.x)
        self._xref = self.init - self.x
        if self.need_prec_update == False:
            self._xref = self.M_right @ self._xref
        self.k += 1
        self._regpar = self.alpha0 * self.alpha_step**self.k
        self._cgstep = 0
        self._kappa = 1

        # Preparations for the CG method
        self._ztilde = self.setting.Hcodomain.gram(self._residual)
        self._stilde = (deriv.adjoint(self._ztilde)
                        + self._regpar * self.setting.Hdomain.gram(self._xref))
        self._s = self.setting.Hdomain.gram_inv(self._stilde)
        if self.need_prec_update == False:
            self._s = self.M_left @ self._s
        self._d = self._s
        self._dtilde = self._stilde
        self._norm_s = np.real(self._stilde @ self._s)
        self._norm_s0 = self._norm_s
        self._norm_h = 0

        self._h = np.zeros(np.shape(self._s))
        self._Th = np.zeros(np.shape(self._residual))
        self._Thtilde = self._Th
        self.inner_num = 0

    def inner_update(self):
        """Updates and computes variables for the CG iteration.

        In this function all variables in each CG iteration , after ``self.x``
        was updated, are updated. Its only purpose is to improve tidiness.
        """
        self._Th = self._Th + self._gamma * self._z
        self._Thtilde = self._Thtilde + self._gamma * self._ztilde
        _, deriv = self.setting.op.linearize(self.x)
        self._stilde += (- self._gamma * (deriv.adjoint(self._ztilde)
                                          + self._regpar * self._dtilde)).real
        self._s = self.setting.Hdomain.gram_inv(self._stilde)
        if self.need_prec_update == False:
            self._s = self.M_left @ self._s
        self._norm_s_old = self._norm_s
        self._norm_s = np.real(self._stilde @ self._s)
        self._beta = self._norm_s / self._norm_s_old
        self._d = self._s + self._beta * self._d
        self._dtilde = self._stilde + self._beta * self._dtilde
        self._norm_h = self.setting.Hdomain.inner(self._h, self._h)
        self._kappa = 1 + self._beta * self._kappa
        self._cgstep += 1
        self.inner_num += 1

    def _next(self):
        r"""Run a single IRGNM_CG iteration.

        The while loop is the CG method, it has four conditions to stop. The
        first three work with the tolerances given in ``self.cgtol``. The last
        condition checks if the maximum number of CG iterations
        (``self.cgmaxit``) is reached.

        The CG method solves by CGNE


        .. math:: A h = b,

        with

        .. math:: A := G_X^{-1} F^{' *} G_Y F' + \mbox{regpar} ~I
        .. math:: b := G_X^{-1} F^{' *} G_Y y + \mbox{regpar}~ \mbox{xref}

        where

        +--------------------+-------------------------------------+
        | :math:`F`          | self.op                             |
        +--------------------+-------------------------------------+
        | :math:`G_X,~ G_Y`  | self.op.domain.gram, self.op.range.gram |
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

        if self.need_prec_update:
            while (
                self.inner_num < self.krylov_num or
                (  # First condition
                    np.sqrt(np.float64(self._norm_s) / self._norm_h / self._kappa)
                    / self._regpar > self.cgtol[0] / (1 + self.cgtol[0]) and
                    # Second condition
                    np.sqrt(np.float64(self._norm_s)
                            / np.real(self.setting.Hdomain.inner(self._Th, self._Th))
                            / self._kappa / self._regpar)
                    > self.cgtol[1] / (1 + self.cgtol[1]) and
                    # Third condition
                    np.sqrt(np.float64(self._norm_s) / self._norm_s0 / self._kappa)
                    > self.cgtol[2] and
                    # Fourth condition
                    self._cgstep <= self.cgmaxit)):

                # Computations and updates of variables

                _, deriv = self.setting.op.linearize(self.x)
                self._z = deriv(self._d)
                self._ztilde = self.setting.Hcodomain.gram(self._z)
                self._gamma = (self._norm_s
                               / np.real(self._regpar
                                         * self._dtilde @ self._d
                                         + self._ztilde @ self._z
                                         )
                               )
                self._h = self._h + self._gamma * self._d
                #            print(np.mean(self._norm_s)/np.mean(self._norm_s0))
                # Updating ``self.x``
                #           self.x += self._h
                if self.inner_num <= self.krylov_num:
                    self.orthonormal[self.inner_num - 1, :] = self._s / np.sqrt(self._norm_s)
                self.inner_update()

            self.lanzcos_update()

        # End of the CG method. ``self.outer_update()`` does all computations
        # of the current Newton iteration.
        else:
            while (
                # First condition
                # np.sqrt(np.float64(self._norm_s)/self._norm_h/self._kappa)
                # /self._regpar > self.cgtol[0] / (1+self.cgtol[0]) and
                # Second condition
                # np.sqrt(np.float64(self._norm_s)
                # /np.real(self.setting.Hdomain.inner(self._Th,self._Th))
                # /self._kappa/self._regpar)
                # > self.cgtol[1] / (1+self.cgtol[1]) and
                # Third condition
                # np.sqrt(np.float64(self._norm_s)/self._norm_s0/self._kappa)
                # > self.cgtol[2] and
                # Fourth condition
                self._cgstep <= self.cgmaxit
            ):
                # Computations and updates of variables
                _, deriv = self.setting.op.linearize(self.x)
                # self._z = deriv(self._d)
                self._z = deriv(self.M_right @ self._d)
                self._ztilde = self.setting.Hcodomain.gram(self._z)
                self._gamma = (self._norm_s
                               / np.real(self._regpar
                                         * self._dtilde @ self._d
                                         + self._ztilde @ self._z
                                         )
                               )
                self._h = self._h + self._gamma * self._d
                self.inner_update()

            self._h = np.linalg.solve(self.M_right, self._h)

            _, deriv = self.op.linearize(self.x)
            self._z_precond = deriv(self._d_precond)
            self._ztilde_precond = self.op.range.gram(self._z_precond)
            self._gamma_precond = (self._norm_s_precond
                                   / np.real(self._regpar
                                             * self.op.domain.inner(self._dtilde_precond,
                                                                    self._d_precond)
                                             + self.op.domain.inner(self._ztilde_precond,
                                                                    self._z_precond)
                                             )
                                   )
            self._h_precond = self._h_precond + self._gamma_precond * self._d_precond
            #            print(np.mean(self._norm_s)/np.mean(self._norm_s0))
            # Updating ``self.x``
            #           self.x += self._h

            self.inner_update_precond()

        #        self._h=scipy.optimize.minimize(self.M+self._regpar*np.identity(self.op.domain.shape[0]), self._h_precond)
        self._h = np.dot(np.linalg.inv(self.M), self._h_precond)

        self.x += self._h
        self.outer_update()
        self.need_prec_update = False
        if (int(np.sqrt(self.k)))**2 == self.k:
            self.need_prec_update = True
        self.outer_update()
        return True

    def lanzcos_update(self):
        """perform lanzcos method to calculate the preconditioner"""
        self.L = np.zeros((self.krylov_num, self.krylov_num))
        _, self.deriv = self.setting.op.linearize(self.x)
        for i in range(0, self.krylov_num):
            self.L[i, :] = np.dot(self.orthonormal, self.setting.Hdomain.gram_inv(
                self.deriv.adjoint(
                    self.setting.Hcodomain.gram(self.deriv((self.orthonormal[i, :]))))))
        # TODO: Only compute the three biggest eigenvalues with Lanczos method
        # self.lamb, self.U=np.linalg.eig(self.L)
        # self.diag_lamb=np.zeros(self.L.shape)
        # for i in range(0, self.eigval_num):
        #    self.diag_lamb[i, i]=1/(self._regpar+self.lamb[i])-1/self._regpar
        from scipy.sparse.linalg import eigsh
        self.lamb, self.U = eigsh(self.L, self.eigval_num, which='LM')
        self.diag_lamb = np.diag(1 / (self._regpar + self.lamb) - 1 / self._regpar)

        self.lanczos_krylov = np.float64(self.U @ self.diag_lamb @ self.U.transpose())
        self.lanczos = self.orthonormal.transpose() @ self.lanczos_krylov @ self.orthonormal
        self.M_left = 1 / self._regpar * np.identity(self.data.shape[0]) + self.lanczos
        self.M_right = np.eye(np.size(self.data))

    def lanczos(self, L, v, maxit):
        """perform lanczos method to calculate tridiagonal decomposition"""
        epsilon = np.dot(v, L @ v)
        w = L @ v - epsilon * v
        zeta = np.linalg.norm(w)
        v_old = v

        V = np.zeros((maxit, maxit))
        Epsilon = np.zeros(maxit)
        Zeta = np.zeros(maxit - 1)

        V[0, :] = v
        Epsilon[0] = epsilon

        counter = 1
        while (zeta != 0 and counter < maxit):
            v = w / zeta
            epsilon = np.dot(v, L @ v)
            w = L @ v - epsilon * v - zeta * v_old
            zeta = np.linalg.norm(w)
            v_old = v

            V[counter, :] = v
            Epsilon[counter] = epsilon
            Zeta[counter - 1] = zeta
            counter += 1
        return [V, Epsilon, Zeta]
