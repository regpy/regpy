from itreg.operators import NonlinearOperator, Params
from itreg.spaces import UniformGrid
from itreg.util import set_defaults

import numpy as np


class MediumScattering(NonlinearOperator):
    """Acoustic scattering problem for inhomogeneous medium.

    The forward problem is solved Vainikko's fast solver of the Lippmann
    Schwinger equation.

    References
    ----------
    T. Hohage: On the numerical solution of a 3D inverse medium scattering
    problem. Inverse Problems, 17:1743-1763, 2001.

    G. Vainikko: Fast solvers of the Lippmann-Schwinger equation in: Direct and
    inverse problems of mathematical physics edited by R.P.Gilbert, J.Kajiwara,
    and S.Xu, Kluwer, 2000.
    """

    def __init__(self, gridsize, radius, wave_number, inc_directions,
                 meas_directions, support=None, amplitude=False,
                 coarsesize=None, gmres_args={}):
        assert len(gridsize) in (2, 3)
        assert all(s % 2 == 0 for s in gridsize)
        grid = UniformGrid(*(np.linspace(-2*radius, 2*radius, s, endpoint=False)
                             for s in gridsize))

        if support is None:
            support = (np.linalg.norm(grid.coords, axis=0) <= radius)
        else:
            support = np.asarray(support, dtype=bool)
        assert support.shape == grid.shape

        inc_directions = np.asarray(inc_directions)
        assert inc_directions.ndim == 2
        assert inc_directions.shape[1] == grid.ndim
        inc_directions = inc_directions / np.linalg.norm(inc_directions, axis=1)[:, np.newaxis]

        meas_directions = np.asarray(meas_directions)
        assert meas_directions.ndim == 2
        assert meas_directions.shape[1] == grid.ndim
        meas_directions = meas_directions / np.linalg.norm(meas_directions, axis=1)[:, np.newaxis]

        if amplitude:
            range_dtype = float
        else:
            range_dtype = complex
        range = UniformGrid(axisdata=(meas_directions, inc_directions),
                            names=('meas', 'inc'),
                            dtype=range_dtype)

        if coarsesize:
            coarse = UniformGrid(*(np.linspace(-2*radius, 2*radius, s, endpoint=False)
                                   for s in coarsesize))
        else:
            coarse = None

        gmres_args = set_defaults(gmres_args, restart=10, tol=1e-14, maxiter=100)

        super().__init__(Params(grid, range, wave_number=wave_number,
                                amplitude=amplitude, coarse=coarse,
                                gmres_args=gmres_args))

    def _eval(self, x, differentiate):
        # data.contrast=1j*np.zeros((np.prod(params.scattering.N)))
        # np.put(data.contrast, params.domain.parameters_domain.ind_support, x)
        # if params.scattering.N_coarse:
        #     Nfac = np.prod(params.scattering.N_coarse)/np.prod(params.scattering.N)
        #     contrast_hat = np.fft.fftn(np.reshape(data.contrast,params.scattering.N, order='F'))
        #     if params.domain.dim==2:
        #         data.contrast_coarse = Nfac*np.fft.ifftn(contrast_hat[params.scattering.prec.dual_x_coarse.astype(int),:][:,params.scattering.prec.dual_y_coarse.astype(int)])
        #     if params.domain.dim==3:
        #         data.contrast_coarse = Nfac*np.fft.ifftn(contrast_hat[params.scattering.prec.dual_x_coarse.astype(int),:, :][:,params.scattering.prec.dual_y_coarse.astype(int),:][:, :, params.scattering.prec.dual_z_coarse.astype(int)])

        x = x[self.params.support]

        totalfield = np.empty(self.domain.shape + (self.range.shape.inc,), dtype=complex)
        farfield = np.empty((self.range.shape.meas, self.range.shape.inc), dtype=complex)
        rhs = self.domain.zeros(dtype=complex)

        # TODO parallelize
        for j in range(self.range.shape.inc):
            # Solve Lippmann-Schwinger equation v + a(k*v) = a*u_inc for the
            # unknown v = a u_total. The Fourier coefficients of the periodic
            # convolution kernel k are precomputed.

            # rhs=complex(0,1)*np.zeros(np.prod(params.scattering.N))
            # np.put(rhs, params.domain.parameters_domain.ind_support, x*params.scattering.prec.incMatrix[:, j])
            # rhs=np.reshape(rhs, params.scattering.N, order='F')
            rhs[self.params.support] = x * self.params.inc_matrix[:, j]

            if not params.scattering.N_coarse:
                LippmannSchwingerOperator = scsla.LinearOperator(
                    (self.domain.size, self.domain.size),
                    matvec=(lambda x: MediumScatteringBase.LippmannSchwingerOp(params, data, x)))
                v, info = scsla.gmres(LippmannSchwingerOperator, rhs.ravel(),
                                      **self.params.gmres_args)
                v = v.reshape(self.domain.shape)
                if info > 0:
                    self.log.warn('Gmres failed to converge')
                elif info < 0:
                    self.log.warn('Illegal Gmres input or breakdown')
                else:
                    self.log.info('Gmres converged')
            else:
                rhs = np.fft.fftn(rhs)
                v = MediumScatteringBase.SolveTwoGrid(params, data, rhs)

            # np.dot(params.scattering.prec.farfieldMatrix, v.reshape(np.prod(params.scattering.N), order='F')[params.domain.parameters_domain.ind_support])

            # TODO really only v[support] here?
            farfield[:, j] = self.params.farfieldMatrix @ v[self.params.support]

            # The total field can be recovered from v in a stable manner by the formula
            # u_total = ui - k*v
            if differentiate:
                aux = np.fft.ifftn(params.scattering.prec.K_hat * np.fft.fftn(v))
                totalfield[:, j] = self.params.inc_matrix[:, j] - aux.T[self.params.support]

        if self.params.amplitude:
            return np.abs(farfield)**2
        else:
            return farfield
