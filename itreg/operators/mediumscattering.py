from functools import partial
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import scipy.sparse.linalg as spla
from scipy.special import hankel1, jv as besselj

from . import NonlinearOperator, Params
from .. import spaces
from .. import util


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

    def __init__(self, gridshape, radius, wave_number, inc_directions,
                 meas_directions, support=None, amplitude=False,
                 coarseshape=None, coarseiterations=3, gmres_args={}):
        assert len(gridshape) in (2, 3)
        assert all(isinstance(s, int) for s in gridshape)
        # TODO should this be a complex space?
        grid = spaces.UniformGrid(*(np.linspace(-2*radius, 2*radius, s, endpoint=False)
                                    for s in gridshape))

        if support is None:
            support = (np.linalg.norm(grid.coords, axis=0) <= radius)
        elif callable(support):
            support = np.asarray(support(grid, radius), dtype=bool)
        else:
            support = np.asarray(support, dtype=bool)
        assert support.shape == grid.shape

        inc_directions = np.asarray(inc_directions)
        assert inc_directions.ndim == 2
        assert inc_directions.shape[1] == grid.ndim
        inc_directions = inc_directions / np.linalg.norm(inc_directions, axis=1)[:, np.newaxis]
        inc_matrix = np.exp(-1j * wave_number * (inc_directions @ grid.coords[:, support]))

        meas_directions = np.asarray(meas_directions)
        assert meas_directions.ndim == 2
        assert meas_directions.shape[1] == grid.ndim
        meas_directions = meas_directions / np.linalg.norm(meas_directions, axis=1)[:, np.newaxis]
        farfield_matrix = np.exp(-1j * wave_number * (meas_directions @ grid.coords[:, support]))

        if grid.ndim == 2:
            # TODO This appears to be missing a factor -exp(i pi/4) / sqrt(8 pi wave_number)
            farfield_matrix *= wave_number**2 * grid.volume_elem
            compute_kernel = partial(compute_kernel_2D, 2 * wave_number * radius)
        elif grid.ndim == 3:
            # TODO The sign appears to be wrong
            farfield_matrix *= wave_number**2 * grid.volume_elem / (4*np.pi)
            compute_kernel = partial(compute_kernel_3D, 2 * wave_number * radius)

        if coarseshape:
            if not all(c < s for c, s in zip(coarseshape, gridshape)):
                raise ValueError('coarse grid is not coarser than fine grid')
            assert all(isinstance(c, int) for c in coarseshape)
            coarsegrid = spaces.UniformGrid(*(np.linspace(-2*radius, 2*radius, c, endpoint=False)
                                              for c in coarseshape)),
            coarse = dict(
                grid=coarsegrid,
                kernel=compute_kernel(coarsegrid.shape),
                dualcoords=np.ix_(*(ifftshift(np.arange(-(c//2), (c+1)//2)) for c in coarseshape)),
                iterations=coarseiterations)
        else:
            coarse = None

        gmres_args = util.set_defaults(gmres_args, restart=10, tol=1e-14, maxiter=100)

        super().__init__(Params(
            domain=grid,
            codomain=spaces.UniformGrid(axisdata=(meas_directions, inc_directions),
                                        names=('meas', 'inc'),
                                        dtype=float if amplitude else complex),
            inc_matrix=inc_matrix,
            farfield_matrix=farfield_matrix,
            wave_number=wave_number,
            support=support,
            amplitude=amplitude,
            coarse=coarse,
            gmres_args=gmres_args,
            kernel=compute_kernel(grid.shape)))

    def _alloc(self):
        self._totalfield = np.empty((np.sum(self.params.support), self.codomain.shape.inc,),
                                    dtype=complex)
        # These belong to self, not params, since they implicitly depend on
        # self._contrast
        self._lippmann_schwinger = spla.LinearOperator(
            (self.domain.size,) * 2,
            matvec=self._lippmann_schwinger_op,
            rmatvec=self._lippmann_schwinger_adjoint,
            dtype=complex)
        if self.params.coarse:
            self._lippmann_schwinger_coarse = spla.LinearOperator(
                (self.params.coarse.grid.size,) * 2,
                matvec=self._lippmann_schwinger_coarse_op,
                rmatevec=self._lippmann_schwinger_coarse_adjoint,
                dtype=complex)

    def _eval(self, contrast, differentiate=False):
        contrast[~self.params.support] = 0
        self._contrast = contrast
        if self.params.coarse:
            # TODO take real part? what about even case? for 1d, highest
            # fourier coeff must be real then, which is not guaranteed by
            # subsampling here.
            aux = fftn(self._contrast)[self.params.coarse.dualcoords]
            self._coarse_contrast = (
                (self.params.coarse.grid.size / self.domain.size) *
                ifftn(aux))
        farfield = self.codomain.empty(dtype=complex)
        rhs = self.domain.zeros(dtype=complex)
        # TODO parallelize
        for j in range(self.codomain.shape.inc):
            # Solve Lippmann-Schwinger equation v + a(k*v) = a*u_inc for the
            # unknown v = a u_total. The Fourier coefficients of the periodic
            # convolution kernel k are precomputed.
            rhs[self.params.support] = self.params.inc_matrix[j, :] * contrast[self.params.support]
            if self.params.coarse:
                v = self._solve_two_grid(rhs)
            else:
                v = (self
                     ._gmres(self._lippmann_schwinger, rhs)
                     .reshape(self.domain.shape))
            farfield[:, j] = self.params.farfield_matrix @ v[self.params.support]
            # The total field can be recovered from v in a stable manner by the formula
            # u_total = ui - k*v
            if differentiate:
                self._totalfield[:, j] = (self.params.inc_matrix[j, :] -
                                          ifftn(self.params.kernel * fftn(v))[self.params.support])
        if self.params.amplitude:
            self._farfield = farfield
            return np.abs(farfield)**2
        else:
            return farfield

    def _derivative(self, contrast):
        contrast = contrast[self.params.support]
        farfield = self.codomain.empty(dtype=complex)
        rhs = self.domain.zeros(dtype=complex)
        for j in range(self.codomain.shape.inc):
            rhs[self.params.support] = self._totalfield[:, j] * contrast
            if self.params.coarse:
                v = self._solve_two_grid(rhs)
            else:
                v, info = (self
                           ._gmres(self._lippmann_schwinger, rhs)
                           .reshape(self.domain.shape))
            farfield[:, j] = self.params.farfield_matrix @ v[self.params.support]
        if self.params.amplitude:
            return 2 * (self._farfield.real * farfield.real +
                        self._farfield.imag * farfield.imag)
        else:
            return farfield

    def _adjoint(self, farfield):
        if self.params.amplitude:
            farfield = 2 * self._farfield * farfield
        v = self.domain.zeros(dtype=complex)
        farfield_matrix_H = self.params.farfield_matrix.conj().T
        contrast = self.domain.zeros()
        for j in range(self.codomain.shape.inc):
            v[self.params.support] = farfield_matrix_H @ farfield[:, j]
            if self.params.coarse:
                rhs = self._solve_two_grid_adjoint(v)
            else:
                rhs = (self
                       ._gmres(self._lippmann_schwinger.adjoint(), v)
                       .reshape(self.domain.shape))
            rhs_supp = rhs[self.params.support]
            contrast[self.params.support] += (
                self._totalfield[:, j].real * rhs_supp.real +
                self._totalfield[:, j].imag * rhs_supp.imag)
        return contrast

    def _solve_two_grid(self, rhs):
        rhs = fftn(rhs)
        v = self.domain.zeros(dtype=complex)
        rhs_coarse = rhs[self.coarse.dualcoords]
        for remaining_iters in range(self.params.coarse.iterations, 0, -1):
            v_coarse = (self
                        ._gmres(self._lippmann_schwinger_coarse, rhs_coarse)
                        .reshape(self.params.coarse.grid.shape))
            v[self.params.coarse.dualcoords] = v_coarse
            if remaining_iters > 0:
                rhs_coarse = fftn(self._coarse_contrast * ifftn(
                    self.params.coarse.kernel * v_coarse))
                v = rhs - fftn(self._contrast * ifftn(self.params.kernel * v))
                rhs_coarse += v[self.params.coarse.dualcoords]
        return ifftn(v)

    def _solve_two_grid_adjoint(self, v):
        v = fftn(v)
        rhs = self.domain.zeros(dtype=complex)
        v_coarse = v[self.params.coarse.dualcoords]
        for remaining_iters in range(self.params.coarse.iterations, 0, -1):
            rhs_coarse = (self
                          ._gmres(self._lippmann_schwinger_coarse.adjoint(), v_coarse)
                          .reshape(self.params.coarse.grid.shape))
            rhs[self.params.coarse.dualcoords] = rhs_coarse
            if remaining_iters > 0:
                v_coarse = self.params.coarse.kernel * fftn(
                    self._coarse_contrast * ifftn(rhs_coarse))
                rhs = v - self.params.kernel * fftn(self._contrast * ifftn(rhs))
                v_coarse += rhs[self.params.coarse.dualcoords]
        return ifftn(rhs)

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, rhs.ravel(), **self.params.gmres_args)
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        else:
            self.log.info('Gmres converged')
        return result

    def _lippmann_schwinger_op(self, v):
        """Lippmann-Schwinger operator in spatial domain on fine grid
        """
        v = v.reshape(self.domain.shape)
        v += self._contrast * ifftn(self.params.kernel * fftn(v))
        return v.ravel()

    def _lippmann_schwinger_adjoint(self, v):
        """Adjoint Lippmann-Schwinger operator in spatial domain on fine grid
        """
        v = v.reshape(self.domain.shape)
        v += ifftn(np.conj(self.params.kernel) * fftn(np.conj(self._contrast) * v))
        return v.ravel()

    def _lippmann_schwinger_coarse_op(self, v):
        """Lippmann-Schwinger operator in frequency domain on coarse grid
        """
        v = v.reshape(self.params.coarse.grid.shape)
        v += fftn(self._coarse_contrast * ifftn(self.params.coarse.kernel * v))
        return v.ravel()

    def _lippmann_schwinger_coarse_adjoint(self, v):
        """Lippmann-Schwinger operator in frequency domain on coarse grid
        """
        v = v.reshape(self.params.coarse.grid.shape)
        v += np.conj(self.params.coarse.kernel) * fftn(np.conj(self._coarse_contrast) * ifftn(v))
        return v.ravel()


def compute_kernel_2D(R, shape):
    J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
    piabsJ = np.pi * np.linalg.norm(J, axis=0)
    Jzero = tuple(s//2 for s in shape)

    K_hat = (2*R)**(-1) * R**2 / (piabsJ**2 - R**2) * (
        1 + 1j*np.pi/2 * (
            piabsJ * besselj(1, piabsJ) * hankel1(0, R) -
            R * besselj(0, piabsJ) * hankel1(1, R)))
    K_hat[Jzero] = -1/(2*R) + 1j*np.pi/4 * hankel1(1, R)
    K_hat[piabsJ == R] = 1j*np.pi*R/8 * (
        besselj(0, R) * hankel1(0, R) + besselj(1, R) * hankel1(1, R))
    return 2 * R * np.fft.fftshift(K_hat)


def compute_kernel_3D(R, shape):
    J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
    piabsJ = np.pi * np.linalg.norm(J, axis=0)
    Jzero = tuple(s//2 for s in shape)

    K_hat = (2*R)**(-3/2) * R**2 / (piabsJ**2 - R**2) * (
        1 - np.exp(1j*R) * (np.cos(piabsJ) - 1j*R * np.sin(piabsJ) / piabsJ))
    K_hat[Jzero] = -(2*R)**(-1.5) * (1 - np.exp(1j*R) * (1 - 1j*R))
    K_hat[piabsJ == R] = -1j/4 * (2*R)**(-1/2) * (1 - np.exp(1j*R) * np.sin(R) / R)
    return 2 * R * np.fft.fftshift(K_hat)
