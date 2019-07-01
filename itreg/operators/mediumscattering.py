from functools import partial
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import scipy.sparse.linalg as spla
from scipy.special import hankel1, jv as besselj

from . import NonlinearOperator
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
                 meas_directions, support=None, amplitude=False, refractive=True,
                 absorptive=True, coarseshape=None, coarseiterations=3, gmres_args={}):
        assert len(gridshape) in (2, 3)
        assert all(isinstance(s, int) for s in gridshape)
        assert refractive or absorptive
        grid = spaces.UniformGrid(
            *(np.linspace(-2*radius, 2*radius, s, endpoint=False)
              for s in gridshape),
            dtype=complex if refractive and absorptive else float)

        if support is None:
            support = (np.linalg.norm(grid.coords, axis=0) <= radius)
        elif callable(support):
            support = np.asarray(support(grid, radius), dtype=bool)
        else:
            support = np.asarray(support, dtype=bool)
        assert support.shape == grid.shape
        self.support = support

        self.amplitude = amplitude
        self.refractive = refractive
        self.absorptive = absorptive

        inc_directions = np.asarray(inc_directions)
        assert inc_directions.ndim == 2
        assert inc_directions.shape[1] == grid.ndim
        inc_directions = inc_directions / np.linalg.norm(inc_directions, axis=1)[:, np.newaxis]
        self.inc_matrix = np.exp(-1j * wave_number * (inc_directions @ grid.coords[:, support]))

        meas_directions = np.asarray(meas_directions)
        assert meas_directions.ndim == 2
        assert meas_directions.shape[1] == grid.ndim
        meas_directions = meas_directions / np.linalg.norm(meas_directions, axis=1)[:, np.newaxis]
        self.farfield_matrix = np.exp(
            -1j * wave_number * (meas_directions @ grid.coords[:, support]))

        if grid.ndim == 2:
            # TODO This appears to be missing a factor -exp(i pi/4) / sqrt(8 pi wave_number)
            self.farfield_matrix *= wave_number**2 * grid.volume_elem
            compute_kernel = partial(compute_kernel_2D, 2 * wave_number * radius)
        elif grid.ndim == 3:
            # TODO The sign appears to be wrong
            self.farfield_matrix *= wave_number**2 * grid.volume_elem / (4*np.pi)
            compute_kernel = partial(compute_kernel_3D, 2 * wave_number * radius)

        self.kernel = compute_kernel(grid.shape)

        if coarseshape:
            if not all(c < s for c, s in zip(coarseshape, gridshape)):
                raise ValueError('coarse grid is not coarser than fine grid')
            assert all(isinstance(c, int) for c in coarseshape)
            self.coarse = True
            self.coarsegrid = spaces.UniformGrid(
                *(np.linspace(-2*radius, 2*radius, c, endpoint=False)
                  for c in coarseshape))
            self.coarsekernel = compute_kernel(self.coarsegrid.shape),
            # TODO use coarsegrid.dualgrid here, move fftshift down (and use
            # coarsegrid.fft there)
            self.dualcoords = np.ix_(
                *(ifftshift(np.arange(-(c//2), (c+1)//2)) for c in coarseshape))
            self.coarseiterations = coarseiterations
        else:
            self.coarse = False

        self.gmres_args = util.set_defaults(gmres_args, restart=10, tol=1e-14, maxiter=100)

        super().__init__(
            domain=grid,
            codomain=spaces.UniformGrid(
                axisdata=(meas_directions, inc_directions),
                dtype=float if amplitude else complex))

        # all attributes defined above are constants
        self._consts.update(self.attrs)

        # pre-allocate to save time in _eval
        self._totalfield = np.empty((np.sum(self.support), self.codomain.shape[1]),
                                    dtype=complex)
        self._lippmann_schwinger = spla.LinearOperator(
            (self.domain.csize,) * 2,
            matvec=self._lippmann_schwinger_op,
            rmatvec=self._lippmann_schwinger_adjoint,
            dtype=complex)
        if self.coarse:
            self._lippmann_schwinger_coarse = spla.LinearOperator(
                (self.coarsegrid.csize,) * 2,
                matvec=self._lippmann_schwinger_coarse_op,
                rmatvec=self._lippmann_schwinger_coarse_adjoint,
                dtype=complex)

    def _eval(self, contrast, differentiate=False):
        contrast[~self.support] = 0
        if self.absorptive and not self.refractive:
            contrast = 1j * contrast
        self._contrast = contrast
        if self.coarse:
            # TODO take real part? what about even case? for 1d, highest
            # fourier coeff must be real then, which is not guaranteed by
            # subsampling here.
            aux = fftn(self._contrast)[self.dualcoords]
            self._coarse_contrast = (
                (self.coarsegrid.size / self.domain.size) *
                ifftn(aux))
        farfield = self.codomain.empty(dtype=complex)
        rhs = self.domain.zeros(dtype=complex)
        # TODO parallelize
        for j in range(self.codomain.shape[1]):
            # Solve Lippmann-Schwinger equation v + a(k*v) = a*u_inc for the
            # unknown v = a u_total. The Fourier coefficients of the periodic
            # convolution kernel k are precomputed.
            rhs[self.support] = self.inc_matrix[j, :] * contrast[self.support]
            if self.coarse:
                v = self._solve_two_grid(rhs)
            else:
                v = (self
                     ._gmres(self._lippmann_schwinger, rhs)
                     .reshape(self.domain.shape))
            farfield[:, j] = self.farfield_matrix @ v[self.support]
            # The total field can be recovered from v in a stable manner by the formula
            # u_total = ui - k*v
            if differentiate:
                self._totalfield[:, j] = (self.inc_matrix[j, :] -
                                          ifftn(self.kernel * fftn(v))[self.support])
        if self.amplitude:
            self._farfield = farfield
            return np.abs(farfield)**2
        else:
            return farfield

    def _derivative(self, contrast):
        contrast = contrast[self.support]
        if self.absorptive and not self.refractive:
            contrast = 1j * contrast
        farfield = self.codomain.empty(dtype=complex)
        rhs = self.domain.zeros(dtype=complex)
        for j in range(self.codomain.shape[1]):
            rhs[self.support] = self._totalfield[:, j] * contrast
            if self.coarse:
                v = self._solve_two_grid(rhs)
            else:
                v, info = (self
                           ._gmres(self._lippmann_schwinger, rhs)
                           .reshape(self.domain.shape))
            farfield[:, j] = self.farfield_matrix @ v[self.support]
        if self.amplitude:
            return 2 * np.real(self._farfield.conj() * farfield)
        else:
            return farfield

    def _adjoint(self, farfield):
        if self.amplitude:
            farfield = 2 * self._farfield * farfield
        v = self.domain.zeros(dtype=complex)
        farfield_matrix_H = self.farfield_matrix.conj().T
        contrast = self.domain.zeros()
        for j in range(self.codomain.shape[1]):
            v[self.support] = farfield_matrix_H @ farfield[:, j]
            if self.coarse:
                rhs = self._solve_two_grid_adjoint(v)
            else:
                rhs = (self
                       ._gmres(self._lippmann_schwinger.adjoint(), v)
                       .reshape(self.domain.shape))
            aux = self._totalfield[:, j].conj() * rhs[self.support]
            if self.refractive:
                if self.absorptive:
                    contrast[self.support] += aux
                else:
                    contrast[self.support] += np.real(aux)
            else:
                contrast[self.support] += np.imag(aux)
        return contrast

    def _solve_two_grid(self, rhs):
        rhs = fftn(rhs)
        v = self.domain.zeros(dtype=complex)
        rhs_coarse = rhs[self.dualcoords]
        for remaining_iters in range(self.coarseiterations, 0, -1):
            v_coarse = (self
                        ._gmres(self._lippmann_schwinger_coarse, rhs_coarse)
                        .reshape(self.coarsegrid.shape))
            v[self.dualcoords] = v_coarse
            if remaining_iters > 0:
                rhs_coarse = fftn(self._coarse_contrast * ifftn(
                    self.coarsekernel * v_coarse))
                v = rhs - fftn(self._contrast * ifftn(self.kernel * v))
                rhs_coarse += v[self.dualcoords]
        return ifftn(v)

    def _solve_two_grid_adjoint(self, v):
        v = fftn(v)
        rhs = self.domain.zeros(dtype=complex)
        v_coarse = v[self.dualcoords]
        for remaining_iters in range(self.coarseiterations, 0, -1):
            rhs_coarse = (self
                          ._gmres(self._lippmann_schwinger_coarse.adjoint(), v_coarse)
                          .reshape(self.coarsegrid.shape))
            rhs[self.dualcoords] = rhs_coarse
            if remaining_iters > 0:
                v_coarse = self.coarsekernel * fftn(
                    self._coarse_contrast * ifftn(rhs_coarse))
                rhs = v - self.kernel * fftn(self._contrast * ifftn(rhs))
                v_coarse += rhs[self.dualcoords]
        return ifftn(rhs)

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, rhs.ravel(), **self.gmres_args)
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        return result

    def _lippmann_schwinger_op(self, v):
        """Lippmann-Schwinger operator in spatial domain on fine grid
        """
        v = v.reshape(self.domain.shape)
        v = v + self._contrast * ifftn(self.kernel * fftn(v))
        return v.ravel()

    def _lippmann_schwinger_adjoint(self, v):
        """Adjoint Lippmann-Schwinger operator in spatial domain on fine grid
        """
        v = v.reshape(self.domain.shape)
        v = v + ifftn(np.conj(self.kernel) * fftn(np.conj(self._contrast) * v))
        return v.ravel()

    def _lippmann_schwinger_coarse_op(self, v):
        """Lippmann-Schwinger operator in frequency domain on coarse grid
        """
        v = v.reshape(self.coarsegrid.shape)
        v = v + fftn(self._coarse_contrast * ifftn(self.coarsekernel * v))
        return v.ravel()

    def _lippmann_schwinger_coarse_adjoint(self, v):
        """Lippmann-Schwinger operator in frequency domain on coarse grid
        """
        v = v.reshape(self.coarsegrid.shape)
        v = v + np.conj(self.coarsekernel) * fftn(np.conj(self._coarse_contrast) * ifftn(v))
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
    return 2 * R * fftshift(K_hat)


def compute_kernel_3D(R, shape):
    J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
    piabsJ = np.pi * np.linalg.norm(J, axis=0)
    Jzero = tuple(s//2 for s in shape)

    K_hat = (2*R)**(-3/2) * R**2 / (piabsJ**2 - R**2) * (
        1 - np.exp(1j*R) * (np.cos(piabsJ) - 1j*R * np.sin(piabsJ) / piabsJ))
    K_hat[Jzero] = -(2*R)**(-1.5) * (1 - np.exp(1j*R) * (1 - 1j*R))
    K_hat[piabsJ == R] = -1j/4 * (2*R)**(-1/2) * (1 - np.exp(1j*R) * np.sin(R) / R)
    return 2 * R * fftshift(K_hat)
