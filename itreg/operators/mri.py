import numpy as np

from . import CoordinateProjection, DirectSum, FourierTransform, Multiplication, Operator
from .. import util
from ..spaces import discrs


class CoilMult(Operator):
    """
    Operator that implements the multiplication between density and coil profiles.
    """

    def __init__(self, grid, ncoils):
        # TODO: are density and/or coil profiles complex?
        assert isinstance(grid, discrs.UniformGrid)
        assert grid.ndim == 2
        self.grid = grid
        self.coilgrid = discrs.UniformGrid(ncoils, *grid.axes, dtype=grid.dtype)
        self.ncoils = ncoils
        super().__init__(
            domain=self.grid + self.coilgrid,
            codomain=self.coilgrid
        )

    def _eval(self, x, differentiate=False):
        density, coils = self.domain.split(x)
        if differentiate:
            # We need to copy here since .split() returns views into x if possible.
            self._density = density.copy()
            self._coils = coils.copy()
        return density * coils

    def _derivative(self, x):
        density, coils = self.domain.split(x)
        return density * self._coils + self._density * coils

    def _adjoint(self, y):
        density = self._density
        coils = self._coils
        if self.domain.is_complex:
            # Only conj() in complex case. For real case, we can avoid the copy.
            density = np.conj(density)
            coils = np.conj(coils)
        return self.domain.join(
            np.sum(coils * y, axis=0),
            density * y
        )

    def __repr__(self):
        return util.make_repr(self, self.grid, self.ncoils)


def cartesian_sampling(domain, mask):
    return CoordinateProjection(domain, mask)


def parallel_mri(grid, ncoils, centered=False):
    """
    Construct a parallel MRI operator.

    Parameters
    ----------
    grid : discrs.UniformGrid
        The grid on which the density is defined
    ncoils : int
        The number of coils
    centered : bool
        Whether to use a centered FFT. If true, the operator will use fftshift.

    Returns
    -------
    Operator
    """
    cmult = CoilMult(grid, ncoils)
    ft = FourierTransform(cmult.codomain, axes=range(1, cmult.codomain.ndim), centered=centered)
    return ft * cmult


def sobolev_smoother(codomain, sobolev_index):
    """
    Partial reimplementation of the Sobolev gram matrix. Can be composed with forward operator (from the right) to
    substitute

        coils = ifft(aux / sqrt(sobolev_weights)),

    making `aux` the new unknown. This can be used to avoid the numerically unstable gram matrix for high Sobolev
    indices.
    """
    # TODO Combine with Sobolev space implementation as much as possible
    grid, coilsgrid = codomain
    ft = FourierTransform(coilsgrid, axes=(1, 2))
    mul = Multiplication(
        ft.codomain,
        grid.volume_elem * (
            1 + np.linalg.norm(ft.codomain.coords[1:], axis=0)**2
        )**(-sobolev_index / 2)
    )
    return DirectSum(grid.identity, ft.inverse * mul, codomain=codomain)


def estimate_sampling_pattern(data):
    return np.all(data == 0, axis=0)


def normalize(density, coils):
    """
    Normalize density and coils to handle the inherent non-injectivity of the CoilMult operator.
    """
    return density * np.linalg.norm(coils, axis=0)
