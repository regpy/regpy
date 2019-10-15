from copy import copy

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.pyplot import gca
from scipy.spatial.qhull import ConvexHull

from regpy.solvers import Solver
from regpy.operators.mediumscattering import MediumScatteringOneToMany
from regpy.operators.nfft import NFFT
from regpy.util import bounded_voronoi


class IterativeBorn(Solver):
    """Solver based on Born approximation with non-zero background potential

    Literature
    ----------
    [1] R.G. Novikov, Sbornik Mathematics 206, 120-34, 2015
    [2] <Add article where [1] is implemented>
    """

    def __init__(self, op, data, cutoffs):
        # TODO make useable for other operators
        assert isinstance(op, MediumScatteringOneToMany)
        super().__init__()
        self.op = op

        # Compute properly scaled scattering vectors on the Ewald sphere
        scattering_vecs = np.array([
            op.wave_number * (x - y)
            for x, Y in zip(op.inc_directions, op.farfield_directions)
            for y in Y
        ])

        scattering_vecs, self.node_indices = np.unique(scattering_vecs.round(decimals=6), axis=0, return_index=True)

        # Compute uniform grid surrounding the Ewald sphere
        self.bound = 2 * op.wave_number
        # TODO is using the spatial grid shape here relevant?
        x, y = np.meshgrid(*(np.linspace(-self.bound, self.bound, n, endpoint=False) for n in op.domain.shape))
        outer_ind = np.sqrt(x**2 + y**2) > self.bound
        outer_nodes = np.stack([x[outer_ind], y[outer_ind]], axis=1)

        # Compute nodes, Voronoi diagram and approximate weights
        nodes = np.concatenate([scattering_vecs, outer_nodes])
        self.outer = np.concatenate([
            np.zeros_like(scattering_vecs, dtype=bool),
            np.ones_like(outer_nodes, dtype=bool)
        ])
        regions, vertices = bounded_voronoi(
            nodes, left=-self.bound, down=-self.bound,
            right=self.bound, up=self.bound
        )

        self.weights = np.array(
            [ConvexHull([vertices[i] for i in reg]).volume for reg in regions]
        )

        # For submanifold_indicator
        self.node_dists = np.linalg.norm(nodes, axis=1) / (2 * self.bound)

        # Save the patches of the Voronoi diagram for plotting
        self.patches = PatchCollection(
            [Polygon([vertices[i] for i in reg]) for reg in regions],
            edgecolors=None
        )

        self.NFFT = NFFT(op.domain, nodes, self.weights)

        self.rhs = self._pad_data(data)

        try:
            self.cutoffs = iter(cutoffs)
        except TypeError:
            self.cutoffs = iter([cutoffs])
        try:
            self.cutoff = next(self.cutoffs)
        except StopIteration:
            raise ValueError('no cutoffs given') from None

        self.subind = self._submanifold_indicator(self.cutoff)
        self.x_hat = self.rhs * self.subind
        self._update_xy()

    def _update_xy(self):
        self.x = self._inverse(self.x_hat)
        self.x[~self.op.support] = 0
        self.y = self._pad_data(self.op(self.x))

    def _forward(self, x):
        y = np.conj(self.NFFT(np.conj(x)))
        y[self.outer] = 0
        return y

    def _inverse(self, x_hat):
        return np.conj(self.NFFT.inverse(np.conj(x_hat)))

    def _pad_data(self, x):
        y = self.NFFT.codomain.zeros()
        y[:len(self.node_indices)] = x.ravel('F')[self.node_indices]
        return y

    def _next(self):
        try:
            self.cutoff = next(self.cutoffs)
            self.subind = self._submanifold_indicator(self.cutoff)
            # TODO assert cutoff <= 0.5?
        except StopIteration:
            # If no next cutoff is available, just keep the final one
            pass
        self.x_hat = (self.x_hat - self.y + self.rhs) * self.subind
        self._update_xy()

    def _submanifold_indicator(self, radius):
        return self.node_dists <= radius

    def display(self, f, ax=None):
        """Display a function on the Ewald sphere"""
        if ax is None:
            ax = gca()
        self.patches.set_array(np.real(f))
        ax.add_collection(copy(self.patches))
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)
        # Return mappable for caller to be able to setup colorbar
        return self.patches

    def derivative(self, dx, cutoff=0.5):
        """Derivative of the map F: x[j] -> x[j+1]"""
        subind = self._submanifold_indicator(cutoff)
        dx = dx.copy()
        dx[~self.op.support] = 0
        dx_hat = self._forward(dx)
        _, deriv = self.op.linearize(self.x)
        dy = self._pad_data(deriv(dx))
        dw_hat = (dx_hat - dy) * subind
        dw = self._inverse(dw_hat)
        dw[~self.op.support] = 0
        return dw

    def datanorm(self, f):
        return np.sqrt(np.sum(np.abs(f**2) * self.weights))
