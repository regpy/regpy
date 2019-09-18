"""Ewald sphere geometry framework for NFFT
"""

from copy import copy
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pynfft.nfft import NFFT as PYNFFT
from pynfft.solver import Solver
from scipy.spatial import ConvexHull

from . import bounded_voronoi


class Rep(Enum):
    """Enumeration specifying representation of data"""

    # Data defined on the whole Ewald sphere
    EwaldSphere = auto()

    # Data defined on pairs (indident_direction, measurement_direction)
    PairsOfDirections = auto()

    # Data defined in the coordinate (domain) space
    CoordinateDomain = auto()


class NFFT:
    def __init__(self, inc_directions, meas_directions, grid, wave_number):
        # Computing nodes of the Ewald sphere scaled to [-0.5,0.5)
        all_nodes = np.array([wave_number * (x - y) for x, Y in zip(inc_directions, meas_directions) for y in Y])
        # TODO just use the first return value, ignore indices?
        _, self.node_indices = np.unique(all_nodes.round(decimals=6), axis=0, return_index=True)
        self.nodes = all_nodes[self.node_indices]
        self.ewald_node_count = self.nodes.shape[0]

        # Computing the uniform grid surrounding the Ewald sphere
        # TODO what for?
        # x = np.arange(p['GRID_SHAPE'][0]) / p['GRID_SHAPE'][0] - 0.5
        # y = np.arange(p['GRID_SHAPE'][1]) / p['GRID_SHAPE'][1] - 0.5
        # X, Y = np.meshgrid(x, y)
        x, y = np.meshgrid(*(np.linspace(-2*wave_number, 2*wave_number, n, endpoint=False) for n in grid.shape))
        outer_ind = np.sqrt(x ** 2 + y ** 2) > 2*wave_number
        outer_nodes = np.stack([x[outer_ind], y[outer_ind]], axis=1)
        self.nodes = np.concatenate([self.nodes, outer_nodes])

        self.wave_number = wave_number

        # Computing the weights of nodes to compute Riemann sums over the Ewald sphere
        # Compute the bounded by [-0.5,0.5)^2 Voronoi diagram of the Ewald sphere
        regions, vertices = bounded_voronoi(self.nodes, left=-2*wave_number, down=-2*wave_number, right=2*wave_number, up=2*wave_number)

        # Physical Ewald sphere has radius 2sqrt(E)
        # Scaling the Ewald sphere to radius 2*wave_numver*support_radius / (pi*shape[0])
        # corresponds to scaling the x-domain to [-.5,.5)
        # self.scaling_factor = 8 * p['WAVE_NUMBER'] * p['SUPPORT_RADIUS'] / (np.pi * p['GRID_SHAPE'][0])
        scaling_factor = 1 / (2*np.pi) * grid.extents[0] / grid.shape[0]

        # Computing areas of cells of the Voronoi diagram
        self.weights = scaling_factor**2 * np.array(
            [ConvexHull([vertices[i] for i in reg]).volume for reg in regions]
        )

        # Saving the patches of the Voronoi diagram
        self.patches = PatchCollection(
            [Polygon([vertices[i] for i in reg]) for reg in regions],
            edgecolors=None
        )

        # Initialize the NFFT
        self.plan = PYNFFT(N=grid.shape, M=self.nodes.shape[0])

        # NFFT Precomputations
        self.plan.x = scaling_factor * self.nodes
        self.plan.precompute()

        # NFFT scaling factor
        # self.nfft_factor = (4 * p['SUPPORT_RADIUS'] / (2 * np.pi)) ** 2 / np.prod(p['GRID_SHAPE'])
        self.nfft_factor = grid.volume_elem / (2 * np.pi) ** grid.ndim

        # Initialize the Solver for computing the inverse NFFT
        # TODO unused?
        self.solver = Solver(self.plan)
        self.solver.w = self.weights

    def forward(self, f_hat):
        """Computes the forward NFFT

        Parameters
        ----------
        f_hat : ndarray
            function on the rectangular grid

        Returns
        -------
        function on the Ewald sphere
        """
        self.plan.f_hat = f_hat
        return self.plan.trafo()

    def adjoint(self, f):
        """ Computes the adjoint NFFT

        Returns
        -------
        function on the grid
        """

        self.plan.f = f
        f_hat = self.plan.adjoint()
        return f_hat

    def inverse(self, f):
        """Computes the inverse NFFT

        Parameters
        ----------
        f : function on the Ewald sphere
        """
        return self.adjoint(self.weights * f)

    def convert(self, x, from_rep, to_rep):
        """Changes the representation of data between different formats

        Parameters
        ----------
        x : ndarray
            data
        from_rep : Rep
            initial data representation
        to_rep : Rep
           target data representation
        """

        assert isinstance(from_rep, Rep) and isinstance(to_rep, Rep)

        if from_rep == Rep.PairsOfDirections and to_rep == Rep.EwaldSphere:
            # TODO why not just `return x.flatten('F')[self.node_indices]`? Maybe since solver must have data
            # for all nodes, while keeping the outer nodes at 0 all the time, since it's subtracted from x_hat,
            # which is defined on all nodes. Would be simpler in-place, though. Also, it's multiplied by indicator
            # right afterwards, setting outer nodes to 0 anyway.
            y = np.zeros(self.nodes.shape[0], dtype=complex)
            y[:self.ewald_node_count] = x.flatten('F')[self.node_indices]
            return y

        elif from_rep == Rep.CoordinateDomain and to_rep == Rep.EwaldSphere:
            # TODO is the submanifold indicator here just the node_indices?
            return self.nfft_factor * np.conj(self.forward(np.conj(x))) * self.submanifold_indicator(0.5)

        elif from_rep == Rep.EwaldSphere and to_rep == Rep.CoordinateDomain:
            return np.conj(self.inverse(np.conj(x))) / self.nfft_factor

        else:
            raise NotImplementedError('Can not convert from {} to {}'.format(from_rep, to_rep))

    def submanifold_indicator(self, radius):
        return np.linalg.norm(self.nodes, axis=1) <= 4 * radius * self.wave_number

    def display(self, f, ax=None):
        """Display a function on the Ewald sphere"""
        # TODO move to solver
        if ax is None:
            ax = plt.gca()
        self.patches.set_array(np.real(f))
        ax.add_collection(copy(self.patches))
        ax.set_xlim(-2 * self.wave_number, 2 * self.wave_number)
        ax.set_ylim(-2 * self.wave_number, 2 * self.wave_number)
        # Return mappable for caller to be able to setup colorbar
        return self.patches

    def norm(self, f):
        return np.sqrt(np.sum(np.abs(f ** 2) * self.weights))

    def zeros(self, **kwargs):
        return np.zeros(self.nodes.shape[0], **kwargs)
