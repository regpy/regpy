import numpy as np
import pynfft

from regpy.operators import Operator
from regpy.discrs import Discretization, UniformGrid
from regpy.util import memoized_property


class NFFT(Operator):
    def __init__(self, grid, nodes, weights):
        assert isinstance(grid, UniformGrid)
        assert nodes.shape[1] == grid.ndim

        # pynfft computes a sum of the form
        #      sum_k f_k exp(-2 pi i k x)
        # where k ranges from -n/2 to n/2-1. Our frequencies are actually defined
        # `grid`, so we would like
        #      sum_k f_k exp(-i k x)
        # where k ranges over `grid`. This is equivalent to rescaling the nodes x
        # by the following factor (also handling the multidimensional case):
        scaling_factor = 1 / (2 * np.pi) * grid.extents / np.asarray(grid.shape)
        nodes = scaling_factor * nodes
        # The nodes' inversion weights need to be scaled accordingly.
        self.weights = np.prod(scaling_factor) * weights

        super().__init__(
            domain=grid,
            codomain=Discretization(nodes.shape[0], dtype=complex),
            linear=True
        )

        # Initialize the NFFT
        self.plan = pynfft.NFFT(N=grid.shape, M=nodes.shape[0])
        self.plan.x = nodes
        self.plan.precompute()

        # NFFT quadrature factor
        self.nfft_factor = grid.volume_elem / (2 * np.pi)**grid.ndim

        # Initialize the Solver for computing the inverse NFFT
        # TODO unused?
        self.solver = pynfft.Solver(self.plan)
        self.solver.w = self.weights

    def _eval(self, x):
        self.plan.f_hat = x
        return self.nfft_factor * self.plan.trafo()

    def _adjoint(self, y):
        self.plan.f = y
        return self.nfft_factor * self.plan.adjoint()

    @memoized_property
    def inverse(self):
        # TODO add solver-based inverse
        return ApproxInverseNFFT(self)


class ApproxInverseNFFT(Operator):
    def __init__(self, op):
        self.op = op
        super().__init__(
            domain=op.codomain,
            codomain=op.domain,
            linear=True
        )

    def _eval(self, y):
        self.op.plan.f = self.op.weights * y
        return self.op.plan.adjoint() / self.op.nfft_factor

    def _adjoint(self, x):
        self.op.plan.f_hat = x
        return np.conj(self.op.weights) * self.op.plan.trafo() / self.op.nfft_factor
