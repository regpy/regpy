import numpy as np
import pynfft


class NFFT:
    def __init__(self, nodes, grid, weights):
        # TODO explain
        scaling_factor = 1 / (2 * np.pi) * grid.extents[0] / grid.shape[0]
        nodes = scaling_factor * nodes
        self.weights = scaling_factor**2 * weights

        # Initialize the NFFT
        self.plan = pynfft.NFFT(N=grid.shape, M=nodes.shape[0])
        self.plan.x = nodes
        self.plan.precompute()

        # NFFT scaling factor
        self.nfft_factor = grid.volume_elem / (2 * np.pi)**grid.ndim

        # Initialize the Solver for computing the inverse NFFT
        # TODO unused?
        self.solver = pynfft.Solver(self.plan)
        self.solver.w = self.weights

    def __call__(self, f_hat):
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
        return self.nfft_factor * self.plan.trafo()

    def adjoint(self, f):
        """ Computes the adjoint NFFT

        Returns
        -------
        function on the grid
        """
        self.plan.f = f
        return self.nfft_factor * self.plan.adjoint()

    def inverse(self, f):
        """Computes the inverse NFFT

        Parameters
        ----------
        f : function on the Ewald sphere
        """
        self.plan.f = self.weights * f
        return self.plan.adjoint() / self.nfft_factor

    def norm(self, f):
        return np.sqrt(np.sum(np.abs(f**2) * self.weights))
