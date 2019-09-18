from . import Solver
from itreg.util.nfft_ewald import NFFT, Rep
from ..operators import CoordinateProjection
from ..operators.mediumscattering import MediumScatteringOneToMany


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
        self.projection = CoordinateProjection(
            op.domain,
            op.support
        )
        self.NFFT = NFFT(op.inc_directions, op.farfield_directions, self.projection, op.domain, op.wave_number)
        self.rhs = self.NFFT.convert(data, Rep.PairsOfDirections, Rep.EwaldSphere)
        try:
            self.cutoffs = iter(cutoffs)
        except TypeError:
            self.cutoffs = iter([cutoffs])
        self.cutoff = next(self.cutoffs)

    def _next(self):
        subind = self.NFFT.submanifold_indicator(self.cutoff)
        try:
            self.cutoff = next(self.cutoffs)
        except StopIteration:
            # If no next cutoff is available, just keep the final one
            pass
        try:
            self.x_hat = (self.x_hat - self.y + self.rhs) * subind
        except AttributeError:
            # First iteration: compute the Born approximation to the solution
            # TODO: move to constructor
            self.x_hat = self.rhs * subind
        self.x = self.projection.adjoint(self.NFFT.convert(self.x_hat, Rep.EwaldSphere, Rep.CoordinateDomain))
        self.y = self.NFFT.convert(self.op(self.x), Rep.PairsOfDirections, Rep.EwaldSphere)

    def display(self, f):
        return self.NFFT.display(f)

    def derivative(self, dx, cutoff=0.5):
        """Derivative of the map F: x[j] -> x[j+1]"""
        subind = self.NFFT.submanifold_indicator(cutoff)
        dx_hat = self.NFFT.convert(self.projection(dx), Rep.CoordinateDomain, Rep.EwaldSphere)
        _, deriv = self.op.linearize(self.x)
        dy = self.NFFT.convert(deriv(dx), Rep.PairsOfDirections, Rep.EwaldSphere)
        dw_hat = (dx_hat - dy) * subind
        dw = self.projection.adjoint(self.NFFT.convert(dw_hat, Rep.EwaldSphere, Rep.CoordinateDomain))
        return dw
