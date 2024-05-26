import numpy as np
from regpy.operators.parallel_operators import DistributedVectorOfOperators
from regpy.operators import CoordinateProjection, DirectSum, FourierTransform, PtwMultiplication, Operator
from regpy import util, vecsps

class CoilMult(Operator):
    """Operator that implements the multiplication between density and coil profiles. The domain
    is a direct sum of the `grid` (for the densitiy) and a `regpy.vecsps.UniformGridFcts` of `ncoils`
    copies of `grid`, stacked along the 0th dimension.

    Parameters
    ----------
    grid : regpy.vecsps.UniformGridFcts
        The grid on which the density is defined.
    ncoils : int
        The number of coils.
    """

    def __init__(self, grid, ncoils):
        assert isinstance(grid, vecsps.UniformGridFcts)
        assert grid.ndim == 2
        self.grid = grid
        """The density grid."""
        if(ncoils>1):
            self.coilgrid = vecsps.UniformGridFcts(ncoils, *grid.axes, dtype=grid.dtype)
        else:
            self.coilgrid=vecsps.UniformGridFcts(*grid.axes, dtype=grid.dtype)
        """The coil grid, a stack of copies of `grid`."""
        self.ncoils = ncoils
        """The number of coils."""
        super().__init__(
            domain=self.grid + self.coilgrid,
            codomain=self.coilgrid
        )

    def _eval(self, x, differentiate=False, adjoint_derivative=False):
        density, coils = self.domain.split(x)
        if differentiate or adjoint_derivative:
            r"""We need to copy here since `.split()` returns views into `x` if possible."""            
            self._density = density.copy()
            self._coils = coils.copy()
        return density * coils

    def _derivative(self, x):
        density, coils = self.domain.split(x)
        return density * self._coils + self._density * coils

    def _adjoint(self, y):
        density = self._density
        coils = self._coils
        if self.grid.is_complex:
            r"""Only `conj()` in complex case. For real case, we can avoid the copy."""
            density = np.conj(density)
            coils = np.conj(coils)
        if(self.ncoils>1):
            return self.domain.join(
                np.sum(coils * y, axis=0),
                density * y)
        return self.domain.join(coils*y,density*y)

    def __repr__(self):
        return util.make_repr(self, self.grid, self.ncoils)
    
class DomainConversion(Operator):
    r"""
    Converts a domain comprised of two uniform grids such that the second grid is split along its 0-th axis into a direct sum
    of grids. This is used to use the same smoothing operator for parallel and non-paralell implementation.

    Parameters
    ----------
    codomain : regpy.vecsps.DirectSum
        The grid used as the domain of the parallelized mri operator.
    """

    def __init__(self,codomain):
        self.ncoils=len(codomain)-1
        if(self.ncoils>1):
            coilgrid = vecsps.UniformGridFcts(self.ncoils, *codomain[0].axes, dtype=codomain[0].dtype)
        else:
            coilgrid=vecsps.UniformGridFcts(*codomain[0].axes, dtype=codomain[0].dtype)
        super().__init__(codomain[0]+coilgrid,codomain,True)


    def _eval(self,x):
        density,coils=self.domain.split(x)
        return self.codomain.join(density,*[coils[n,:,:] for n in range(self.ncoils)])
    
    def _adjoint(self, y):
        y_elms=self.codomain.split(y)
        coils=self.domain[1].zeros()
        for i in range(self.ncoils):
            coils[i,:,:]=y_elms[i+1]
        return self.domain.join(y_elms[0],coils)


def cartesian_sampling(domain, mask):
    """Constructs a cartesian sampling operator. This simple uses all arguments to construct
    a `regpy.operators.CoordinateProjection` and returns it.
    """
    return CoordinateProjection(domain, mask)


def parallel_mri(grid, ncoils, centered=False):
    """Construct a parallel MRI operator by composing a `regpy.operators.FourierTransform` and a
    `CoilMult`. Subsampling patterns need to added by composing with e.g. a `cartesian_sampling`.

    Parameters
    ----------
    grid : vecsps.UniformGridFcts
        The grid on which the density is defined.
    ncoils : int
        The number of coils.
    centered : bool
        Whether to use a centered FFT. If true, the operator will use fftshift.

    Returns
    -------
    Operator
    """
    cmult = CoilMult(grid, ncoils)
    ft = FourierTransform(cmult.codomain, axes=range(1, cmult.codomain.ndim), centered=centered)
    return ft * cmult


def full_parallel_mri_parallelized(grid,ncoils,mask,centered=False):
    """Construct a parallelized parallel MRI operator by composing a `regpy.operators.FourierTransform` and a
    `CoilMult` on each component. Subsampling patterns are added by composing with a `cartesian_sampling`.

    Parameters
    ----------
    grid : vecsps.UniformGridFcts
        The grid on which the density is defined.
    ncoils : int
        The number of coils.
    centered : bool
        Whether to use a centered FFT. If true, the operator will use fftshift.

    Returns
    -------
    Operator
    """
    ops=[]
    distribution_mat=np.zeros((ncoils,ncoils+1),dtype=bool)
    distribution_mat[:,0]=True
    cmult = CoilMult(grid, 1)
    ft = FourierTransform(cmult.codomain, axes=range(cmult.codomain.ndim), centered=centered)
    sampling=cartesian_sampling(ft.codomain, mask=mask)
    domain=grid
    for i in range(ncoils):
        ops.append(sampling*ft*cmult)
        distribution_mat[i,i+1]=True
        domain+=cmult.domain[1]
    return DistributedVectorOfOperators(ops,domain,distribution_mat)


def sobolev_smoother(codomain, sobolev_index, factor=None, centered=False):
    """Partial reimplementation of the Sobolev Gram matrix. Can be composed with forward operator
    (from the right) to substitute

        coils = ifft(aux / sqrt(sobolev_weights)),

    making `aux` the new unknown. This can be used to avoid the numerically unstable Gram matrix
    for high Sobolev indices.

    Parameters
    ----------
    codomain :
        Codomain of the operator
    sobolev_index : int
    centered : bool
        Whether to use a centered FFT. If true, the operator will use fftshift.
    factor : float
        If factor is None (default): Implicit scaling based on the codomain. Otherwise,
        the coordinates are normalized and this factor is applied.
    """
    # TODO Combine with Sobolev space implementation as much as possible
    grid, coilsgrid = codomain
    ft = FourierTransform(coilsgrid, axes=(1, 2), centered=centered)
    if factor is None:
       mulfactor = grid.volume_elem * (
                    1 + np.linalg.norm(ft.codomain.coords[1:], axis=0)**2
                                      )**(-sobolev_index / 2)
    else:
        mulfactor = ( 1 + factor * np.linalg.norm(ft.codomain.coords[1:]/2./np.amax(np.abs(ft.codomain.coords[1:])), axis=0)**2
                                                 )**(-sobolev_index / 2)

    mul = PtwMultiplication(ft.codomain, mulfactor)
    return DirectSum(grid.identity, ft.inverse * mul, codomain=codomain)


def estimate_sampling_pattern(data):
    """Estimate the sampling pattern from measured data. If some measurement point is zero in all
    coil profiles it is assumed to be outside of the sampling pattern. This method has a very low
    probability of failing, especially non-integer data.

    Parameters
    ----------
    data : array-like
        The measured data, with coils stacked along dimension 0.

    Returns
    -------
    boolean array
        The subsampling mask.
    """
    return np.all(data != 0, axis=0)

def normalize(density, coils):
    """Normalize density and coils to handle the inherent non-injectivity of the `CoilMult` operator.
    """
    scaling_factor = np.linalg.norm(coils, axis=0)
    return density * scaling_factor, coils /scaling_factor


