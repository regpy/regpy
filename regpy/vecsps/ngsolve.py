r"""Finite element vector spaces using NGSolve

This module implements a `regpy.vecsps.VectorSpace` instance for NGSolve spaces. This gives the basic
interface to use FES spaces defined in `ngsolve` to be used as `VectorSpaces`in `regpy`.  
Operators are using such spaces are implemented in the `regpy.operators.ngsolve` module. Hilbert spaces 
and Functionals defined on such spaces can be found in `regpy.hilbert.ngsolve` and `regpy.functionals.ngsolve`
respectively. 
"""

import ngsolve as ngs
import numpy as np

from regpy.vecsps import VectorSpace, DirectSum
from regpy.util import is_complex_dtype


class NgsSpace(VectorSpace):
    r"""A vector space wrapping an `ngsolve.FESpace`.

    Parameters
    ----------
    fes : ngsolve.FESpace
       The wrapped NGSolve vector space.
    bdr : 
        Boundary of the NGSolve vector space.
    """
    def __init__(self, fes, bdr=None):
        assert isinstance(fes, ngs.FESpace)
        super().__init__(fes.ndof)
        self.fes = fes
        self.bdr = bdr
        # Checks if FES is Vector valued and stores the dimension in self.codim
        from netgen.libngpy._meshing import NgException
        try:
            self.codim = len(fes.components)
            # assert self.codim == fes.mesh.dim
            self._fes_util = ngs.L2(self.fes.mesh, order=0, complex = self.is_complex)**self.codim
        except NgException:
            self.codim = 1
            self._fes_util = ngs.L2(self.fes.mesh, order=0, complex = self.is_complex)
        self._gfu_util = ngs.GridFunction(self._fes_util)
        self._gfu_fes = ngs.GridFunction(fes)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.fes == other.fes
        else:
            return NotImplemented

    def ones(self):
        if self.codim == 1:
            self._gfu_fes.Set(1)
        else:
            self._gfu_fes.Set(tuple(1 for _ in range(self.codim)))
        return self._gfu_fes.vec.FV().NumPy().copy()

    def rand(self, rand=np.random.random_sample):
        r = rand(self._fes_util.ndof)
        if self.is_complex and not is_complex_dtype(r.dtype):
            c = np.empty(self._fes_util.ndof, dtype=complex)
            c.real = r
            c.imag = rand(self._fes_util.ndof)
            self._gfu_util.vec.FV().NumPy()[:] = c            
        else:
            self._gfu_util.vec.FV().NumPy()[:] = r
        self._gfu_fes.Set(self._gfu_util)
        return self._gfu_fes.vec.FV().NumPy().copy()
    
    def randn(self):
        return self.rand(np.random.standard_normal)

    def to_ngs(self, array):
        gf = ngs.GridFunction(self.fes)
        gf.vec.FV().NumPy()[:] = array
        return gf

    def from_ngs(self, ngs_elem):
        if isinstance(ngs_elem,ngs.comp.GridFunction):
            return ngs_elem.vec.FV().NumPy().copy()
        else:
            self._gfu_fes.Set(ngs_elem)
            return self._gfu_fes.vec.FV().NumPy().copy()
    
    def draw(self, coefficient_array, name):
        assert isinstance(name, str)
        gfu_fes = ngs.GridFunction(self.fes)
        gfu_fes.vec.FV().NumPy()[:] = coefficient_array
        coefficientfunction = ngs.CoefficientFunction( gfu_fes )
        ngs.Draw(coefficientfunction, self.fes.mesh, name)

    def is_on_boundary(self,array):
        if self.bdr is None:
            return False
        gf = self.to_ngs(array)
        ngs.Projector(self.fes.FreeDofs(), range=True).Project(gf.vec)
        return np.all(gf.vec.FV().NumPy() == 0)

    def __add__(self, other):
        if isinstance(other, VectorSpace):
            product_space = DirectSum(self, other, flatten=True)
            if all(product_space.summands[i]==product_space.summands[0] for i in range(len(product_space.summands))):
                product_space.fes = product_space.summands[0].fes
                product_space.bdr = product_space.summands[0].bdr
            return product_space
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, VectorSpace):
            product_space = DirectSum(other, self, flatten=True)
            if all(product_space.summands[i]==product_space.summands[0] for i in range(len(product_space.summands))):
                product_space.fes = product_space.summands[0].fes
                product_space.bdr = product_space.summands[0].bdr
            return product_space            
        else:
            return NotImplemented

    def __pow__(self, power):
        assert isinstance(power, int)
        domain = self
        for i in range(power-1):
            domain = DirectSum(domain, self, flatten=True)
        domain.fes = self.fes
        domain.bdr = self.bdr
        return domain

    @property
    # By default, even a complex fes would read as a real vector space,
    # since NGSolve parses complexes as double-sized reals.
    # This overwrites the usual check.
    def is_complex(self):
        return self.fes.is_complex


    
# Registering functionals and Hilbert spaces defined in `regpy.hilbert.ngsolve`
# and `regpy,functionals.ngsolve`. So that the can be used by using the abstract 
# Hilbert space `AbstractSpace` or abstract Functionals `AbstractFunctional` 

from regpy.hilbert import L2, L2Boundary, Sobolev, SobolevBoundary, Hm0
from regpy.hilbert.ngsolve import L2FESpace, SobolevFESpace, H10FESpace, L2BoundaryFESpace, SobolevBoundaryFESpace

L2.register(NgsSpace, L2FESpace)
Sobolev.register(NgsSpace,SobolevFESpace)
Hm0.register(NgsSpace,H10FESpace)
L2Boundary.register(NgsSpace, L2BoundaryFESpace)
SobolevBoundary.register(NgsSpace,SobolevBoundaryFESpace)


from regpy.functionals.ngsolve import NgsL1,NgsTV
from regpy.functionals import L1, TV

L1.register(NgsSpace, NgsL1)
TV.register(NgsSpace,NgsTV)
