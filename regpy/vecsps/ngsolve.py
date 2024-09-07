"""Finite element vector spaces using NGSolve

This module implements a `regpy.vecsps.VectorSpaceBase` instance for NGSolve spaces. This gives the basic
interface to use FES spaces defined in `ngsolve` to be used as `VectorSpaceBases`in `regpy`.  
Operators are using such spaces are implemented in the `regpy.operators.ngsolve` module. Hilbert spaces 
and Functionals defined on such spaces can be found in `regpy.hilbert.ngsolve` and `regpy.functionals.ngsolve`
respectively. 
"""

import ngsolve as ngs
import numpy as np
from copy import copy,deepcopy
from dataclasses import dataclass, field
from typing import Optional

from regpy.vecsps import VectorBase, VectorSpaceBase, DirectSum
from regpy.util import is_complex_dtype, classlogger

@dataclass 
class NgsBaseVector:
    vec: ngs.la.BaseVector
    make_copy: Optional[bool] = field(default=False)

    def copy(self):
        return copy(self)

    def __post_init__(self):
        if isinstance(self.vec,ngs.la.BaseVector):
            if self.make_copy:
                self.vec = deepcopy(self.vec)
            pass
        elif isinstance(self.vec,ngs.la.DynamicVectorExpression):
            if self.make_copy:
                self.vec = deepcopy(self.vec.Evaluate())
            else:
                self.vec = self.vec.Evaluate()
        else:
            raise TypeError("Could not treat {} type only ngs.la.BaseVector or ngs.la.DynamicVectorExpression".format(type(self.vec)))
        self.size = self.vec.size
        self.is_complex = self.vec.is_complex

    def conj(self):
        z = self.vec.CreateVector()
        for i in range(self.size): 
            z[i] = self.vec[i].real - 1j*self.vec[i].imag
        return NgsBaseVector(z)
    
    @property
    def real(self):
        z = self.vec.CreateVector()
        for i in range(self.size): 
            z[i] = self.vec[i].real
        return NgsBaseVector(z)
    
    @property
    def imag(self):
        z = self.vec.CreateVector()
        for i in range(self.size): 
            z[i] = self.vec[i].imag
        return NgsBaseVector(z)

    def __iadd__(self,other):
        assert isinstance(other,NgsBaseVector) and other.size == self.vec.size 
        self.vec.data += other.vec
        return self

    def __isub__(self,other):
        assert isinstance(other,NgsBaseVector) and other.size == self.vec.size 
        self.vec.data -= other.vec
        return self
    
    def __add__(self,other):
        assert isinstance(other,NgsBaseVector) and other.size == self.vec.size 
        v = self.vec.CreateVector()
        v.data = self.vec + other.vec
        return NgsBaseVector(v)
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self,other):
        return self + (-1*other)
    
    def __rsub__(self,other):
        return (-1*self) + other
    
    def __imul__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        self.vec.data *= other
        return self
    
    def __itruediv__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        self.vec.data /= other
        return self
    
    def __mul__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        v = self.vec.CreateVector()
        v.data = other * self.vec
        return NgsBaseVector(v)
        
    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        v = self.vec.CreateVector()
        v.data = (1/other)*self.vec
        return NgsBaseVector(v)
    
    def __getitem__(self,i):
        return self.vec[i]
    
    def __setitem__(self,i,val):
        self.vec[i] = val

    def __iter__(self):
        return self.vec

    def iter_basis(self):
        v = NgsBaseVector(self.vec.CreateVector())
        for i in self.size:
            v[i] = 1 
            yield v
            if self.is_complex:
                v[i] = 1j
                yield v
            v[i] = 0                

    def __and__(self,x,y):
        assert isinstance(y,NgsBaseVector) and x.size == y.size
        return (x_i == y_i for x_i,y_i in zip(x,y))
    
    def __or__(self,x,y):
        assert isinstance(y,NgsBaseVector) and x.size == y.size
        return (x_i != y_i for x_i,y_i in zip(x,y))

    def __xor__(self,x,y):
        assert isinstance(y,NgsBaseVector) and x.size == y.size
        return (x_i != y_i for x_i,y_i in zip(x,y))
    
    def __copy__(self):
        return deepcopy(self)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class NgsVector(VectorBase):
    def __init__(self, fes, bdr=None):
        assert isinstance(fes, ngs.FESpace)
        self.fes = fes
        self.bdr = bdr
        super().__init__(NgsBaseVector, (fes.ndof,), fes.is_complex)
        # Checks if FES is Vector valued and stores the dimension in self.codim
        from netgen.libngpy._meshing import NgException
        try:
            self.codim = len(fes.components)
            assert self.codim == fes.mesh.dim
            self._fes_util = ngs.VectorL2(self.fes.mesh, order=0, complex = self.is_complex)
        except NgException:
            self.codim = 1
            self._fes_util = ngs.L2(self.fes.mesh, order=0, complex = self.is_complex)
        self._gfu_util = ngs.GridFunction(self._fes_util)
        self._gfu_fes = ngs.GridFunction(fes)

    def zeros(self):
        h = self._gfu_fes.vec.CreateVector()
        h *= 0
        return NgsBaseVector(h,make_copy=True)
    
    def ones(self):
        self._gfu_fes.Set(1)
        return NgsBaseVector(ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec),make_copy=True)
    
    def empty(self):
        h = self._gfu_fes.vec.CreateVector()
        h *= 0
        return NgsBaseVector(h,make_copy=True)
    
    def rand(self,random_generator = None):
        random_generator = random_generator or np.random.random_sample 
        r = random_generator(self._fes_util.ndof)
        if self.is_complex and not is_complex_dtype(r.dtype):
            c = np.empty(self._fes_util.ndof, dtype=complex)
            c.real = r
            c.imag = random_generator(self._fes_util.ndof)
            self._gfu_util.vec.FV().NumPy()[:] = c            
        else:
            self._gfu_util.vec.FV().NumPy()[:] = r
        self._gfu_fes.Set(self._gfu_util)
        return NgsBaseVector(ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec),make_copy=True)
    
    def poisson(self,x):
        assert not self.is_complex
        self._gfu_fes.vec.FV().NumPy[:] =  np.random.poisson(x.vec.FV().NumPy())
        return NgsBaseVector(ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec),make_copy=True)
    

    def is_vector(self,x):
        if not isinstance(x,NgsBaseVector):
            return False
        elif x.size != self.fes.ndof:
            return False
        elif x.is_complex:
            return self.is_complex
        else:
            return True
        
    def vdot(self, x, y):
        return ngs.InnerProduct(x.vec,y.vec)

    def to_complex(self):
        if self.is_complex:
            return copy(self)
        return NgsVector(type(fes)(fes.mesh,order=fes.globalorder,bdr=self.bdr,complex=True),bdr=self.bdr)

    def to_real(self):
        if not self.is_complex:
            return copy(self)
        return NgsVector(type(fes)(fes.mesh,order=fes.globalorder,bdr=self.bdr,complex=False),bdr=self.bdr)
    
    def flatten(self, x):
        return x

    def fromflat(self, x):
        return x
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other,type(self)):
            return False
        return self.fes == other.fes
    
    def iter_basis(self):
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array is returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of complex a vector space after each
        each array modified in its place with a real one it returns the same vector with \(1i\)
        in its place.   
        """
        elm = self.zeros()
        for idx in range(self.shape[0]):
            elm[idx] = 1
            yield elm
            if self.is_complex:
                elm[idx] = 1j
                yield elm
            elm[idx] = 0

    @staticmethod
    def logical_and(x,y):
        return x & y 
    
    @staticmethod
    def logical_or(x,y):
        return x | y 
    
    def logical_not(self,x):
        return not x
    
    def logical_xor(self,x,y):
        return x ^ y


class NgsVectorSpace(VectorSpaceBase):
    """A vector space wrapping an `ngsolve.FESpace`.

    Parameters
    ----------
    fes : ngsolve.FESpace
       The wrapped NGSolve vector space.
    bdr : 
        Boundary of the NGSolve vector space.
    """

    log = classlogger

    def __init__(self, fes, bdr=None):
        super().__init__(NgsVector(fes=fes,bdr=bdr))
        self.fes = self.vec_type.fes
        self.bdr = self.vec_type.bdr
        self.codim = self.vec_type.codim
        self._gfu_fes = ngs.GridFunction(self.fes)
        self._help_x = NgsBaseVector(self._gfu_fes.vec)

    def is_on_boundary(self,x):
        if self.bdr is None:
            return False
        t = x.vec.CreateVector()
        t.data = x.vec
        ngs.Projector(self.fes.FreeDofs(), range=True).Project(t)
        return np.all(t.FV().NumPy() == 0)
    
    def to_gf(self, x):
        gf = ngs.GridFunction(self.fes)
        gf.vec.data = x.vec
        return gf
    
    def from_ngs(self, ngs_elem):
        if isinstance(ngs_elem,ngs.comp.GridFunction):
            return NgsBaseVector(ngs_elem.vec,make_copy=True)
        else:
            self._gfu_fes.Set(ngs_elem)
            return self._help_x.copy()

    
# Registering functionals and Hilbert spaces defined in `regpy.hilbert.ngsolve`
# and `regpy,functionals.ngsolve`. So that the can be used by using the abstract 
# Hilbert space `AbstractSpace` or abstract Functionals `AbstractFunctional` 

from regpy.hilbert import L2, L2Boundary, Sobolev, SobolevBoundary, Hm0
from regpy.hilbert.ngsolve import L2FESpace, SobolevFESpace, H10FESpace, L2BoundaryFESpace, SobolevBoundaryFESpace

L2.register(NgsVectorSpace, L2FESpace)
Sobolev.register(NgsVectorSpace,SobolevFESpace)
Hm0.register(NgsVectorSpace,H10FESpace)
L2Boundary.register(NgsVectorSpace, L2BoundaryFESpace)
SobolevBoundary.register(NgsVectorSpace,SobolevBoundaryFESpace)


from regpy.functionals.ngsolve import NgsL1,NgsTV
from regpy.functionals import L1, TV

L1.register(NgsVectorSpace, NgsL1)
TV.register(NgsVectorSpace,NgsTV)
