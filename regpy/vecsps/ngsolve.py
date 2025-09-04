r"""Finite element vector spaces using NGSolve

This module implements a `regpy.vecsps.VectorSpaceBase` instance for NGSolve spaces. This gives the basic
interface to use FES spaces defined in `ngsolve` to be used as `VectorSpaceBases`in `regpy`.  
Operators are using such spaces are implemented in the `regpy.operators.ngsolve` module. Hilbert spaces 
and Functionals defined on such spaces can be found in `regpy.hilbert.ngsolve` and `regpy.functionals.ngsolve`
respectively. 
"""

__all__ = ['NgsBaseVector','NgsVectorSpace']

from copy import copy,deepcopy
from dataclasses import dataclass, field
from typing import Optional

import ngsolve as ngs
import numpy as np
from pyngcore.pyngcore import BitArray

from regpy.util import is_complex_dtype

from .base import VectorSpaceBase

@dataclass 
class NgsBaseVector:
    vec: ngs.la.BaseVector
    make_copy: Optional[bool] = field(default=False)

    __array_ufunc__ = None

    def copy(self):
        return copy(self)
    
    @property
    def is_complex_dtype(self):
        return self.vec.is_complex

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
        if self.is_complex_dtype:
            z = ngs.la.BaseVector(size = self.size)
            for i in range(self.size): 
                z[i] = self.vec[i][0].real
            return NgsBaseVector(z)
        return self.copy()
    
    @property
    def imag(self):
        if self.is_complex_dtype:
            z = ngs.la.BaseVector(size = self.size)
            for i in range(self.size): 
                z[i] = self.vec[i][0].imag
            return NgsBaseVector(z)
        return NgsBaseVector(self.vec.CreateVector())

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
    
    def __neg__(self):
        return -1*self
    
    def __imul__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        self.vec.data *= other
        return self
    
    def __itruediv__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        self.vec.data /= other
        return self
    
    def __mul__(self,other):
        from regpy.operators.base import Operator,PtwMultiplication
        if isinstance(other,float) or isinstance(other,int) or isinstance(other,complex):
            v = self.vec.CreateVector()
            v.data = other * self.vec
            return NgsBaseVector(v)
        elif isinstance(other,Operator):
            return PtwMultiplication(other.codomain, self) * other
        else:
            raise NotImplementedError(f"Multiplication of TupleVector with {type(other)} is not defined. It has to be either a number eg float, int or complex or an Operator.")
        
    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):
        assert isinstance(other,float) or isinstance(other,int)
        v = self.vec.CreateVector()
        v.data = (1/other)*self.vec
        return NgsBaseVector(v)
    
    def __getitem__(self,i):
        if isinstance(i,BitArray):
            v = self.vec.CreateVector()
            v[i] = self.vec
            return NgsBaseVector(v)
        elif isinstance(i,int):
            return self.vec[i]
        else:
            return NgsBaseVector(self.vec[i])
    
    def __setitem__(self,i,val):
        if isinstance(val, NgsBaseVector) and self.size == val.size:
            self.vec[i] = val.vec
        else:
            try:
                self.vec[i] = val
            except TypeError:
                raise TypeError(f"Not able to set {val} to NgsBaseVector. It has to be either an NgsBaseVector of same size or Something compatible to set to an ngsolve.la.BaseVector.")

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


class NgsVectorSpace(VectorSpaceBase):
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
        self.fes = fes
        self.bdr = bdr
        super().__init__(vec_type=NgsBaseVector, shape=(fes.ndof,), complex=fes.is_complex)
        # Checks if FES is Vector valued and stores the dimension in self.codim
        from netgen.libngpy._meshing import NgException
        try:
            self.codim = len(fes.components)
            # assert self.codim == fes.mesh.dim
            if (isinstance(fes,ngs.VectorH1),isinstance(fes,ngs.VectorL2),isinstance(fes,ngs.VectorValued)):
                self._fes_util = ngs.VectorL2(self.fes.mesh, order = 0, dim = fes.dim, complex = self.is_complex)
            else:
                l_fes = []
                for f in fes.components:
                    if isinstance(f,ngs.ProductSpace):
                        if (isinstance(f,ngs.VectorH1),isinstance(f,ngs.VectorL2),isinstance(f,ngs.VectorValued)):
                            l_fes.append(ngs.VectorL2(self.fes.mesh, order = 0, complex = self.is_complex))
                        else:
                            raise ValueError
                    else:
                        l_fes.append(ngs.L2(self.fes.mesh, order=0, dim = f.dim, complex = self.is_complex))
                self._fes_util = ngs.ProductSpace(*l_fes)
        except NgException:
            self.codim = 1
            self._fes_util = ngs.L2(self.fes.mesh, order=0, complex = self.is_complex)
        except ValueError:
            self.log.warning("Tried to initialize with a product space of product spaces, which are not VectorH1, VectorL2 or VectorValued. Thus fes_util is not available and thus random generator will not work!")
            self._fes_util = None
        if self._fes_util is not None:
            self._gfu_util = ngs.GridFunction(self._fes_util)
        self._gfu_fes = ngs.GridFunction(self.fes)
        self._help_x = NgsBaseVector(self._gfu_fes.vec)
        self._no_pickle = {*self._no_pickle,"fes"}

    def zeros(self):
        h = self._gfu_fes.vec.CreateVector()
        h *= 0
        return NgsBaseVector(h,make_copy=True)
    
    def ones(self):
        if self.codim == 1:
            self._gfu_fes.Set(1)
        else:
            for gfu_i in self._gfu_fes.components:
                gfu_i.Set(tuple(1 for _ in range(gfu_i.dim)))
        return NgsBaseVector(ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec),make_copy=True)
    
    def empty(self):
        h = self._gfu_fes.vec.CreateVector()
        h *= 0
        return NgsBaseVector(h,make_copy=True)
    
    def rand(self,random_generator = None):
        if self._fes_util is None:
            raise RuntimeError("the utility fes was not created random generator is not available!")
        random_generator = random_generator or np.random.random_sample 
        r = random_generator(self._fes_util.ndof)
        if self.is_complex and not is_complex_dtype(r.dtype):
            c = np.empty(self._fes_util.ndof, dtype=complex)
            c.real = r
            c.imag = random_generator(self._fes_util.ndof)
            self._gfu_util.vec.FV().NumPy()[:] = c            
        else:
            self._gfu_util.vec.FV().NumPy()[:] = r
        if self.codim == 1:
            self._gfu_fes.Set(self._gfu_util)
        else:
            for gfu_i,gfu_util_i in zip(self._gfu_fes.components,self._gfu_util.components):
                gfu_i.Set(gfu_util_i)
        return NgsBaseVector(ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec),make_copy=True)
    
    def poisson(self,x, n = 1):
        assert not self.is_complex
        self._gfu_util.Set(self.to_gf(x))
        assert np.all(self._gfu_util.vec.FV().NumPy()>=0), f"Not all values in {self._gfu_util.vec.FV().NumPy()} are positive."
        self._gfu_util.vec.FV().NumPy()[:] =  np.sum(np.random.poisson(lam = self._gfu_util.vec.FV().NumPy(), size = (n,self._fes_util.ndof)),axis = 0)/n
        self._gfu_fes.Set(self._gfu_util)
        return NgsBaseVector(ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec),make_copy=True)

    def __contains__(self,x):
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

    def complex_space(self):
        if self.is_complex:
            return copy(self)
        return NgsVectorSpace(type(fes)(fes.mesh,order=fes.globalorder,bdr=self.bdr,complex=True),bdr=self.bdr)

    def real_space(self):
        if not self.is_complex:
            return copy(self)
        return NgsVectorSpace(type(fes)(fes.mesh,order=fes.globalorder,bdr=self.bdr,complex=False),bdr=self.bdr)
    
    def flatten(self, x:NgsBaseVector) -> np.ndarray:
        if self.is_complex:
            return np.concatenate([x.vec.FV().NumPy().real,x.vec.FV().NumPy().imag])
        else:
            return x.vec.FV().NumPy().copy()

    def fromflat(self, vec:np.ndarray) -> NgsBaseVector:
        if vec.ndim == 1:
            if self.is_complex and vec.size == self.shape[0] * 2:
                x = self.zeros()
                x.vec.FV().NumPy()[:] = vec[:self.shape[0]] + 1j*vec[self.shape[0]]
            elif vec.size == self.shape[0]:
                x = self.zeros()
                x.vec.FV().NumPy()[:] = vec
            else:
                raise ValueError("provided vector has non fitting shape.")
        else:
            raise ValueError("Provided vector must be one dimensional")
        return x
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other,type(self)):
            return False
        return self.fes == other.fes
    
    def IfPos(self, x):
        """Analyses which components contribute to the positive part of the 
        function corresponding to the vector.

        Parameters
        ----------
        x : NgsBaseVector
            The vector to analyse.

        Returns
        -------
        mask : BitArray
            A BitArray of masks for the vector components contributing to the positive part of a function.
        """
        if not x in self:
            raise ValueError("The vector {} is not an element of the vector space {}".format(x,self))
        if not self.is_complex: 
            self._gfu_fes.vec.data = x.vec
            gfu_help = ngs.GridFunction(self.fes)
            gfu_help.Set(ngs.IfPos(self._gfu_fes,1,0))
            return BitArray([v_i==0 for v_i in gfu_help.vec])
        else:
            return TypeError("The vector space {} is complex, use IfPos only works for real valued functions.".format(self))
    
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

    def logical_and(self,x,y):
        return x & y 
    
    def logical_or(self,x,y):
        return x | y 
    
    def logical_not(self,x):
        return not x
    
    def logical_xor(self,x,y):
        return x ^ y

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
    
    def from_ngs(self, ngs_elem, definedon : ngs.comp.Region|None = None):
        if isinstance(ngs_elem,ngs.comp.GridFunction):
            return NgsBaseVector(ngs_elem.vec,make_copy=True)
        else:
            self._gfu_fes.Set(ngs_elem,definedon=definedon)
            return self._help_x.copy()

