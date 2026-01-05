import numpy as np
from collections.abc import Callable
from regpy.util import Errors
from abc import ABC, abstractmethod
from .numpy import UniformGridFcts,NumPyVectorSpace

__all__ = ["GenCurve","StarCurve","Kite","Peanut","Round_rect","Apple","Three_lobes","Pinched_ellipse","Smoothed_rectangle","Nonsym_shape","Circle","ParameterizedCurve","ParameterizedCurveSpc","GenTrigSpc","GenTrig","StarTrigRadialFcts","StarTrigCurve"]

class GenCurve:
    r"""Base class for parameterized smooth, non self-intersecting, closed curves in :math:`R^2`. 
    The parametrization is given by a function :math:`z(t)`\, 
    :math:`0\leq t\leq 2*\pi` and should be counter-clockwise (for the correct orientation of the normal 
    vector). Note that :math:`z(t)` must return two _call_samples :math:`[x(t),y(t)]`\.

    Subclasses should implement `_call` with the optional argument `der` 
    to determine which derivative to compute.

    After initializing the curve additional derivatives can be computed by 
    resetting the `der` property. The number of evaluation points can also be 
    reset by setting the 'n` property with some new number resulting in a recompute
    of all the evaluations. 

    Parameters
    ----------
    n : int 
        number of discretization point
    der : int, optional
        number of derivatives to initially compute.
    """

    BlockDerivative = None
    """ class variable of type regpy.operator.convolution.Derivative to compute derivatives of (possibly several) 
    complex value periodic functions on :math:`[0,2 pi]` sampled at n equidistant points
    """

    def __init__(self, n : int ,nderivs : int = 0):
        self._z = []
        """List of all evaluations of z(t) and its derivatives. """
        self._nderivs = -1

        self.n = n
        self.nderivs = nderivs 



    def __call__(self,der :  int = 0)->np.ndarray:
        res = self._call(der=der)
        if res.ndim != 2 or res.shape[0] != 2:
            raise RuntimeError(Errors.runtime_error(f"Calling the GenCurve {self} did not construct a array of Nx2 dimension!"))
        return res
    
    def _call(self,der=0):
        raise NotImplementedError
    
    @property
    def nderivs(self)->int:
        """number of derivatives to compute"""
        return self._nderivs

    @nderivs.setter
    def nderivs(self,nderivs_new):
        if not isinstance(nderivs_new,int) or nderivs_new >3:
            raise ValueError(Errors.value_error("The number of derivatives needs to be an integer between 0 and 3!"))
        if self.nderivs < nderivs_new:
            for i in range(self.nderivs+1,nderivs_new+1):
                self._z.append(self(i))
                self._nderivs += 1

    @property    
    def n(self)->int:
        """number of evaluation points"""
        return self._n
    
    @n.setter
    def n(self,n_new):
        if not isinstance(n_new,int) or n_new <= 0:
            raise ValueError(Errors.value_error("The number of discretization points of the GenCurve needs to be a positive integer!"))
        self.t = np.linspace(0, 2*np.pi, n_new,endpoint=False)
        self._n = n_new
        for i in range(0,self.nderivs+1):
            self._z[i]= self(i)

    @property
    def z(self)->np.ndarray:
        """Values of z(t) at equidistant grid of self.n points."""
        if self.nderivs >= 0:
            return self._z[0]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation the self.nderivs >=0 please change that!",self,"z"))
    
    @property
    def zp(self)->np.ndarray:
        """Values of z(t) its first derivatives at equidistant grid of self.n points."""
        if self.nderivs >= 1:
            return self._z[1]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the first derivative the self.nderivs >=1 please change that!",self,"zp"))
    
    @property
    def zpabs(self)->np.ndarray:
        """Absolute _call_samples |z'(t)| at equidistant grid of self.n points."""
        if self.zp is not None:
            return np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
    @property
    def normal(self)->np.ndarray:
        if self.zp is not None:
            return np.append(self.zp[1,:], -self.zp[0,:]).reshape((2, self.n))
        
    @property
    def zpp(self)->np.ndarray:
        """Values of z(t) its second derivatives at equidistant grid of self.n points."""
        if self.nderivs >= 2:
            return self._z[2]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the second derivative the self.nderivs >=2 please change that!",self,"zpp"))

    @property
    def zppp(self)->np.ndarray:
        """Values of z(t) its third derivatives at equidistant grid of self.n points."""
        if self.nderivs >= 3:
            return self._z[3]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the third derivative the self.nderivs >=3 please change that!",self,"zppp"))

    def param_derivative(self,u:np.ndarray) -> np.ndarray : 
        """
        Computes the derivative(s) of one or several complex periodic functions :math:`u:[0,2 pi] \to C`,
        which are given by their radial_samples at self.nval equidistant point on :math:`[0,2 pi]` 
        
        Parameters:
        u: np.ndarray
            two-dimnensional complex array with first dimension self.nval 
        """
        from regpy.operators.convolution import Derivative        
        
        if not isinstance(u,np.ndarray) or not np.issubdtype(u.dtype,complex):
            raise TypeError(Errors.type_error('u must be a complex ndarray'))
        if not len(u.shape) in (1,2) or not u.shape[0]==self.n:
            raise ValueError(Errors.value_error(f'u must have two dimensions, the first one equal to self.n. Given shape: {u.shape}. n: {self.n}'))
                
        if GenCurve.BlockDerivative is None or  (GenCurve.BlockDerivative.domain.shape!=u.shape):
            der_domain = UniformGridFcts((0.,2*np.pi,self.n), periodic=True,dtype=complex,
                                         shape_codomain=(u.shape[1],) if len(u.shape) ==2 else () 
                                         )
            GenCurve.BlockDerivative = Derivative(der_domain,(1,))
        return GenCurve.BlockDerivative(u)


    def arc_length_der(self, h: np.ndarray)->np.ndarray:
        if len(h.shape)==1:
            return self.param_derivative(h) / self.zpabs
        elif len(h.shape)==2:
            return self.param_derivative(h) / self.zpabs[:,np.newaxis]
        else:
            raise ValueError(Errors.value_error('shape of h must have length 1 or 2.'))

class StarCurve(GenCurve):
    r"""Base class for star-shaped curve (w.r.t the origin) in :math:`R^2`, 
    parameterized by 

    .. math::
        z(t) = radial(t)*[cos(t);sin(t)] 0<=t<=2pi

    with a positive, :math:`2\pi`\-periodic function :math:`radial`\. 

    Subclasses should implement `radial` with the optional argument `der` 
    to determine which derivative to compute.

    After initializing the curve additional derivatives can be computed by 
    resetting the `nderivs` property. The number of evaluation points can also be 
    reset by setting the 'n` property with some new number resulting in a recompute
    of all the evaluations. 

    Parameters
    ----------
    n : int 
        number of discretization point
    nderivs : int, optional
        number of derivatives to initially compute.
    """
    def __init__(self, n:int, nderivs:int = 0)->None:
        super().__init__(n,nderivs=nderivs)

    def __call__(self,der:int=0)->np.ndarray:
        res = self.radial(der=der)
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        if res.ndim != 1:
            raise RuntimeError(Errors.runtime_error(f"Calling radial of StarCurve {self} did not construct a array of one dimension!"))
        if der == 0:
            return np.array([res*cost,res*sint])
        elif der == 1:            
            return np.array([res*cost,res*sint]) + np.array([[0,-1],[1,0]])@self.z
        elif der == 2:
            return np.array([res*cost, res*sint]) + 2*np.array([[0,-1],[1,0]])@self.zp + self.z
        elif der == 3:
            return np.array([res*cost ,res*sint]) + 3*np.array([[0,-1],[1,0]])@self.zpp + 3 * self.zp + np.array([[0,1],[-1,0]])@self.z
        return res
    
    def radial(self,der:int=0)->np.ndarray:
        raise NotImplementedError
    
    @property
    def zpabs(self)->np.ndarray:
        r""":math:`|z'(t)|`"""
        if self.zp is not None:
            return np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
    @property
    def normal(self)->np.ndarray:
        r"""Outer normal vector(not normalized)"""
        if self.zp is not None:
            return np.vstack((self.zp[1,:], -self.zp[0,:]))

    @abstractmethod
    def radial(self, der:int=0)->np.ndarray:
        """return radial fucntion or its der-th derivative on the equidistant grid."""

    
################################ special GenCurves and StarCurves ####################################

class Kite(GenCurve):
    r"""Subclass of the `GenCurve` that gives a kite form. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """
    def __init__(self, n:int, nderivs:int = 0):
        super().__init__(n,nderivs=nderivs)

    def _call(self, der:int=0):
        if der==0:
            return np.vstack((np.cos(self.t)+0.65*np.cos(2*self.t)-0.65,   1.5*np.sin(self.t)))
        elif der==1:
            return np.vstack((-np.sin(self.t)-1.3*np.sin(2*self.t)    ,    1.5*np.cos(self.t)))
        elif der==2:
            return np.vstack((-np.cos(self.t)-2.6*np.cos(2*self.t)    ,   -1.5*np.sin(self.t)))
        elif der==3:
            return np.vstack((np.sin(self.t)+5.2*np.sin(2*self.t)     ,   -1.5*np.cos(self.t)))
        else:
            raise ValueError('derivative not implemented')



class Peanut(StarCurve):
    r"""Subclass of the `StarCurve` that gives a peanut-shaped curve. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """    
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        if der==0:
            return 1./2.*(3*cost**2+1)**(1./2)
        elif der==1:
            return -3./2./(4.*cost**2+sint**2)**(1./2)*cost*sint
        elif der==2:
            return  -3./2*(3.*cost**4+2.*cost**2-1)/(3*cost**2+1)**(3./2)
        elif der==3:
            return  3./2.*cost*sint*(9.*cost**4+6*cost**2+13)/(3*cost**2+1)**(5./2)
        else:
            raise ValueError('derivative not implemented')
        return res

class Round_rect(StarCurve):
    r"""Subclass of the `StarCurve` that gives a curve with a rounded rectangular shape. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        co = 2/3
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        if der==0:
            return  (sint**10 + (co*cost)**10)**(-0.1)
        elif der==1:
            return  -1/10/(sint**10+co**10*cost**10)**(11/10)*(10*sint**9*cost-10*co**10*cost**9*sint)
        elif der==2:
            return  11/100/(sint**10+co**10*cost**10)**(21/10)*(10*sint**9*cost-10*co**10*cost**9*sint) \
                **2-1/10/(sint**10+co**10*cost**10)**(11/10)*(90*sint**8*cost**2-10*sint**10+90*co**10 \
                *cost**8*sint**2-10*co**10*cost**10)
        elif der==3:
            return  -231/1000/(sint**10+co**10*cost**10)**(31/10)**(10*sint**9*cost-10*co**10*cost**9*sint)**3+33 \
                /100/(sint**10+co**10*cost**10)**(21/10)*(10*sint**9*cost-10*co**10*cost**9*sint) \
                *(90*sint**8*cost**2-10*sint**10+90*co**10*cost**8*sint**2-10*co**10*cost**10)-1/10 \
                /(sint**10+co**10*cost**10)**(11/10)*(720*sint**7*cost**3-280*sint**9*cost-720*co**10 \
                *cost**7*sint**3+280*co**10*cost**9*sint)
        else:
            raise ValueError('derivative not implemented')


class Apple(StarCurve):
    r"""Subclass of the `StarCurve` that gives an apple-shaped curve. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        cos2t = np.cos(2*self.t)
        sin2t = np.sin(2*self.t)
        if der==0:
            return  (0.5+0.4*cost+0.1*sin2t)/(1+0.7*cost)
        elif der==1:
            return  (-2/5*sint+1/5*cos2t)/(1+7/10*cost)+7/10*(1/2+2/5*cost+1/10*sin2t)/(1+7/10*cost)**2*sint
        elif der==2:
            return  (-2/5*cost-2/5*sin2t)/(1+7/10*cost)+7/5*(-2/5*sint+1/5*cos2t)/(1+7/10*cost) \
                **2*sint+49/50*(1/2+2/5*cost+1/10*sin2t)/(1+7/10*cost)**3*sint**2+7/10*(1/2+2/5  \
                *cost+1/10*sin2t)/(1+7/10*cost)**2*cost
        elif der==3:
            return  (2/5*sint-4/5*cos2t)/(1+7/10*cost)+21/10*(-2/5*cost-2/5*sin2t)/(1+7/10*cost)**2 \
                *sint+147/50*(-2/5*sint+1/5*cos2t)/(1+7/10*cost)**3*sint**2+21/10*(-2/5*sint+1/5 \
                *cos2t)/(1+7/10*cost)**2*cost+1029/500*(1/2+2/5*cost+1/10*sin2t)/(1+7/10*cost) \
                **4*sint**3+147/50*(1/2+2/5*cost+1/10*sin2t)/(1+7/10*cost)**3*sint*cost-7/10 \
                *(1/2+2/5*cost+1/10*sin2t)/(1+7/10*cost)**2*sint
        else:
            raise ValueError('derivative not implemented')


class Three_lobes(StarCurve):
    r"""Subclass of the `StarCurve` that gives a curve with three lobes. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        cos3t = np.cos(3*self.t)
        sin3t = np.sin(3*self.t)
        if der==0:
            return  0.5 + 0.25*np.exp(-sin3t) - 0.1*sint
        elif der==1:
            return  -3/4*cos3t*np.exp(-sin3t)-1/10*cost
        elif der==2:
            return  9/4*sin3t*np.exp(-sin3t)+9/4*cos3t**2*np.exp(-sin3t)+1/10*sint
        elif der==3:
            return  27/4*cos3t*np.exp(-sin3t)-81/4*sin3t*cos3t*np.exp(-sin3t)-27/4*cos3t**3*np.exp(-sin3t)+1/10*cost
        else:
            raise ValueError('derivative not implemented')


class Pinched_ellipse(StarCurve):
    r"""Subclass of the `StarCurve` that gives curve with shape of a pinched ellipse. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        if der==0:
            return  3/2*np.sqrt(1/4*cost**2 + sint**2)
        elif der==1:
            return  9/4/(-3*cost**2+4)**(1/2)*cost*sint
        elif der==2:
            return  9/4*(3*cost**4-8*cost**2+4)/(3*cost**2-4)/(-3*cost**2+4)**(1/2)
        elif der==3:
            return  -9/4*cost*sint*(9*cost**4-24*cost**2+28)/(3*cost**2-4)**2/(-3*cost**2+4)**(1/2)
        else:
            raise ValueError('derivative not implemented')


class Smoothed_rectangle(StarCurve):
    r"""Subclass of the `StarCurve` that gives a curve with the shape of a smoothed rectangle. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        if der==0:
            return  (cost**10 +2/3*sint**10)**(-1/10)
        elif der==1:
            return  -1/10/(cost**10+2/3*sint**10)**(11/10)*(-10*cost**9*sint+20/3*sint**9*cost)
        elif der==2:
            return  11/100/(cost**10+2/3*sint**10)**(21/10)*(-10*cost**9*sint+20/3*sint**9*cost)**2 \
                -1/10/(cost**10+2/3*sint**10)**(11/10)*(90*cost**8*sint**2-10*cost**10 \
                +60*sint**8*cost**2-20/3*sint**10)
        elif der==3:
            return  -231/1000/(cost**10+2/3*sint**10)**(31/10)*(-10*cost**9*sint+20/3*sint**9*cost)**3 \
                +33/100/(cost**10+2/3*sint**10)**(21/10)*(-10*cost**9*sint+20/3*sint**9*cost)* \
                (90*cost**8*sint**2-10*cost**10+60*sint**8*cost**2-20/3*sint**10) \
                -1/10/(cost**10+2/3*sint**10)**(11/10)*(-720*cost**7*sint**3+280*cost**9*sint \
                +480*sint**7*cost**3-560/3*sint**9*cost)
        else:
            raise ValueError('derivative not implemented')


class Nonsym_shape(StarCurve):
    r"""Subclass of the `StarCurve` that gives a non symmetric curve. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        if der==0:
            return (1 + 0.9*cost + 0.1*np.sin(2*self.t))/(1 + 0.75*cost)
        elif der==1:
            return  4/5*(-3*sint+8*cost**2-4+3*cost**3)/(16+24*cost+9*cost**2)
        elif der==2:
            return  -4/5*(12*cost-9*cost**2+64*sint*cost+36*sint*cost**2+9*sint*cost**3+24*sint+18) \
                /(64+144*cost+108*cost**2+27*cost**3)
        elif der==3:
            return  -4/5*(144*sint*cost+114*sint-40+240*cost**3+192*cost-27*sint*cost**2+368*cost**2 \
                +144*cost**4+27*cost**5)/(256+768*cost+864*cost**2+432*cost**3+81*cost**4)
        else:
            raise ValueError('derivative not implemented')


class Circle(StarCurve):
    r"""Subclass of the `StarCurve` that gives a circle. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """     
    def __init__(self,n:int,nderivs:int=0):
        super().__init__(n,nderivs=nderivs)

    def radial(self,der=0):
        if der==0:
            return np.ones_like(self.t)
        else:
            return np.zeros_like(self.t)



######################### parameterized curves ##############################

class ParameterizedCurve():
    r""" This is an abstract class for GenCurves parameterized by coefficients in some vector space.

    Typical forward operators are naturally defined on a (shape) space of curves. In particular, they 
    are invariant under re-parametrizations of the curve. However, such shape spaces are not vector spaces, and we define forward operators on some vector space of coefficients paraemeterizing a certain class of curves (e.g., star-shape curves with radial functions given by trigonometric 
    polynomials of a cetrain degree, see `StarTrigRadialFcts`).  

    One may think of the mapping taking coefficients to curves as an operator, and of the parameterized forward operators as compositions of this operator with the actual forward operator defined 
    on a shape space. However, as mentioned above, shape spaces aren't vector spaces, and it is often  
    computationally more convenient, not to implement the discrete Frechet derivative of the parameterized forward operator, but to discretize a continuous characterization of the Frechet derivative. For this reason the forward operators on curve spaces in this toolbox are not implemented as such compositions.

    Due to invariance under reparameterizations, Frechet derivatives of operators on curve spaces only 
    depend on the normal component of a perturbation of the curve, and this quantity is used 
    in our implementations of the derivatives of parameterized forward operators. More precisely, 
    these derivatives are compositions of a discretization of the continuous derivative on curve space 
    and the operator `self.der_normal`.
    """
    def __init__(self,coeff:np.ndarray,der_normal,**kwargs):
        from regpy.operators import Operator
        if not coeff in der_normal.domain:
            raise ValueError(Errors.not_a_vecsp(coeff,der_normal.domain))
        if not isinstance(der_normal.domain,ParameterizedCurveSpc):
            raise ValueError(Errors.not_instance(der_normal.domain,ParameterizedCurveSpc))
        if not isinstance(der_normal.codomain,UniformGridFcts):
            raise ValueError(Errors.not_instance(der_normal.codomain,UniformGridFcts))
        if not len(der_normal.codomain.shape)==1:
            raise ValueError(Errors.generic_message('codomain must have shape of length 1.'))
        self.coeff = coeff
        """The coefficient vector characterizing the curve.
        """
        self.der_normal = der_normal
        """Linear Operator given by the inner product of the derivative of the coefficient-to-curve "operator" (i.e. the derivative of `ParameterizedCurveSpc.coeff2curve`) and the normal vector of length 1 of the curve. """


class ParameterizedCurveSpc(ABC):
    r""" Abstract base class of vector spaces of coefficients of `ParameterizedCurve`s. 
    """
    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def coeff2curve(self, coeff : np.ndarray, nderivs : int =0)->ParameterizedCurve:
        r"""Compute a curve for the given coefficients.
        """
        pass

class GenTrig(GenCurve,ParameterizedCurve):
    r"""The class GenTrig describes boundaries of domains in R^2 which are
    parameterized by 

    .. math::
        z(t) = [z_1(t), z_2(t)]      0<=t<=2pi

     where z_1 and z_2 are real trigonometric polynomials with 'self.spc.n_sample' real coefficients.
     z and its derivatives are sampled at n equidistant points.
     
     Parameters
     ----------
     coeff : np.ndarray
        Equidistant (in parameter space!) samples of the cartesian components of the parameterization of the curve 
     spc : regpy.vecspc.curve.GenTrigSpc 
        Underlying `ParameterizedCurveSpc`
     nderivs : int
        Number of derivatives to compute 
     """

    def __init__(self, coeff:np.ndarray, spc:ParameterizedCurveSpc, nderivs:int=0):
        from regpy.operators import Operator
        class NormalComponent(Operator):
            """Operator which computes the normal component of a vector field on the curve
            """
            def __init__(self,curve):
                super().__init__(domain= UniformGridFcts((0,2*np.pi,curve.n), periodic=True, dtype=float,shape_codomain=(2,)), 
                                codomain= UniformGridFcts((0.,2*np.pi,curve.n), periodic=True,dtype=float),
                                linear=True
                                )
                self.curve = curve

            def _eval(self, h):
                return np.sum(h.T*self.curve.normal,0)/self.curve.zpabs

            def _adjoint(self, g):
                return ((g/self.curve.zpabs)[np.newaxis,:]*self.curve.normal).T

        if len(coeff.shape)!=2 or not np.issubdtype(coeff.dtype,np.floating):
            raise ValueError(Errors.value_error(f'coeff must be a 2xN array of real numbers. Got shape {coeff.shape} of type {coeff.dtype}.'))
        if not isinstance(spc,GenTrigSpc):
            raise TypeError(Errors.type_error('spc must be a GenTrigSpc'))
        self.spc = spc
        self.coeff = coeff
        GenCurve.__init__(self,n=spc.n,nderivs=nderivs)
        ParameterizedCurve.__init__(self,
                                    coeff=coeff,
                                    der_normal = NormalComponent(self) * spc.der_op(0)) 
        
    def _call(self,der=0):
        return self.spc.der_op(der)(self.coeff).T


class GenTrigSpc(UniformGridFcts,ParameterizedCurveSpc):
    r"""Class for the `VectorSpaceBase` instance of `GenTrig` instances. 
    It is a space of vector-valued trigonometric polynomials. 
    The class provides method `bd_eval` which generates a curve `GenTrig` from a given coefficient (or sample) vector.  

    Parameters
    ----------
    n_sample : int
        Number of coefficients of each of the cartesian components.
    n: int
        Number of points to evaluate the parameterization on
    """
    def __init__(self, n_sample:int, n:int|None=None):  
        if not isinstance(n_sample, int,) or n_sample<=0:
            raise TypeError(Errors.not_instance(n,int,add_info="The GenTrigSpc need n to be a positive integer!"))
        self.n_sample = n_sample
        if n is None:
            n= n_sample 
        super().__init__(np.linspace(0, 2*np.pi, n_sample, endpoint=False),shape_codomain=(2,))
        self.n = n

    @property    
    def n(self)->int:
        """number of evaluation points"""
        return self._n
    
    @n.setter
    def n(self,n_new:int):
        from regpy.operators.convolution import Derivative  
        if not isinstance(n_new,int) or n_new < 0:
            raise ValueError(Errors.value_error("The number of discretization points of the GenTrigSpace needs to be a positive integer!"))
        if (self.n_sample-n_new)%2!=0:
            raise ValueError(Errors.value_error(f"n_sample-n must be even. Got {n_new} and {self.n_sample}"))
        self._n = n_new
        self._der_ops = [Derivative(self,(order,),Fourier_truncation_amount=(self.n_sample-n_new)//2) for order in range(1)]

    def der_op(self,order:int):
        """ Returns the order's derivative operator on a vector-valued space of codomain shape 2, which also performs a Fourier interpolation from a grid of size n_sample to a grid of size n. 
        """
        from regpy.operators.convolution import Derivative         
        if order>=len(self._der_ops):
            self._der_ops += [Derivative(self,(order,),Fourier_truncation_amount=(self.n_sample-self.n)//2) for order in range(len(self._der_ops),order+1)]
        return self._der_ops[order]

    def coeff2curve(self, coeff:np.ndarray, nderivs:int=0)->GenTrig:
        r"""Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `GenTrig`.
        
        Parameters
        ----------
        coeff : array-like
            samples from which to generate the curve
        nderivs : int
            Number of derivatives to compute 
        """
        if self.n==0:
            raise RuntimeError('self.n has not been set, yet.')
        return GenTrig(coeff, self, nderivs)
    
    def circle(self, radius:float =1.,nderivs:int=0)->GenTrig:
        t = np.linspace(0, 2*np.pi,self.n_sample,endpoint=False)
        return GenTrig(radius*np.vstack((np.cos(t), np.sin(t))).T,self,nderivs=nderivs)        

class StarTrigCurve(StarCurve,ParameterizedCurve): 
    r"""A class representing star shaped 2d curves with radial function parametrized in a
    trigonometric basis. Should usually be instantiated via `StarTrigRadialFcts.coeff2curve`.

    Parameters
    ----------
    vecsp : StarTrigRadialFcts
        The underlying vector space.
    coeff : array-like
        The samples of the radial function.
    nderivs : int, optional
        How many derivatives to compute. At most 3 derivatives are implemented.
    """

    def __init__(self, vecsp, coeff:np.ndarray, nderivs:int=1):
        from regpy.operators import PtwMultiplication
            
        if not isinstance(nderivs, int) or nderivs <0 or nderivs >3:
            raise ValueError(Errors.value_error(f"The number of derivative in StarTrigCurve needs to be an integer between 0 and 3"))
        self.vecsp = vecsp
        """The StarTrigFcts vector space."""
        #self.dim = len(coeff)

        self.coeff = coeff
        self._radial = np.asanyarray([vecsp.der_op(order)(coeff) for order in range(nderivs+1)])
        """Sampled radial function and its derivatives, shaped `(nderivs + 1, nvals)`."""
        StarCurve.__init__(self,n=self.vecsp.n,nderivs=nderivs)      
        mult = PtwMultiplication(UniformGridFcts((0,2*np.pi,self.n),periodic=True),
                                 self._radial[0,:] / self.zpabs
                                 ) 
        ParameterizedCurve.__init__(self,coeff = coeff,
                                    der_normal = mult * vecsp.der_op(0)
                                    )

    def radial(self,der:int=0):
        if der>self._radial.shape[0]:
            return RuntimeError(f'Value of der {der} greater than self.nderivs {self.nderivs}. Initialize with larger value of nderivs!')
        return self._radial[der,:]


class StarTrigRadialFcts(UniformGridFcts,ParameterizedCurveSpc):
    r"""Class for VectorSpaceBase` instance of `StarTrigCurve` instances. It provides 
    the method `eval_curve` which gives a curve `StarTrigCurve`.  

    The space consists of star-shaped curves with radial functions given by real trigonometric 
    polynomials of some maximal degree. These trigonometric polynomials are determined by their radial_samples on 
    an equidistant grid. 

    Parameters
    ----------
    dim : int
        Dimension of the space of trigonometric polynomials 
    n: int
        number of points on the curves
    """
    def __init__(self, dim:int,n:int|None=None):
        if not isinstance(dim, int) or dim<=0:
            raise TypeError(Errors.not_instance(dim,int,add_info="StarTrigRadialFcts need dim to be a positive integer!"))      
        super().__init__(np.linspace(0, 2*np.pi, dim, endpoint=False))
        self.dim = dim
        self.n = n if n is not None else dim

    @property    
    def n(self)->int:
        """number of evaluation points"""
        return self._n
    
    @n.setter
    def n(self,n_new:int):
        from regpy.operators.convolution import Derivative
        if not isinstance(n_new,int) or n_new < 0:
            raise ValueError(Errors.value_error("The number of discretization points  needs to be a positive integer!"))
        self._der_ops = [Derivative(self,(order,),Fourier_truncation_amount=(self.dim-n_new)//2) for order in range(1)]        
        self._n = n_new

    def der_op(self,order:int):
        """ Returns the order's derivative operator, which also performs a Fourier interpolation from a grid of size n_sample to a grid of size n. 
        """
        from regpy.operators.convolution import Derivative         
        if order>=len(self._der_ops):
            self._der_ops += [Derivative(self,(order,),Fourier_truncation_amount=(self.dim-self.n)//2) for order in range(len(self._der_ops),order+1)]
        return self._der_ops[order]

    def coeff2curve(self, coeff:np.ndarray, nderivs:int=1)->StarTrigCurve:
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarTrigCurve`.
        
        Parameters
        ----------
        coeff : np.ndararray
            sample of the radial function at self.dim equidistant points
        nderivs : int, optional
            Number of derivatives to compute , Defaults : 0
        """
        if self.n==0:
            raise RuntimeError('self.n has not been set, yet.')
        return StarTrigCurve(self, coeff,  nderivs)

    def radialfct2curve(self, f:Callable[[np.floating],np.floating],nderivs:int=1):
        coeff = f(np.linspace(0, 2*np.pi, self.dim, endpoint=False))
        return StarTrigCurve(self, coeff,  nderivs)
    
    def circle(self, radius:float=1.,nderivs:int=1)->StarTrigCurve:
        return StarTrigCurve(self, radius*self.ones(),nderivs
                             )