import numpy as np

from regpy.util import Errors

from .numpy import UniformGridFcts

__all__ = ["GenCurve","kite","StarCurve","peanut","round_rect","apple","three_lobes","pinched_ellipse","smoothed_rectangle","nonsym_shape","circle","GenTrigSpc","GenTrig","StarTrigRadialFcts","StarTrigCurve"]

class GenCurve:
    r"""Base class for Parameterized smooth closed curve in :math:`R^2` 
    without self-crossing parametrization by function :math:`z(t)`\, 
    :math:`0\leq t\leq 2*\pi` (counter-clockwise). Note :math:`z(t)` must return two 
    values :math:`[x(t),y(t)]`\.

    Subclasses should implement `_call` with the optional argument `der` 
    to determine which derivative to compute.

    After initializing the curve additional derivatives can be computed by 
    resetting the `der` property. The number of evaluation points can also be 
    reset by setting the 'n` property with some new number resulting in a recompute
    of all the evaluations. 

    Parameters
    ----------
    name : str 
        name of the curves
    n : int 
        number of discretization point
    der : int, optional
        number of derivatives to initially compute.
    """
    
    def __init__(self, name, n, der = 0):
        self.name=name
        "Name of the true curve function"
        
        self._z = []
        """List of all evaluations of z(t) and its derivatives. """
        self._der = -1

        self.n = n
        self.der = der 



    def __call__(self,der=0):
        res = self._call(der=der)
        if res.ndim != 2 or res.shape[0] != 2:
            raise RuntimeError(Errors.runtime_error(f"Calling the GenCurve {self} did not construct a array of Nx2 dimension!"))
        return res
    
    def _call(self,der=0):
        raise NotImplementedError
    
    @property
    def der(self):
        """number of derivatives to compute"""
        return self._der

    @der.setter
    def der(self,der_new):
        if not isinstance(der_new,int) or der_new >3:
            raise ValueError(Errors.value_error("The number of derivatives needs to be an integer between 0 and 3!"))
        if self.der < der_new:
            for i in range(self.der+1,der_new+1):
                self._z.append(self(i))
                self._der += 1

    @property    
    def n(self):
        """number of evaluation points"""
        return self._n
    
    @n.setter
    def n(self,n_new):
        if not isinstance(n_new,int) or n_new <= 0:
            raise ValueError(Errors.value_error("The number of discretization points of the GenCurve needs to be a positive integer!"))
        self.t = 2*np.pi*np.linspace(0, n_new-1, n_new)/n_new
        self._n = n_new
        for i in range(0,self.der+1):
            self._z[i]= self(i)

    @property
    def z(self):
        """Values of z(t) at equidistant grid"""
        if self.der >= 0:
            return self._z[0]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation the self.der >=0 please change that!",self,"z"))
    
    @property
    def zp(self):
        """Values of z(t) its first derivatives at equidistant grid"""
        if self.der >= 1:
            return self._z[1]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the first derivative the self.der >=1 please change that!",self,"zp"))
    
    @property
    def zpabs(self):
        if self.zp is not None:
            return np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
    @property
    def normal(self):
        if self.zp is not None:
            return np.append(self.zp[1,:], -self.zp[0,:]).reshape((2, self.n))
        
    @property
    def zpp(self):
        """Values of z(t) its second derivatives at equidistant grid"""
        if self.der >= 2:
            return self._z[2]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the second derivative the self.der >=2 please change that!",self,"zpp"))

    @property
    def zppp(self):
        """Values of z(t) its third derivatives at equidistant grid"""
        if self.der >= 3:
            return self._z[3]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the third derivative the self.der >=3 please change that!",self,"zppp"))


class kite(GenCurve):
    r"""Subclass of the `GenCurve` that gives a kite form. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    der : int, optional
        Number of derivatives to initially compute. Default: 0
    """
    def __init__(self, n, der = 0):
        super().__init__("kite",n,der=der)

    def _call(self, der=0):
        if der==0:
            return np.append(np.cos(self.t)+0.65*np.cos(2*self.t)-0.65,   1.5*np.sin(self.t)).reshape(2, self.n)
        elif der==1:
            return np.append(-np.sin(self.t)-1.3*np.sin(2*self.t)    ,    1.5*np.cos(self.t)).reshape(2, self.n)
        elif der==2:
            return np.append(-np.cos(self.t)-2.6*np.cos(2*self.t)    ,   -1.5*np.sin(self.t)).reshape(2, self.n)
        elif der==3:
            return np.append(np.sin(self.t)+5.2*np.sin(2*self.t)     ,   -1.5*np.cos(self.t)).reshape(2, self.n)
        else:
            raise ValueError('derivative not implemented')


class StarCurve(GenCurve):
    r"""Base class for radial curve in :math:`R^2` 
    parameterized by 

    .. math::
        z(t) = q(t)*[cos(t);sin(t)] 0<=t<=2pi

    with a positive, :math:`2\pi`\-periodic function :math:`q`\. 

    Subclasses should implement `_call` with the optional argument `der` 
    to determine which derivative to compute.

    After initializing the curve additional derivatives can be computed by 
    resetting the `der` property. The number of evaluation points can also be 
    reset by setting the 'n` property with some new number resulting in a recompute
    of all the evaluations. 

    Parameters
    ----------
    name : str 
        name of the curves
    n : int 
        number of discretization point
    der : int, optional
        number of derivatives to initially compute.
    """
    def __init__(self, name, n, der = 0):
        super().__init__(name,n,der=der)

    def __call__(self,der=0):
        res = self._call(der=der)
        if res.ndim != 1:
            raise RuntimeError(Errors.runtime_error(f"Calling the StarCurve {self} did not construct a array of one dimension!"))
        if der == 0:
            return np.array([res*np.cos(self.t),res*np.sin(self.t)])
        elif der == 1:
            cost = np.cos(self.t)
            sint = np.sin(self.t)
            return np.array([res*cost,res*sint]) + np.array([[0,-1],[1,0]])@self.z
        elif der == 2:
            cost = np.cos(self.t)
            sint = np.sin(self.t)
            return np.array([res*cost, res*sint]) + 2*np.array([[0,-1],[1,0]])@self.zp + self.z
        elif der == 3:
            cost = np.cos(self.t)
            sint = np.cos(self.t)
            return np.array([res*cost ,res*sint]) + 3*np.array([[0,-1],[1,0]])@self.zpp + 3 * self.zp + np.array([[0,1],[-1,0]])@self.z
        return res
    
    def _call(self,der=0):
        raise NotImplementedError
    
    @property
    def zpabs(self):
        r""":math:`|z'(t)|`"""
        if self.zp is not None:
            return np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
    @property
    def normal(self):
        r"""Outer normal vector(not normalized)"""
        if self.zp is not None:
            return np.append(self.zp[1,:], -self.zp[0,:]).reshape((2, self.n))

    def radial(self, n):
        t=2*np.pi*np.linspace(0, n-1, n)/n
        rad = eval(self.name)(t, 0)
        return rad

class peanut(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("peanut",n,der=der)

    def _call(self,der):
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

class round_rect(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("round_rect",n,der=der)

    def _call(self,der):
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


class apple(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("apple",n,der=der)

    def _call(self,der):
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


class three_lobes(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("three_lobes",n,der=der)

    def _call(self,der):
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


class pinched_ellipse(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("pinched_ellipse",n,der=der)

    def _call(self,der):
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


class smoothed_rectangle(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("smoothed_rectangle",n,der=der)

    def _call(self,der):
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


class nonsym_shape(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("nonsym_shape",n,der=der)

    def _call(self,der):
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


class circle(StarCurve):
    
    def __init__(self,n,der=0):
        super().__init__("circle",n,der=der)

    def _call(self,der):
        if der==0:
            return np.ones_like(self.t)
        else:
            return np.zeros_like(self.t)


class GenTrigSpc(UniformGridFcts):
    r"""Class for the `VectorSpaceBase` instance of `GenTrig` instances. 
    It is a space of vector-valued trigonometric polynomials. 
    The class provides method `bd_eval` which generates a curve `GenTrig` from a given coefficient (or sample) vector.  

    Parameters
    ----------
    n : int
        Number of coefficients of each of the cartesian components.
    """
    def __init__(self, n):
        if not isinstance(n, int,) or n<=0:
            raise TypeError(Errors.not_instance(n,int,add_info="The GenTrigSpc need n to be a positive integer!"))
        self.n = n
        super().__init__(np.linspace(0, 2*np.pi, n, endpoint=False),shape_codomain=(2,))

    def bd_eval(self, samples, nvals=None, nderivs=0):
        r"""Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `GenTrig`.
        
        Parameters
        ----------
        samples : array-like
            samples from which to generate the curve
        nvals : int 
            Number of points to evaluate the parameterization on
        nderivs : int
            Number of derivatives to compute 
        """
        gentrig=GenTrig(samples, nvals, nderivs)
        
        return gentrig
    
class GenTrig:
    r"""The class GenTrig describes boundaries of domains in R^2 which are
    parameterized by

    .. math::
        z(t) = [z_1(t), z_2(t)]      0<=t<=2pi

     where z_1 and z_2 are real trigonometric polynomials with N coefficients.
     z and its derivatives are sampled at n equidistant points.
     
     Parameters
     ----------
     samples : array-like
        Equidistant (in parameter space!) samples of the cartesian components of the parameterization of the curve 
     nvals : int 
        Number of points at which to evaluate the curve
     nderivs : int
        Number of derivatives to compute 
     """

    def __init__(self, samples, nvals, nderivs):
        if len(samples.shape)!=2 or not np.issubdtype(samples.dtype,np.floating):
            raise ValueError('samples must be a 2xN array of real numbers')
        self.samples = samples
        """Equidistant samples of the trigonometric polynomials""" 
        N = self.samples.shape[1]
        self.nvals = nvals
        self.nderivs = nderivs
        
        """Evaluates the first der derivatives of the parametrization of
        the curve on n equidistant points"""

        coeffhat = np.vstack((trig_interpolate(samples[0,:], self.nvals), \
                             trig_interpolate(samples[1,:], self.nvals)))
        self.z = np.vstack((np.real(np.fft.ifft(np.fft.fftshift(coeffhat[0,:]))), \
                           np.real(np.fft.ifft(np.fft.fftshift(coeffhat[1,:])))))
        
        if self.nderivs>=1:
            """Array indices"""
            self.zp = np.vstack((np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))*coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))*coeffhat[1,:])))))
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            """Outer normal vector"""
            self.normal = np.vstack((self.zp[1,:], -self.zp[0,:]))

        if self.nderivs>=2:
            """Array indices"""
            self.zpp = np.vstack((np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**2 * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**2 * coeffhat[1,:])))))
        if self.nderivs>=3:
            self.zppp = np.vstack((np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**3 * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**3 * coeffhat[1,:])))))
        
        if self.nderivs>3:
            raise ValueError('only derivatives up to order 3 implemented')

    def der_normal(self, h):
        N = h.shape[1]
        n = self.z.shape[1]

        if N == n:
            hn = h

        else:
            h_hat = np.array([trig_interpolate(h[0,:], n),\
                              trig_interpolate(h[1,:], n)])

            hn = np.array([np.real(np.fft.ifft(np.fft.fftshift(h_hat[0,:]))),\
                           np.real(np.fft.ifft(np.fft.fftshift(h_hat[1,:])))])

        der=np.sum(hn*self.normal,0)/self.zpabs
        return der

    def adjoint_der_normal(self, g):

        N = self.coeff.shape[1]
        n = int(len(g))
        
        adj_n=np.array([g/self.zpabs,g/self.zpabs])*self.normal
    
        if N == n:
            adj = np.array([adj_n[0,:],\
                             adj_n[1,:].transpose()])
        else:
            adj_hat = np.array([trig_interpolate(adj_n[0, :], N), \
                                trig_interpolate(adj_n[1,:], N)])*n/N
            adj_hat=adj_hat.T 
         
            adj = np.append(np.array([np.fft.ifft(np.fft.fftshift(adj_hat[:,0]))]),\
                            np.array([np.fft.ifft(np.fft.fftshift(adj_hat[:,1]))]))
            
        return adj.real
        
    def arc_length_der(self, h):
            n = int(len(self.zpabs))
            dhds = np.fft.ifft(np.fft.fftshift((1j*np.linspace(-n/2, n/2-1, n)).transpose()*trig_interpolate(
                h, n)))/self.zpabs.transpose()
            return dhds

    def coeff_to_curve(self, coeff, n):
        N = int(len(coeff)/2)

        val = coeff[N:2*N]
        val1 = coeff[0:N]
        
        coeffhat = np.array([trig_interpolate(val1, N),\
                    trig_interpolate(val, N)])
        
        pts = np.array([np.real(np.fft.ifft(np.fft.fftshift(coeffhat[0,:]))), \
            np.real(np.fft.ifft(np.fft.fftshift(coeffhat[1,:])))])
        
        return pts

class StarTrigRadialFcts(UniformGridFcts):
    r"""Class for VectorSpaceBase` instance of `StarTrigCurve` instances. It provides 
    the method `eval_curve` which gives a curve `StarTrigCurve`.  

    The space consists of star-shaped curves with radial functions given by real trigonometric 
    polynomials of some maximal degree. These trigonometric polynomials are determined by their values on 
    an equidistant grid. 

    Parameters
    ----------
    n : int
        Dimension of the space of trigonometric polynomials 
    """
    def __init__(self, n):
        if not isinstance(n, int) or n<=0:
            raise TypeError(Errors.not_instance(n,int,add_info="The StarTrigD need n to be a positive integer!"))
        super().__init__(np.linspace(0, 2*np.pi, n, endpoint=False))

    def eval_curve(self, coeffs, nvals=None, nderivs=0):
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarTrigCurve`.
        
        Parameters
        ----------
        coeffs : array-like
            Coefficients for which to evaluate the curve
        nvals : int, optional
            Number of points to evaluate on, Defaults : None
        nderivs : int, optional
            Number of derivatives to compute , Defaults : 0
        """
        return StarTrigCurve(self, coeffs, nvals, nderivs)

    def sample(self, f):
        return np.asarray(
            np.broadcast_to(f(np.linspace(0, 2*np.pi, self.size, endpoint=False)), self.shape),
            dtype=self.dtype
        )

class StarTrigCurve: 
    r"""A class representing star shaped 2d curves with radial function parametrized in a
    trigonometric basis. Should usually be instantiated via `StarTrigRadialFcts.eval_curve`.

    Parameters
    ----------
    vecsp : StarTrigRadialFcts
        The underlying vector space.
    coeffs : array-like
        The coefficient array of the radial function.
    nvals : int, optional
        How many points on the curve to compute. The points will be at equispaced angles in
        `[0, 2pi)`. If omitted, the number of points will match the number of `coeffs`.
    nderivs : int, optional
        How many derivatives to compute. At most 3 derivatives are implemented.
    """

    def __init__(self, vecsp, values, nvals=None, nderivs=0):
        if not isinstance(nderivs, int) or nderivs <0 or nderivs >3:
            raise ValueError(Errors.value_error(f"The number of derivative in StarTrigCurve needs to be an integer between 0 and 3"))
        self.vecsp = vecsp
        """The vector space."""
        self.values = values
        """The coefficients."""
        self.nvals = nvals or self.vecsp.size
        """The number of computed values."""
        self.nderivs = nderivs
        """The number of computed derivatives."""

        self._frqs = 1j*np.arange(self.vecsp.size // 2 + 1)
        self.radius = (self.nvals / self.vecsp.size) * np.fft.irfft(
            (self._frqs ** np.arange(self.nderivs + 1)[:, np.newaxis])*np.fft.rfft(values),
            self.nvals,
            axis=1
        )
        """The values of the radial function and its derivatives, shaped `(nderivs + 1, nvals)`."""

        t = np.linspace(0, 2 * np.pi, self.nvals, endpoint=False)
        cost = np.cos(t)
        sint = np.sin(t)

        self.curve = np.zeros((self.nderivs + 1, 2, self.nvals))
        """The points on the curve and its derivatives, shaped `(nderivs + 1, 2, nvals)`."""

        binom = np.ones(self.nderivs + 1, dtype=int)
        for n in range(self.nderivs + 1):
            binom[1:n] += binom[:n-1]
            aux = binom[:n+1, np.newaxis] * self.radius[n::-1]
            even = np.sum(aux[::4], axis=0) - np.sum(aux[2::4], axis=0)
            odd = np.sum(aux[1::4], axis=0) - np.sum(aux[3::4], axis=0)
            self.curve[n, 0] = even * cost - odd * sint
            self.curve[n, 1] = even * sint + odd * cost

        if self.nderivs == 0:
            return

        self.normal = np.stack([self.curve[1, 1], -self.curve[1, 1]])
        """The (unnormalized) outer normal vector as `(2, nvals)` array. Its norm identical to that
        of the tangent vector `curve[1]`."""
        self.tangent_norm = np.linalg.norm(self.normal, axis=0)
        """The absolute values of the tangent and normal vectors as `(nvals,)` array."""

    def derivative(self, h):
        return (self.nvals / self.vecsp.size) * np.fft.irfft(
            np.fft.rfft(h), self.nvals
        )

    def adjoint(self, g):
        return (self.nvals / self.vecsp.size) * adjoint_rfft(
            adjoint_irfft(g, self.vecsp.size // 2 + 1),
            self.vecsp.size
        )

    def der_normal(self, h):
        return (self.radius[0] / self.tangent_norm) * self.derivative(h)

    def adjoint_der_normal(self, g):
        return self.adjoint((self.radius[0] / self.tangent_norm)*g)

    def arc_length_der(self, h):
        return (self.nvals / self.vecsp.size) * np.fft.irfft(
            self._frqs * np.fft.rfft(h), self.nvals
        ) / self.tangent_norm

def trig_interpolate(val, n):
    """Computes `n` Fourier coeffients to the point values given by `val`
    such that `ifft(fftshift(coeffs))` is an interpolation of `val`."""
    n_val = len(val)
    coeff_val = np.fft.fft(val)
    if n == n_val:
        return np.fft.ifftshift(coeff_val)
    elif n > n_val:
        coeffs = np.zeros(n, dtype=complex)
        coeffs[:n_val // 2] = coeff_val[:n_val // 2]
        coeffs[-(n_val // 2) - n_val % 2 + 1:] = coeff_val[-(n_val // 2) -n_val % 2 + 1:]
        coeffs[n_val // 2] = 0.5 * coeff_val[n_val // 2]
        coeffs[-(n_val // 2) - n_val % 2] = 0.5 * coeff_val[n_val // 2]
        return n / n_val * np.fft.ifftshift(coeffs)
    else:
        coeffs = np.zeros(n, dtype=complex)
        coeffs[:n // 2] = coeff_val[:n // 2]
        coeffs[-(n // 2) - n % 2 + 1:] = coeff_val[-(n // 2)- n % 2 + 1:]
        coeffs[n // 2] = 0.5 * (coeff_val[n // 2] + coeff_val[-(n // 2) - n % 2])
        return n / n_val * np.fft.ifftshift(coeffs)

def adjoint_rfft(y, size, n=None):
   
    if n is None:
        n = size
    if n // 2 + 1 != y.size:
        raise ValueError(Errors.value_error(f"The size of y, y.size = {y.size}, for the adjoint_rfft is not n//2+1 where n = {n}"))

    result = np.fft.irfft(y, n)
    result *= n / 2
    result += y[0].real / 2
    if n % 2 == 0:
        aux = y[-1].real / 2
        result[::2] += aux
        result[1::2] -= aux

    if n == size:
        return result
    elif size < n:
        return result[:size]
    else:
        aux = np.zeros(size, dtype=result.dtype)
        aux[:n] = result
        return aux

def adjoint_irfft(y, size=None):
    r"""Compute the adjoint of `numpy.fft.irfft`\. More concretely, the adjoint of

    .. math::
        x \mapsto \mathrm{irfft}(x, n)

    is

    .. math::
        y \mapsto \mathrm{adjoint_irfft}(y, x.size)

    Since the size of `x` can not be determined from `y`\, it needs to be given explicitly. The
    parameter `n`, however, is determined as the output size of `irfft`\, so it does not not need to
    be specified for the adjoint.

    Parameters
    ----------
    y : array-like
        The input array.
    size : int, optional
        The size of the output, i.e. the size of the original input to `irfft`. If omitted,
        `x.size // 2 + 1` will be used, i.e. we assume the `irfft` is inverse to a plain `rfft(x)`,
        without additional padding or truncation.

    Returns
    -------
    array of shape (size,)
    """

    if size is None:
        size = y.size // 2 + 1
    
    result = np.fft.rfft(y)
    result[0] -= np.sum(y) / 2
    if y.size % 2 == 0:
        result[-1] -= (np.sum(y[::2]) - np.sum(y[1::2])) / 2
    result *= 2 / y.size
   
    if size == result.size:
        return result
    elif size < result.size:
        return result[:size]
    else:
        aux = np.zeros(size, dtype=result.dtype)
        aux[:result.size] = result
        return aux
