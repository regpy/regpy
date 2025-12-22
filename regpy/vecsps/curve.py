import numpy as np
from regpy.util import Errors

from .numpy import UniformGridFcts

__all__ = ["GenCurve","kite","StarCurve","peanut","round_rect","apple","three_lobes","pinched_ellipse","smoothed_rectangle","nonsym_shape","circle","GenTrigSpc","GenTrig","StarTrigRadialFcts","StarTrigCurve"]

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
    name : str 
        name of the curves
    n : int 
        number of discretization point
    der : int, optional
        number of derivatives to initially compute.
    """
    
    def __init__(self, name, n,nderivs = 0):
        self.name=name
        "Name of the true curve function"
        
        self._z = []
        """List of all evaluations of z(t) and its derivatives. """
        self._nderivs = -1

        self.n = n
        self.nderivs = nderivs 



    def __call__(self,der=0):
        res = self._call(der=der)
        if res.ndim != 2 or res.shape[0] != 2:
            raise RuntimeError(Errors.runtime_error(f"Calling the GenCurve {self} did not construct a array of Nx2 dimension!"))
        return res
    
    def _call(self,der=0):
        raise NotImplementedError
    
    @property
    def nderivs(self):
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
    def n(self):
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
    def z(self):
        """Values of z(t) at equidistant grid of self.n points."""
        if self.nderivs >= 0:
            return self._z[0]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation the self.nderivs >=0 please change that!",self,"z"))
    
    @property
    def zp(self):
        """Values of z(t) its first derivatives at equidistant grid of self.n points."""
        if self.nderivs >= 1:
            return self._z[1]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the first derivative the self.nderivs >=1 please change that!",self,"zp"))
    
    @property
    def zpabs(self):
        """Absolute _call_samples |z'(t)| at equidistant grid of self.n points."""
        if self.zp is not None:
            return np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
    @property
    def normal(self):
        if self.zp is not None:
            return np.append(self.zp[1,:], -self.zp[0,:]).reshape((2, self.n))
        
    @property
    def zpp(self):
        """Values of z(t) its second derivatives at equidistant grid of self.n points."""
        if self.nderivs >= 2:
            return self._z[2]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the second derivative the self.nderivs >=2 please change that!",self,"zpp"))

    @property
    def zppp(self):
        """Values of z(t) its third derivatives at equidistant grid of self.n points."""
        if self.nderivs >= 3:
            return self._z[3]
        else:
            raise RuntimeError(Errors.runtime_error("To return the evaluation of the third derivative the self.nderivs >=3 please change that!",self,"zppp"))

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
    name : str 
        name of the curve
    n : int 
        number of discretization point
    nderivs : int, optional
        number of derivatives to initially compute.
    """
    def __init__(self, name, n, nderivs = 0):
        super().__init__(name,n,nderivs=nderivs)

    def __call__(self,der=0):
        res = self.radial(der=der)
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
    
    def radial(self,der=0):
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
            return np.vstack((self.zp[1,:], -self.zp[0,:]))

    def radial(self, n):
        t=np.linspace(0, 2*np.pi, n,endpoint=False)
        rad = eval(self.name)(t, 0)
        return rad
    
################################ special GenCurves and StarCurves ####################################

class kite(GenCurve):
    r"""Subclass of the `GenCurve` that gives a kite form. 

    Parameters
    ----------
    n : int
        number of evaluation points on the parameterized curve.
    nderivs : int, optional
        Number of derivatives to initially compute. Default: 0
    """
    def __init__(self, n, nderivs = 0):
        super().__init__("kite",n,nderivs=nderivs)

    def _call(self, der=0):
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



class peanut(StarCurve):
    
    def __init__(self,n,nderivs=0):
        super().__init__("peanut",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("round_rect",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("apple",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("three_lobes",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("pinched_ellipse",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("smoothed_rectangle",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("nonsym_shape",n,nderivs=nderivs)

    def radial(self,der):
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
    
    def __init__(self,n,nderivs=0):
        super().__init__("circle",n,nderivs=nderivs)

    def radial(self,der):
        if der==0:
            return np.ones_like(self.t)
        else:
            return np.zeros_like(self.t)



######################### parameterized curves ##############################

class GenTrigSpc(UniformGridFcts):
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
    def __init__(self, n_sample,n):
        if not isinstance(n_sample, int,) or n_sample<=0:
            raise TypeError(Errors.not_instance(n,int,add_info="The GenTrigSpc need n to be a positive integer!"))
        self.n_sample = n_sample
        self.n = n
        super().__init__(np.linspace(0, 2*np.pi, n_sample, endpoint=False),shape_codomain=(2,))

    def coeff2curve(self, coeff, nderivs=0):
        r"""Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `GenTrig`.
        
        Parameters
        ----------
        coeff : array-like
            samples from which to generate the curve
        nderivs : int
            Number of derivatives to compute 
        """
        gentrig=GenTrig(coeff, self, nderivs)
        
        return gentrig
    
    def circle(self, radius =1.,nderivs=0):
        t = np.linspace(0, 2*np.pi,self.n_sample,endpoint=False)
        return GenTrig(radius*np.vstack((np.cos(t), np.sin(t))).T,self,nderivs=0)

    def param_derivative(self,u):
        """
        Computes the derivative(s) of one or several complex periodic functions :math:`u:[0,2\pi] \to \mathbb{C}`,
        which are given by their radial_samples at self.nval equidistant point on :math:`[0,2\pi]` 
        
        Parameters:
        u: np.ndarray
            two-dimnensional complex array with first dimension self.nval 
        """
        from regpy.operators.convolution import Derivative        
        
        if not isinstance(u, np.ndarray) or not np.issubdtype(u.dtype,complex):
            raise TypeError(Errors.type_error('u must be complex np.ndarray.'))
        if not len(u.shape) in (1,2) or not u.shape[0]==self.n:
            raise ValueError(Errors.value_error(f'u must have two dimensions, the first one equal to self.n. Given shape: {u.shape}. n: {self.n}'))
                
        if not hasattr(self,'_complexBlockDerivative') or (self._complexBlockDerivative.domain.shape!=u.shape):
            der_domain = UniformGridFcts((0.,2*np.pi,self.n), periodic=True,dtype=complex,
                                         shape_codomain=(u.shape[1],) if len(u.shape) ==2 else () 
                                         )
            self._complexBlockDerivative = Derivative(der_domain,(1,))
        return self._complexBlockDerivative(u)

class GenTrig(GenCurve):
    r"""The class GenTrig describes boundaries of domains in R^2 which are
    parameterized by 

    .. math::
        z(t) = [z_1(t), z_2(t)]      0<=t<=2pi

     where z_1 and z_2 are real trigonometric polynomials with N coefficients.
     z and its derivatives are sampled at n equidistant points.
     
     Parameters
     ----------
     coeff : np.ndarray
        Equidistant (in parameter space!) samples of the cartesian components of the parameterization of the curve 
     spc : regpy.vecspc.curve.GenTrigSpc 
        Underlying curve space
     nderivs : int
        Number of derivatives to compute 
     """

    def __init__(self, coeff, spc, nderivs):
        if len(coeff.shape)!=2 or not np.issubdtype(coeff.dtype,np.floating):
            raise ValueError(Errors.value_error(f'coeff must be a 2xN array of real numbers. Got shape {coeff.shape} of type {coeff.dtype}.'))
        self.coeff = coeff
        """Equidistant samples of the trigonometric polynomials""" 
        if not isinstance(spc,GenTrigSpc):
            raise TypeError(Errors.type_error('spc must be a GenTrigSpc'))
        self.spc = spc
        self.coeffhat = np.vstack((trig_interpolate(coeff[:,0], spc.n), \
                                   trig_interpolate(coeff[:,1], spc.n))).T
        self._freq = 1j*np.linspace(-spc.n/2, spc.n/2-1, spc.n)
        super().__init__("GenTrig",spc.n,nderivs=nderivs) 
        
    def _call(self,der=0):
        return np.vstack((np.real(np.fft.ifft(np.fft.fftshift(self._freq**der *self.coeffhat[:,0]))), \
                np.real(np.fft.ifft(np.fft.fftshift(self._freq**der * self.coeffhat[:,1])))))

    def der_normal(self, h):
        """ If h is a perturbation of the self.sample, this function returns the normal component 
        of the resulting perturbation of self.z

        Parameters:
        -------
        h: np.ndarray
            perturbation of self.z_sample
        """
        
        if h not in self.spc:
            raise ValueError(Errors.not_in_vecsp(h,self.spc))
        n = self.spc.n

        if self.spc.n_sample == n:
            hn = h
        else:
            h_hat = np.array([trig_interpolate(h[:,0], n),\
                              trig_interpolate(h[:,1], n)])
            hn = np.array([np.real(np.fft.ifft(np.fft.fftshift(h_hat[0,:]))),\
                           np.real(np.fft.ifft(np.fft.fftshift(h_hat[1,:])))])

        return np.sum(hn*self.normal,0)/self.zpabs

    def adjoint_der_normal(self, g):
        """ adjoint of the linear mapping der_normal

        Paraameters:
        -----------------
        g: np.nd_array
        """
        n = self.spc.n
        n_sample = self.spc.n_sample    

        if not isinstance(g,np.ndarray) or not g.shape == (n,):
            raise ValueError('g must be a vector of length self.n.')

        adj_n=np.array([g/self.zpabs,g/self.zpabs])*self.normal
    
        if n_sample == n:
            adj = adj_n.T
        else:
            adj_hat = np.array([trig_interpolate(adj_n[0,:], n_sample), \
                                trig_interpolate(adj_n[1,:], n_sample)])*n/n_sample        
            adj = np.array([np.fft.ifft(np.fft.fftshift(adj_hat[0,:])),\
                            np.fft.ifft(np.fft.fftshift(adj_hat[1,:]))])
            
        return adj.T.real
        

    def arc_length_der(self, h):
        if len(h.shape)==1:
            return self.spc.param_derivative(h) / self.zpabs
        elif len(h.shape)==2:
            return self.spc.param_derivative(h) / self.zpabs[:,np.newaxis]
        else:
            raise ValueError(Errors.value_error('shape of h must have length 1 or 2.'))

class StarTrigRadialFcts(UniformGridFcts):
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
    def __init__(self, dim,n):
        if not isinstance(dim, int) or dim<=0:
            raise TypeError(Errors.not_instance(dim,int,add_info="StarTrigRadialFcts need dim to be a positive integer!"))
        if not isinstance(n, int) or n<=0:
            raise TypeError(Errors.not_instance(n,int,add_info="The StarTrigRadialFcts need n to be a positive integer!"))        
        self.n = n
        self.dim = dim
        super().__init__(np.linspace(0, 2*np.pi, dim, endpoint=False))

    def coeff2curve(self, coeff, nderivs=0):
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarTrigCurve`.
        
        Parameters
        ----------
        coeff : np.ndararray
            sample of the radial function at self.dim equidistant points
        nderivs : int, optional
            Number of derivatives to compute , Defaults : 0
        """
        return StarTrigCurve(self, coeff,  nderivs)

    def radialfct2curve(self, f,nderivs=0):
        coeff = f(np.linspace(0, 2*np.pi, self.dim, endpoint=False))
        return StarTrigCurve(self, coeff,  nderivs)
    
    def circle(self, radius=1.,nderivs=0):
        return StarTrigCurve(self, radius*self.ones(),nderivs)

class StarTrigCurve(StarCurve): 
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

    def __init__(self, vecsp, coeff, nderivs=0):
        if not isinstance(nderivs, int) or nderivs <0 or nderivs >3:
            raise ValueError(Errors.value_error(f"The number of derivative in StarTrigCurve needs to be an integer between 0 and 3"))
        self.vecsp = vecsp
        """The vector space."""
        self.coeff = coeff
        """The coefficients."""
        self.dim = len(coeff)

        self._frqs = 1j*np.arange(self.dim // 2 + 1)
        self._radial = (self.vecsp.n / self.dim) * np.fft.irfft(
            (self._frqs ** np.arange(nderivs + 1)[:, np.newaxis])*np.fft.rfft(coeff),
            self.vecsp.n,
            axis=1
        )
        """Sampled radial function and its derivatives, shaped `(nderivs + 1, nvals)`."""
        super().__init__('StarTrigCurve',self.vecsp.n,nderivs)

    def radial(self,der=0):
        if der>self._radial.shape[0]:
            return RuntimeError(f'Value of der {der} greater than self.nderivs {self.nderivs}. Initialize with larger value of nderivs!')
        return self._radial[der,:]

    def derivative(self, h):
        return (self.n / self.dim) * np.fft.irfft(
            np.fft.rfft(h), self.n
        )

    def adjoint(self, g):
        return (self.n / self.dim) * adjoint_rfft(
            adjoint_irfft(g, len(self.coeff) // 2 + 1),
            self.dim
        )

    def der_normal(self, h):
        return (self._radial[0,:] / self.zabs) * self.derivative(h)

    def adjoint_der_normal(self, g):
        return self.adjoint((self._radial[0,:] / self.zabs)*g)

    def arc_length_der(self, h):
        return (self.n / len(self.coeff)) * np.fft.irfft(
            self._frqs * np.fft.rfft(h), self.n
        ) / self.zpabs

def trig_interpolate(val, n):
    """Computes `n` Fourier coeffients to the point radial_samples given by `val`
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
