import numpy as np

from regpy.vecsps import UniformGridFcts

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
        assert res.ndim == 2 and res.shape[0] == 2
        return res
    
    def _call(self,der=0):
        raise NotImplementedError
    
    @property
    def der(self):
        """number of derivatives to compute"""
        return self._der

    @der.setter
    def der(self,der_new):
        assert isinstance(der_new,int) and der_new <=3
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
        assert isinstance(n_new,int)
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
            raise ValueError
    
    @property
    def zp(self):
        """Values of z(t) its first derivatives at equidistant grid"""
        if self.der >= 1:
            return self._z[1]
        else:
            raise ValueError
    
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
            raise ValueError

    @property
    def zppp(self):
        """Values of z(t) its third derivatives at equidistant grid"""
        if self.der >= 3:
            return self._z[3]
        else:
            raise ValueError


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
        assert res.ndim == 1
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


class GenTrigDiscr(UniformGridFcts):
    r"""Class for the `VectorSpace` instance of `GenTrig` instances. It provides method `bd_eval` which 
    gives evaluates a curve `GenTrig` by name.  

    Parameters
    ----------
    n : int
        Number of discretization points. 
    """
    def __init__(self, n):
        assert isinstance(n, int)
        self.n = n
        super().__init__(np.linspace(0, 2*np.pi, n, endpoint=False))

    def bd_eval(self, coeffs, nvals=None, nderivs=0):
        r"""Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `GenTrig`.
        
        Parameters
        ----------
        coeffs : array-like
            Coefficients for which to evaluate the curve
        nvals : int 
            Number of points to evaluate on
        nderivs : int
            Number of derivatives to compute 
        """
        gentrig=GenTrig(coeffs, nvals, nderivs)
        self.z=gentrig.z
        self.zpabs=gentrig.zpabs
        self.zp=gentrig.zp
        self.zpp=gentrig.zpp
        self.zppp=gentrig.zppp
        self.normal=gentrig.normal
        self.der_normal=gentrig.der_normal
        self.adjoint_der_normal=gentrig.adjoint_der_normal
        
        return gentrig
    
class GenTrig:
    r"""The class GenTrig describes boundaries of domains in R^2 which are
    parameterized by

    .. math::
        z(t) = [z_1(t), z_2(t)]      0<=t<=2pi

     where z_1 and z_2 are trigonometric polynomials with N coefficient.
     Here N must be even, so the highest order monomial is cos(t*N/2),
     but sin(t*N/2) does not occur.
     z and its derivatives are sampled at n equidistant points.
     Application of the Gramian matrix and its inverse w.r.t. the
     Sobolev norm ||z||_{H^s} are implemented.
     
     Parameters
     ----------
     coeffs : array-like
        Coefficients for which to evaluate the curve
     nvals : int 
        Number of points to evaluate on
     nderivs : int
        Number of derivatives to compute 
     """

    def __init__(self, coeffs, nvals, nderivs):
        self.coeff = coeffs
        """Coefficients of the trigonometric polynomials""" 
        self.nvals = nvals
        self.nderivs = nderivs
        
        """Evaluates the first der derivatives of the parametrization of
        the curve on n equidistant time points"""
        
        N = int(len(self.coeff)/2)
        val = self.coeff[N:2*N]
        val1 = self.coeff[0:N]

        coeffhat = np.append(trig_interpolate(val1, self.nvals), \
                             trig_interpolate(val, self.nvals)).reshape(2, self.nvals)
        self.z = np.append(np.real(np.fft.ifft(np.fft.fftshift(coeffhat[0,:]))), \
            np.real(np.fft.ifft(np.fft.fftshift(coeffhat[1,:])))).reshape(2, coeffhat[0,:].shape[0])
        
        if self.nderivs>=1:
            """Array indices"""
            self.zp = np.append(np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))*coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))*coeffhat[1,:])))).reshape(2, coeffhat[0,:].shape[0])
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            """Outer normal vector"""
            self.normal = np.append(self.zp[1,:], -self.zp[0,:]).reshape(2, self.zp[0, :].shape[0])

        if self.nderivs>=2:
            """Array indices"""
            self.zpp = np.append(np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**2 * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**2 * coeffhat[1,:])))).reshape(2, coeffhat[0, :].shape[0])

        if self.nderivs>=3:
            self.zppp = np.append(np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**3 * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2-1, self.nvals))**3 * coeffhat[1,:])))).reshape(2, coeffhat[0, :].shape[0])
        
        if self.nderivs>3:
            raise ValueError('only derivatives up to order 3 implemented')

    def der_normal(self, h):
        N = int(len(h)/2)
        n = self.z.shape[1]

        if N == n:
            hn = np.array([h[0:n],\
                           h[n:2*n]])

        else:
            val = h[N:2*N]
            val1 = h[0:N]  

            h_hat = np.array([trig_interpolate(val1, n),\
                     trig_interpolate(val, n)])

            hn = np.array([np.real(np.fft.ifft(np.fft.fftshift(h_hat[0,:]))),\
                np.real(np.fft.ifft(np.fft.fftshift(h_hat[1,:])))])

        der=np.sum(hn*self.normal,0)/self.zpabs
        return der

    def adjoint_der_normal(self, g):

        N = int(len(self.coeff)/2)
        n = int(len(g))
        
        adj_n=np.array([g/self.zpabs,g/self.zpabs])*self.normal
    
        if N == n:
            adj = np.array([adj_n[0,:],\
                             adj_n[1,:].transpose()])
        else:
            val = adj_n[0, :]
            val1 = adj_n[1,:]
            adj_hat = np.array([trig_interpolate(val, N), \
                       trig_interpolate(val1, N)])*n/N
            
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

class StarTrigDiscr(UniformGridFcts):
    r"""Class for the `VectorSpace` instance of `StarTrigCurve` instances. It provides 
    method `eval_curve` which gives a curve `StarTrigCurve`.  

    Parameters
    ----------
    n : int
        Number of discretization points. 
    """
    def __init__(self, n):
        assert isinstance(n, int)
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
    trigonometric basis. Should usually be instantiated via `StarTrigDiscr.eval_curve`.

    Parameters
    ----------
    vecsp : StarTrigDiscr
        The underlying vector space.
    coeffs : array-like
        The coefficient array of the radial function.
    nvals : int, optional
        How many points on the curve to compute. The points will be at equispaced angles in
        `[0, 2pi)`. If omitted, the number of points will match the number of `coeffs`.
    nderivs : int, optional
        How many derivatives to compute. At most 3 derivatives are implemented.
    """

    def __init__(self, vecsp, coeffs, nvals=None, nderivs=0):
        assert isinstance(nderivs, int) and 0 <= nderivs <= 3
        self.vecsp = vecsp
        """The vector space."""
        self.coeffs = coeffs
        """The coefficients."""
        self.nvals = nvals or self.vecsp.size
        """The number of computed values."""
        self.nderivs = nderivs
        """The number of computed derivatives."""

        self._frqs = 1j*np.arange(self.vecsp.size // 2 + 1)
        self.radius = (self.nvals / self.vecsp.size) * np.fft.irfft(
            (self._frqs ** np.arange(self.nderivs + 1)[:, np.newaxis])*np.fft.rfft(coeffs),
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
    assert n // 2 + 1 == y.size

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
