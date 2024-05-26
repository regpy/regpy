import numpy as np

from regpy.vecsps import UniformGridFcts

class GenCurveDiscr(UniformGridFcts):
    """Class for the `VectorSpace` instance of GenCurve instances. It provides method `bd_eval` which 
    gives the capability to evaluate a curve GenCurve by name.  

    Parameters
    ----------
    n : int
        number of discretization points 
    """
    def __init__(self, n):
        assert isinstance(n, int)
        self.n = n 
        super().__init__(np.linspace(0, 2*np.pi, n, endpoint=False))

    def bd_eval(self, name, der = 0):
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarCurve`.
        
        Parameters
        ----------
        name : str
            name of the method to evaluate as string
        der : int, optional
            Number of derivatives to compute , Defaults : 0
        """
        gencurve=GenCurve(name, self.n, der)
        self.z=gencurve.z
        self.zpabs=gencurve.zpabs
        self.zp=gencurve.zp
        self.zpp=gencurve.zpp
        self.zppp=gencurve.zppp 
        self.normal=gencurve.normal
        return gencurve

class GenCurve:
    r"""Parameterized smooth closed curve in R^2 without self-crossing
    parametrization by function z(t), 0<=t<=2*pi (counter-clockwise). 
    Note \(z(t)\) must return two values [x(t),z(t)]

    Parameters
    ----------
    name : str 
        name of the curves
    n : int 
        number of discretization point
    der : int
        up to which derivative to compute, Defaults: 0

    Raises
    ------
    ValueError
        If der > 3, because only Derivative up to order 3 is implemented
    """
    
    def __init__(self, name, n, der = 0):
        self.name=name
        self.z=None
        self.zp=None
        self.zpp=None
        self.zppp=None
        """Values of z(t) and its derivatives at equidistant grid"""
        self.zpabs=None   
        """|z'(t)|"""
        self.normal=None 
        """Outer normal vector(not normalized)"""

        t=2*np.pi*np.linspace(0, n-1, n)/n
        self.z = eval(self.name)(t,0)

        if der>=1:
            self.zp = eval(self.name)(t,1)
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            self.normal = np.append(self.zp[1,:],
                -self.zp[0,:]).reshape(2, n)

        if der>=2:
            self.zpp = eval(self.name)(t,2)

        if der>=3:
            self.zppp = eval(self.name)(t,3)

        if der>3:
            raise ValueError('only derivatives up to order 3 implemented')


class StarCurveDiscr(UniformGridFcts):
    """Class for the `VectorSpace` instance of `StarCurve` instances. It provides method `bd_eval` which 
    gives evaluates a curve `StarCurve` by name.  

    Parameters
    ----------
    n : int
        Number of discretization points. 
    """
    def __init__(self, n):
        assert isinstance(n, int)
        self.n = n
        super().__init__(np.linspace(0, 2*np.pi, n, endpoint=False))

    def bd_eval(self, name, der=0):
        """Compute a curve for the given coefficients. All parameters will be passed to the
        constructor of `StarCurve`.
        
        Parameters
        ----------
        name : str
            name of the method to evaluate
        der : int, optional
            Number of derivatives to compute , Defaults : 0
        """
        starcurve=StarCurve(name, self.n, der)
        self.z=starcurve.z
        self.zpabs=starcurve.zpabs
        self.zp=starcurve.zp
        self.zpp=starcurve.zpp
        self.zppp=starcurve.zpp
        self.normal=starcurve.normal
        return starcurve

class StarCurve:
    r"""Radial curve parameterized by 
    \[
      z(t) = q(t)*[cos(t);sin(t)] 0<=t<=2pi
    \]
     with a positive, 2pi-periodic function q. 
     
    Parameters
    ----------
    name : str 
        name of the curves
    n : int 
        number of discretization point
    der : int
        up to which derivative to compute, Defaults: 0

    Raises
    ------
    ValueError
        If der > 3, because only Derivative up to order 3 is implemented
    """
    def __init__(self, name, n, der):
      
        self.name=name

        t=2*np.pi*np.linspace(0, n-1, n)/n  
        cost = np.cos(t)
        sint = np.sin(t)

        self.q = np.zeros((der+1,n))
        """The first row of q contains values of q(t) at equidistant points
        the second row values of q', the third row of q'' and so on"""

        for j in range(0, der+1):
            self.q[j, :]=eval(self.name)(t, j)
        q=self.q
        self.z = np.append(q[0, :]*cost,\
            q[0,:]*sint).reshape((2, n))
            
        if der>=1:
            self.zp = np.append(q[1,:]*cost - q[0,:]*sint,\
                q[1,:]*sint + q[0,:]*cost).reshape((2, n))
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            self.normal = np.append(self.zp[1,:],
                -self.zp[0,:]).reshape((2, n))
            """Outer normal vector"""

        if der>=2:
            self.zpp = np.append(q[2,:]*cost - 2*q[1,:]*sint - q[0,:]*cost,\
                q[2,:]*sint + 2*q[1,:]*cost - q[0,:]*sint).reshape((2, n))

        if der>=3:
            self.zppp = np.append(q[3,:]*cost - 3*q[2,:]*sint - 3*q[1,:]*cost + q[0,:]*sint,\
                q[3,:]*sint + 3*q[2,:]*cost - 3*q[1,:]*sint - q[0,:]*cost).reshape((2,n))
                
        if der>3:
            raise ValueError('only derivatives up to order 3 implemented')

    def radial(self, n):
        t=2*np.pi*np.linspace(0, n-1, n)/n
        rad = eval(self.name)(t, 0)
        return rad

class GenTrigDiscr(UniformGridFcts):
    """Class for the `VectorSpace` instance of `GenTrig` instances. It provides method `bd_eval` which 
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
        """Compute a curve for the given coefficients. All parameters will be passed to the
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
     \[
          z(t) = [z_1(t), z_2(t)]      0<=t<=2pi
     \]
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
            self.zppp = np.append(np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2, self.nvals))**3 * coeffhat[0,:]))), \
                np.real(np.fft.ifft(np.fft.fftshift((1j*np.linspace(-self.nvals/2, self.nvals/2, self.nvals))**3 * coeffhat[1,:])))).reshape(2, coeffhat[1, :].shape[0])
        
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
    """Class for the `VectorSpace` instance of `StarTrigCurve` instances. It provides method `eval_curve` which 
    gives evaluates a curve `StarTrigCurve` by name.  

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
    """A class representing star shaped 2d curves with radial function parametrized in a
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
    if n % 2 != 0:
        ValueError('n should be even')
    N = len(val)
    coeffhat = np.fft.fft(val)
    coeffs = np.zeros(n, dtype=complex)
    if n >= N:
        coeffs[:N // 2] = coeffhat[:N // 2]
        coeffs[-(N // 2) + 1:] = coeffhat[N // 2 + 1:]
        if n > N:
            coeffs[N // 2] = 0.5 * coeffhat[N // 2]
            coeffs[-(N // 2)] = 0.5 * coeffhat[N // 2]
        else:
            coeffs[N // 2] = coeffhat[N // 2]
    else:
        coeffs[:n // 2] = coeffhat[:n // 2]
        coeffs[n // 2 + 1:] = coeffhat[-(n // 2) + 1:]
        coeffs[n // 2] = 0.5 * (coeffhat[n // 2] + coeffhat[-(n // 2)])
    coeffs = n / N * np.fft.ifftshift(coeffs)
    return coeffs

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
    """Compute the adjoint of `numpy.fft.irfft`. More concretely, the adjoint of

        x |-> irfft(x, n)

    is

        y |-> adjoint_irfft(y, x.size)

    Since the size of `x` can not be determined from `y`, it needs to be given explicitly. The
    parameter `n`, however, is determined as the output size of `irfft`, so it does not not need to
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

def peanut(t,der):
      res=np.zeros(t.shape[0])
      if der==0:
        res = 1./2.*(3*np.cos(t)**2+1)**(1./2)
      elif der==1:
        res = -3./2./(4.*np.cos(t)**2+np.sin(t)**2)**(1./2)*np.cos(t)*np.sin(t)
      elif der==2:
        res = -3./2*(3.*np.cos(t)**4+2.*np.cos(t)**2-1)/(3*np.cos(t)**2+1)**(3./2)
      elif der==3:
        res = 3./2.*np.cos(t)*np.sin(t)*(9.*np.cos(t)**4+6*np.cos(t)**2+13)/(3*np.cos(t)**2+1)**(5./2)
      else:
        raise ValueError('derivative not implemented')
      return res

def round_rect(t,der):
      co = 2/3
      if der==0:
        res = (np.sin(t)**10 + (co*np.cos(t))**10)**(-0.1)
      elif der==1:
        res = -1/10/(np.sin(t)**10+co**10*np.cos(t)**10)**(11/10)*(10*np.sin(t)**9*np.cos(t)-10*co**10*np.cos(t)**9*np.sin(t))
      elif der==2:
        res = 11/100/(np.sin(t)**10+co**10*np.cos(t)**10)**(21/10)*(10*np.sin(t)**9*np.cos(t)-10*co**10*np.cos(t)**9*np.sin(t)) \
            **2-1/10/(np.sin(t)**10+co**10*np.cos(t)**10)**(11/10)*(90*np.sin(t)**8*np.cos(t)**2-10*np.sin(t)**10+90*co**10 \
            *np.cos(t)**8*np.sin(t)**2-10*co**10*np.cos(t)**10)
      elif der==3:
        res = -231/1000/(np.sin(t)**10+co**10*np.cos(t)**10)**(31/10)**(10*np.sin(t)**9*np.cos(t)-10*co**10*np.cos(t)**9*np.sin(t))**3+33 \
            /100/(np.sin(t)**10+co**10*np.cos(t)**10)**(21/10)*(10*np.sin(t)**9*np.cos(t)-10*co**10*np.cos(t)**9*np.sin(t)) \
            *(90*np.sin(t)**8*np.cos(t)**2-10*np.sin(t)**10+90*co**10*np.cos(t)**8*np.sin(t)**2-10*co**10*np.cos(t)**10)-1/10 \
            /(np.sin(t)**10+co**10*np.cos(t)**10)**(11/10)*(720*np.sin(t)**7*np.cos(t)**3-280*np.sin(t)**9*np.cos(t)-720*co**10 \
            *np.cos(t)**7*np.sin(t)**3+280*co**10*np.cos(t)**9*np.sin(t))
      else:
        raise ValueError('derivative not implemented')
      return res

def apple(t, der):
      res=np.zeros(t.shape[0])
      if der==0:
        res = (0.5+0.4*np.cos(t)+0.1*np.sin(2*t))/(1+0.7*np.cos(t))
      elif der==1:
        res = (-2/5*np.sin(t)+1/5*np.cos(2*t))/(1+7/10*np.cos(t))+7/10*(1/2+2/5*np.cos(t)+1/10*np.sin(2*t))/(1+7/10*np.cos(t))**2*np.sin(t)
      elif der==2:
        res = (-2/5*np.cos(t)-2/5*np.sin(2*t))/(1+7/10*np.cos(t))+7/5*(-2/5*np.sin(t)+1/5*np.cos(2*t))/(1+7/10*np.cos(t)) \
            **2*np.sin(t)+49/50*(1/2+2/5*np.cos(t)+1/10*np.sin(2*t))/(1+7/10*np.cos(t))**3*np.sin(t)**2+7/10*(1/2+2/5  \
            *np.cos(t)+1/10*np.sin(2*t))/(1+7/10*np.cos(t))**2*np.cos(t)
      elif der==3:
        res = (2/5*np.sin(t)-4/5*np.cos(2*t))/(1+7/10*np.cos(t))+21/10*(-2/5*np.cos(t)-2/5*np.sin(2*t))/(1+7/10*np.cos(t))**2 \
            *np.sin(t)+147/50*(-2/5*np.sin(t)+1/5*np.cos(2*t))/(1+7/10*np.cos(t))**3*np.sin(t)**2+21/10*(-2/5*np.sin(t)+1/5 \
            *np.cos(2*t))/(1+7/10*np.cos(t))**2*np.cos(t)+1029/500*(1/2+2/5*np.cos(t)+1/10*np.sin(2*t))/(1+7/10*np.cos(t)) \
            **4*np.sin(t)**3+147/50*(1/2+2/5*np.cos(t)+1/10*np.sin(2*t))/(1+7/10*np.cos(t))**3*np.sin(t)*np.cos(t)-7/10 \
            *(1/2+2/5*np.cos(t)+1/10*np.sin(2*t))/(1+7/10*np.cos(t))**2*np.sin(t)
      else:
        raise ValueError('derivative not implemented')
      return res

def three_lobes(t, der):
     res=np.zeros(t.shape[0])
     if der==0:
        res = 0.5 + 0.25*np.exp(-np.sin(3*t)) - 0.1*np.sin(t)
     elif der==1:
        res = -3/4*np.cos(3*t)*np.exp(-np.sin(3*t))-1/10*np.cos(t)
     elif der==2:
        res = 9/4*np.sin(3*t)*np.exp(-np.sin(3*t))+9/4*np.cos(3*t)**2*np.exp(-np.sin(3*t))+1/10*np.sin(t)
     elif der==3:
        res = 27/4*np.cos(3*t)*np.exp(-np.sin(3*t))-81/4*np.sin(3*t)*np.cos(3*t)*np.exp(-np.sin(3*t))-27/4*np.cos(3*t)**3*np.exp(-np.sin(3*t))+1/10*np.cos(t)
     else:
        raise ValueError('derivative not implemented')
     return res

def pinched_ellipse(t, der):
     res=np.zeros(t.shape[0])
     if der==0:
       res = 3/2*np.sqrt(1/4*np.cos(t)**2 + np.sin(t)**2)
     elif der==1:
       res = 9/4/(-3*np.cos(t)**2+4)**(1/2)*np.cos(t)*np.sin(t)
     elif der==2:
       res = 9/4*(3*np.cos(t)**4-8*np.cos(t)**2+4)/(3*np.cos(t)**2-4)/(-3*np.cos(t)**2+4)**(1/2)
     elif der==3:
        res = -9/4*np.cos(t)*np.sin(t)*(9*np.cos(t)**4-24*np.cos(t)**2+28)/(3*np.cos(t)**2-4)**2/(-3*np.cos(t)**2+4)**(1/2)
     else:
        raise ValueError('derivative not implemented')
     return res

def smoothed_rectangle(t, der):
     res=np.zeros(t.shape[0])
     if der==0:
        res = (np.cos(t)**10 +2/3*np.sin(t)**10)**(-1/10)
     elif der==1:
        res = -1/10/(np.cos(t)**10+2/3*np.sin(t)**10)**(11/10)*(-10*np.cos(t)**9*np.sin(t)+20/3*np.sin(t)**9*np.cos(t))
     elif der==2:
        res = 11/100/(np.cos(t)**10+2/3*np.sin(t)**10)**(21/10)*(-10*np.cos(t)**9*np.sin(t)+20/3*np.sin(t)**9*np.cos(t))**2 \
            -1/10/(np.cos(t)**10+2/3*np.sin(t)**10)**(11/10)*(90*np.cos(t)**8*np.sin(t)**2-10*np.cos(t)**10 \
            +60*np.sin(t)**8*np.cos(t)**2-20/3*np.sin(t)**10)
     elif der==3:
        res = -231/1000/(np.cos(t)**10+2/3*np.sin(t)**10)**(31/10)*(-10*np.cos(t)**9*np.sin(t)+20/3*np.sin(t)**9*np.cos(t))**3 \
            +33/100/(np.cos(t)**10+2/3*np.sin(t)**10)**(21/10)*(-10*np.cos(t)**9*np.sin(t)+20/3*np.sin(t)**9*np.cos(t))* \
            (90*np.cos(t)**8*np.sin(t)**2-10*np.cos(t)**10+60*np.sin(t)**8*np.cos(t)**2-20/3*np.sin(t)**10) \
            -1/10/(np.cos(t)**10+2/3*np.sin(t)**10)**(11/10)*(-720*np.cos(t)**7*np.sin(t)**3+280*np.cos(t)**9*np.sin(t) \
            +480*np.sin(t)**7*np.cos(t)**3-560/3*np.sin(t)**9*np.cos(t))
     else:
        raise ValueError('derivative not implemented')
     return res

def nonsym_shape(t, der):
     res=np.zeros(t.shape[0])
     if der==0:
        res =(1 + 0.9*np.cos(t) + 0.1*np.sin(2*t))/(1 + 0.75*np.cos(t))
     elif der==1:
        res = 4/5*(-3*np.sin(t)+8*np.cos(t)**2-4+3*np.cos(t)**3)/(16+24*np.cos(t)+9*np.cos(t)**2)
     elif der==2:
        res = -4/5*(12*np.cos(t)-9*np.cos(t)**2+64*np.sin(t)*np.cos(t)+36*np.sin(t)*np.cos(t)**2+9*np.sin(t)*np.cos(t)**3+24*np.sin(t)+18) \
            /(64+144*np.cos(t)+108*np.cos(t)**2+27*np.cos(t)**3)
     elif der==3:
        res = -4/5*(144*np.sin(t)*np.cos(t)+114*np.sin(t)-40+240*np.cos(t)**3+192*np.cos(t)-27*np.sin(t)*np.cos(t)**2+368*np.cos(t)**2 \
            +144*np.cos(t)**4+27*np.cos(t)**5)/(256+768*np.cos(t)+864*np.cos(t)**2+432*np.cos(t)**3+81*np.cos(t)**4)
     else:
        raise ValueError('derivative not implemented')
     return res

def circle(t, der):
     if der==0:
        res=np.ones(t.shape[0])
     else:
        res=np.zeros(t.shape[0])
     return res

def kite(t, der):
    res=np.zeros((2,t.shape[0]))
    n=t.shape[0]

    if der==0:
        res = np.append(np.cos(t)+0.65*np.cos(2*t)-0.65,   1.5*np.sin(t)).reshape(2, n)
    elif der==1:
        res = np.append(-np.sin(t)-1.3*np.sin(2*t)    ,    1.5*np.cos(t)).reshape(2, n)
    elif der==2:
        res = np.append(-np.cos(t)-2.6*np.cos(2*t)    ,   -1.5*np.sin(t)).reshape(2, n)
    elif der==3:
        res = np.append(np.sin(t)+5.2*np.sin(2*t)     ,   -1.5*np.cos(t)).reshape(2, n)
    else:
        raise ValueError('derivative not implemented')
    return res


