import numpy as np

class StarTrig:
    """ The class StarTrig describes boundaries of domains in R^2 which are
     star-shaped w.r.t. the origin and which are parameterized by
          z(t) = q(t)[cos(t);sin(t)]      0<=t<=2pi
     where q is a trigonometric polynomial with N coefficient.
     Here N must be even, so the highest order monomial is cos(t*N/2),
     but sin(t*N/2) does not occur.
     z and its derivatives are sampled at n equidistant points.
     Application of the Gramian matrix and its inverse w.r.t. the
     Sobolev norm ||q||_{H^s} are implemented."""
    
    def __init__(self, N_fk, sobo_Index, type=None, coeff=None, **kwargs):
        # Fourier coefficients of q(t)
        self.coeff=None 
        # Sobolev index (for GramX)
        self.sobo_Index=sobo_Index
        self.type=None
        self.coeff=None

    
    def StarTrig(self, N, s):
        self.name = 'StarTrig'
        self.type = 'StarTrig'
        if N%2==1:
            raise ValueError('N should be even')
        self.coeff = np.zeros(N)
        self.sobo_Index = s

    def compute_FK(self, val, n):
    
        """ computes n Fourier coeffients to the point values given by by val
         such that ifft(fftshift(coeffhat)) is an interpolation of val"""

        if n%2==1:
            ValueError('length of t should be even')

        N = len(val)
#        print(N)
        coeffhat = np.fft.fft(val)
        coeffhat2 = 1j*np.zeros(n)
        if (n>=N):
            coeffhat2[0:int(N/2)]= coeffhat[0:int(N/2)]
            coeffhat2[n-int(N/2)+1:n] = coeffhat[int(N/2)+1:N]
            if (n>N):
                coeffhat2[int(N/2)] = 0.5*coeffhat[int(N/2)]
                coeffhat2[n-int(N/2)] = 0.5*coeffhat[int(N/2)]
            else: #n==N
                coeffhat2[int(N/2)] = coeffhat[int(N/2)]

        else:
            coeffhat2[0:int(n/2)] = coeffhat[0:int(n/2)]
            coeffhat2[int(n/2)+1:n] = coeffhat[N-int(n/2)+1:N]
            coeffhat2[int(n/2)] = 0.5*(coeffhat[int(n/2)]+coeffhat[N-int(n/2)])

        coeffhat2 = n/N*np.fft.fftshift(coeffhat2)
        return coeffhat2


    def radial(self, n, der):
        
        """ evaluates all derivatives of the radial function up to order der
        at n equidistant points"""
        
        coeffhat = self.compute_FK(self.coeff,n)
        self.q=np.zeros((der+1, coeffhat.shape[0]))
        for d in range(0, der+1):
            self.q[d,:] = np.real(np.fft.ifft(np.fft.fftshift( (1j*np.linspace(-n/2, n/2-1, n).transpose())**d * coeffhat))).transpose()

    def bd_eval(self, n, der):
    
        self.radial(n,der)
        q=self.q
        t = 2*np.pi*np.linspace(0, n-1, n)/n
        cost = np.cos(t)
        sint = np.sin(t)
        
        self.z = np.append(q[0,:]*cost, \
            q[1,:]*sint).reshape(2, q[0, :].shape[0])
        if der>=1:
            self.zp = np.append(q[1,:]*cost - q[0,:]*sint, \
                q[1,:]*sint + q[1,:]*cost).reshape(2, q[0, :].shape[0])
            self.zpabs = np.sqrt(self.zp[0,:]**2 + self.zp[1,:]**2)
            #outer normal vector
            self.normal = np.append(self.zp[1,:], \
                -self.zp[0,:]).reshape(2, self.zp[0, :].shape[0])

        if der>=2:
            self.zpp = np.append(q[2,:]*cost - 2*q[1,:]*sint - q[0,:]*cost, \
                q[2,:]*sint + 2*q[1,:]*cost - q[0,:]*sint).reshape(2, n)

        if der>=3:
            self.zppp = np.append(q[3,:]*cost - 3*q[2,:]*sint - 3*q[1,:]*cost + q[0,:]*sint, \
                q[3,:]*sint + 3*q[2,:]*cost - 3*q[1,:]*sint - q[0,:]*cost).reshape(2, n)
        if der>3:
            raise ValueError('only derivatives up to order 3 implemented')

    def der_normal(self, h):        
    
        """computes the normal part of the perturbation of the curve caused by
        perturbing the coefficient vector curve.coeff in direction h"""
        
        n=np.size(self.q[0, :])
        h_long = np.fft.ifft(np.fft.fftshift(self.compute_FK(h,n)))
        der =  self.q[0,:].transpose() * h_long / self.zpabs.transpose()
        return der

    def adjoint_der_normal(self, g):    
    
        """applies the adjoint of the linear mapping h->der_normal(curve,h) to g"""
        
        N = len(self.coeff)
#        print(N)
        n = len(g)
        adj_long = g*self.q[0,:].transpose() / self.zpabs.transpose()
        adj = np.fft.ifft(np.fft.fftshift(self.compute_FK(adj_long,N))) * n/N
        return adj.real
        
        """ The following two methods are not needed for operators depending
             only on the curve, but not on its parametrization.
             They are included for test purposes."""
             
    def StarTrig_derivative(self, h):
        
    
        """computes the perturbation of the curve caused by perturbing the coefficient
         vector curve.coeff in direction h"""
        n=np.size(self.q[0, :])
        h_long = np.fft.ifft(np.fft.fftshift(self.compute_FK(h,n))).transpose()
        t = 2*np.pi*n.linspace(0, n-1, n)/n
        der =  np.append(h_long*np.cos(t), h_long*np.sin(t)).reshape(2, n)
        return der

    def StarTrig_adjoint_derivative(self, g):   
    
        """applies the adjoint of the linear mapping h->derivative(curve,h) to g"""
        N = len(self.coeff)
        n = len(g)
        t = 2*np.pi*np.linspace(0, n-1)/n
        adj_long = g[0,:]*np.cos(t)+g[1,:]*np.sin(t)
        adj = np.fft.ifft(np.fft.fftshift(self.compute_FK(adj_long,N))) * n/N
        return adj
    
    def arc_length_der(self, h):    
        """ computes the derivative of h with respect to arclength"""
        n=np.size(self.q[0, :])
        dhds = np.fft.ifft(np.fft.fftshift((1j*np.linspace(-n/2, n/2-1, n).transpose()) *self.compute_FK(h,n)))/self.zpabs.transpose()
        return dhds
    
    def L2err(self, q1, q2):
        
        
        res = self.params.domain.norm(q1-q2)/np.sqrt(len(q1))
        return res
        
    def coeff2Curve(self,coeff,n):
        radial = np.fft.ifft(np.fft.fftshift(self.compute_FK(coeff,n)))
        t = 2*np.pi/n * np.linspace(0, n-1, n)
        pts = np.append(radial.transpose()*np.cos(t), \
            radial.transpose()*np.sin(t)).reshape(2, n)
        return pts.real