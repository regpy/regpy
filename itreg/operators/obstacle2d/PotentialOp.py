from .. import NonlinearOperator

from .Obstacle2dBaseOp import Obstacle2dBaseOp
from .Obstacle2dBaseOp import bd_params

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class PotentialOp(NonlinearOperator):
    """ identification of the shape of a heat source from measurements of the
     heat flux.
     see F. Hettlich & W. Rundell "Iterative methods for the reconstruction
     of an inverse potential problem", Inverse Problems 12 (1996) 251–266
     or
     sec. 3 in T. Hohage "Logarithmic convergence rates of the iteratively
     regularized Gauss–Newton method for an inverse potential
     and an inverse scattering problem" Inverse Problems 13 (1997) 1279–1299
    """

    def __init__(self, domain, codomain=None, error='deterministic', **kwargs):

            codomain = codomain or domain
            self.radius = 1.5            # radius of outer circle
            self.Nfwd = 64               # nr. of discretization points for forward solver
            self.Nfwd_synth = 256        # nr of discretization points for computation of synthetic data
            self.N_meas = 64             # number of measurement points
            """transpose"""
            self.meas_angles=2*np.pi*np.linspace(0, self.N_meas-1, self.N_meas).transpose()/self.N_meas             # angles of measure points

            self.cosin=np.zeros((self.Nfwd, self.Nfwd))
            self.sinus=np.zeros((self.Nfwd, self.Nfwd))
            self.sin_fl=np.zeros((self.Nfwd, self.Nfwd))
            self.cos_fl=np.zeros((self.Nfwd, self.Nfwd))
            self.obstacle=Obstacle2dBaseOp()
            self.obstacle.Obstacle2dBasefunc()
            self.bd=self.obstacle.bd
            self.error=error

            N = self.Nfwd
            t= 2*np.pi*np.linspace(0, N-1, N)/N
            t_fl = self.meas_angles

            for j in range(0, N):
                self.cosin[j,:] = np.cos((j+1)*t)
                self.sinus[j,:] = np.sin((j+1)*t)
                self.sin_fl[:,j] = np.sin((j+1)*t_fl)
                self.cos_fl[:,j] = np.cos((j+1)*t_fl)
            super().__init__(
                    domain=domain,
                    codomain=codomain)

    def _eval(self, coeff, **kwargs):
        """self.bd.coeff"""
        self.bd.coeff = coeff
        N = self.Nfwd
        self.bd.bd_eval(N,1)
        q=self.bd.q[0, :]
        if q.max() >= self.radius:
            raise ValueError('reconstructed object penetrates measurement circle')

        if q.min()<=0:
            raise ValueError('reconstructed radial function negative')

        qq = q**2

        flux = 1/(2*self.radius*N) * np.sum(qq)*np.ones(len(self.meas_angles))
        fac = 2/(N*self.radius)
        for j in range(0, int((N-1)/2)):
            fac= fac/self.radius
            qq = qq * q
            flux = flux + (fac/(j+3)) * self.cos_fl[:,j] * np.sum(qq*self.cosin[j,:]) \
                + (fac/(j+3)) * self.sin_fl[:,j] * np.sum(qq*self.sinus[j,:])

        if (N % 2==0):
            fac = fac/self.radius
            qq = qq * q
            flux = flux + fac * self.cos_fl[:,int(N/2)] * np.sum(qq*self.cosin[int(N/2),:])
        return flux

    def _derivative(self, h_coeff, **kwargs):

        N = self.Nfwd
        """transpose ?"""
        h = self.bd.der_normal(h_coeff).transpose()
        q = self.bd.q[0,:]
        qq = self.bd.zpabs

        der = 1/(self.radius*N) * np.sum(qq*h)*np.ones(len(self.meas_angles))
        fac = 2/(N*self.radius)
        for j in range(0, int((N-1)/2)):
            fac= fac/self.radius
            qq = qq*q
            der = der + fac * self.cos_fl[:,j] * np.sum(qq*h*self.cosin[j,:]) \
                    + fac * self.sin_fl[:,j] * np.sum(qq*h*self.sinus[j,:])

        if (N % 2==0):
            fac = fac/self.radius
            qq = qq*q
            der = der + fac * self.cos_fl[:,int(N/2)] * np.sum(qq*h*self.cosin[int(N/2),:])
        return der.real


    def _adjoint(self, g, **kwargs):
        N = self.Nfwd
        q = self.bd.q[0,:]
        qq = self.bd.zpabs

        """transpose?"""
        adj = 1/(self.radius*N) *np.sum(g) * qq.transpose()
        fac = 2/(N*self.radius)
        for j in range(0, int((N-1)/2)):
            fac= fac/self.radius
            qq = qq*q
            """transpose?"""
            adj = adj + fac * np.sum(g*self.cos_fl[:,j]) * (self.cosin[j,:]*qq).transpose() \
                    + fac * np.sum(g*self.sin_fl[:,j]) * (self.sinus[j,:]*qq).transpose()

        if (N % 2==0):
            fac = fac/self.radius
            qq = qq*q
            """transpose?"""
            adj = adj + fac * np.sum(g*self.cos_fl[:,int(N/2)]) * (self.cosin[int(N/2),:]*qq).transpose()

        adj = self.bd.adjoint_der_normal(adj).real
        return adj

    def accept_proposed(self, positions):
        """self.bd.coeff"""
        self.bd.coeff=positions

        self.bd.bd_eval(N,1)
        q=self.bd.q[0, :]
        if q.max() >= self.radius:
            return False

        if q.min()<=0:
            return False
        return True

def create_synthetic_data(self, noiselevel):

        N = self.op.Nfwd_synth
        t = 2*np.pi/N * np.arange(0, N, 1)
        t_fl = self.op.meas_angles
        q = self.op.obstacle.bd_ex.radial(self.op.obstacle.bd_ex, N)
        qq = q**2

        flux = 1/(2*self.op.radius*N) * sum(qq)*np.ones(len(t_fl))
        fac = 2/(N*self.op.radius)
        for j in range(0, int((N-1)/2)):
            fac= fac/self.op.radius
            qq = qq*q
            flux = flux + (fac/(j+3)) * np.cos((j+1)*t_fl) * np.sum(qq*np.cos((j+1)*t)) \
                + (fac/(j+3)) * np.sin((j+1)*t_fl) * np.sum(qq*np.sin((j+1)*t))

        if N%2==0:
            fac = fac/self.op.radius
            qq = qq*q
            flux = flux + fac * np.cos(N/2*t_fl) * np.sum(qq*np.cos(N/2*t))
        noise = np.random.randn(len(flux))
        data = flux + noiselevel * noise/self.Hcodomain.norm(noise)
        return data

def create_impulsive_noise(n,eta,var=None):
    """Create Mc such that |Mc|<eta
    """
    Mc = set('')
    while(len(Mc) < int(eta*n)):
        s = np.ceil(np.random.rand()*n)
        t = np.ceil(np.random.rand()*n)
        st=np.arange(s, t)
        if s < t and len(Mc) + len(st) <= int(eta*n):
            Mc = Mc.union(st)
        if (len(Mc) == int(eta*n)-1):
            break

    if var is None:
        """Now create random noise on Mc such that noise = \pm 1/\eta with equal
        probability"""
        res = np.zeros((n))
        res[np.asarry(list(Mc)).astype(int)] = (2*np.random.uniform(1,2,len(Mc))-3)/eta
    else:
        """Create Gaussian noise on Mc with variance var^2"""
        res = np.zeros(n)
        res[np.asarray(list(Mc)).astype(int)] = var*np.random.randn(len(Mc))
    return res


class plots():
    def __init__(self,
                 op,
                 reco,
                 reco_data,
                 data,
                 exact_data,
                 exact_solution,
#                 nr_plots,
#                 fig_rec=None,
                 figsize=(8, 8),
#                 Nx=None,
#                 Ny=None,
                 n=64
                 ):

        self.op=op
#        self.Nx=Nx or self.op.params.Nx
#        self.Ny=Ny or self.op.params.Ny
        self.reco=reco
        self.reco_data=reco_data
        self.data=data
        self.exact_data=exact_data
        self.exact_solution=exact_solution
#        self.nr_plots=nr_plots
#        self.fig_rec=fig_rec
        self.figsize=figsize
        self.n=n

    def plotting(self):
#    function F = plot(F,x_k,x_start,y_k,y_obs,k)
#            nr_plots = self.nr_plots

            fig, axs = plt.subplots(1, 2,sharey=True,figsize=self.figsize)
            axs[0].set_title('Domain')
            axs[1].set_title('Heat source')
            axs[1].plot(self.exact_data)
            axs[1].plot(self.data)
            axs[1].plot(self.reco_data)
            ymin=0.7*min(self.reco_data.min(), self.data.min(), self.exact_data.min())
            ymax=1.3*max(self.reco_data.max(), self.data.max(), self.exact_data.max())
            axs[1].set_ylim((ymin, ymax))
            bd=self.op.bd
            pts=bd.coeff2Curve(self.reco, self.n)
            pts_2=bd.coeff2Curve(self.exact_solution, self.n)
            poly = Polygon(np.column_stack([pts[0, :], pts[1, :]]), animated=True, fill=False)
            poly_2=Polygon(np.column_stack([pts_2[0, :], pts_2[1, :]]), animated=True, fill=False)
            axs[0].add_patch(poly)
            axs[0].add_patch(poly_2)
            xmin=1.5*min(pts[0, :].min(), pts_2[0, :].min())
            xmax=1.5*max(pts[0, :].max(), pts_2[0, :].max())
            ymin=1.5*min(pts[1, :].min(), pts_2[1, :].min())
            ymax=1.5*max(pts[1, :].max(), pts_2[1, :].max())
            axs[0].set_xlim((xmin, xmax))
            axs[0].set_ylim((ymin, ymax))
            plt.show()
