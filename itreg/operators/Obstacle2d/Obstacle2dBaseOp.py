import numpy as np
from .Curves.GenCurve import GenCurve
from .Curves.GenTrig import GenTrig
from .Curves.StarTrig import StarTrig
from .Curves.StarCurve import StarCurve

class Obstacle2dBaseOp():
    def __init__(self, **kwargs):
        
            self.op_name = 'PotentialOp'
            self.syntheticdata_flag = True
            self.sobo_Index = 1.6   # Sobolev index of preimage space
            self.N_FK = 64  # number of Fourier coefficients
            self.noiselevel = 0.05
            self.init_guess=None
            self.bd_type = 'StarTrig'
            self.bd=bd_params
            self.true_curve = 'apple'
            self.other_X_err=None 
            self.bd_ex=bd_params
            self.xdag=None
            self.xdag_pts_plot=None
            self.Ydim=None
            self.Xdim=None
            self.fig_rec=None
            self.rec_plot_handle=None
    
    
    def Obstacle2dBasefunc(self):
        """set_parameters ?"""

        
        """compute precomputable quantities
         initialize boundary curve structure"""
        self.other_X_err = {(lambda x: self.L2err(x)), 'L2err'}
        
        if self.syntheticdata_flag is True:
            if self.true_curve=='kite':
                self.bd_ex = GenCurve
#                self.bd_ex.GenCurve_func(self.true_curve)
                self.bd_ex.name=self.true_curve
            else:
                self.bd_ex=StarCurve
                #self.bd_ex.StarCurve(self.true_curve)
                self.bd_ex.name=self.true_curve
            
            N_plot =64
            self.bd_ex.bd_eval(self.bd_ex, N_plot,0)
            self.xdag_pts_plot = self.bd_ex.z
            #   t_plot = 2*pi/N_plot*[0:N_plot];
            #   rad_plot =  F.bd_ex.radial(F.bd_ex,N_plot);
            #   rad_plot = [rad_plot rad_plot(1)];
            #   F.xdag_pts_plot = [rad_plot.*cos(t_plot);
            #       rad_plot.*sin(t_plot)];

        
        if self.bd_type=='StarTrig':
        
            self.bd = StarTrig(self.N_FK,self.sobo_Index)
            self.init_guess = np.ones(self.N_FK)  #initial guess = unit circle
            self.Xdim = self.N_FK;
            if self.syntheticdata_flag is True:
                """transpose ?"""
                self.xdag = self.bd_ex.radial(self.bd_ex, self.N_FK).transpose()
                
        elif self.bd_type=='GenTrig':
            self.bd = GenTrig(self.N_FK,self.sobo_Index)
            """tranpose ?"""
            t = 2*np.pi*np.linspace(0, self.N_FK-1, self.N_FK).transpose()/self.N_FK
            self.init_guess = np.append(np.cos(t),np.sin(t)).reshape(2, t.shape[0])  #initial guess = unit circle
            self.Xdim = 2*self.N_FK
            self.xdag = np.zeros(self.Xdim)
        else:
            ValueError('unknown boundary curve type')


    def plot(self, x_k, x_start, y_k, y_obs, k):
        scat_plot(self, x_k, x_start, y_k, y_obs, k)
        
        
    def  L2err(self,F2,h):
        res = F2.bd.L2err(h,F2.xdag)
        return res
        
    
class bd_params:
    
    def __init__(self, **kwargs):
        self.bd_coeff=None
        self.bd_eval=None
        self.name=None
        self.z=None
        self.q=None
        
       