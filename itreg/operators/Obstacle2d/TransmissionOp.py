# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:31:36 2019

@author: Björn Müller
"""

class TransmissionOp(NonlinearOperator):
    """ 2 dimensional obstacle scattering problem with Neumann boundary condition
    % see T. Hohage & C. Schormann "A Newton-type method for a transmission
    % problem in inverse scattering Inverse Problems" Inverse Problems:14, 1207-1227, 1998."""
    
    
    def __init__(self, domain, codomain=None, **kwargs):
        codomain = codomain or domain
        super().__init__(domain, codomain)
        self.kappa = 3            # wave number
        self.N_ieq = 32           # 2*F_ieq is the number of discretization points
        self.N_ieq_synth = 32     # 2*N_ieq is the number of discretization points for
        """the boundary integral equation when computing synthetic data (choose different
         to N_ieq to avoid invere crime)"""
        self.meas_directions = 64 # measurement directions
        self.inc_directions = np.asarry([1,0]).reshape((2,1))
        self.obstacle=Obstacle2dBaseOp()
        self.obstacle.Obstacle2dBasefunc()
        self.bd=self.obstacle.bd
    
        self.kappa_ex = 3          #exterior wave number
        self.kappa_in = 4         # interior wave number
        self.rho = 4.3-6*complex(0,1)         # density ratio
        self.wSL_ex = -1
        self.wDL_ex = 1
        self.wSL_in
        self.wDL_in = -1
        self.wSL
        self.wDL
        self.op_name = 'TransmissionOp'
        self.default=default(self.kappa, self.rho, self.wSL_ex, self.wDL_ex)
        
        
        self.op_name = 'TransmissionOp'
        self.syntheticdata_flag = True
        self.kappa_ex = 2    # wave number
        self.kappa_in = 4
        self.rho = 20.3-16*complex(0,1)
         #146+155i; %0.06-0.05i; %4.3-6i; %10+3i; %0.1+0.2i; %
        # directions of incident waves
        N_inc = 4
        t=2*np.pi*np.arange(0, N_inc)/N_inc
        #t = 0.5;
        self.inc_directions = np.append(np.cos(t), np.sin(t)).reshape((2, N_inc)) 
        
        N_meas = 64
        t= 2*np.pi*np.arange(0, N_meas)/N_meas
        self.N_ieq = 128
        self.meas_directions = np.append(np.cos(t), np.sin(t)).reshape((2, N_meas))
        
        
        self.true_curve = 'peanut' 
        #'peanut','round_rect', 'apple', %'non_sym_shape',
        #'three_lobes','pinched_ellipse','smoothed_rectangle',
        self.noiselevel = 0;



        

                    
        self.Ydim = 2* np.size(self.meas_directions,1) * np.size(self.inc_directions,1)
        

    
    
        
        
    def _eval(self,coeff, differentiate=False):
        """solve the forward transmission problem for the obstacle parameterized by
        % coeff. Quantities needed again for the computation of derivatives and
        % adjoints are stored as members of F."""
        
        self.bd.coeff = coeff
        """compute the grid points of the boundary parameterized by coeff and derivatives
        %of the parametrization and save these quantities as members of F.bd"""
        self.bd.bd_eval(2*self.N_ieq,3)
        
        Iop_data_ex = setup_iop_data(self.bd,self.kappa_ex)
        Iop_data_in = setup_iop_data(self.bd,self.kappa_in)
        """ constructing operator Iop=A+L according to (4.9). Note that the operators in 'int_op'
        have an additional factor of 2|z'(x)| compared to the operators considered in the thesis."""
#Bring this matrix in the right form
        Iop = np.append(self.wDL_ex*op_K(self.bd,Iop_data_ex)+self.wDL_in*op_K(self.bd,Iop_data_in)+(self.wDL_ex-self.wDL_in-4)*np.diag(self.bd.zpabs) \
            self.wSL_ex*op_S(self.bd,Iop_data_ex)+self.wSL_in*op_S(self.bd,Iop_data_in), \
            self.wDL_ex*op_T(self.bd,Iop_data_ex)+self.wDL_in*op_T(self.bd,Iop_data_in) \
            self.wSL_ex*op_K(self.bd,Iop_data_ex).T+self.wSL_in*op_K(self.bd,Iop_data_in).T+(self.wSL_in-2*self.rho-self.wSL_ex-2)*np.diag(self.bd.zpabs)).reshape((2*op_K(self.bd,Iop_data_ex).shape[0], op_K(self.bd,Iop_data_ex).shape[1]))
        R  = [-self.wDL_in*op_K(self.bd,Iop_data_in)+(self.wDL_in+2)*np.diag(self.bd.zpabs) \
            -self.wSL_in*op_S(self.bd,Iop_data_in); \
            -self.wDL_in*op_T(self.bd,Iop_data_in) \
            -self.wSL_in*op_K(self.bd,Iop_data_in).T+(2*self.rho-self.wSL_in)*np.diag(self.bd.zpabs)]
        self.Iop = Iop\R  #here the extra factors of 2|z'(x)| cancel
        
        #set up the matrix mapping the density to the far field pattern
        self.FF_combined = farfield_matrix_trans(self.bd,self.meas_directions,self.kappa_ex,self.wSL_ex,self.wDL_ex)
#introduce farfield in another way
        farfield = []
        self.dudn = np.zeros(4*self.N_ieq,np.size(self.inc_directions,1))
        
        """defining the various u's and their normal derivatives needed for the
        implementation of (5.3) needed for the derivative and adjoint operator"""
        self.ue       = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        self.duednu   = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        self.ui       = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        self.duidnu   = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        self.uinc     = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        self.duincdnu = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        self.u        = np.zeros(2*self.N_ieq,np.size(self.inc_directions,1))
        
        for l in range(0, np.size(self.inc_directions,1)):
            #defining the inhomogenities according to (4.4)
            self.uinc[:,l]      = (np.exp(1*complex(0,1)*self.kappa_ex*self.inc_directions[:,l].T.dot(self.bd.z))).T
            self.duincdnu[:,l]  = (np.exp(1*complex(0,1)*self.kappa_ex*self.inc_directions[:,l].T.dot(self.bd.z)).*(1*complex(0,1)*self.kappa_ex*self.inc_directions[:,l].T.dot(self.bd.normal))./self.bd.zpabs).T
            rhs              = np.append(self.uinc[:,l], self.duincdnu[:,l]).reshape((self.inc_directions.shape))
            #computing the far field operator (FF_combined) at phi=(A+L)^(-1)*Rf
            #according to eq. (4.19)
            self.dudn[:,l]      = self.Iop*rhs
            complex_farfield = self.FF_combined * self.dudn[:,l]
            farfield         = np.append(farfield, np.append(complex_farfield.real, complex_farfield.imag)).reshape((2*np.size(self.inc_directions,0), np.size(self.inc_directions,1)))
            #computing the various u's and their normal derivatives needed for the
            #implementation of (5.3) needed for the derivative and adjoint operator
            self.ue[:,l]        = self.dudn[0:2*self.N_ieq,l]
            self.duednu[:,l]    = self.dudn[2*self.N_ieq:4*self.N_ieq,l]
            self.ui[:,l]        = self.ue[:,l]+self.uinc[:,l]     #according to (4.4)
            self.duidnu[:,l]    = self.rho*(self.duednu[:,l]+self.duincdnu[:,l])
            self.u[:,l]         = self.ui[:,l]-self.rho*(self.ue[:,l]+self.uinc[:,l])


        
    def _derivative(self,h):
            der = []
            hn  = self.bd.der_normal(h)
            
            for l in range(0,np.size(self.inc_directions,1)):
                
                #implementing the rhs according to (5.3)
                duds  = self.bd.arc_length_der(self.u[:,l])
                rhs_a = hn*(self.duednu[:,l]+self.duincdnu[:,l]-self.duidnu[:,l])
                rhs_b = self.bd.arc_length_der[hn*duds.asytpe(int)]+ self.kappa_in**2*hn.*self.ui[:,l] - self.rho*self.kappa_ex**2*hn*(self.ue[:,l]+self.uinc[:,l])
                rhs_b = rhs_b/F.rho;   #compare (5.3b) and (4.3b)
                rhs   = np.append(rhs_a, rhs_b).reshape((2*rhs_a.shape[0], rhs_a.shape[1]))
                
                #solving again (4.19) with rhs as just constructed
                phi = self.Iop*rhs
                complex_farfield = self.FF_combined.dot(phi)
                der=np.append(complex_farfield.real, complex_farfield.imag)
            return der   

        
        def _adjoint(self,g):
            
            #implementing the adjoint operator via the decomposition as described in theorem 15/16.
            
            res = np.zeros(2*self.N_ieq)
            N_FF = np.size(self.meas_directions,1)
            
            for l in range(0, np.size(self.inc_directions,1)):
                
                g_complex = (g[2*(l)*N_FF+npa.arange(0, N_FF)]+ 1*complex(0, 1)*g[2*(l)*N_FF+np.arange(N_FF:2*N_FF))
                #apply adjoint of the far field operator
                phi = self.FF_combined.T.dot(g_complex)
                #apply the adjoint of the integral operator
                rhs   = self.Iop.T.dot(phi)
                rhs_a = rhs[0:2*self.N_ieq]
                rhs_b = rhs[(2*self.N_ieq),4*self.N_ieq)]
                #apply the adjoint of M
                res = res +np.real(np.conjugate(self.duednu[:,l]+self.duincdnu[:,l]-self.duidnu[:,l])*rhs_a \
                    -np.conjugate(self.bd.arc_length_der(self.u[:,l])/self.rho)*self.bd.arc_length_der(rhs_b/self.bd.zpabs.T)*self.bd.zpabs.T \
                    +np.conjugate(self.kappa_in**2*self.ui[:,l]/self.rho-self.kappa_ex**2*(self.ue[:,l]+self.uinc[:,l]))*rhs_b)
                

            #apply the adjoint of N
            adj = self.bd.adjoint_der_normal(res)
            return adj


function [data,F]=TransmissionOp_create_synthetic_data(F)
            bd = F.bd_ex.bd_eval(2*F.N_ieq_synth,3);
            %compute the grid points of the exact boundary and derivatives
            %(up to order 3) of the parametrization and save these quantities as
            %members of bd_ex
            
            Iop_data_ex = setup_iop_data(bd,F.kappa_ex);
            Iop_data_in = setup_iop_data(bd,F.kappa_in);
            
            %constructing operator Iop=(A+L)^(-1)*R according to (4.9). Note that the operators in 'int_op'
            %have an additional factor of 2|z'(x)| compared to the operators considered in the thesis.
            Iop = [F.wDL_ex*op_K(bd,Iop_data_ex)+F.wDL_in*op_K(bd,Iop_data_in)+(F.wDL_ex-F.wDL_in-4)*diag(bd.zpabs) ...
                F.wSL_ex*op_S(bd,Iop_data_ex)+F.wSL_in*op_S(bd,Iop_data_in); ...
                F.wDL_ex*op_T(bd,Iop_data_ex)+F.wDL_in*op_T(bd,Iop_data_in) ...
                F.wSL_ex*op_K(bd,Iop_data_ex).'+F.wSL_in*op_K(bd,Iop_data_in).'+(F.wSL_in-2*F.rho-F.wSL_ex-2)*diag(bd.zpabs)];
            R  = [-F.wDL_in*op_K(bd,Iop_data_in)+(F.wDL_in+2)*diag(bd.zpabs) ...
                -F.wSL_in*op_S(bd,Iop_data_in); ...
                -F.wDL_in*op_T(bd,Iop_data_in) ...
                -F.wSL_in*op_K(bd,Iop_data_in).'+(2*F.rho-F.wSL_in)*diag(bd.zpabs)];
            Iop = Iop\R;  %here the extra factors of 2|z'(x)| cancel
            
            %set up the matrix mapping the density to the far field pattern
            FF_combined = farfield_matrix_trans(bd,F.meas_directions,F.kappa_ex,F.wSL_ex,F.wDL_ex);
            farfield = [];
            
            for l=1:size(F.inc_directions,2)
                
                % implementation of inhomogenities f_1=rhs_a and f_2=rhs_b according to
                % eq. (4.4).
                rhs_a = exp(1i*F.kappa_ex*F.inc_directions(:,l)'*bd.z);
                rhs_b = exp(1i*F.kappa_ex*F.inc_directions(:,l)'*bd.z).*(1i*F.kappa_ex*F.inc_directions(:,l)'*bd.normal)./bd.zpabs;
                rhs   = [rhs_a rhs_b];
                %computing the far field operator (FF_combined) at phi=(A+L)^(-1)*Rf
                %according to eq. (4.12)
                phi = Iop*rhs.';
                complex_farfield = FF_combined * phi;
                farfield = [farfield;real(complex_farfield);imag(complex_farfield)];
            end
            noise = randn(size(farfield));
            data  = farfield + F.noiselevel * noise/sqrt(noise'*F.applyGramY(F,noise));
            if F.plotWhat.field == true
                plot_total_field_trans(F,bd,phi,F.inc_directions(:,end),F.wSL_ex,F.wDL_ex,F.wSL_in,F.wDL_in);
            end
        end

class default(object):
    def __init__(self, kappa_ex, wSL_in, wSL, wDL):
        self.kappa_ex=kappa_ex
        self.wSL_in=wSL_in
        self.wSL=wSL
        self.wDL=wDL