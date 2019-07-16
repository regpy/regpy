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
        Iop = [self.wDL_ex*op_K(self.bd,Iop_data_ex)+self.wDL_in*op_K(self.bd,Iop_data_in)+(self.wDL_ex-self.wDL_in-4)*np.diag(self.bd.zpabs) \
            self.wSL_ex*op_S(self.bd,Iop_data_ex)+self.wSL_in*op_S(self.bd,Iop_data_in); \
            self.wDL_ex*op_T(self.bd,Iop_data_ex)+self.wDL_in*op_T(self.bd,Iop_data_in) \
            self.wSL_ex*op_K(self.bd,Iop_data_ex).T+self.wSL_in*op_K(self.bd,Iop_data_in).T+(self.wSL_in-2*self.rho-self.wSL_ex-2)*diag(self.bd.zpabs)]
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
            F.uinc(:,l)      = (exp(1i*F.kappa_ex*F.inc_directions(:,l)'*F.bd.z)).';
            F.duincdnu(:,l)  = (exp(1i*F.kappa_ex*F.inc_directions(:,l)'*F.bd.z).*(1i*F.kappa_ex*F.inc_directions(:,l)'*F.bd.normal)./F.bd.zpabs).';
            rhs              = [F.uinc(:,l); F.duincdnu(:,l)];
            %computing the far field operator (FF_combined) at phi=(A+L)^(-1)*Rf
            %according to eq. (4.19)
            F.dudn(:,l)      = F.Iop*rhs;
            complex_farfield = F.FF_combined * F.dudn(:,l);
            farfield         = [farfield;real(complex_farfield);imag(complex_farfield)];
            %computing the various u's and their normal derivatives needed for the
            %implementation of (5.3) needed for the derivative and adjoint operator
            F.ue(:,l)        = F.dudn(1:2*F.N_ieq,l);
            F.duednu(:,l)    = F.dudn((2*F.N_ieq+1):4*F.N_ieq,l);
            F.ui(:,l)        = F.ue(:,l)+F.uinc(:,l);     %according to (4.4)
            F.duidnu(:,l)    = F.rho*(F.duednu(:,l)+F.duincdnu(:,l));
            F.u(:,l)         = F.ui(:,l)-F.rho*(F.ue(:,l)+F.uinc(:,l));
        end
            
        end
        
        function [der,F] = TransmissionOp_derivative(F,h)
            der = [];
            hn  = F.bd.der_normal(h);
            
            for l=1:size(F.inc_directions,2)
                
                %implementing the rhs according to (5.3)
                duds  = F.bd.arc_length_der(F.u(:,l));
                rhs_a = hn.*(F.duednu(:,l)+F.duincdnu(:,l)-F.duidnu(:,l));
                rhs_b = F.bd.arc_length_der(hn.*duds)+ F.kappa_in^2*hn.*F.ui(:,l) - F.rho*F.kappa_ex^2*hn.*(F.ue(:,l)+F.uinc(:,l));
                rhs_b = rhs_b/F.rho;   %compare (5.3b) and (4.3b)
                rhs   = [rhs_a; rhs_b];
                
                %solving again (4.19) with rhs as just constructed
                phi = F.Iop*rhs;
                complex_farfield = F.FF_combined * phi;
                der = [der;real(complex_farfield);imag(complex_farfield)];
            end
            
        end
        
        function [adj,F] = TransmissionOp_adjoint(F,g)
            
            %implementing the adjoint operator via the decomposition as described in theorem 15/16.
            
            res = zeros(2*F.N_ieq,1);
            N_FF = size(F.meas_directions,2);
            
            for l=1:size(F.inc_directions,2);
                
                g_complex = (g(2*(l-1)*N_FF+[1:N_FF]) + 1i*g(2*(l-1)*N_FF+[N_FF+1:2*N_FF]));
                %apply adjoint of the far field operator
                phi = F.FF_combined'*g_complex;
                %apply the adjoint of the integral operator
                rhs   = F.Iop'*phi;
                rhs_a = rhs(1:2*F.N_ieq);
                rhs_b = rhs((2*F.N_ieq+1):4*F.N_ieq);
                %apply the adjoint of M
                res = res +real(conj(F.duednu(:,l)+F.duincdnu(:,l)-F.duidnu(:,l)).*rhs_a ...
                    -conj(F.bd.arc_length_der(F.u(:,l))/F.rho).*F.bd.arc_length_der(rhs_b./F.bd.zpabs').*F.bd.zpabs' ...
                    +conj(F.kappa_in^2*F.ui(:,l)/F.rho-F.kappa_ex^2*(F.ue(:,l)+F.uinc(:,l))).*rhs_b);
                
            end
            %apply the adjoint of N
            adj = F.bd.adjoint_der_normal(res);
        end
    end
end


class default(object):
    def __init__(self, kappa_ex, wSL_in, wSL, wDL):
        self.kappa_ex=kappa_ex
        self.wSL_in=wSL_in
        self.wSL=wSL
        self.wDL=wDL