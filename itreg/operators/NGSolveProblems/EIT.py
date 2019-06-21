#TODO: Insert Netgen and netgen visualization
#TODO: Circular domain
#TODO: Insert projection onto boundary values
#TODO: Make sure int_domega u=0 in evaluation (maybe define new fes)
#TODO: Make landweber converging


from itreg.operators import NonlinearOperator

import numpy as np
import scipy.optimize as sco
import netgen.gui
#%gui tk
from ngsolve import *
from netgen.geom2d import SplineGeometry

from ngsolve.meshes import MakeQuadMesh

class EIT(NonlinearOperator):
    

    def __init__(self, domain, g, pts, codomain=None):
        
        codomain = codomain or domain
        self.g=g
        self.pts=pts

        
        #Define mesh and finite element space
#        geo=SplineGeometry()
#        geo.AddCircle((0,0),0.5,bc="circle")
#        self.mesh = geo.GenerateMesh()
        self.mesh=MakeQuadMesh(10)
   
#Variables for setting of boundary values later     
        self.ind=[v.point in pts for v in self.mesh.vertices]
        self.fes_in = H1 (self.mesh, order=1)
        self.gfu_in = GridFunction(self.fes_in)
        
        self.fes = H1(self.mesh, order=2, dirichlet="top|right|bottom|left")

        #grid functions for later use 
        self.gfu = GridFunction(self.fes)  # solution, return value of _eval
        self.gfu_bdr=GridFunction(self.fes) #grid function holding boundary values, g/sigma=du/dn
        
        self.gfu_integrator = GridFunction(self.fes) #grid function for defining integrator (bilinearform)
        self.gfu_rhs = GridFunction(self.fes) #grid function for defining right hand side (linearform), f
        
        self.gfu_inner=GridFunction(self.fes) #grid function for inner computation in derivative and adjoint
        self.gfu_toret=GridFunction(self.fes) #grid function for returning values in adjoint and derivative
       
#        self.gfu_dir=GridFunction(self.fes) #grid function for solving the dirichlet problem in adjoint
        self.gfu_error=GridFunction(self.fes) #grid function used in _target to compute the error in forward computation
        self.gfu_tar=GridFunction(self.fes) #grid function used in _target, holding the arguments
        
        u = self.fes.TrialFunction()  # symbolic object
        v = self.fes.TestFunction()   # symbolic object 

        #Define Bilinearform, will be assembled later        
        self.a = BilinearForm(self.fes, symmetric=True)
        self.a += SymbolicBFI(grad(u)*grad(v)*self.gfu_integrator)
        

        #Define Linearform, will be assembled later        
        self.f=LinearForm(self.fes)
        self.f += SymbolicLFI(self.gfu_rhs*v)
        
        self.r=self.f.vec.CreateVector()
        
        self.b=LinearForm(self.fes)
        self.gfu_b = GridFunction(self.fes)
        self.b+=SymbolicLFI(self.gfu_b*v.Trace(), BND)
        
#        self.b2=LinearForm(self.fes)
#        self.b2+=SymbolicLFI(div(v*grad(self.gfu))

        super().__init__(domain, codomain)
        
    def _eval(self, diff, differentiate, **kwargs):
        #Assemble Bilinearform
        self.gfu_integrator.vec.FV().NumPy()[:]=diff  
        self.a.Assemble()
        #print(self.a.mat)
            
        #Assemble Linearform, boundary term
        self.gfu_b.Set(self.g)
        self.b.Assemble()
        #print(self.b.vec)
            
        #Solve system
        #self.gfu.vec.data=self._Solve(self.a, self.b.vec)
        #res=sco.least_squares(self._Target, np.zeros(441), max_nfev=50)
        
        res=sco.minimize((lambda u: self._target(u, self.b.vec)), np.zeros(441), constraints={"fun": self._constraint, "type": "eq"})
        
        #print(res.x)
        #print(self._Target(np.zeros(441)))
        #print(self._Target(self.gfu.vec.FV().NumPy()))
        #print(self._Target(res.x))
        #c = Preconditioner(self.a, 'local')
        #inv=CGSolver(self.a.mat, c.mat, maxsteps=1000)
        #c.Update()
        #self.gfu.vec.data=inv * self.b.vec
        
        if differentiate:
            sigma=CoefficientFunction(self.gfu_integrator)
            self.gfu_bdr.Set(self.g/sigma)

#        return self.gfu.vec.FV().NumPy().copy()
#        return self._GetBoundaryValues(self.gfu)
        self.gfu.vec.FV().NumPy()[:]=res.x
        return self._get_boundary_values(self.gfu)


    def _derivative(self, h, **kwargs):
        #Bilinearform already defined from _eval

        #Translate arguments in Coefficient Function            
        self.gfu_inner.vec.FV().NumPy()[:]=h
 
        #Define rhs (f)              
        rhs=div(self.gfu_inner*grad(self.gfu))
               
        self.gfu_rhs.Set(rhs)
        self.f.Assemble()
        
        #Define boundary term
        #self.gfu_b.Set(-self.gfu_inner*self.gfu_bdr)
        #self.b.Assemble()
        
        #self.gfu_toret.vec.data=self._Solve(self.a, self.f.vec)#+self.b.vec)
        
        res=sco.minimize((lambda u: self._target(u, self.f.vec)), np.zeros(441), constraints={"fun": self._constraint, "type": "eq"})

        self.gfu_toret.vec.FV().NumPy()[:]=res.x
#        return res.x            
#        return self.gfu_toret.vec.FV().NumPy().copy()
        return self._get_boundary_values(self.gfu_toret)

            

    def _adjoint(self, argument, **kwargs):  
        #Bilinearform already defined from _eval
            
        #Definition of Linearform
        #But it only needs to be defined on boundary
        self._set_boundary_values(argument)
#        self.gfu_dir.Set(self.gfu_in)
        
        #Note: Here the linearform f for the dirichlet problem is just zero
        #Update for boundary values
#        self.r.data=-self.a.mat * self.gfu_dir.vec
        
        #Solve system
#        self.gfu_toret.vec.data=self.gfu_dir.vec.data+self._Solve(self.a, self.r)
        
#        return self.gfu_toret.vec.FV().NumPy().copy()
        self.gfu_rhs.Set(self.gfu_in)
        self.f.Assemble()
        
        res=sco.minimize((lambda u: self._target(u, self.f.vec)), 0.0001*np.ones(441), constraints={"fun": self._constraint, "type": "eq"})
        self.gfu_inner.vec.FV().NumPy()[:]=res.x
        
        toret=-grad(self.gfu_inner)*grad(self.gfu)
        
        self.gfu_toret.Set(toret)
        return self.gfu_toret.vec.FV().NumPy().copy()

        
    def _solve(self, bilinear, rhs, boundary=False):
        return bilinear.mat.Inverse(freedofs=self.fes.FreeDofs()) * rhs
    
    def _get_boundary_values(self, gfu):
        myfunc=CoefficientFunction(gfu)
        vals = np.asarray([myfunc(self.mesh(*p)) for p in self.pts])
        return vals
    
    def _set_boundary_values(self, vals):
        self.gfu_in.vec.FV().NumPy()[self.ind]=vals
        return 
    
###############################################################################
    
    def _target(self, u, linearform_vec):
        self.gfu_tar.vec.FV().NumPy()[:]=u
        self.gfu_error.vec.data = self.a.mat * self.gfu_tar.vec-linearform_vec
        return self.gfu_error.vec.Norm()**2#+1*Integrate(coff, self.mesh, BND)**2 

        
    def _constraint(self, u):
        tar=GridFunction(self.fes)
        tar.vec.FV().NumPy()[:]=u
        coff=CoefficientFunction(tar)
        return Integrate(coff, self.mesh, BND)

    

