import logging

import ngsolve as ngs
from netgen.occ import *
from ngsolve.webgui import Draw 

import numpy as np

import regpy.stoprules as rules
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.landweber import Landweber
from regpy.hilbert import L2, Sobolev
from regpy.vecsps.ngsolve import NgsSpace
from regpy.operators.ngsolve import SecondOrderEllipticCoefficientPDE



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)



class convection(SecondOrderEllipticCoefficientPDE):

    def __init__(self, domain, sol_domain, eps,rhs):
        super().__init__(domain, sol_domain)
        self.eps=eps
        self.rhs=rhs

    def _bf(self,a,u,v):
        
        return ngs.InnerProduct(a, ngs.grad(u)) * v * ngs.dx

    def _bf_0(self):       
       return self.eps*ngs.InnerProduct(ngs.grad(self.u),ngs.grad(self.v)) * ngs.dx

    def _lf(self):
        lf = ngs.LinearForm(self.codomain.fes)
        lf += 10*self.rhs * self.v * ngs.dx
        return lf.Assemble()


mesh = ngs.Mesh( unit_square.GenerateMesh(maxh=0.1))
w = (ngs.x*(1-ngs.x),ngs.y*(1-ngs.y))
eps=0.01
wind = ngs.CoefficientFunction(w)

coef_f=ngs.CF(1)

bdr="left"

fes_domain=ngs.VectorH1(mesh,order=3)
domain=NgsSpace(fes_domain)

fes_codomain = ngs.H1(mesh, order=3, dirichlet=bdr)
codomain = NgsSpace(fes_codomain, bdr=bdr) 

op = convection(domain,codomain,eps,coef_f)

exact_solution_coeff = wind
exact_solution = domain.from_ngs( exact_solution_coeff )

exact_data = op(exact_solution)
noise = 0.00001 * codomain.randn()
data = exact_data*(1+noise)

init=domain.from_ngs((0.25,ngs.y*(1-ngs.y)))
init_data = op(init)

setting = RegularizationSetting(op=op, penalty=Sobolev, data_fid=L2)

landweber = Landweber(setting, data, init, stepsize=0.00001)
stoprule = (
        rules.CountIterations(3000) +
        rules.Discrepancy(setting.h_codomain.norm, data, noiselevel=setting.h_codomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)
print("exact solution")
Draw(exact_solution_coeff, mesh, "exact",min=0,max=0.5)

# Draw reconstructed solution
print("reconstructed solution")
Draw(domain.to_ngs(reco), mesh,"reco",min=0,max=0.5)

# Draw data space
print("noisy data")
Draw(codomain.to_ngs(data),mesh, "data")
print("reconstructed data")
Draw(codomain.to_ngs(reco_data),mesh, "reco_data")


