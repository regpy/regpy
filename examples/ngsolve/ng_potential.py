import logging

import numpy as np

import ngsolve as ngs
from ngsolve.webgui import Draw 
import netgen.geom2d as geom2d

import regpy.stoprules as rules

from regpy.operators.ngsolve import SecondOrderEllipticCoefficientPDE
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.landweber import Landweber
from regpy.hilbert import L2, Sobolev
from regpy.vecsps.ngsolve import NgsSpace
from regpy.util.operator_tests import test_derivative
from regpy.util.operator_tests import test_adjoint_derivative


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)


class potential(SecondOrderEllipticCoefficientPDE):

    def __init__(self, domain, sol_domain,bdr_val=None,a_bdr_val=None):
        super().__init__(domain, sol_domain,bdr_val=bdr_val,a_bdr_val=a_bdr_val)

    def _bf(self,a,u,v):
        return a*ngs.grad(u)*ngs.grad(v)*ngs.dx

    def _bf_0(self):
       return 0.01*self.u*self.v * ngs.dx

    def _lf(self):
        lf = ngs.LinearForm(self.codomain.fes)
        lf += 100 * self.v * ngs.dx
        return lf.Assemble()


geo = geom2d.SplineGeometry()
p1,p2,p3,p4 = [ geo.AddPoint (x,y,maxh=0.01) for x,y in [(0,0), (1,0), (1,1), (0,1)] ]

geo.Append (["line", p1, p2], bc="bottom")
geo.Append (["line", p2, p3], bc="right")
geo.Append (["line", p3, p4], bc="top")
geo.Append (["line", p4, p1], bc="left")

#definition of FE spaces    
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2 ))

fes_domain = ngs.H1(mesh, order=3)
domain = NgsSpace(fes_domain)


bdr = "top|bottom"
fes_codomain = ngs.H1(mesh, order=3, dirichlet=bdr)
codomain = NgsSpace(fes_codomain, bdr=bdr)

op = potential(domain,codomain)

exact_solution_coeff = 1+10*0.5*ngs.exp(-2*(ngs.x-0.5)**2-2*(ngs.y-0.5)**2)
exact_solution = domain.from_ngs( exact_solution_coeff )
exact_data = op(exact_solution)

noise = 0.00001 * codomain.randn()

data = exact_data*(1+noise)

init = domain.from_ngs ( 4 )
init_data = op(init)

setting = RegularizationSetting(op=op, penalty=Sobolev, data_fid=L2)

landweber = Landweber(setting, data, init, stepsize=5)
stoprule = (
        rules.CountIterations(200) +
        rules.Discrepancy(setting.h_codomain.norm, data, noiselevel=setting.h_codomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)
print("exact solution")
Draw(exact_solution_coeff, mesh, "exact",min=1,max=6)

# Draw reconstructed solution
print("reconstructed solution")
Draw(domain.to_ngs(reco), mesh,"reco",min=1,max=6)

# Draw data space
print("noisy data")
Draw(codomain.to_ngs(data),mesh, "data")
print("reconstructed data")
Draw(codomain.to_ngs(reco_data),mesh, "reco_data")


