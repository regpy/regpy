import ngsolve as ngs
from netgen.geom2d import unit_square

from regpy.vecsps.ngsolve import *
from regpy.vecsps import DirectSum, NumPyVectorSpace


def test_NgsVectorSpace():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes_domain = ngs.H1(mesh, order=6, dirichlet = bdr)
    domain = NgsVectorSpace(fes_domain,bdr = bdr)

    # test basic methods
    one = domain.ones()
    zero = domain.zeros()
    assert domain.norm(one-zero)-domain.norm(one)<1e-14
    _ = domain.rand()
    _ = domain.randn()
    _ = domain.poisson(domain.from_ngs(ngs.x*ngs.y**2))

    # general construction
    gfu = ngs.GridFunction(fes_domain)
    gfu.Set(ngs.x**2*ngs.y)
    v = NgsBaseVector(gfu.vec,make_copy=True)
    w = domain.from_ngs(ngs.x*ngs.y**2)
    assert v in domain
    assert w in domain
    assert v!=w

    # construction of masks
    mask_regpy = domain.IfPos(v-0.5*one)
    v[mask_regpy] = v
    
    #test complex spaces
    fes_domain = ngs.H1(mesh, order=6, dirichlet = bdr, complex= True)
    domain = NgsVectorSpace(fes_domain,bdr = bdr)

    rand = domain.rand()
    _ = rand.real
    _ = rand.imag

    #test vector spaces
    fes_domain = ngs.VectorH1(mesh, order=3, dirichlet = bdr)
    domain = NgsVectorSpace(fes_domain,bdr = bdr)

    rand = domain.randn()
    rand_real = rand.real
    rand_imag = rand.imag

    assert ngs.sqrt(ngs.Integrate(domain.to_gf((rand_real - rand))**2,mesh)) < 1e-15




