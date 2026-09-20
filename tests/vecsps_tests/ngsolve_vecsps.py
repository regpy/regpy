from math import isnan

import numpy as np
import ngsolve as ngs
from netgen.geom2d import unit_square
import pytest

from regpy.vecsps.ngsolve import *
from regpy.util import set_rng_seed

from .base_vecsps import vecsps_basics,vector_basics

set_rng_seed(15873098306879350073259142812684978477)

class TestNgsVectorSpace():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=4, dirichlet = bdr)
    fine_mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    fine_fes = ngs.H1(fine_mesh, order=6, dirichlet = bdr)
    domain = NgsVectorSpace(fes,bdr = bdr)
    gf = ngs.GridFunction(fes)
    gf.Set(ngs.x**2*ngs.y)
    gf_fine = ngs.GridFunction(fine_fes)
    gf_fine.Set(ngs.x**2*ngs.y)

    @pytest.mark.parametrize("fes",[ 
        ngs.H1(mesh, order=3, dirichlet = bdr),
        ngs.H1(mesh, order=3, dirichlet = bdr, complex= True),
        ngs.VectorH1(mesh, order=3, dirichlet = bdr)
    ])
    def test_vecsps_basics(self, fes):
        vecsps_basics(NgsVectorSpace,fes, test_methods=True,bdr = self.bdr)

    @pytest.mark.parametrize("fes",[ 
        ngs.H1(mesh, order=3, dirichlet = bdr),
        ngs.H1(mesh, order=3, dirichlet = bdr, complex= True),
        ngs.VectorH1(mesh, order=3, dirichlet = bdr)
    ])
    def test_vector_basics(self, fes):
        vector_basics(NgsVectorSpace,fes, test_comparison= False, ones_is_onevec= False,bdr = self.bdr)
        d = NgsVectorSpace(fes,bdr=self.bdr)
        d_r = d.real_space()
        v_1 = d.ones()
        assert d_r.norm(v_1.imag+v_1.conj().imag) == pytest.approx(0),f"Tying to compare v.imag with v.conj().imag failed for {v_1}"


    @pytest.mark.parametrize("ngs_object",[ 
        gf,
        ngs.x**2*ngs.y,
        gf.vec,
        gf_fine,
    ])
    def test_from_ngs(self,ngs_object):
        # general construction
        gfu = ngs.GridFunction(self.fes)
        gfu.Set(ngs.x**2*ngs.y)
        v = NgsBaseVector(gfu.vec,make_copy=True)
        w = self.domain.from_ngs(ngs_object)
        assert v in self.domain
        assert w in self.domain
        assert v!=w

    def test_mask(self):
        v = self.domain.from_ngs(ngs.x**2*ngs.y)
        difference = v - 0.5 * self.domain.ones()
        mask = self.domain.IfPos(difference)
        assert self.domain.is_mask(mask)
        selected = np.asarray(list(self.domain.iter_mask(mask)), dtype=bool)
        coefficients = np.asarray(list(self.domain.iter_coefficients(v)))
        original = coefficients.copy()
        expected_mask = np.asarray(
            list(self.domain.iter_coefficients(difference))
        ) > 0
        np.testing.assert_array_equal(selected, expected_mask)
        assert selected.any() and (~selected).any()

        result = self.domain.zeros()
        self.domain.copyto(result, v, where=mask)
        np.testing.assert_allclose(
            list(self.domain.iter_coefficients(result)),
            np.where(selected, coefficients, 0.),
            rtol=0, atol=0,
        )
        np.testing.assert_array_equal(
            list(self.domain.iter_coefficients(v)), original
        )

class TestNgsVectorSpaceWithInnerProd():
    bdr = "left|top|right|bottom"
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ngs.H1(mesh, order=4, dirichlet = bdr)
    fine_mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    fine_fes = ngs.H1(fine_mesh, order=6, dirichlet = bdr)
    domain = NgsVectorSpaceWithInnerProduct(fes,bdr = bdr)
    gf = ngs.GridFunction(fes)
    gf.Set(ngs.x**2*ngs.y)
    gf_fine = ngs.GridFunction(fine_fes)
    gf_fine.Set(ngs.x**2*ngs.y)

    @pytest.mark.parametrize("fes",[ 
        ngs.H1(mesh, order=3, dirichlet = bdr),
        ngs.H1(mesh, order=3, dirichlet = bdr, complex= True),
        ngs.VectorH1(mesh, order=3, dirichlet = bdr)
    ])
    def test_vecsps_basics(self, fes):
        vecsps_basics(NgsVectorSpaceWithInnerProduct,fes, test_methods=True,bdr = self.bdr)

    @pytest.mark.parametrize("fes",[ 
        ngs.H1(mesh, order=3, dirichlet = bdr),
        ngs.H1(mesh, order=3, dirichlet = bdr, complex= True),
        ngs.VectorH1(mesh, order=3, dirichlet = bdr)
    ])
    def test_vector_basics(self, fes):
        vector_basics(NgsVectorSpaceWithInnerProduct,fes, test_comparison=False, ones_is_onevec= False,bdr = self.bdr)
        d = NgsVectorSpaceWithInnerProduct(fes,bdr=self.bdr)
        d_r = d.real_space()
        v_1 = d.ones()
        assert d_r.norm(v_1.imag+v_1.conj().imag) == pytest.approx(0),f"Tying to compare v.imag with v.conj().imag failed for {v_1}"

    @pytest.mark.parametrize("ngs_object",[ 
        gf,
        ngs.x**2*ngs.y,
        gf.vec,
        gf_fine,
    ])
    def test_from_ngs(self,ngs_object):
        # general construction
        gfu = ngs.GridFunction(self.fes)
        gfu.Set(ngs.x**2*ngs.y)
        v = NgsBaseVector(gfu.vec,make_copy=True)
        w = self.domain.from_ngs(ngs_object)
        assert v in self.domain
        assert w in self.domain
        assert v!=w

    def test_norm(self):
        v = self.domain.from_ngs(ngs.x*ngs.y)
        assert self.domain.norm(v) == pytest.approx(1/3)

    def test_mask(self):
        v = self.domain.from_ngs(ngs.x**2*ngs.y)
        difference = v - 0.5 * self.domain.ones()
        mask = self.domain.IfPos(difference)
        assert self.domain.is_mask(mask)
        selected = np.asarray(list(self.domain.iter_mask(mask)), dtype=bool)
        coefficients = np.asarray(list(self.domain.iter_coefficients(v)))
        original = coefficients.copy()
        expected_mask = np.asarray(
            list(self.domain.iter_coefficients(difference))
        ) > 0
        np.testing.assert_array_equal(selected, expected_mask)
        assert selected.any() and (~selected).any()

        result = self.domain.zeros()
        self.domain.copyto(result, v, where=mask)
        np.testing.assert_allclose(
            list(self.domain.iter_coefficients(result)),
            np.where(selected, coefficients, 0.),
            rtol=0, atol=0,
        )
        np.testing.assert_array_equal(
            list(self.domain.iter_coefficients(v)), original
        )


@pytest.mark.parametrize("as_real", [True, False])
def test_ngs_vector_size_and_scalar_field(as_real):
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = ngs.H1(mesh, order=1, complex=True)
    space = NgsVectorSpace(fes, as_real_space=as_real)
    x = space.ones()
    assert x.size == fes.ndof  # Vector storage size is not deprecated.
    assert space.dimension == fes.ndof * (2 if as_real else 1)
    dynamic = x + 2.0 * x
    assert dynamic.size == fes.ndof
    assert dynamic in space
    np.testing.assert_allclose(space.flatten(dynamic), 3 * space.flatten(x))
    out = space.empty()
    space.copyto(out, dynamic)
    np.testing.assert_allclose(space.flatten(out), space.flatten(dynamic))


@pytest.mark.parametrize("as_real", [True, False])
def test_ngs_coordinate_roundtrip(as_real):
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = ngs.H1(mesh, order=1, complex=True)
    space = NgsVectorSpace(fes, as_real_space=as_real)
    rng = np.random.default_rng(20260920)
    native = rng.normal(size=fes.ndof) + 1j * rng.normal(size=fes.ndof)
    x = space.zeros()
    x.vec.FV().NumPy()[:] = native
    expected = (np.stack((native.real, native.imag), axis=-1).reshape(-1)
                if as_real else native)
    np.testing.assert_allclose(space.flatten(x), expected)
    np.testing.assert_allclose(list(space.iter_coefficients(x)), expected)
    reconstructed = space.fromflat(expected)
    np.testing.assert_allclose(reconstructed.vec.FV().NumPy(), native)
    out = space.empty()
    space.set_coefficients(out, iter(expected))
    np.testing.assert_allclose(out.vec.FV().NumPy(), native)
    basis = [space.flatten(e).copy() for e in space.iter_basis()]
    np.testing.assert_allclose(basis, np.eye(space.dimension))
