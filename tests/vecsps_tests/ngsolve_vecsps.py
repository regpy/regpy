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
