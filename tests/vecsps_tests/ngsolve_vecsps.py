import operator

import numpy as np
import ngsolve as ngs
from netgen.geom2d import unit_square
import pytest

from regpy.vecsps.ngsolve import *
from regpy.util import set_rng_seed
from regpy.hilbert import L2
from regpy.operators import Identity
from regpy.solvers.setting import Setting

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
        # The generic helper calls poisson(VS.ones()). For Dirichlet FEs,
        # the native coefficients of VS.ones() can be negative, so they are
        # not valid Poisson rates. Dedicated noise tests below use explicitly
        # nonnegative native coefficients and cover Poisson in both spaces.
        # Preserve the original generic method checks for complex spaces;
        # those do not attempt real Poisson sampling.
        vecsps_basics(NgsVectorSpace, fes, test_methods=bool(fes.is_complex),
                      bdr=self.bdr)

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
        # The shared helper assumes VS.ones() is a valid Poisson rate;
        # this need not hold for constrained finite-element coefficients.
        # See the dedicated rate-validation tests below.
        vecsps_basics(NgsVectorSpaceWithInnerProduct, fes,
                      test_methods=bool(fes.is_complex), bdr=self.bdr)

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


# ---------------------------------------------------------------------------
# Noise generation contracts for NGSolve and Setting.
# These tests intentionally operate on native FE coefficient vectors.
# An interpolated FE constant (space.ones()) is not guaranteed to have
# nonnegative coefficients and must not be used blindly as a Poisson rate.
# ---------------------------------------------------------------------------


def _noise_space(kind="scalar", *, complex_storage=False,
                 space_class=NgsVectorSpace, constrained=False,
                 as_real_space=True):
    """Small, fresh NGSolve FE vector space for stochastic tests."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.55))
    kwargs = {"order": 1, "complex": complex_storage}
    if constrained:
        kwargs["dirichlet"] = "left|top|right|bottom"
    if kind == "scalar":
        fes = ngs.H1(mesh, **kwargs)
    elif kind == "vector":
        fes = ngs.VectorH1(mesh, **kwargs)
    elif kind == "product":
        # Exercise the per-component auxiliary-space sampling branch.
        fes = ngs.H1(mesh, **kwargs) * ngs.H1(mesh, **kwargs)
    else:
        raise ValueError(f"Unknown FE kind: {kind}")
    return space_class(fes, as_real_space=as_real_space)


def _native_coefficients(space, vector):
    """Snapshot native FE coefficients, including lazy NGSolve expressions.

    ``Setting.data`` can return a read-only NgsBaseVector whose ``vec`` is a
    DynamicVectorExpression. Only an evaluated BaseVector provides ``FV``.
    Evaluate into temporary storage without altering the input, and copy the
    coefficient array so the snapshot remains valid across subsequent runs.

    Do not use ``space.flatten`` here: for a complex space with
    ``as_real_space=True``, flatten splits each native complex coefficient
    into two real coordinates, whereas these tests inspect native storage.
    """
    native = vector.vec.Evaluate() if vector.is_dynamic else vector.vec
    return np.asarray(native.FV().NumPy()).copy()




@pytest.mark.parametrize("complex_storage", [False, True], ids=["real", "complex"])
def test_ngs_native_coefficient_snapshot_handles_lazy_readonly_vectors(complex_storage):
    """The test helper must work on the lazy, read-only type returned by Setting.data."""
    space = _noise_space(complex_storage=complex_storage)
    a = space.zeros()
    b = space.zeros()
    expected = np.arange(1, space.fes.ndof + 1, dtype=float)
    if complex_storage:
        expected = expected + 1j * (expected / 3.)
    a.vec.FV().NumPy()[:] = expected
    b.vec.FV().NumPy()[:] = expected / 2.

    # Adding two NGSolve vectors creates a lazy DynamicVectorExpression.
    lazy = space.read_only(a + b)
    assert lazy.is_dynamic
    assert lazy.read_only
    observed = _native_coefficients(space, lazy)
    np.testing.assert_allclose(observed, 1.5 * expected)
    assert observed.size == space.fes.ndof
    assert np.iscomplexobj(observed) == complex_storage

    # A snapshot is independent and the helper does not materialize or alter
    # the read-only wrapper or its two inputs.
    observed[:] = 0
    assert lazy.is_dynamic
    np.testing.assert_allclose(_native_coefficients(space, lazy), 1.5 * expected)
    np.testing.assert_allclose(_native_coefficients(space, a), expected)
    np.testing.assert_allclose(_native_coefficients(space, b), expected / 2.)


@pytest.mark.parametrize("space_class", [
    NgsVectorSpace, NgsVectorSpaceWithInnerProduct,
], ids=["vector-space", "inner-product-space"])
@pytest.mark.parametrize("use_space_sampling", [False, True],
                         ids=["auxiliary-L2", "native-FE"])
def test_ngs_gaussian_noise_real_seed_and_independent_results(
        space_class, use_space_sampling):
    space = _noise_space(space_class=space_class)
    options = {"seed": 12345, "use_space_sampling": use_space_sampling,
               "mass_matrix_weighting": False}
    first = space.randn(**options)
    first_snapshot = _native_coefficients(space, first)
    second = space.randn(**options)
    assert first in space
    assert second in space
    assert np.all(np.isfinite(first_snapshot))
    np.testing.assert_array_equal(_native_coefficients(space, second),
                                  first_snapshot)
    # A second call must not overwrite storage retained by the first result.
    np.testing.assert_array_equal(_native_coefficients(space, first),
                                  first_snapshot)


@pytest.mark.parametrize("mode", ["vector", "product"])
@pytest.mark.parametrize("use_space_sampling", [False, True],
                         ids=["auxiliary-L2", "native-FE"])
def test_ngs_gaussian_noise_vector_and_product_spaces(mode, use_space_sampling):
    """Regress the old _gfu_i / _gfu attribute bugs in NgsVectorSpace.rand."""
    space = _noise_space(mode)
    options = {"seed": 2718, "use_space_sampling": use_space_sampling,
               "mass_matrix_weighting": False}
    first = space.randn(**options)
    second = space.randn(**options)
    assert first in space
    assert np.all(np.isfinite(_native_coefficients(space, first)))
    np.testing.assert_array_equal(_native_coefficients(space, first),
                                  _native_coefficients(space, second))
    assert _native_coefficients(space, first).size == space.fes.ndof


def test_ngs_native_mass_weighted_gaussian_uses_adjoint_cholesky():
    """If M = L L*, the native white draw must be transformed by L^{- *}."""
    space = _noise_space()
    unweighted = space.randn(seed=141, use_space_sampling=True,
                             mass_matrix_weighting=False)
    weighted = space.randn(seed=141, use_space_sampling=True,
                           mass_matrix_weighting=True)
    assert weighted in space
    assert space.mass_matrix_cholesky is not None
    L = space.mass_matrix_cholesky
    # No Dirichlet restrictions in this fixture: P_free is the identity.
    np.testing.assert_allclose(
        L.conjugate().transpose() @ _native_coefficients(space, weighted),
        _native_coefficients(space, unweighted),
        rtol=1e-8, atol=1e-8,
    )


@pytest.mark.parametrize("mass_weighted", [False, True],
                         ids=["unweighted", "mass-weighted"])
def test_ngs_gaussian_noise_respects_constrained_dofs(mass_weighted):
    space = _noise_space(constrained=True)
    # Enforcing homogeneous Dirichlet constraints is now explicitly opt-in.
    sampled = space.randn(seed=38, use_space_sampling=True,
                          mass_matrix_weighting=mass_weighted,
                          project_free_dofs=True)
    freedofs = space.fes.FreeDofs()
    free = np.fromiter((bool(freedofs[i]) for i in range(space.fes.ndof)),
                       dtype=bool, count=space.fes.ndof)
    assert np.any(~free)  # A non-vacuous boundary-projection check.
    np.testing.assert_array_equal(_native_coefficients(space, sampled)[~free],
                                  np.zeros(np.count_nonzero(~free)))


@pytest.mark.parametrize("use_space_sampling", [False, True],
                         ids=["auxiliary-L2", "native-FE"])
def test_ngs_gaussian_default_retains_constrained_dofs(use_space_sampling):
    """Boundary coefficients belong to the vector space and data space.

    NgsVectorSpace.randn() retains them by default, whereas a homogeneous
    Dirichlet random *PDE variation* uses project_free_dofs=True.
    """
    space = _noise_space(constrained=True)
    options = {"seed": 38, "use_space_sampling": use_space_sampling,
               "mass_matrix_weighting": False}
    free_dofs = space.fes.FreeDofs()
    free = np.fromiter((bool(free_dofs[i]) for i in range(space.fes.ndof)),
                       dtype=bool, count=space.fes.ndof)
    assert np.any(~free)

    full = space.randn(**options)
    repeated = space.randn(**options)
    projected = space.randn(**options, project_free_dofs=True)
    full_coeffs = _native_coefficients(space, full)
    projected_coeffs = _native_coefficients(space, projected)

    np.testing.assert_array_equal(full_coeffs,
                                  _native_coefficients(space, repeated))
    assert np.any(np.abs(full_coeffs[~free]) > 0)
    np.testing.assert_array_equal(projected_coeffs[~free], 0)
    np.testing.assert_allclose(projected_coeffs[free], full_coeffs[free])


@pytest.mark.parametrize("use_space_sampling", [False, True],
                         ids=["auxiliary-L2", "native-FE"])
def test_ngs_complex_gaussian_has_independent_real_imaginary_draws(
        use_space_sampling):
    space = _noise_space(complex_storage=True)
    kwargs = {"seed": 3004, "use_space_sampling": use_space_sampling,
              "mass_matrix_weighting": False}
    first = space.randn(**kwargs)
    second = space.randn(**kwargs)
    coefficients = _native_coefficients(space, first)
    assert np.iscomplexobj(coefficients)
    assert np.any(coefficients.imag != 0)
    assert not np.array_equal(coefficients.real, coefficients.imag)
    np.testing.assert_array_equal(coefficients,
                                  _native_coefficients(space, second))


@pytest.mark.parametrize("space_class", [
    NgsVectorSpace, NgsVectorSpaceWithInnerProduct,
], ids=["vector-space", "inner-product-space"])
@pytest.mark.parametrize("kind", ["scalar", "vector", "product"])
def test_ngs_poisson_accepts_nonnegative_native_rates_and_is_reproducible(
        space_class, kind):
    # Poisson sampling always uses native dofs, regardless of FE type.
    # The inner-product variant supports scalar/vector but not product FE.
    if kind == "product" and space_class is NgsVectorSpaceWithInnerProduct:
        pytest.skip("NgsVectorSpaceWithInnerProduct does not accept product FEs")
    space = _noise_space(kind, space_class=space_class)
    rates = np.linspace(0., 4., space.fes.ndof)
    x = space.fromflat(rates)
    original = _native_coefficients(space, x)
    first = space.poisson(x, seed=51)
    second = space.poisson(x, seed=51)
    assert first in space
    output = _native_coefficients(space, first)
    np.testing.assert_array_equal(output, _native_coefficients(space, second))
    np.testing.assert_array_equal(original, _native_coefficients(space, x))
    assert np.all(np.isfinite(output)) and np.all(output >= 0.)
    assert np.all(output == np.floor(output))
    assert output[0] == 0.  # A zero rate must produce a zero count.
    averaged = space.poisson(x, n=3, seed=62)
    assert averaged in space
    assert np.all(_native_coefficients(space, averaged) >= 0.)
    np.testing.assert_array_equal(_native_coefficients(space, averaged),
                                  _native_coefficients(
                                      space, space.poisson(x, n=3, seed=62)))


@pytest.mark.parametrize("invalid_rate", [-0.1, np.nan, np.inf])
def test_ngs_poisson_rejects_invalid_rate_coefficients_without_mutation(
        invalid_rate):
    space = _noise_space()
    rates = np.ones(space.fes.ndof)
    rates[0] = invalid_rate
    x = space.fromflat(rates)
    before = _native_coefficients(space, x)
    with pytest.raises(ValueError, match="nonnegative"):
        space.poisson(x, seed=4)
    np.testing.assert_array_equal(_native_coefficients(space, x), before)


def test_ngs_poisson_clips_tiny_negative_roundoff_without_modifying_rates():
    space = _noise_space()
    rates = np.ones(space.fes.ndof)
    rates[0] = -1.e-15
    x = space.fromflat(rates)
    result = space.poisson(x, tol=3.e-15, seed=27)
    assert _native_coefficients(space, result)[0] == 0.
    assert _native_coefficients(space, x)[0] == -1.e-15


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_ngs_poisson_rejects_invalid_sample_count(n):
    space = _noise_space()
    with pytest.raises(ValueError, match="positive integer"):
        space.poisson(space.fromflat(np.ones(space.fes.ndof)), n=n)


@pytest.mark.parametrize("as_real", [False, True],
                         ids=["complex-coordinates", "real-coordinates"])
def test_ngs_poisson_rejects_complex_storage_even_for_real_coefficients(
        as_real):
    space = _noise_space(complex_storage=True)
    space.as_real_space = as_real
    rates = space.zeros()
    rates.vec.FV().NumPy()[:] = 2. + 0.j
    with pytest.raises(TypeError, match="real"):
        space.poisson(rates, seed=16)
    with pytest.raises(TypeError, match="real"):
        space.rand("poisson", lam=2., use_space_sampling=True, seed=16)


def test_ngs_vector_bitwise_operations_are_not_supported():
    """Coordinate selection belongs to the mask API, not vector operators."""
    space = _noise_space()
    a = space.zeros()
    b = space.ones()
    for operation in (operator.and_, operator.or_, operator.xor):
        with pytest.raises(TypeError):
            operation(a, b)


def _noise_setting(*, complex_storage=False, exact_coefficients=2.,
                   codomain_free=False, as_real_space=True):
    """Setting with native FE observations, not interpolated 'ones'."""
    space = _noise_space(complex_storage=complex_storage,
                         as_real_space=as_real_space)
    coefficients = np.full(space.fes.ndof, exact_coefficients,
                           dtype=complex if complex_storage else float)
    exact = space.zeros()
    exact.vec.FV().NumPy()[:] = coefficients
    setting = Setting(Identity(space), penalty=L2(space),
                      data_fid=L2(space), exact_data=exact, regpar=0.1,
                      codomain_free=codomain_free)
    return space, exact, setting


@pytest.mark.parametrize("codomain_free", [False, True],
                         ids=["explicit-codomain", "codomain-free"])
def test_ngs_setting_poisson_generates_reproducible_real_data(codomain_free):
    space, exact, setting = _noise_setting(codomain_free=codomain_free)
    original = _native_coefficients(space, exact)
    setting.generate_Poisson_data(expected_nr_counts=100., seed=53)
    first = _native_coefficients(space, setting.data)
    assert setting.data in space
    assert np.all(np.isfinite(first)) and np.all(first >= 0.)
    assert np.isfinite(setting._noise_model_param["absolute_noise_level"])
    assert setting._noise_model_param["model_name"] == "Poisson"
    assert setting._noise_model_param["expected_nr_counts"] == 100.
    setting.generate_Poisson_data(expected_nr_counts=100., seed=53)
    np.testing.assert_array_equal(_native_coefficients(space, setting.data),
                                  first)
    np.testing.assert_array_equal(_native_coefficients(space, exact), original)


@pytest.mark.parametrize("as_real_space", [True, False],
                         ids=["real-coordinates", "complex-scalar-field"])
def test_ngs_setting_poisson_rejects_complex_storage(as_real_space):
    space, _, setting = _noise_setting(complex_storage=True,
                                      as_real_space=as_real_space)
    # Even zero imaginary coefficients cannot turn complex FE storage into
    # an admissible Poisson rate space.
    assert space.as_real_space is as_real_space
    with pytest.raises(TypeError, match="real"):
        setting.generate_Poisson_data(seed=17)
    assert setting.data is None


def test_ngs_setting_poisson_rejects_negative_rates():
    space, _, setting = _noise_setting(exact_coefficients=-0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        setting.generate_Poisson_data(seed=19)
    assert setting.data is None


@pytest.mark.parametrize("white_noise", [False, True],
                         ids=["normalized", "white"])
def test_ngs_setting_gaussian_noise_is_seeded_and_reusable(white_noise):
    space, exact, setting = _noise_setting()
    original = _native_coefficients(space, exact)
    setting.add_Gaussian_noise(absolute_noise_level=0.05,
                               white_noise=white_noise, seed=31)
    first = _native_coefficients(space, setting.data)
    assert setting.data in space
    assert np.all(np.isfinite(first))
    assert setting._noise_model_param["model_name"] == "Gaussian"
    assert setting._noise_model_param["white_noise"] is white_noise
    assert setting._noise_model_param["estimated_absolute_noise_level"] == .05
    if not white_noise:
        difference = setting.data - exact
        assert setting.h_codomain.norm(difference) == pytest.approx(.05,
                                                                   rel=1e-8)
    setting.add_Gaussian_noise(absolute_noise_level=0.05,
                               white_noise=white_noise, seed=31)
    np.testing.assert_allclose(_native_coefficients(space, setting.data), first,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(_native_coefficients(space, exact), original)
    # Model information must still be usable by the synthetic-data dispatcher.
    setting.generate_synthetic_data()
    assert setting.data in space


@pytest.mark.parametrize("as_real_space", [True, False],
                         ids=["real-coordinates", "complex-scalar-field"])
def test_ngs_setting_complex_gaussian_is_supported(as_real_space):
    """Complex storage is forbidden for Poisson, not for Gaussian noise."""
    space, exact, setting = _noise_setting(complex_storage=True,
                                           as_real_space=as_real_space)
    setting.add_Gaussian_noise(absolute_noise_level=0.08,
                               white_noise=False, seed=842)
    observation = _native_coefficients(space, setting.data)
    assert np.iscomplexobj(observation)
    assert np.all(np.isfinite(observation))
    assert setting.h_codomain.norm(setting.data - exact) == pytest.approx(
        0.08, rel=1e-8)


# ---------------------------------------------------------------------------
# NGSolve inner products: complex and lazy arithmetic must be compatible with
# Setting's synthetic data. Keep a genuine numerical contract, not just a
# "does not raise" check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("complex_storage, as_real_space", [
    (False, True), (True, True), (True, False),
], ids=["real-storage", "complex-real-coordinates", "complex-scalar-field"])
@pytest.mark.parametrize("space_class", [
    NgsVectorSpace, NgsVectorSpaceWithInnerProduct,
], ids=["native", "mass-matrix"])
def test_ngs_vdot_accepts_lazy_operands_and_preserves_conjugation(
        complex_storage, as_real_space, space_class):
    space = _noise_space(complex_storage=complex_storage,
                         space_class=space_class,
                         as_real_space=as_real_space)
    assert space.as_real_space is as_real_space
    n = space.fes.ndof
    first = space.zeros()
    second = space.zeros()
    u = np.linspace(0.1, 0.9, n)
    v = np.linspace(-0.3, 0.6, n)
    if complex_storage:
        u = u + 1j * np.linspace(0.7, -0.2, n)
        v = v + 1j * np.linspace(-0.1, 0.8, n)
    first.vec.FV().NumPy()[:] = u
    second.vec.FV().NumPy()[:] = v

    # Arithmetic returns lazy NGSolve vector expressions.
    lazy_first = first + 2.0 * second
    lazy_second = second - first
    saved_first = _native_coefficients(space, first)
    saved_second = _native_coefficients(space, second)

    if space_class is NgsVectorSpaceWithInnerProduct:
        matrix = space.mass
        mass_applied = space.mass_vec.CreateVector()
        mass_applied.data = matrix * lazy_first.vec.Evaluate()
        expected = np.vdot(mass_applied.FV().NumPy(),
                           _native_coefficients(space, lazy_second))
    else:
        expected = np.vdot(_native_coefficients(space, lazy_first),
                           _native_coefficients(space, lazy_second))

    np.testing.assert_allclose(space.vdot(lazy_first, lazy_second),
                               expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(space.vdot(lazy_second, lazy_first),
                               np.conjugate(expected), rtol=1e-12, atol=1e-12)
    # Check the combinations with only one dynamic argument, too.
    assert np.isfinite(space.vdot(lazy_first, second))
    assert np.isfinite(space.vdot(first, lazy_second))
    np.testing.assert_array_equal(_native_coefficients(space, first), saved_first)
    np.testing.assert_array_equal(_native_coefficients(space, second), saved_second)


@pytest.mark.parametrize("as_real_space", [True, False],
                         ids=["real-coordinates", "complex-scalar-field"])
@pytest.mark.parametrize("space_class", [
    NgsVectorSpace, NgsVectorSpaceWithInnerProduct,
], ids=["native", "mass-matrix"])
def test_ngs_complex_l2_norm_of_lazy_observation_difference(
        as_real_space, space_class):
    # Complex *storage* triggers the backend bug regardless of the scalar-field
    # choice. Exercise both, including the mass-matrix vdot subclass.
    space = _noise_space(complex_storage=True, space_class=space_class,
                         as_real_space=as_real_space)
    assert space.as_real_space is as_real_space
    a = space.zeros()
    b = space.zeros()
    a.vec.FV().NumPy()[:] = (1. + 0.25j)
    b.vec.FV().NumPy()[:] = np.linspace(0.1, 0.7, space.fes.ndof) * (1. - 2j)
    lazy_delta = a - b
    native_delta = (_native_coefficients(space, a) -
                    _native_coefficients(space, b))
    if space_class is NgsVectorSpaceWithInnerProduct:
        mass_applied = space.mass_vec.CreateVector()
        mass_applied.data = space.mass * lazy_delta.vec.Evaluate()
        norm_sq = np.vdot(native_delta, mass_applied.FV().NumPy()).real
        target = np.sqrt(norm_sq)
    else:
        target = np.linalg.norm(native_delta)
    np.testing.assert_allclose(space.norm(lazy_delta), target,
                               rtol=1e-12, atol=1e-12)
