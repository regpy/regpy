import numpy as np
from numpy import pi 
from numpy.linalg import norm
from scipy.fft import fftshift, ifftshift
from scipy.signal import resample
import pytest

from regpy.vecsps.numpy import *
from regpy.vecsps import GridFcts
from regpy.operators.convolution import *
from regpy.operators.numpy import FourierTransform
from regpy.util import Errors

from .base_operator import op_basics_wrapper,op_evaluation_and_ot
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)


def _centered_axis(size, spacing=1.0):
    return np.arange(-(size // 2), (size + 1) // 2) * spacing


class TestPaddingOperator():

    @pytest.mark.parametrize("vs, pad_amount",[ 
        (UniformGridFcts(3,3),[2,3]),
        (UniformGridFcts(2,3,dtype=complex),2)
    ])
    def test_op_basic(self,vs,pad_amount):
        op_basics_wrapper(PaddingOperator,vs,test_methods=True, pad_amount = pad_amount)

    @pytest.mark.parametrize("vs, pad_amount, x, res",[ 
        (UniformGridFcts(3,3),0, np.ones((3,3)),np.ones((3,3))),
        (UniformGridFcts(2,2),[1,2], np.ones((2,2)), np.asarray([[0,0,0,0,0,0],[0,0,1.,1.,0,0],[0,0,1.,1.,0,0],[0,0,0,0,0,0]])),
        (UniformGridFcts(2,3,dtype=complex),2,np.ones((2,3))*1j, np.asarray([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,1j,1j,1j,0,0],[0,0,1j,1j,1j,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]))
    ])
    def test_ot_eval(self,vs,pad_amount,x,res):
        op=PaddingOperator(vs,pad_amount=pad_amount)
        op_evaluation_and_ot(op,x=x,res=res)

    def test_vector_valued_padding_and_positive_axis(self):
        grid = UniformGridFcts(
            _centered_axis(3),
            _centered_axis(4),
            dtype=float,
            shape_codomain=(2,),
        )
        op = PaddingOperator(
            grid,
            pad_amount=[1, 2, 0],
            positive_axis=1,
        )
        x = np.arange(np.prod(grid.shape), dtype=float).reshape(grid.shape)
        expected = np.pad(x, ((1, 1), (0, 2), (0, 0)))

        assert op.pad_amount == ((1, 1), (0, 2), (0, 0))
        assert op.codomain.shape == (5, 6, 2)
        op_evaluation_and_ot(op, x=x, res=expected, sample_N=1)
        np.testing.assert_array_equal(op.adjoint(expected), x)

    def test_nonzero_padding_value_and_derivative(self):
        grid = UniformGridFcts(_centered_axis(6), dtype=float)
        op = PaddingOperator(grid, pad_amount=2, pad_value=3.)
        x = np.arange(6., dtype=float)
        h = np.linspace(-1., 1., 6)
        value, derivative = op.linearize(x)

        assert op.linear is False
        np.testing.assert_array_equal(
            value,
            np.pad(x, 2, constant_values=3.),
        )
        np.testing.assert_array_equal(derivative(h), np.pad(h, 2))
        np.testing.assert_array_equal(
            derivative.adjoint(derivative(h)), h
        )

    @pytest.mark.parametrize("operator", [PaddingOperator, TruncationOperator])
    def test_zero_amount_returns_independent_vector(self, operator):
        grid = UniformGridFcts(_centered_axis(6), dtype=float)
        x = np.arange(6., dtype=float)
        op = operator(grid, 0)

        result = op(x)

        np.testing.assert_array_equal(result, x)
        assert result is not x
        assert not np.shares_memory(result, x)

    def test_truncation_of_nonuniform_grid(self):
        grid = GridFcts(
            np.array([0., 0.2, 1., 2.]),
            dtype=float,
            shape_codomain=(2,),
            use_cell_measure=False,
        )
        grid.measure = np.array([1., 2., 3., 4.])
        op = TruncationOperator(grid, 1, positive_axis=0)
        x = np.arange(8., dtype=float).reshape(4, 2)

        assert type(op.codomain) is GridFcts
        np.testing.assert_array_equal(op.codomain.axes[0], grid.axes[0][:-1])
        np.testing.assert_array_equal(
            op.codomain.measure[..., 0], grid.measure[:-1, 0]
        )
        np.testing.assert_array_equal(op(x), x[:-1])
        expected_adjoint = np.zeros_like(x)
        expected_adjoint[:-1] = x[:-1]
        np.testing.assert_array_equal(op.adjoint(x[:-1]), expected_adjoint)

    def test_padding_and_truncation_validation(self):
        scalar_grid = UniformGridFcts(_centered_axis(4), dtype=float)
        vector_grid = scalar_grid.vector_valued_space(2)

        with pytest.raises(ValueError):
            PaddingOperator(scalar_grid, -1)
        with pytest.raises(ValueError):
            PaddingOperator(vector_grid, [1, 1])
        with pytest.raises(ValueError):
            TruncationOperator(scalar_grid, 2)
        with pytest.raises(ValueError):
            TruncationOperator(vector_grid, [1, 1])

@pytest.mark.parametrize("vs, pad_amount, kernel",[ 
        (UniformGridFcts(8,12),2,np.arange(12*9).reshape(12,9)),
        (UniformGridFcts(8,12),2,lambda a,b:a*b*1j),
        (UniformGridFcts(10,10,dtype=np.complex128),2,np.arange(14*14).reshape(14,14)),
        (UniformGridFcts(10,10,dtype=np.complex128),2,lambda a,b:a*np.conj(b)),
    ])
class TestConvolutionOperator():
    def test_op_basic(self,vs,pad_amount,kernel):
        op_basics_wrapper(ConvolutionOperator,vs,test_methods=True,rel_tol_norm = 1e-3,
                          fourier_multiplier=kernel, pad_amount = pad_amount)

    def test_ot_eval(self,vs,pad_amount,kernel):
        op=ConvolutionOperator(vs,fourier_multiplier=kernel,pad_amount=2)
        op_evaluation_and_ot(op)

class TestConvolutionOperatorRegression:
    @pytest.mark.parametrize("size", [1, 2, 3, 4, 7, 8])
    def test_real_identity_convolution(self, size):
        grid = UniformGridFcts(_centered_axis(size), dtype=float)
        op = ConvolutionOperator(
            grid, fourier_multiplier=lambda frequency: 1.0
        )
        rng = np.random.default_rng(size)
        x = rng.normal(size=grid.shape)

        op_evaluation_and_ot(op, x=x, res=x, sample_N=1)

    @pytest.mark.parametrize(
        "shape, convolution_axes",
        [
            ((5, 7), (0,)),
            ((4, 6, 3), (1, 0)),
        ],
    )
    def test_partial_and_reordered_transform_axes(
            self, shape, convolution_axes):
        grid = UniformGridFcts(
            *[_centered_axis(size) for size in shape], dtype=float
        )
        ft = FourierTransform(grid, axes=convolution_axes)
        multiplier = 1.0 / (
            1.0 + sum(coord**2 for coord in ft.codomain.coords)
        )
        op = ConvolutionOperator(
            grid,
            fourier_multiplier=multiplier,
            convolution_axes=convolution_axes,
        )
        rng = np.random.default_rng(sum(shape) + sum(convolution_axes))
        x = rng.normal(size=grid.shape)
        expected = ft.left_inverse(multiplier * ft(x))

        op_evaluation_and_ot(op, x=x, res=expected, sample_N=1)

    @pytest.mark.parametrize(
        "source_size, target_size",
        [
            (2, 4),
            (4, 2),
            (4, 8),
            (8, 4),
            (5, 9),
            (9, 5),
            (8, 12),
            (12, 8),
        ],
    )
    def test_real_fourier_resizing_matches_scipy_resample(
            self, source_size, target_size):
        grid = UniformGridFcts(_centered_axis(source_size), dtype=float)
        amount = (source_size - target_size) // 2
        op = ConvolutionOperator(
            grid,
            fourier_multiplier=lambda frequency: 1.0,
            Fourier_truncation_amount=amount,
        )
        rng = np.random.default_rng(source_size * 100 + target_size)
        x = rng.normal(size=grid.shape)
        expected = fftshift(resample(ifftshift(x), target_size))

        assert op.codomain.shape == (target_size,)
        op_evaluation_and_ot(
            op, x=x, res=expected, sample_N=1,
            rel_tol=2e-11, tol=2e-11,
        )

    @pytest.mark.parametrize(
        "source_size, target_size",
        [
            pytest.param(9, 5, id="Fourier-truncation"),
            pytest.param(5, 9, id="Fourier-zero-extension"),
        ],
    )
    def test_fourier_truncation_and_zero_extension(
            self, source_size, target_size):
        grid = UniformGridFcts(_centered_axis(source_size), dtype=float)
        amount = (source_size - target_size) // 2
        op = ConvolutionOperator(
            grid,
            fourier_multiplier=lambda frequency: 1.0,
            Fourier_truncation_amount=amount,
        )
        source_ft = FourierTransform(grid, centered=True)
        target_ft = FourierTransform(op.codomain, centered=True)
        rng = np.random.default_rng(1000 + source_size * 10 + target_size)
        x = rng.normal(size=grid.shape)

        source_spectrum = source_ft(x)
        expected_spectrum = target_ft.codomain.zeros()
        retained_size = min(
            source_spectrum.shape[0], expected_spectrum.shape[0]
        )
        expected_spectrum[:retained_size] = (
            np.sqrt(target_size / source_size)
            * source_spectrum[:retained_size]
        )
        expected = target_ft.left_inverse(expected_spectrum)

        op_evaluation_and_ot(op, x=x, res=expected, sample_N=1)
        actual_spectrum = target_ft(op(x))
        np.testing.assert_allclose(
            actual_spectrum, expected_spectrum, rtol=2e-12, atol=2e-12
        )
        if target_size > source_size:
            np.testing.assert_allclose(
                actual_spectrum[source_spectrum.shape[0]:], 0,
                rtol=0, atol=2e-12,
            )

    @pytest.mark.parametrize(
        "source_size, target_size, coefficient, nyquist_factor",
        [
            pytest.param(4, 8, 2.0, 0.5, id="split-source-Nyquist"),
            pytest.param(8, 4, 1.5, 2.0, id="collapse-to-target-Nyquist"),
        ],
    )
    def test_even_length_nyquist_splitting_and_collapsing(
            self, source_size, target_size, coefficient, nyquist_factor):
        grid = UniformGridFcts(_centered_axis(source_size), dtype=float)
        amount = (source_size - target_size) // 2
        op = ConvolutionOperator(
            grid,
            fourier_multiplier=lambda frequency: 1.0,
            Fourier_truncation_amount=amount,
        )
        source_ft = FourierTransform(grid, centered=True)
        target_ft = FourierTransform(op.codomain, centered=True)

        source_spectrum = source_ft.codomain.zeros()
        retained_index = min(
            source_ft.codomain.shape_domain[0],
            target_ft.codomain.shape_domain[0],
        ) - 1
        source_spectrum[retained_index] = coefficient
        x = source_ft.left_inverse(source_spectrum)

        expected_spectrum = target_ft.codomain.zeros()
        expected_spectrum[retained_index] = (
            np.sqrt(target_size / source_size)
            * nyquist_factor
            * coefficient
        )
        expected = target_ft.left_inverse(expected_spectrum)

        op_evaluation_and_ot(op, x=x, res=expected, sample_N=1)
        np.testing.assert_allclose(
            target_ft(op(x)), expected_spectrum, rtol=2e-12, atol=2e-12
        )


class TestConvolutionMetadata:
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_scalar_self_adjoint_detection_and_opt_out(self, dtype):
        grid = UniformGridFcts(_centered_axis(8), dtype=dtype)
        spectrum_size = 5 if dtype is float else 8
        multiplier = np.linspace(1., 2., spectrum_size)

        detected = ConvolutionOperator(grid, multiplier, as_svd=False)
        opted_out = ConvolutionOperator(
            grid,
            multiplier,
            as_svd=False,
            self_adjoint=False,
        )

        assert detected.self_adjoint is True
        assert detected.adjoint is detected
        assert opted_out.self_adjoint is False
        op_evaluation_and_ot(detected, sample_N=1)

    def test_roundoff_is_ignored_but_complex_multiplier_is_not(self):
        grid = UniformGridFcts(_centered_axis(8), dtype=complex)
        real_multiplier = np.arange(1., 9.)

        rounding_perturbation = ConvolutionOperator(
            grid,
            real_multiplier + 1e-14j,
            as_svd=False,
        )
        genuinely_complex = ConvolutionOperator(
            grid,
            real_multiplier + 1e-3j,
            as_svd=False,
        )

        assert rounding_perturbation.self_adjoint is True
        assert genuinely_complex.self_adjoint is False
        with pytest.raises(ValueError):
            ConvolutionOperator(
                grid,
                real_multiplier + 1e-3j,
                as_svd=False,
                self_adjoint=True,
            )

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_matrix_valued_self_adjoint_detection(self, dtype):
        grid = UniformGridFcts(
            _centered_axis(8), dtype=dtype, shape_codomain=(2,)
        )
        spectrum_size = 5 if dtype is float else 8
        hermitian = np.zeros((spectrum_size, 2, 2), dtype=complex)
        hermitian[..., 0, 0] = 1.
        hermitian[..., 1, 1] = 2.
        hermitian[..., 0, 1] = 0.2j
        hermitian[..., 1, 0] = -0.2j
        nonhermitian = hermitian.copy()
        nonhermitian[..., 1, 0] = 0.1j

        detected = ConvolutionOperator(
            grid,
            hermitian,
            kernel_matrix_shape=(2, 2),
            as_svd=False,
        )
        rejected = ConvolutionOperator(
            grid,
            nonhermitian,
            kernel_matrix_shape=(2, 2),
            as_svd=False,
        )

        assert detected.self_adjoint is True
        assert rejected.self_adjoint is False
        op_evaluation_and_ot(detected, sample_N=1)
        with pytest.raises(ValueError):
            ConvolutionOperator(
                grid,
                nonhermitian,
                kernel_matrix_shape=(2, 2),
                as_svd=False,
                self_adjoint=True,
            )

    def test_padding_resizing_and_affine_padding(self):
        grid = UniformGridFcts(_centered_axis(8), dtype=float)
        padded = ConvolutionOperator(
            grid,
            lambda frequency: 1.,
            pad_amount=2,
            as_svd=False,
        )
        affine = ConvolutionOperator(
            grid,
            lambda frequency: 1.,
            pad_amount=2,
            pad_value=1.,
            as_svd=False,
        )
        resized = ConvolutionOperator(
            grid,
            lambda frequency: 1.,
            Fourier_truncation_amount=1,
            as_svd=False,
        )

        assert padded.self_adjoint is True
        assert affine.linear is False
        assert affine.self_adjoint is False
        assert resized.self_adjoint is False
        with pytest.raises(ValueError):
            ConvolutionOperator(
                grid,
                lambda frequency: 1.,
                pad_amount=2,
                pad_value=1.,
                as_svd=False,
                self_adjoint=True,
            )
        with pytest.raises(ValueError):
            ConvolutionOperator(
                grid,
                lambda frequency: 1.,
                Fourier_truncation_amount=1,
                as_svd=False,
                self_adjoint=True,
            )

    def test_padding_is_normalized_over_domain_and_value_axes(self):
        grid = UniformGridFcts(
            _centered_axis(4),
            _centered_axis(6),
            dtype=float,
            shape_codomain=(2,),
        )
        op = ConvolutionOperator(
            grid,
            lambda frequency: 1.,
            convolution_axes=(1,),
            pad_amount=[0, 2, 0],
            as_svd=False,
        )
        x = grid.randn()

        np.testing.assert_array_equal(op.pad_amount, [0, 2, 0])
        np.testing.assert_allclose(op(x), x, rtol=1e-13, atol=1e-13)
        assert op.self_adjoint is True

        with pytest.raises(ValueError):
            ConvolutionOperator(
                grid,
                lambda frequency: 1.,
                convolution_axes=(1,),
                pad_amount=[1, 2, 0],
                as_svd=False,
            )
        with pytest.raises(ValueError):
            ConvolutionOperator(
                grid,
                lambda frequency: 1.,
                convolution_axes=(1,),
                pad_amount=[0, 2, 1],
                as_svd=False,
            )

    def test_flags_of_concrete_convolution_operators(self):
        grid = UniformGridFcts(_centered_axis(8), dtype=float)

        assert GaussianBlur(grid).self_adjoint is True
        assert ExponentialConvolution(grid, a=0.5).self_adjoint is True
        assert Laplacian(grid).self_adjoint is True
        assert Derivative(grid, order=(1,)).self_adjoint is False
        assert Derivative(grid, order=(2,)).self_adjoint is True
        assert PeriodicShift(grid, shift=(0.,)).self_adjoint is True
        assert PeriodicShift(grid, shift=(1.,)).self_adjoint is False


class TestConvolutionSVDAndInverse:
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_automatic_svd_and_explicit_opt_out(self, dtype):
        grid = UniformGridFcts(_centered_axis(8), dtype=dtype)
        spectrum_size = 5 if dtype is float else 8
        multiplier = np.arange(1., spectrum_size + 1)
        svd = ConvolutionOperator(grid, multiplier)
        plain = ConvolutionOperator(grid, multiplier, as_svd=False)
        x = grid.randn()

        assert svd.is_svd is True
        assert svd.svd_data["is_eig"] is True
        assert plain.is_svd is False
        singular_values = np.asarray(svd.svd_data["M"].factor)
        np.testing.assert_allclose(singular_values.imag, 0.)
        assert np.all(singular_values.real > 0.)
        np.testing.assert_allclose(svd(x), plain(x), rtol=1e-13, atol=1e-13)

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_svd_with_phase_and_zero_singular_value(self, dtype):
        grid = UniformGridFcts(_centered_axis(8), dtype=dtype)
        spectrum_size = 5 if dtype is float else 8
        multiplier = np.arange(1., spectrum_size + 1).astype(complex)
        multiplier *= 1. + 0.2j
        multiplier[1] = 0.
        svd = ConvolutionOperator(grid, multiplier)
        plain = ConvolutionOperator(grid, multiplier, as_svd=False)
        x = grid.randn()

        assert svd.is_svd is True
        assert svd.svd_data["is_eig"] is False
        singular_values = np.asarray(svd.svd_data["M"].factor)
        np.testing.assert_allclose(singular_values.imag, 0.)
        assert np.all(singular_values.real > 0.)
        np.testing.assert_allclose(svd(x), plain(x), rtol=1e-13, atol=1e-13)

    def test_explicit_svd_rejects_unsupported_convolutions(self):
        real_grid = UniformGridFcts(_centered_axis(8), dtype=float)
        real_grid_2d = UniformGridFcts(
            _centered_axis(4), _centered_axis(4), dtype=float
        )
        vector_grid = real_grid.vector_valued_space(2)

        with pytest.raises(TypeError):
            ConvolutionOperator(real_grid, np.ones(5), as_svd=1)
        with pytest.raises(ValueError):
            ConvolutionOperator(
                real_grid,
                lambda frequency: 1.,
                pad_amount=1,
                as_svd=True,
            )
        with pytest.raises(NotImplementedError):
            ConvolutionOperator(
                real_grid,
                lambda frequency: 1.,
                Fourier_truncation_amount=1,
                as_svd=True,
            )
        with pytest.raises(NotImplementedError):
            ConvolutionOperator(
                vector_grid,
                np.broadcast_to(np.eye(2), (5, 2, 2)),
                kernel_matrix_shape=(2, 2),
                as_svd=True,
            )
        with pytest.raises(NotImplementedError):
            ConvolutionOperator(
                real_grid_2d,
                lambda frequency_x, frequency_y: 1.,
                as_svd=True,
            )

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_exact_scalar_inverse(self, dtype):
        grid = UniformGridFcts(_centered_axis(8), dtype=dtype)
        spectrum_size = 5 if dtype is float else 8
        multiplier = np.linspace(1., 2., spectrum_size)
        op = ConvolutionOperator(grid, multiplier, as_svd=False)
        x = grid.randn()

        assert op.invertible is True
        np.testing.assert_allclose(
            op.inverse(op(x)), x, rtol=2e-13, atol=2e-13
        )
        assert op.inverse.inverse is op

    def test_singular_and_padded_convolutions_have_no_exact_inverse(self):
        grid = UniformGridFcts(_centered_axis(8), dtype=float)
        singular = ConvolutionOperator(
            grid,
            np.array([1., 2., 0., 4., 5.]),
            as_svd=False,
        )
        padded = ConvolutionOperator(
            grid,
            np.arange(1., 8.),
            pad_amount=2,
            as_svd=False,
        )

        assert not singular.invertible
        assert not padded.invertible
        with pytest.raises(NotImplementedError):
            _ = singular.inverse
        with pytest.raises(NotImplementedError):
            singular.conv_inverse()
        with pytest.raises(NotImplementedError):
            _ = padded.inverse
        assert isinstance(padded.conv_inverse(), ConvolutionOperator)

    def test_convolution_calculus_adjoint(self):
        grid = UniformGridFcts(_centered_axis(8), dtype=complex)
        multiplier = np.arange(1., 9.) * (1. + 0.3j)
        op = ConvolutionOperator(grid, multiplier, as_svd=False)
        x = grid.randn()

        calculus_adjoint = op.conv_adjoint()
        np.testing.assert_allclose(
            calculus_adjoint(x), op.adjoint(x), rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            calculus_adjoint.fourier_multiplier,
            np.conjugate(multiplier),
        )


@pytest.mark.parametrize("vs, sigma, pad_amount, convolution_axes",[ 
        (UniformGridFcts(10,10),5,2,[1]),
        (UniformGridFcts(10,10,dtype=np.complex128),5,2,[1]),
    ])
class TestGaussianBlur():
    def test_op_basic(self,vs,sigma,pad_amount,convolution_axes):
        op_basics_wrapper(GaussianBlur,vs,test_methods=True, sigma = sigma, pad_amount=pad_amount,convolution_axes=convolution_axes)

    def test_ot_eval(self,vs,sigma,pad_amount,convolution_axes):
        op=GaussianBlur(vs,sigma=sigma,pad_amount=pad_amount,convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)

@pytest.mark.parametrize("vs, a, pad_amount, convolution_axes",[ 
        (UniformGridFcts(10,10),0.5,2,[0]),
        (UniformGridFcts(10,10,dtype=np.complex128),0.5,2,None),
    ])
class TestExponentialConvolution():
    def test_op_basic(self,vs,a,pad_amount,convolution_axes):
        op_basics_wrapper(ExponentialConvolution,vs,test_methods=True, a = a, pad_amount=pad_amount,convolution_axes=convolution_axes)

    def test_ot_eval(self,vs,a,pad_amount,convolution_axes):
        op=ExponentialConvolution(vs,a=a,pad_amount=pad_amount,convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)

@pytest.mark.parametrize("vs, fresnel_number, pad_amount, convolution_axes",[ 
        (UniformGridFcts(10,10,dtype=np.complex128),2.5,2,None),
    ])
class TestFresnelPropagator():
    def test_op_basic(self,vs,fresnel_number, pad_amount, convolution_axes):
        op_basics_wrapper(FresnelPropagator,vs,fresnel_number,test_methods=True, rel_tol_norm=1e-3, pad_amount=pad_amount,convolution_axes=convolution_axes)
    
    def test_ot_eval(self,vs,fresnel_number, pad_amount, convolution_axes):
        op=FresnelPropagator(vs,fresnel_number,pad_amount=pad_amount, convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)


class TestDifferentialOperators():
    errors = []

    grid = UniformGridFcts((-pi,pi,10), (-pi,pi,9),dtype=float,shape_codomain=(1,))
    pad_amount = [2,0]

    @pytest.mark.parametrize("op, vs, pad_amount,Fourier_truncation_amount",[ 
        (gradient,grid.vector_valued_space(1),[2,0],[-2,2]),
        (divergence,grid.vector_valued_space(2),[2,0],None),
        (Laplacian,grid.scalar_space(),[2,0],2),
    ])
    def test_op_basic(self, op, vs, pad_amount,Fourier_truncation_amount):
        op_basics_wrapper(op,vs,pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount)

    @pytest.mark.parametrize("grid, pad_amount,Fourier_truncation_amount",[
        (grid,[0,2],0),
        (grid.complex_space(),[0,2],0),
        (grid,2,[-2,2]),
        (grid.complex_space(),2,[-2,2])
    ])
    def test_compatibility(self,grid,pad_amount,Fourier_truncation_amount):
        grad =  gradient(grid.vector_valued_space(1),pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount)
        div = divergence(grid.vector_valued_space(2),pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount) 
        div_grad = div.composition(grad)
        Lap = Laplacian(grid,pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount,kernel_matrix_shape=(1,1))
        assert np.allclose(Lap.fourier_multiplier,div_grad.fourier_multiplier), Errors.failed_test(f"Comparing the Fourier multiplier of Laplace to div_grad composition is not close!",meth="Differential Operators")

        Lap_scal = Laplacian(grid,pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount)
        Dxsq = Derivative(grid,order=(2,0),pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount) 
        Dysq = Derivative(grid,order=(0,2),pad_amount=pad_amount,Fourier_truncation_amount=Fourier_truncation_amount)         
        Lap_as_sum = Dxsq+Dysq
        assert np.allclose(Lap_scal.fourier_multiplier,Lap_as_sum.fourier_multiplier), Errors.failed_test(f"Comparing the Fourier multiplier of Laplace to sum of second derivative operators is not close!",meth="Differential Operators")

    @pytest.mark.parametrize("op, vs, pad_amount",[ 
        (gradient,grid.vector_valued_space(1),[2,0]),
        (divergence,grid.vector_valued_space(2),[2,0]),
        (Laplacian,grid.scalar_space(),[2,0]),
    ])
    def test_op_eval(self,op,vs,pad_amount):
        op_evaluation_and_ot(op(vs,pad_amount=pad_amount))

    # test identities curl grad = 0,  div curl = 0, and \Delta = grad div - curl curl
    # different implementations of convolution operators with and without padding 
    @pytest.mark.parametrize("type,pad_amount",[ 
        (float,0),
        (float,[2,0,3]),
        (complex,0),
        (complex,[2,0,3]),
    ])
    def test_identies(self,type,pad_amount):
        grid3D = UniformGridFcts((-pi,pi,20), (-pi,pi,24),(-pi,pi,15),dtype=type,shape_codomain=(1,))

        grad = gradient(grid3D.vector_valued_space(1),pad_amount=pad_amount)   
        curlop = curl(grid3D.vector_valued_space(3),pad_amount=pad_amount)
        div = divergence(grid3D.vector_valued_space(3),pad_amount=pad_amount)
        Lap3D = Laplacian(grid3D.vector_valued_space(3),pad_amount=pad_amount,kernel_matrix_shape=(3,3))
    

        curl_grad =  curlop.composition(grad)
        assert norm(curl_grad.fourier_multiplier)==0
        div_curl =  div.composition(curlop)
        assert norm(div_curl.fourier_multiplier)==0
        test = grad.composition(div) - curlop.composition(curlop) 
        test -= Lap3D
        assert norm(test.fourier_multiplier)<=1e-10

        # make sure this also holds true approximately with periodization errors
        X,Y,Z = grid3D.coords
        f3d = grid3D.zeros()
        f3d[...,0] = np.exp(-400*(X**2+Y**2+Z**2))*np.cos(3*X-Z+2*Z)
        g3d = grid3D.vector_valued_space(3).zeros()
        g3d[...,0] = f3d[...,0]
        g3d[...,1] = np.exp(-400*(X**2+Y**2+Z**2))*np.sin(3*X)
        g3d[...,2] = np.exp(-400*(X**2+Y**2+Z**2))*np.sin(-Z+2*Y)

        assert np.allclose(curlop(grad(f3d)),np.zeros_like(g3d))
        assert np.allclose(grad.adjoint(curlop(g3d)),np.zeros_like(f3d))
        assert np.allclose(grad(div(g3d))-curlop.adjoint(curlop(g3d)) ,  Lap3D(g3d),atol=1e-6)

class TestShiftConvolutionCalculus():
    @pytest.mark.parametrize("vs,shift,pad_amount,convolution_axes", [ 
        (UniformGridFcts((-pi,pi,50),dtype=np.complex128),[1.],2,None)
    ])
    def test_op_basic(self,vs,shift,pad_amount,convolution_axes):
        op_basics_wrapper(PeriodicShift,vs,shift,test_methods=True, rel_tol_norm=1e-3, pad_amount=pad_amount,convolution_axes=convolution_axes)
    
    @pytest.mark.parametrize("vs,shift,pad_amount,convolution_axes", [ 
        (UniformGridFcts((-pi,pi,50),dtype=np.complex128),[1.],2,None)
    ])
    def test_ot_eval(self,vs,shift,pad_amount,convolution_axes):
        op=PeriodicShift(vs,shift,pad_amount=pad_amount,convolution_axes=convolution_axes)
        op_evaluation_and_ot(op)

    def test_periodic_shift(self):
        grid =  UniformGridFcts((-pi,pi,300),(-pi,pi,50),periodic=True,dtype=complex)
        X,Y = grid.coords
        f = np.exp(-100*(X**2+Y**2))
        g = np.exp(-100*((X+1)**2+Y**2))
        shift1 = PeriodicShift(grid,[1,0])
        fshift = shift1(f)
        assert norm(fshift-g)<=1e-10

        blur = GaussianBlur(grid,(0.1)**2)
        double_peak = 0.5*shift1.composition(blur) + blur
        assert norm(double_peak(f)-blur(f)-0.5*shift1(blur(f)))<1e-12
