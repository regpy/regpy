import numpy as np
from scipy.sparse import csc_array
import pytest

from regpy.vecsps import MeasureSpaceFcts, TupleVector
from regpy.vecsps.numpy import *
from regpy.operators.numpy import *

from .base_operator import op_basics_wrapper,op_evaluation_and_ot
from regpy.util import set_rng_seed

set_rng_seed(15873098306879350073259142812684978477)

class TestMatrixMultiplication():
    @pytest.mark.parametrize("matrix",[ 
        np.random.rand(3,5),
        np.random.rand(20,21),
        np.random.rand(20,21)+1j*np.random.rand(20,21)
    ])
    def test_on_random_matrix(self,matrix):
        op_basics_wrapper(MatrixMultiplication, matrix, test_methods=True)
        op = MatrixMultiplication(matrix)
        op_evaluation_and_ot(op)

    def test_eval(self):
        matrix = np.arange(4*7).reshape(4,7)
        vec = np.sum(matrix,axis=1)
        op = MatrixMultiplication(matrix=matrix)
        assert vec == pytest.approx(op(op.domain.ones()))

    def test_multidimensional_real_coordinates(self):
        domain = NumPyVectorSpace((2, 2))
        codomain = NumPyVectorSpace((2, 3))
        matrix = np.arange(24., dtype=float).reshape(6, 4) / 10
        op = MatrixMultiplication(
            matrix, domain=domain, codomain=codomain
        )
        x = np.arange(4., dtype=float).reshape(2, 2)
        y = np.arange(6., dtype=float).reshape(2, 3)

        expected = codomain.fromflat(matrix @ domain.flatten(x))
        expected_adjoint = domain.fromflat(
            matrix.T @ codomain.flatten(y)
        )
        np.testing.assert_allclose(op(x), expected)
        np.testing.assert_allclose(op.adjoint(y), expected_adjoint)
        assert op.complex_coordinates is False
        op_evaluation_and_ot(op, sample_N=2)

    def test_genuinely_complex_coordinates(self):
        domain = NumPyVectorSpace((2, 2), dtype=complex, as_real_space=False)
        codomain = NumPyVectorSpace(3, dtype=complex, as_real_space=False)
        matrix = (
            np.arange(12.).reshape(3, 4)
            + 1j * np.arange(12., 24.).reshape(3, 4)
        ) / 10
        op = MatrixMultiplication(
            matrix, domain=domain, codomain=codomain
        )
        x = (
            np.arange(4.).reshape(2, 2)
            + 1j * np.arange(4., 8.).reshape(2, 2)
        )
        y = np.array([1 + 2j, -0.5j, 3 - 1j])

        expected = codomain.fromflat(
            matrix @ domain.flatten(x)
        )
        expected_adjoint = domain.fromflat(
            matrix.conj().T @ codomain.flatten(y)
        )
        assert op.complex_coordinates is True
        np.testing.assert_allclose(op(x), expected)
        np.testing.assert_allclose(op.adjoint(y), expected_adjoint)
        op_evaluation_and_ot(op, sample_N=2)

    def test_realified_complex_coordinates(self):
        space = NumPyVectorSpace(2, dtype=complex)
        matrix = np.diag([1., 2., 3., 4.])
        op = MatrixMultiplication(
            matrix, domain=space, codomain=space
        )
        x = np.array([1 + 2j, 3 + 4j])
        expected = space.fromflat(matrix @ space.flatten(x))

        assert op.complex_coordinates is False
        assert op.self_adjoint is True
        assert op.adjoint is op
        np.testing.assert_allclose(op(x), expected)

    def test_self_adjoint_inference_for_complex_matrix(self):
        space = NumPyVectorSpace(2, dtype=complex)
        hermitian = np.array([
            [2., 1 + 2j],
            [1 - 2j, 3.],
        ])
        self_adjoint = MatrixMultiplication(
            hermitian, domain=space, codomain=space
        )
        not_self_adjoint = MatrixMultiplication(
            np.array([[1., 2j], [0., 1.]]),
            domain=space,
            codomain=space,
        )

        assert self_adjoint.self_adjoint is True
        assert self_adjoint.adjoint is self_adjoint
        assert not_self_adjoint.self_adjoint is False

    def test_coordinate_validation(self):
        real_space = NumPyVectorSpace((2, 2))
        complex_space = NumPyVectorSpace(2, dtype=complex)

        with pytest.raises(ValueError):
            MatrixMultiplication(
                np.eye(3), domain=real_space, codomain=real_space
            )
        with pytest.raises(ValueError):
            MatrixMultiplication(
                np.eye(4, dtype=complex) * (1 + 1j),
                domain=complex_space,
                codomain=complex_space,
            )


@pytest.mark.parametrize("axis", (0, 1, -1))
def test_complex2real(axis):
    domain = NumPyVectorSpace((2, 3), dtype=complex)
    op = Complex2Real(domain, axis=axis)
    x = (
        np.arange(6.).reshape(2, 3)
        + 1j * np.arange(6., 12.).reshape(2, 3)
    )
    expected = np.stack((x.real, x.imag), axis=axis)

    assert complex2real is Complex2Real
    np.testing.assert_allclose(op(x), expected)
    np.testing.assert_allclose(op.adjoint(expected), x)
    np.testing.assert_allclose(op.inverse(expected), x)
    assert op.inverse is op.adjoint
    assert op.invertible
    op_evaluation_and_ot(op, x=x, res=expected, sample_N=2)


def test_complex2real_validation():
    with pytest.raises(TypeError):
        Complex2Real(object())
    with pytest.raises(ValueError):
        Complex2Real(NumPyVectorSpace(4))
    with pytest.raises(TypeError):
        Complex2Real(NumPyVectorSpace(4, dtype=complex), axis=True)
    with pytest.raises(ValueError):
        Complex2Real(NumPyVectorSpace(4, dtype=complex), axis=2)
    
class TestCholeskyInverse():
    op_mat = MatrixMultiplication(np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]),domain= UniformGridFcts(4),codomain=UniformGridFcts(4))
    
    def test_op_basic(self):
        op_basics_wrapper(CholeskyInverse, self.op_mat, test_methods=True, inv_tol=1e-14)
    
    def test_ot_eval(self):
        op = CholeskyInverse(self.op_mat)
        op_evaluation_and_ot(op)
    
class TestSuperLUInverse():
    mat =  csc_array([[1,2,0,4], [1,0,0,1], [1,0,2,1], [2,2,1,0.]])
    op_mat = MatrixMultiplication(mat,domain= UniformGridFcts(4),codomain=UniformGridFcts(4))
    
    def test_op_basic(self):
        op_basics_wrapper(SuperLUInverse, self.op_mat, test_methods=True, inv_tol=1e-14)

    def test_ot_eval(self):
        op = SuperLUInverse(self.op_mat)
        op_evaluation_and_ot(op)
    
class TestPower():
    vs = NumPyVectorSpace((2,4),dtype=complex)
    
    @pytest.mark.parametrize("power, kwargs", [(3,{"integer":True}),(1.0,{"integer":True}),(1.5,{}),(-1.5,{})])
    def test_op_basic(self,power,kwargs):
        op_basics_wrapper(Power,power,self.vs,test_methods=True,**kwargs)

    @pytest.mark.parametrize("power, kwargs, x, res", [
        (1.5,{},(np.arange(8).reshape(2,4) + 1j*np.arange(8).reshape(2,4))**2,np.array([(-1+1j)*(2*i)*i**2 for i in range(8)]).reshape(2,4)),
        (3.0,{"integer":True},np.arange(8).reshape(2,4) +1j*np.arange(8).reshape(2,4),np.array([(-1+1j)*(2*i)*i**2 for i in range(8)]).reshape(2,4))])
    def test_ot_eval(self,power,kwargs,x,res):
        op = Power(power,self.vs,**kwargs)
        op_evaluation_and_ot(op,x=x,res=res)

@pytest.mark.parametrize("vs",[ 
        NumPyVectorSpace((2,4),dtype=complex),
        NumPyVectorSpace((3,5))
    ])
class TestExponential():

    def test_op_basic(self,vs):
        op_basics_wrapper(Exponential,vs,test_methods=True)

    def test_ot_eval(self,vs):
        op=Exponential(domain=vs)
        x=vs.randn()
        res = np.exp(x)

        op_evaluation_and_ot(op,x=x,res=res)

def _complex_exp_jacobian(z):
    derivative = np.exp(z)[..., np.newaxis, np.newaxis]
    return NemytskiiOperator.complex_matrix_to_real_jacobian(derivative)


class TestNemytskiiOperator():

    def test_real_exponential(self):
        domain = MeasureSpaceFcts(
            shape=(4,), shape_codomain=(), dtype=float
        )
        op = NemytskiiOperator(domain, np.exp, np.exp)
        x = np.array([-0.2, 0.1, 0.7, 1.2])

        op_evaluation_and_ot(op, x=x, res=np.exp(x), sample_N=2)

    def test_complex_exponential(self):
        domain = MeasureSpaceFcts(
            shape=(3,), shape_codomain=(), dtype=complex
        )
        op = NemytskiiOperator(
            domain, np.exp, _complex_exp_jacobian
        )
        x = np.array([0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j])
        h = np.array([0.1 - 0.2j, 0.3 + 0.4j, -0.5 + 0.2j])
        y = np.array([0.2 + 0.5j, -0.6 + 0.1j, 0.4 - 0.3j])

        op_evaluation_and_ot(op, x=x, res=np.exp(x), sample_N=2)

        _, derivative = op.linearize(x)
        assert derivative(h) == pytest.approx(np.exp(x) * h)
        assert derivative.adjoint(y) == pytest.approx(
            np.conj(np.exp(x)) * y
        )

    def test_linear_complex_to_real(self):
        domain = MeasureSpaceFcts(
            shape=(3,), shape_codomain=(), dtype=complex
        )
        op = NemytskiiOperator(
            domain,
            np.real,
            lambda z: np.array([[1.0, 0.0]]),
            linear=True,
        )
        x = np.array([0.2 + 0.3j, -0.4 + 0.1j, 0.7 - 0.2j])

        assert op.linear
        assert not op.codomain.is_complex
        op_evaluation_and_ot(op, x=x, res=x.real, sample_N=2)

    def test_vector_valued_map(self):
        domain = MeasureSpaceFcts(
            shape=(5,), shape_codomain=2, dtype=float
        )

        def f(v):
            return np.stack(
                (
                    v[..., 0] ** 2 + v[..., 1],
                    np.sin(v[..., 1]),
                ),
                axis=-1,
            )

        def Df(v):
            jac = np.zeros(v.shape[:-1] + (2, 2))
            jac[..., 0, 0] = 2 * v[..., 0]
            jac[..., 0, 1] = 1
            jac[..., 1, 1] = np.cos(v[..., 1])
            return jac

        op = NemytskiiOperator(domain, f, Df)
        x = np.array([
            [-0.4, 0.2],
            [0.1, -0.7],
            [0.8, 0.5],
            [-0.2, 1.0],
            [0.6, -0.3],
        ])

        op_evaluation_and_ot(op, x=x, res=f(x), sample_N=2)

    def test_multi_input_ufunc(self):
        domain = MeasureSpaceFcts(
            shape=(4,), shape_codomain=2, dtype=float
        )
        op = NemytskiiOperator(
            domain,
            np.add,
            lambda x: np.array([[1.0, 1.0]]),
            linear=True,
        )
        x = np.arange(8.0).reshape(4, 2)

        assert op.codomain.shape_codomain == ()
        op_evaluation_and_ot(
            op, x=x, res=x[..., 0] + x[..., 1], sample_N=2
        )


@pytest.mark.parametrize("size", (7, 8))
def test_real_fourier_embedding(size):
    grid = UniformGridFcts((-1, 1, size), periodic=True)
    transform = FourierTransform(grid, centered_input=False)
    embedding = RealFourierEmbedding(transform)
    coordinates = np.linspace(-0.7, 1.1, size)
    spectrum = embedding(coordinates)

    assert embedding.coordinate_space == embedding.domain
    assert spectrum[0].imag == pytest.approx(0.)
    if size % 2 == 0:
        assert spectrum[-1].imag == pytest.approx(0.)
    np.testing.assert_allclose(
        embedding.left_inverse(spectrum), coordinates,
        rtol=1e-14, atol=1e-14,
    )
    with pytest.raises(NotImplementedError):
        _ = embedding.inverse

    multiplier = np.linspace(0.5, 2., transform.codomain.shape[0])
    coordinate_multiplier = embedding.real_multiplier(multiplier)
    np.testing.assert_allclose(
        embedding(coordinate_multiplier * coordinates),
        multiplier * spectrum,
        rtol=1e-14,
        atol=1e-14,
    )
    with pytest.raises(ValueError):
        embedding.real_multiplier(multiplier + 1j)
    op_evaluation_and_ot(embedding, sample_N=2)


@pytest.mark.parametrize("size", (7, 8))
def test_real_fourier_transform(size):
    grid = UniformGridFcts((-1, 1, size), periodic=True)
    op = RealFourierTransform(
        grid, centered=True, centered_input=True
    )
    x = grid.randn()
    coefficients = op(x)

    assert not coefficients.dtype.kind == "c"
    assert coefficients.shape == (size,)
    assert op.coordinate_space == op.codomain
    assert op.invertible
    assert op.inverse is op.adjoint
    np.testing.assert_allclose(
        op.adjoint(coefficients), x, rtol=1e-13, atol=1e-13
    )
    op_evaluation_and_ot(op, sample_N=2)


def test_fourier_transform_inverse_contract():
    real_grid = UniformGridFcts((-1, 1, 8), periodic=True)
    real_transform = FourierTransform(
        real_grid, centered=True, centered_input=True
    )
    x = real_grid.randn()

    assert real_transform.invertible is False
    with pytest.raises(NotImplementedError):
        _ = real_transform.inverse
    np.testing.assert_allclose(
        real_transform.left_inverse(real_transform(x)),
        x,
        rtol=1e-13,
        atol=1e-13,
    )
    op_evaluation_and_ot(real_transform, sample_N=2)

    complex_grid = real_grid.complex_space()
    complex_transform = FourierTransform(
        complex_grid, centered=True, centered_input=True
    )
    z = complex_grid.randn()
    assert complex_transform.invertible
    assert complex_transform.inverse is complex_transform.adjoint
    np.testing.assert_allclose(
        complex_transform.inverse(complex_transform(z)),
        z,
        rtol=1e-13,
        atol=1e-13,
    )
    op_evaluation_and_ot(complex_transform, sample_N=2)

@pytest.mark.parametrize("vs",[ 
        (NumPyVectorSpace((2,4),dtype=complex),NumPyVectorSpace(3,dtype=complex),NumPyVectorSpace((5,2),dtype=complex)),
        (NumPyVectorSpace((2,4)),NumPyVectorSpace(3),NumPyVectorSpace((5,2)))
    ])

class TestOuterProduct():

    def test_op_basic(self,vs):
        op_basics_wrapper(OuterProduct,*vs,test_methods=True)

    def test_ot_eval(self,vs):
        op=OuterProduct(*vs)
        x=op.domain.randn()
        res = op.codomain.product(*x)
        op_evaluation_and_ot(op,x=x,res=res)
    
@pytest.mark.parametrize("s,spaces",[
    ("ijki,ji,il->ik",(UniformGridFcts(2,3,3,2,dtype=np.complex128),UniformGridFcts(3,2)) ), #non-linear
    ("ijki,il->ji",(UniformGridFcts(2,3,3,2,dtype=np.complex128),)) #linear
 ])
class TestEinSum():
    errors = []
    tensors=(np.arange(12).reshape(2,6),)

    def test_op_basic(self,s,spaces):
        op_basics_wrapper(EinSum,s,*spaces,test_methods=True,tensors=self.tensors)

    def test_ot_eval(self,s, spaces):
        op=EinSum(s,*spaces,tensors=self.tensors)
        x=op.domain.randn()
        if isinstance(x,TupleVector):
            res = np.einsum(s,*x,*self.tensors)
        else:
            res = np.einsum(s,x,*self.tensors)
        op_evaluation_and_ot(op,x=x,res=res)
