import numpy as np

from regpy.functionals import QuadPosSemi
from regpy.util import functional_tests as ft
from regpy.util import set_rng_seed
from regpy.vecsps import UniformGridFcts


set_rng_seed(15873098306879350073259142812684978477)


def test_quadratic_positive_semidef():
    size = 5
    domain = UniformGridFcts(size, size, dtype=np.complex128)

    func = QuadPosSemi(domain, tol=1e-10)
    sample_count = 10
    orthogonals = np.random.randn(sample_count, 2, size, size)
    orthogonals = orthogonals[:, 0] + 1j * orthogonals[:, 1]
    orthogonals = np.linalg.qr(orthogonals)[0]
    diagonals = np.random.uniform(0, 20, size=(sample_count, size))
    samples = [
        orthogonals[i]
        @ np.diag(diagonals[i])
        @ np.conj(orthogonals[i].T)
        for i in range(sample_count)
    ]
    assert np.abs(0.5 * np.sum(diagonals[0] ** 2) - func(samples[0])) < 1e-10
    ft.test_functional(func, u_s=samples, test_conj=False)

    trace_func = QuadPosSemi(domain, trace_val=2, tol=1e-10)
    trace_samples = [2 * sample / np.trace(sample) for sample in samples]
    expected = 0.5 * np.sum(
        (2 * diagonals[0] / np.sum(diagonals[0])) ** 2
    )
    assert np.abs(expected - trace_func(trace_samples[0])) < 1e-10
    ft.test_functional(
        trace_func, u_s=trace_samples, test_second_deriv=False, reference_is_minimizer=False
    )
