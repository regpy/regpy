"""Small 1-D RegPy solver equivalence tests.

Run: pytest -v --tb=short test_regpy_solver_equivalences_fixed.py

Linear settings register nonlinear solvers explicitly with Setting.add_method:
RegPy deliberately does not include them in its default linear registry.
"""
from copy import deepcopy
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2
from regpy.functionals import QuadBil
from regpy.solvers import Setting
from regpy.solvers.setting import get_nonlinear_reg_method_dictionary, get_convex_solver_dictionary
from regpy.operators import Operator, Exponential, PtwMultiplication

N = 32
STEPS = 3
RTOL = 2e-3
ATOL = 2e-5
ALPHA0 = 0.05
RATIO = 0.7
# Increase if the LM-vs-nlBreg convergence diagnostic shows inner errors.
INNER_MAX_IT = 600
CG_STOP = 600
CG_RELTOL = 1e-9
# Set to True to test LM-vs-nlBreg at the original strict tolerance.
STRICT_LM = True


class GaussianBlur1D(Operator):
    def __init__(self, domain, sigma=1.5):
        super().__init__(domain=domain, codomain=domain, linear=True)
        impulse = np.zeros(domain.shape)
        impulse[0] = 1.
        self.multiplier = np.fft.rfft(gaussian_filter1d(impulse, sigma, mode='wrap'))

    def _eval(self, x):
        return np.fft.irfft(self.multiplier * np.fft.rfft(x), n=x.size).reshape(x.shape)

    def _adjoint(self, y):
        return np.fft.irfft(np.conj(self.multiplier) * np.fft.rfft(y), n=y.size).reshape(y.shape)


def make_setting(linear, penalty, codomain_free):
    domain = UniformGridFcts(np.linspace(-1., 1., N, endpoint=False))
    op = GaussianBlur1D(domain) if linear else Exponential(domain) * PtwMultiplication(domain, 0.5)
    xx = np.linspace(-1., 1., N, endpoint=False)
    truth = 0.15 + 0.65 * np.exp(-((xx + 0.2) / 0.24)**2)
    p = L2 if penalty == 'L2' else QuadBil(lb=0., ub=1.)
    setting = Setting(op, p, L2, data=op(truth), codomain_free=codomain_free,
                      default_inner_solver='FISTA', logging_level='WARNING')
    return setting


def alphas():
    return [ALPHA0 * RATIO**k for k in range(STEPS + 2)]


def register_nonlinear_on_linear(setting, method):
    """Public API: opt in to a nonlinear algorithm for a linear forward map."""
    if setting.op.linear and method in ('IRN', 'nlBreg'):
        cls = get_nonlinear_reg_method_dictionary()[method]['class']
        setting.add_method(method, cls, full_method_name=f'{method} on linear operator')


def inner_template(penalty, max_it=INNER_MAX_IT):
    """Independent, explicitly bounded convex inner-solver configuration."""
    label = 'TikhCG' if penalty == 'L2' else 'FISTA'
    template = deepcopy(get_convex_solver_dictionary()[label])
    template['stoprule_params']['max_it'] = max_it
    template['stoprule_params']['logging_level'] = 'WARNING'
    template['method_params']['logging_level'] = 'WARNING'
    return template


def solve(linear, penalty, codomain_free, method, *, inner_max_it=INNER_MAX_IT,
          cg_reltol=CG_RELTOL):
    setting = make_setting(linear, penalty, codomain_free)
    register_nonlinear_on_linear(setting, method)
    params = {'regpars': iter(alphas())}
    if method in ('IRN', 'nlBreg', 'alGr', 'Breg'):
        params['inner_solver'] = inner_template(penalty, inner_max_it)
    if method in ('IRGN_CG', 'LM'):
        # The CG solver has different accuracy controls from convex FISTA/TikhCG.
        params['cgstop'] = CG_STOP
        params['cg_pars'] = {'reltolx': cg_reltol, 'all_tol_criteria': False}
    setting.set_method_params(method, params)
    return np.array(setting.run(method, max_it=STEPS)[0], copy=True)


def compare(a, b, *, label, rtol=RTOL, atol=ATOL):
    delta = np.asarray(a) - np.asarray(b)
    max_abs = float(np.max(np.abs(delta)))
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-15))
    print(f'{label}: max_abs={max_abs:.6g}, relative_L2={relative:.6g}')
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize('codomain_free', [False, True])
def test_nonlinear_irgn_vs_irn(codomain_free):
    compare(solve(False, 'L2', codomain_free, 'IRGN_CG'),
            solve(False, 'L2', codomain_free, 'IRN'),
            label=f'IRGN_CG/IRN cf={codomain_free}')


@pytest.mark.parametrize('codomain_free', [False, True])
def test_nonlinear_lm_vs_nlbreg(codomain_free):
    x = solve(False, 'L2', codomain_free, 'LM')
    y = solve(False, 'L2', codomain_free, 'nlBreg')
    compare(x, y, label=f'LM/nlBreg cf={codomain_free}',
            rtol=RTOL if STRICT_LM else 1.5e-2)


@pytest.mark.parametrize('penalty', ['L2', 'QuadBil'])
@pytest.mark.parametrize('codomain_free', [False, True])
def test_linear_alpha_grid_vs_irn(penalty, codomain_free):
    compare(solve(True, penalty, codomain_free, 'alGr'),
            solve(True, penalty, codomain_free, 'IRN'),
            label=f'alGr/IRN {penalty} cf={codomain_free}')


@pytest.mark.parametrize('penalty', ['L2', 'QuadBil'])
@pytest.mark.parametrize('codomain_free', [False, True])
def test_linear_breg_vs_nlbreg(penalty, codomain_free):
    compare(solve(True, penalty, codomain_free, 'Breg'),
            solve(True, penalty, codomain_free, 'nlBreg'),
            label=f'Breg/nlBreg {penalty} cf={codomain_free}')


@pytest.mark.parametrize('codomain_free', [False, True])
def test_lm_nlbreg_inner_accuracy_diagnostic(codomain_free):
    """Print a coarse/tight error trend without hiding a genuine algorithmic mismatch."""
    errors = []
    for budget, reltol in ((60, 1e-3), (INNER_MAX_IT, CG_RELTOL)):
        x = solve(False, 'L2', codomain_free, 'LM', cg_reltol=reltol)
        y = solve(False, 'L2', codomain_free, 'nlBreg', inner_max_it=budget)
        err = float(np.linalg.norm(x - y))
        errors.append(err)
        print(f'LM/nlBreg cf={codomain_free} inner_max_it={budget} '
              f'cg_reltol={reltol}: L2_error={err:.8g}')
    assert all(np.isfinite(errors)), 'Inner accuracy diagnostic produced non-finite iterates.'
