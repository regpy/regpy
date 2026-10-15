"""Public behavior of Setting.init and incorporating its penalty center.

No assertions on private solver registries or choice of solver defaults.
"""

import numpy as np
import pytest

from regpy.vecsps import UniformGridFcts
from regpy.operators import PtwMultiplication
from regpy.hilbert import L2
from regpy.solvers import Setting
from regpy.functionals.vector_integral_functionals import TVUniformGridFcts


def _space(n=10):
    return UniformGridFcts((0., 1., n), (0., 1., n))


def _setting(*, init=None, incorporate=True):
    domain = _space()
    op = PtwMultiplication(domain, 1.)
    return Setting(
        op, penalty=L2, data_fid=L2, init=init,
        incorporate_init_in_penalty=incorporate,
        regpar=0.1, explicitDataWarning=False,
    )


def test_constructor_init_is_incorporated():
    space = _space()
    x = 0.75 * space.ones()
    setting = _setting(init=x, incorporate=True)
    assert setting.incorporate_init_in_penalty is True
    np.testing.assert_allclose(setting.init, x)
    assert setting.penalty(x) == pytest.approx(0., abs=1e-13)
    assert setting.penalty(space.zeros()) > 0.


def test_init_setter_recenters_penalty_without_stacking_shifts():
    space = _space()
    setting = _setting(incorporate=True)
    first = 0.75 * space.ones()
    second = 1.25 * space.ones()

    setting.init = first
    assert setting.penalty(first) == pytest.approx(0., abs=1e-13)
    assert setting.penalty(second) > 0.

    setting.init = second
    np.testing.assert_allclose(setting.init, second)
    assert setting.penalty(second) == pytest.approx(0., abs=1e-13)
    assert setting.penalty(first) > 0.

    # Returning to the first center must not translate the functional twice.
    setting.change_init(first)
    assert setting.penalty(first) == pytest.approx(0., abs=1e-13)
    assert setting.penalty(second) > 0.


def test_disabled_flag_only_changes_initial_guess():
    space = _space()
    x = 1.5 * space.ones()
    setting = _setting(incorporate=False)
    assert setting.incorporate_init_in_penalty is False
    initial_penalty_value = setting.penalty(x)
    setting.init = x
    np.testing.assert_allclose(setting.init, x)
    assert setting.penalty(x) == pytest.approx(initial_penalty_value)
    assert setting.penalty(space.zeros()) == pytest.approx(0., abs=1e-13)


def test_input_is_copied_before_becoming_penalty_data():
    space = _space()
    x = 0.6 * space.ones()
    setting = _setting(incorporate=True)
    setting.init = x
    x[:] = -1.  # mutating caller buffer must not move the penalty center
    assert setting.penalty(0.6 * space.ones()) == pytest.approx(0., abs=1e-13)
    np.testing.assert_allclose(setting.init, 0.6 * space.ones())


def test_replacing_penalty_keeps_incorporated_center():
    space = _space()
    x = 0.6 * space.ones()
    setting = _setting(incorporate=True)
    setting.init = x
    setting.set_penalty(L2)
    assert setting.penalty(x) == pytest.approx(0., abs=1e-13)
    assert setting.penalty(space.zeros()) > 0.


def test_tv_prox_fixes_nonzero_constants():
    space = _space()
    tv = TVUniformGridFcts(space, beta=0.03)
    x = 1.25 * space.ones()
    output = tv.proximal(x, tau=0.02)
    np.testing.assert_allclose(output, x, rtol=1e-8, atol=1e-8)
