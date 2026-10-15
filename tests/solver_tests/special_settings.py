"""Regression checks for FISTA/SSN stage switching.

Run from a RegPy checkout after installing special_settings_relative_gap_dual_ssn.py
as regpy/solvers/special_settings.py::

    pytest -v -s test_special_settings_fista_ssn.py
"""
import numpy as np
import pytest

from regpy.vecsps import UniformGridFcts
from regpy.operators import Identity
from regpy.functionals import Huber
from regpy.hilbert import  L2
from regpy.solvers import Setting
from regpy.solvers.special_settings import (
    FISTA_SSNewton,
    dFISTA_SSNewton,
    Huber_XSetting,
)
from regpy.stoprules import DualityGapStopping


def small_setting():
    space = UniformGridFcts(np.linspace(-1, 1, 9))
    op = Identity(space)
    data = 0.3 + 0.1 * np.sin(np.linspace(-1, 1, 9))
    return Setting(op, L2, L2, data=data, regpar=0.1,
                   logging_level='WARNING')


@pytest.mark.parametrize('cls', [FISTA_SSNewton, dFISTA_SSNewton])
@pytest.mark.parametrize('ssn_primal', [False, True])
def test_fista_relative_gap_rule(cls, ssn_primal):
    solver = cls(small_setting(), fista_reltol=0.5,
                 ssn_primal=ssn_primal, logging_level='WARNING')
    assert isinstance(solver.stoprule, DualityGapStopping)
    assert solver.stoprule.reltol == 0.5
    assert solver.stoprule.tol == 0.0
    assert solver.stoprule.schedule is None
    assert solver.solver2 is None
    assert solver.is_primal == (cls is FISTA_SSNewton)


@pytest.mark.parametrize('cls', [FISTA_SSNewton, dFISTA_SSNewton])
@pytest.mark.parametrize('ssn_primal', [False, True])
def test_coordinate_transition(cls, ssn_primal):
    setting = small_setting()
    solver = cls(setting, ssn_primal=ssn_primal,
                 logging_level='WARNING')
    current = solver.solver1.x.copy()
    image = solver.solver1.op(current)
    # Natural completion of FISTA also causes the switch; this avoids
    # depending on a particular number of accelerated-gradient iterations.
    solver.solver1.converge()
    assert solver.next()
    assert solver.solver2 is not None
    assert solver.is_primal is ssn_primal
    if cls is FISTA_SSNewton:
        expected = (current if ssn_primal else
                    setting.primal_to_dual((current, image)))
    else:
        expected = (setting.dual_to_primal((current, image)) if ssn_primal
                    else current)
    np.testing.assert_allclose(solver.solver2.x, expected, rtol=1e-12,
                               atol=1e-12)


@pytest.mark.parametrize('bad', [0, 1, -0.1, 1.0, float('nan'), True])
def test_invalid_relative_gap_tolerance(bad):
    with pytest.raises(ValueError):
        FISTA_SSNewton(small_setting(), fista_reltol=bad)


@pytest.mark.parametrize('sigma', [-0.1, 0.0, 0.05])
def test_huber_setting_registers_dual_newton(sigma):
    setting = small_setting()
    xsetting = Huber_XSetting(
        setting.op, data=setting.data, regpar=0.1, huber_sigma=sigma,
        logging_level='WARNING',
    )
    assert xsetting.huber_sigma == sigma
    assert 'dFISTAdSSN' in xsetting._methods
    method = xsetting._methods['dFISTAdSSN']
    assert method['class'] is dFISTA_SSNewton
    assert method['primal'] is True
    assert method['method_params']['ssn_primal'] is False
    assert method['method_params']['fista_reltol'] == 0.5
    assert xsetting.data_fid is not None
