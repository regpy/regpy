r"""Solvers for ill-posed and inverse problems that are modeled by linear forward operators.
    """

import numpy as np
from regpy.solvers import RegularizationSetting

def power_method(setting,op=None,max_iter=int(1e2),stopping_rule=1e-12):
    r"""Approximation of operator norm by the power method.

    Parameters
    ----------
    setting : RegularizationSetting
        Provides op and Gram. 
    op : Operator,optional
        Optionally overrides choice of operator (e.g. for linearization), Defaults: None
    """
    assert isinstance(setting,RegularizationSetting)
    if not setting.is_hilbert_setting():
        raise NotImplementedError
    if op is None:
        op = setting.op
    
    x = setting.op.domain.rand()
    relative_residual = np.inf
    for i in range(max_iter):
        if relative_residual < stopping_rule:
            break
        y = (setting.h_domain.gram_inv * op.adjoint * setting.h_codomain.gram * op)(x)
        lmb = np.sqrt(np.inner(y, (op.adjoint * setting.h_codomain.gram * op)(x)))
        relative_residual = setting.h_domain.norm(y - lmb * x)
        x = y/lmb
    return lmb