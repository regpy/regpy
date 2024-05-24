r"""Solvers for ill-posed and inverse problems that are modeled by linear forward operators.
    """

class PowerMethod:
    r"""Approximation of operator norm by the power method.

    Parameters
    ----------
    setting: RegularizationSetting
        Provides op and Gram. 
    op: Operator
        Optionally overrides choice of operator (e.g. for linearisation).
    """

    def __init__(self,setting=None,op=None,max_iter=int(1e2),stopping_rule=1e-12):
        if not setting.is_hilbert_setting():
            raise NotImplementedError
        if op is not None:
            op = setting.op
                
        x = setting.domain.rand()
        relative_residual = np.inf
        for i in range(max_iter):
            if relative_residual < stopping_rule:
                break
            y = setting.h_domain.gram_inv * op.adjoint * setting.h_codomain.gram * op(x)
            lmb = np.sqrt(np.inner(y, op.adjoint * setting.h_codomain.gram * op(x)))
            relative_residual = setting.h_domain.norm(y - lmb * x)
            x = y/lmb
        return lmb