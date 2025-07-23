import numpy as np


def test_linearity(op, tolerance=1e-10):
    r"""Numerically tests if operator is linear.

    Checks if ::

    .. highlight:: python
    .. code:: python

        op(x+y) == op(x)+op(y)
        r*op(x) == op(r*x)
    
    for random choices of `x` and `y`.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the results. Defaults to
        1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    x = op.domain.randn()
    y = op.domain.randn()
    r= np.random.uniform(-10,10)
    err_sum=np.max(np.abs((op(x)+op(y))-(op(x+y))))
    err_mult=np.max(np.abs(op(r*x)-r*op(x)))
    assert err_sum<tolerance, f'err = {err_sum}'
    assert err_mult<tolerance, f'err = {err_mult}'

def test_adjoint(op, tolerance=1e-10):
    r"""Numerically test validity of :meth:`adjoint` method.

    Checks if

    .. highlight:: python
    .. code:: python

        inner(y, op(x)) == inner(op.adjoint(x), y)

    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the inner products. Defaults to
        1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    x = op.domain.randn()
    fx = op(x)
    y = op.codomain.randn()
    fty = op.adjoint(y)
    err = np.real(np.vdot(y, fx) - np.vdot(fty, x))
    assert np.abs(err) < tolerance, 'err = {}'.format(err)


def test_derivative(op, steps=None,ret_sequence=False):
    r"""Numerically test derivative of operator.

    Computes :math:` ||\frac{F(x+tv)-F(x)}{t}-F'(x)v|| `
    for randomly chosen .:math:`x` and :math:`v` and different :math:`t` given in steps.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    steps : float, optional
        The used steps. Defaults to
        [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7].
    ret_sequence : bool, optional
        If set to true the sequence of errors is returned else it is asserted that this sequence is decreasing.
        Defaults to False.

    Returns
    ------
    list of float (optional)
        List of computed differences, one for each step. Only if ret_sequence is True. 
    """
    if steps is None:
        steps = [10**k for k in range(-1, -8, -1)]
    x = op.domain.randn()
    y, deriv = op.linearize(x)
    h = op.domain.rand()
    normh = np.linalg.norm(h)
    g = deriv(h)
    seq=[np.linalg.norm((op(x + step * h) - y) / step - g) / normh for step in steps]
    if(ret_sequence):
        return seq
    assert all(seq_i >= seq_j for seq_i, seq_j in zip(seq, seq[1:])),f"convergence errors: {seq}"

def test_adjoint_derivative(op, tolerance=1e-10):
    x = op.domain.randn()
    h = op.domain.randn()
    _,deriv,adjoint_derivative = op.linearize(x, adjoint_derivative=True)
    adjoint_deriv_h = adjoint_derivative(h)
    return np.all(np.abs(adjoint_deriv_h-deriv.adjoint(deriv(h)))<tolerance)

    
def test_operator(op,sample_N=5,tolerance=1e-10,steps=None,adjoint_derivative=False):
    """Numerically tests if operator is computed correctly.

    Checks if operator is linear and if adjoint is correct for linear operators. Checks if derivative is correct by computing
    sequence of difference quotients and checking if they are decreasing. Checks if derivative is linear with correct adjoint.
    Optionally checks correctness of adjoint derivative.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the results. Defaults to 1e-10.
    steps : list of float, optional
        Steps used for the computation for the difference quotients. Should be chosen according to the expected regularity of the operator.
        Defaults to [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7].
    adjoint_derivative : bool, optional
        If true the adjoint_derivative is also checked. Defaults to False.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if steps is None:
        steps = [10**k for k in range(-1, -8, -1)]
    if(op.linear):
        for _ in range(sample_N):
            test_linearity(op,tolerance)
            test_adjoint(op,tolerance)
    else:
        for _ in range(sample_N):
            test_derivative(op, steps)
            x = op.domain.rand()
            _, deriv = op.linearize(x)
            test_operator(deriv,sample_N=sample_N,tolerance=tolerance)
            if(adjoint_derivative):
                test_adjoint_derivative(op,tolerance=tolerance)
                



