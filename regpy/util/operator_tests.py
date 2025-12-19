from random import uniform

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

    Returns
    -------
    bool
        True if the operator is linear within the specified tolerance
        and False otherwise.    
    """
    x = op.domain.randn()
    y = op.domain.randn()
    r= uniform(-10,10)
    err_sum=op.codomain.norm((op(x)+op(y))-(op(x+y)))
    err_mult=op.codomain.norm(op(r*x)-r*op(x))
    if err_sum<tolerance or err_mult<tolerance:
        op.log.info(f'Linearity test passed: err_sum = {err_sum}, err_mult = {err_mult}')
        return True
    else:
        op.log.warning(f'Linearity test failed: err_sum = {err_sum}, err_mult = {err_mult}')
        return False
    
def test_affine_linearity(op, tolerance=1e-10):
    """NUmerically test if the operator is affine linear. 
    
    Checks if ::
        op_unshifted(x) := op(x) - op(0) is linear

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator to be tested.
    tolerance : float, optional
        The maximum allowed difference between the results. Defaults to 
        1e-10.

    Returns
    -------
    bool
        True if the operator is linear within the specified tolerance.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if test_linearity(op-op(op.domain.zeros()),tolerance=tolerance):
        op.log.info('Affine linearity test passed.')
        return True
    else:
        op.log.warning(f'Affine linearity test failed!')
        return False

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

    Returns
    -------
    bool
        If the test fails then False and if it passes then True.
    """
    x = op.domain.randn()
    fx = op(x)
    y = op.codomain.randn()
    fty = op.adjoint(y)
    err = (op.codomain.vdot(y, fx) - op.domain.vdot(fty, x)).real
    if abs(err) < tolerance:
        op.log.info(f'Adjoint test passed: err = {err}')
        return True
    else:
        op.log.warning(f'Adjoint test failed: err = {err}')
        return False

def test_adjoint_eval(op, tolerance=1e-10):
    r"""Numerically test validity of :meth:`adjoint_eval` method.

    Checks if

    .. highlight:: python
    .. code:: python

        norm(adjoint_eval(x)-adjoint(eval(x))) < tolerance

    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the inner products. Defaults to
        1e-10.

    Returns
    -------
    bool
        If the test fails then False and if it passes then True.
    """
    x = op.domain.randn()
    Tast_T_x = op.adjoint(op(x))
    TastT_x = op.adjoint_eval(x)
    err = op.domain.norm(Tast_T_x-TastT_x)
    if abs(err) < tolerance:
        op.log.info(f'AdjointEval test passed: err = {err}')
        return True
    else:
        op.log.warning(f'AdjointEval test failed: err = {err}')
        return False

def test_derivative(op, steps=None,ret_sequence=False,x=None):
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
    x: domain of op or None [default none]
        The point at which the derivative is tested. Chosen randomly if None.

    Returns
    ------
    bool
        If the sequence of differences is decreasing then True, otherwise False. 
    """
    if steps is None:
        steps = [10**k for k in range(-1, -8, -1)]
    if x is None:
        x = op.domain.randn()
    y, deriv = op.linearize(x)
    h = op.domain.rand()
    normh = op.domain.norm(h)
    g = deriv(h)
    seq=[op.codomain.norm((op(x + step * h) - y) / step - g) / normh for step in steps]
    if all(seq_i >= seq_j for seq_i, seq_j in zip(seq, seq[1:])):
        op.log.info(f'Derivative test passed: {seq}')
        return True
    else:
        op.log.warning(f'Derivative test failed: {seq}')
        return False


def test_adjoint_derivative(op, tolerance=1e-10):
    """Numerically test if adjoint derivative is correct implementation aligns
    with the normal composition of the adjoint and derivative.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator to be tested
    tolerance : float, optional
        The tolerance, by default 1e-10

    Returns
    -------
    bool
        True if the adjoint derivative is correct, False otherwise.
    """
    x = op.domain.randn()
    h = op.domain.randn()
    _,deriv = op.linearize(x, return_adjoint_eval=True)
    adjoint_deriv_h = deriv.adjoint_eval(h)
    diff = adjoint_deriv_h-deriv.adjoint(deriv(h))
    if (diff < tolerance).all() and (diff > -tolerance).all():
        op.log.info('Adjoint derivative test passed.')
        return True
    else:
        op.log.warning(f'Adjoint derivative test failed: {diff}')
        return False

    
def test_operator(op,sample_N=5,tolerance=1e-10,steps=None,adjoint_derivative=False,x_s = None):
    """Numerically tests if operator is computed correctly.

    Checks if operator is linear and if adjoint is correct for linear operators. Checks if derivative is correct by computing
    sequence of difference quotients and checking if they are decreasing. Checks if derivative is linear with correct adjoint.
    Optionally checks correctness of adjoint derivative.

    Parameters
    ----------
    op : regpy.operators.Operator
        The operator.
    sample_N : int
        Number of runs for each test.
    tolerance : float, optional
        The maximum allowed difference between the results. Defaults to 1e-10.
    steps : list of float, optional
        Steps used for the computation for the difference quotients. Should be chosen according to the expected regularity of the operator.
        Defaults to [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7].
    adjoint_derivative : bool, optional
        If true the adjoint_derivative is also checked. Defaults to False.
    x_s : list or tuple
        List or tuple of sample_N elements of which to test the derivative.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if steps is None:
        steps = [10**k for k in range(-1, -8, -1)]
    if(op.linear):
        op.log.info('Testing linearity of operator.')
        for _ in range(sample_N):
            if not test_linearity(op,tolerance) or not test_adjoint(op,tolerance):
                raise AssertionError('Linearity or adjoint test failed for linear operator.')
            if not test_adjoint_eval(op,tolerance):
                raise AssertionError(f'Applying the adjoint and eval deviates more then {tolerance} form the implementation of adjoint_eval with respect to the vector space l2 norm.')
    elif test_affine_linearity(op,tolerance=tolerance):
        op.log.info('Testing affine linearity of operator. Skipping test of derivative.')
        for _ in range(sample_N):
            x = op.domain.rand()
            _, deriv = op.linearize(x)
            try:
                test_operator(deriv,sample_N=sample_N,tolerance=tolerance)
            except AssertionError:
                raise AssertionError('Test for the derivative operator test failed for affine linear operator.')
            if(adjoint_derivative) and not test_adjoint_derivative(op,tolerance=tolerance):
                raise AssertionError('Adjoint derivative test failed for affine linear operator.')
    else:
        op.log.info('Testing non-linear operator for exactness of linearization and derivative.')
        if x_s is None:
            x_s = [op.domain.rand() for _ in range(sample_N)]
        elif isinstance(x_s,list): 
            if len(x_s) < sample_N:
                sample_N = len(x_s)
                op.log.info(f'You tried to test for sample_n = {sample_N} but gave only {len(x_s)} x_s. Only testing on the x_s')
            pass
        else:
            x_s = [x_s]
            sample_N = 1
        for k in range(sample_N):
            if not test_derivative(op, steps = steps,x = x_s[k]):
                raise AssertionError('Derivative test failed for non-linear operator.')
            _, deriv = op.linearize(x_s[k])
            try:
                test_operator(deriv,sample_N=sample_N,tolerance=tolerance)
            except AssertionError:
                raise AssertionError('Test for the derivative operator test failed for non-linear operator.')
            if(adjoint_derivative) and not test_adjoint_derivative(op,tolerance=tolerance):
                raise AssertionError('Adjoint derivative test failed for non-linear operator.')
    op.log.info('All tests passed.')
