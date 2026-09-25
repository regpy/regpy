from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Iterable
    from numpy.typing import ArrayLike
    from regpy.vecsps.base import ReadOnlyVector, Scalar, Vector, VectorSpaceBase
    from regpy.operators.base import Operator
    import logging
from random import uniform


def _copy_vector(space: VectorSpaceBase, 
                 value: Scalar | ArrayLike
                 ) -> Vector:
    r"""Create an independent copy using the vector-space copy protocol.
    
    Parameters
    ----------
    space: VectorSpaceBase
        Space.
    value: Scalar | ArrayLike
        Value.
    
    
    Returns
    -------
    Vector
        Writable result vector.
    """
    result = space.empty()
    space.copyto(result, value)
    return result


def _comparison_bound(space: VectorSpaceBase, reference: ReadOnlyVector, tolerance: float) -> float:
    r"""Scale an absolute tolerance by the norm of the reference vector.
    
    Parameters
    ----------
    space: VectorSpaceBase
        Space.
    reference: ReadOnlyVector
        Input vector; its coefficients are only read.
    tolerance: float
        Tolerance used in the consistency check.
    
    
    Returns
    -------
    float
        Computed scalar value.
    """
    return tolerance * max(1., space.norm(reference))


def _compare_vectors(space: VectorSpaceBase, actual: ReadOnlyVector, reference: ReadOnlyVector, tolerance: float, log: logging.Logger, label: str) -> bool:
    err = space.norm(actual-reference)
    bound = _comparison_bound(space, reference, tolerance)
    if err <= bound:
        return True
    log.warning(
        f"Allocating and in-place {label} differ: "
        f"err={err}, tolerance={bound}."
    )
    return False


def _input_unchanged(space: VectorSpaceBase, current: ReadOnlyVector, reference: ReadOnlyVector, tolerance: float, log: logging.Logger, label: str) -> bool:
    err = space.norm(current-reference)
    bound = _comparison_bound(space, reference, tolerance)
    if err <= bound:
        return True
    log.warning(
        f"{label} modified its input although a distinct output buffer was "
        f"provided: err={err}, tolerance={bound}."
    )
    return False


def _spaces_compatible(space1: VectorSpaceBase, space2: VectorSpaceBase) -> bool:
    r"""Return whether one vector can be used as input and output.
    
    Parameters
    ----------
    space1: VectorSpaceBase
        Space1.
    space2: VectorSpaceBase
        Space2.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    try:
        return space1.compatible(space2)
    except (AttributeError, TypeError, ValueError):
        return space1 == space2


def _test_out_action(
        label: str, allocating_call: Callable[[ReadOnlyVector], Vector], inplace_call: Callable[[ReadOnlyVector, Vector], Vector],
        input_space: VectorSpaceBase, output_space: VectorSpaceBase, value: Scalar | ArrayLike,
        log: logging.Logger, tolerance: float=1e-10, test_aliasing: bool=True) -> bool | None:
    r"""Compare allocating and ``out=`` variants of a vector-valued call.
    
    A return value of ``None`` means that the allocating variant is not
    implemented and the check was skipped.
    
    Parameters
    ----------
    label: str
        Label.
    allocating_call: Callable[[ReadOnlyVector], Vector]
        Input vector; its coefficients are only read.
    inplace_call: Callable[[ReadOnlyVector, Vector], Vector]
        Input vector; its coefficients are only read.
    input_space: VectorSpaceBase
        Input space.
    output_space: VectorSpaceBase
        Output space.
    value: Scalar | ArrayLike
        Value.
    log: logging.Logger
        Log.
    tolerance: float [default: 1e-10]
        Tolerance used in the consistency check.
    test_aliasing: bool [default: True]
        Whether to test aliased input and output buffers.
    
    
    Returns
    -------
    bool | None
        Result described above.
    """
    original = _copy_vector(input_space, value)
    try:
        reference_raw = allocating_call(value)
    except NotImplementedError:
        log.info(f"Skipping in-place test for {label}: method not implemented.")
        return None

    reference = _copy_vector(output_space, reference_raw)
    passed = _input_unchanged(
        input_space, value, original, tolerance, log,
        f"Allocating {label}"
    )

    out = output_space.empty()
    try:
        returned = inplace_call(value, out)
    except NotImplementedError:
        log.warning(
            f"Allocating {label} is implemented, but its in-place variant is not."
        )
        return False

    if returned is not out:
        log.warning(
            f"In-place {label} did not return the supplied output object."
        )
        passed = False
    passed &= _compare_vectors(
        output_space, out, reference, tolerance, log, label
    )
    passed &= _input_unchanged(
        input_space, value, original, tolerance, log,
        f"In-place {label}"
    )

    if test_aliasing and _spaces_compatible(input_space, output_space):
        alias = _copy_vector(input_space, original)
        try:
            returned_alias = inplace_call(alias, alias)
        except NotImplementedError:
            log.warning(
                f"In-place {label} does not support the aliasing call "
                f"``out is input``."
            )
            return False
        if returned_alias is not alias:
            log.warning(
                f"Aliasing in-place {label} did not return the supplied buffer."
            )
            passed = False
        passed &= _compare_vectors(
            output_space, alias, reference, tolerance, log,
            f"aliasing in-place {label}"
        )

    return passed


def _test_linearize_variant(
        op: Operator, x: ReadOnlyVector, return_adjoint_eval: bool,
        tolerance: float=1e-10, test_aliasing: bool=True) -> bool | None:
    r"""Compare allocating and in-place variants of ``Operator.linearize``.
    
    Parameters
    ----------
    op: Operator
        Operator used by this construction.
    x: ReadOnlyVector
        Input vector; its coefficients are only read.
    return_adjoint_eval: bool
        Return adjoint eval.
    tolerance: float [default: 1e-10]
        Tolerance used in the consistency check.
    test_aliasing: bool [default: True]
        Whether to test aliased input and output buffers.
    
    
    Returns
    -------
    bool | None
        Result described above.
    """
    label = (
        "linearize(return_adjoint_eval=True)"
        if return_adjoint_eval else "linearize"
    )
    output_space = op.domain if return_adjoint_eval else op.codomain
    original = _copy_vector(op.domain, x)

    try:
        reference_raw, _ = op.linearize(
            x, return_adjoint_eval=return_adjoint_eval
        )
    except NotImplementedError:
        op.log.info(f"Skipping in-place test for {label}: method not implemented.")
        return None

    reference = _copy_vector(output_space, reference_raw)
    passed = _input_unchanged(
        op.domain, x, original, tolerance, op.log,
        f"Allocating {label}"
    )

    out = output_space.empty()
    try:
        result, _ = op.linearize(
            x, out=out, return_adjoint_eval=return_adjoint_eval
        )
    except NotImplementedError:
        op.log.warning(
            f"Allocating {label} is implemented, but its in-place variant is not."
        )
        return False

    if result is not out:
        op.log.warning(
            f"In-place {label} did not return the supplied output object."
        )
        passed = False
    passed &= _compare_vectors(
        output_space, out, reference, tolerance, op.log, label
    )
    passed &= _input_unchanged(
        op.domain, x, original, tolerance, op.log,
        f"In-place {label}"
    )

    if test_aliasing and _spaces_compatible(op.domain, output_space):
        alias = _copy_vector(op.domain, original)
        try:
            alias_result, _ = op.linearize(
                alias, out=alias,
                return_adjoint_eval=return_adjoint_eval
            )
        except NotImplementedError:
            op.log.warning(
                f"In-place {label} does not support the aliasing call "
                f"``out is input``."
            )
            return False
        if alias_result is not alias:
            op.log.warning(
                f"Aliasing in-place {label} did not return the supplied buffer."
            )
            passed = False
        passed &= _compare_vectors(
            output_space, alias, reference, tolerance, op.log,
            f"aliasing in-place {label}"
        )

    return passed


def _test_linear_operator_inplace(
        op: Operator, tolerance: float=1e-10, test_aliasing: bool=True, test_evaluation: bool=True) -> bool:
    r"""Check all public allocating/in-place paths of a linear operator.
    
    Parameters
    ----------
    op: Operator
        Operator used by this construction.
    tolerance: float [default: 1e-10]
        Tolerance used in the consistency check.
    test_aliasing: bool [default: True]
        Whether to test aliased input and output buffers.
    test_evaluation: bool [default: True]
        Test evaluation.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    passed = True

    if test_evaluation:
        x = op.domain.randn()
        result = _test_out_action(
            "evaluation",
            lambda value: op(value),
            lambda value, out: op(value, out=out),
            op.domain, op.codomain, x, op.log,
            tolerance=tolerance, test_aliasing=test_aliasing
        )
        if result is not None:
            passed &= result

    y = op.codomain.randn()
    result = _test_out_action(
        "adjoint evaluation",
        lambda value: op.adjoint(value),
        lambda value, out: op.adjoint(value, out=out),
        op.codomain, op.domain, y, op.log,
        tolerance=tolerance, test_aliasing=test_aliasing
    )
    if result is not None:
        passed &= result

    data = op.codomain.randn()
    result = _test_out_action(
        "adjoint_data",
        lambda value: op.adjoint_data(value),
        lambda value, out: op.adjoint_data(value, out=out),
        op.codomain, op.domain, data, op.log,
        tolerance=tolerance, test_aliasing=test_aliasing
    )
    if result is not None:
        passed &= result

    x = op.domain.randn()
    result = _test_out_action(
        "adjoint_eval",
        lambda value: op.adjoint_eval(value),
        lambda value, out: op.adjoint_eval(value, out=out),
        op.domain, op.domain, x, op.log,
        tolerance=tolerance, test_aliasing=test_aliasing
    )
    if result is not None:
        passed &= result

    return passed


def test_inplace_operator_methods(
        op: Operator, x: ReadOnlyVector | None=None, tolerance: float=1e-10, test_aliasing: bool=True) -> bool:
    r"""Compare allocating and in-place variants of the Operator API.
    
    The following public paths are checked when implemented:
    
    * evaluation,
    * both variants of :meth:`Operator.linearize`,
    * derivative evaluation,
    * adjoint evaluation,
    * :meth:`Operator.adjoint_data`,
    * :meth:`Operator.adjoint_eval`, including the adjoint derivative.
    
    Besides numerical equality, the test checks that the supplied output
    object is returned and that a non-aliasing call does not modify its input.
    If the input and output spaces are compatible, ``out is input`` is tested
    as well.
    
    Parameters
    ----------
    op: Operator
        Operator used by this construction.
    x: ReadOnlyVector | None [default: None]
        Input vector; its coefficients are only read.
    tolerance: float [default: 1e-10]
        Tolerance used in the consistency check.
    test_aliasing: bool [default: True]
        Whether to test aliased input and output buffers.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    if x is None:
        x = op.domain.randn()

    passed = True
    result = _test_out_action(
        "evaluation",
        lambda value: op(value),
        lambda value, out: op(value, out=out),
        op.domain, op.codomain, x, op.log,
        tolerance=tolerance, test_aliasing=test_aliasing
    )
    if result is not None:
        passed &= result

    result = _test_linearize_variant(
        op, x, return_adjoint_eval=False,
        tolerance=tolerance, test_aliasing=test_aliasing
    )
    if result is not None:
        passed &= result

    result = _test_linearize_variant(
        op, x, return_adjoint_eval=True,
        tolerance=tolerance, test_aliasing=test_aliasing
    )
    if result is not None:
        passed &= result

    if op.linear:
        passed &= _test_linear_operator_inplace(
            op, tolerance=tolerance,
            test_aliasing=test_aliasing,
            test_evaluation=False
        )
    else:
        # Create one final, non-aliasing linearization. Earlier calls may have
        # revoked their derivative proxy.
        derivative_out = op.codomain.empty()
        try:
            _, derivative = op.linearize(x, out=derivative_out)
        except NotImplementedError:
            op.log.info(
                "Skipping in-place derivative checks: linearization is not "
                "implemented."
            )
        else:
            passed &= _test_linear_operator_inplace(
                derivative, tolerance=tolerance,
                test_aliasing=test_aliasing,
                test_evaluation=True
            )

    if passed:
        op.log.info("Allocating/in-place equivalence tests passed.")
    return passed


def test_linearity(op: Operator, tolerance: float=1e-10) -> bool:
    r"""Numerically tests if operator is linear.
    
    Checks if ::
    
    .. highlight:: python
    .. code:: python
    
        op(x+y) == op(x)+op(y)
        r*op(x) == op(r*x)
    
    for random choices of `x` and `y`.
    
    Parameters
    ----------
    op: Operator
        The operator.
    tolerance: float [default: 1e-10]
        The maximum allowed difference between the results.
    
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
    if err_sum<tolerance and err_mult<tolerance:
        op.log.info(f'Linearity test passed: err_sum = {err_sum}, err_mult = {err_mult}')
        return True
    else:
        op.log.warning(f'Linearity test failed: err_sum = {err_sum}, err_mult = {err_mult}')
        return False
    
def test_affine_linearity(op: Operator, tolerance: float=1e-10) -> bool:
    r"""NUmerically test if the operator is affine linear. 
    
    Checks if ::
        op_unshifted(x) := op(x) - op(0) is linear
    
    Parameters
    ----------
    op: Operator
        The operator to be tested.
    tolerance: float [default: 1e-10]
        The maximum allowed difference between the results.
    
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

def test_adjoint(op: Operator, tolerance: float=1e-10) -> bool:
    r"""Numerically test validity of :meth:`adjoint` method.
    
    Checks if
    
    .. highlight:: python
    .. code:: python
    
        inner(y, op(x)) == inner(op.adjoint(y), x)
    
    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.
    
    Parameters
    ----------
    op: Operator
        The operator.
    tolerance: float [default: 1e-10]
        The maximum allowed difference between the inner products.
    
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


def test_self_adjoint(op: Operator, tolerance: float=1e-10) -> bool:
    r"""Test the public contract of a declared self-adjoint operator.
    
    Besides the numerical adjoint identity checked by :func:`test_adjoint`,
    self-adjoint operators must be linear, map a vector space to itself, and
    expose themselves through :attr:`Operator.adjoint` rather than construct
    an adjoint proxy.
    
    Parameters
    ----------
    op: Operator
        An operator whose :attr:`~regpy.operators.Operator.self_adjoint` flag
        is expected to be true.
    tolerance: float [default: 1e-10]
        The maximum allowed error in the numerical adjoint test.
    
    Returns
    -------
    bool
        Whether the structural and numerical self-adjointness checks pass.
    """
    if not op.self_adjoint:
        op.log.warning('Self-adjointness test requested for an unmarked operator.')
        return False
    if not op.linear:
        op.log.warning('A self-adjoint operator must be linear.')
        return False
    if op.domain != op.codomain:
        op.log.warning(
            'A self-adjoint operator must have identical domain and codomain.'
        )
        return False
    if op.adjoint is not op:
        op.log.warning(
            'The adjoint property of a self-adjoint operator must return the '
            'operator itself.'
        )
        return False
    return test_adjoint(op, tolerance=tolerance)

def test_adjoint_eval(op: Operator, tolerance: float=1e-10) -> bool:
    r"""Numerically test validity of :meth:`adjoint_eval` method.
    
    Checks if
    
    .. highlight:: python
    .. code:: python
    
        norm(adjoint_eval(x)-adjoint(eval(x))) < tolerance
    
    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.
    
    Parameters
    ----------
    op: Operator
        The operator.
    tolerance: float [default: 1e-10]
        The maximum allowed difference between the inner products.
    
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

def test_derivative(op: Operator, steps: float | None=None,ret_sequence: bool=False,x: ReadOnlyVector | None=None) -> bool:
    r"""Numerically test derivative of operator.
    
    Computes :math:` ||\frac{F(x+tv)-F(x)}{t}-F'(x)v|| `
    for randomly chosen .:math:`x` and :math:`v` and different :math:`t` given in steps.
    
    Parameters
    ----------
    op: Operator
        The operator.
    steps: float | None [default: None]
        The used steps.
    ret_sequence: bool [default: False]
        Ret sequence.
    x: ReadOnlyVector | None [default: None]
        The point at which the derivative is tested. Chosen randomly if None.
    
    Returns
    -------
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


def test_adjoint_derivative(op: Operator, tolerance: float=1e-10) -> bool:
    r"""Numerically test if adjoint derivative is correct implementation aligns
    with the normal composition of the adjoint and derivative.
    
    Parameters
    ----------
    op: Operator
        The operator to be tested
    tolerance: float [default: 1e-10]
        The tolerance, by
    
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
    err = op.domain.norm(diff)
    if err < tolerance:
        op.log.info(f'Adjoint derivative test passed: err = {err}')
        return True
    else:
        op.log.warning(f'Adjoint derivative test failed: err = {err}')
        return False

    
def test_operator(
        op: Operator, sample_N: int=5, tolerance: float=1e-10, steps: Iterable[float] | None=None,
        adjoint_derivative: bool=False, x_s: Iterable[ReadOnlyVector] | None=None,
        test_inplace: bool=True, test_inplace_aliasing: bool=True) -> None:
    r"""Numerically tests if operator is computed correctly.
    
    Checks if an operator is linear and if its adjoint is correct for linear
    operators. For operators marked as self-adjoint, it additionally checks
    that domain and codomain coincide and that :attr:`Operator.adjoint`
    returns the operator itself. For nonlinear operators, it checks the
    derivative by computing a sequence of difference quotients, and verifies
    that the derivative is linear with a correct adjoint. Optionally checks
    correctness of the adjoint derivative.
    
    Parameters
    ----------
    op: Operator
        The operator.
    sample_N: int [default: 5]
        Number of runs for each test.
    tolerance: float [default: 1e-10]
        The maximum allowed difference between the results.
    steps: Iterable[float] | None [default: None]
        Steps used for the computation for the difference quotients. Should be chosen according to the expected regularity of the operator.
    adjoint_derivative: bool [default: False]
        If true the adjoint_derivative is also checked.
    x_s: Iterable[ReadOnlyVector] | None [default: None]
        List or tuple of sample_N elements of which to test the derivative.
    test_inplace: bool [default: True]
        Compare allocating and ``out=`` variants of evaluation, linearization,
        derivative, adjoint, adjoint-data, and adjoint-evaluation methods.
    test_inplace_aliasing: bool [default: True]
        Also test ``out is input`` whenever domain and codomain are compatible.
    
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
            if test_inplace and not test_inplace_operator_methods(
                    op, tolerance=tolerance,
                    test_aliasing=test_inplace_aliasing):
                raise AssertionError(
                    'Allocating/in-place equivalence test failed for linear operator.'
                )
            if not test_linearity(op,tolerance) or not test_adjoint(op,tolerance):
                raise AssertionError('Linearity or adjoint test failed for linear operator.')
            if op.self_adjoint and not test_self_adjoint(op, tolerance):
                raise AssertionError(
                    'Public self-adjointness contract failed for linear operator.'
                )
            if not test_adjoint_eval(op,tolerance):
                raise AssertionError(f'Applying the adjoint and eval deviates more than {tolerance} from the implementation of adjoint_eval with respect to the vector space l2 norm.')
    elif test_affine_linearity(op,tolerance=tolerance):
        op.log.info('Testing affine linearity of operator. Skipping test of derivative.')
        for _ in range(sample_N):
            x = op.domain.rand()
            if test_inplace and not test_inplace_operator_methods(
                    op, x=x, tolerance=tolerance,
                    test_aliasing=test_inplace_aliasing):
                raise AssertionError(
                    'Allocating/in-place equivalence test failed for affine operator.'
                )
            _, deriv = op.linearize(x)
            try:
                test_operator(
                    deriv, sample_N=sample_N, tolerance=tolerance,
                    test_inplace=test_inplace,
                    test_inplace_aliasing=test_inplace_aliasing
                )
            except AssertionError:
                raise AssertionError('Test for the derivative operator test failed for affine linear operator.')
            if(adjoint_derivative) and not test_adjoint_derivative(op,tolerance=tolerance):
                raise AssertionError('Adjoint derivative test failed for affine linear operator.')
    else:
        op.log.info('Testing nonlinear operator for exactness of linearization and derivative.')
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
            if test_inplace and not test_inplace_operator_methods(
                    op, x=x_s[k], tolerance=tolerance,
                    test_aliasing=test_inplace_aliasing):
                raise AssertionError(
                    'Allocating/in-place equivalence test failed for nonlinear operator.'
                )
            if not test_derivative(op, steps = steps,x = x_s[k]):
                raise AssertionError('Derivative test failed for nonlinear operator.')
            _, deriv = op.linearize(x_s[k])
            try:
                test_operator(
                    deriv, sample_N=sample_N, tolerance=tolerance,
                    test_inplace=test_inplace,
                    test_inplace_aliasing=test_inplace_aliasing
                )
            except AssertionError:
                raise AssertionError('Test for the derivative operator test failed for nonlinear operator.')
            if(adjoint_derivative) and not test_adjoint_derivative(op,tolerance=tolerance):
                raise AssertionError('Adjoint derivative test failed for nonlinear operator.')
    op.log.info('All tests passed.')
