from random import uniform

import numpy as np

from .general import Errors

def sample_essential_domain(func,u=None,eps_perturbation=None):
    r""" Returns a grid function in the essential domain of an IntegralFunctional. 
    The values of this grid function are chosen to roughly span the essential domain of the function defining the 
    integral functional.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional. Needs to be separable!
    u : array-like or None, optional
        A supplied point is retained and must satisfy the coordinate bounds.
        Otherwise, sample each coordinate in its own interval.
    eps_perturbation : float or None, optional
        Nonnegative perturbation length. If provided, also return a direction h.
        Coordinates with equal lower and upper bounds have h=0.
        
    Returns
    -------
    array-like
        An element u of func.domain
        If eps_perturbation is not None, return (u, h), with
        u + eps_perturbation*h satisfying the coordinate domain bounds.
        Sampling and perturbation use coordinatewise bounds, not their
        intersection. Finite intervals are sampled in their interior whenever
        representable; a singleton interval is sampled at its sole point.
    """
    from regpy.vecsps import MeasureSpaceFcts, DirectSum
    if not func.is_separable:
        raise ValueError(Errors.value_error("Cannot sample in the essential domain if the functional is not separable!"))
    if eps_perturbation is not None:
        if (not np.isscalar(eps_perturbation)
                or not np.isreal(eps_perturbation)
                or not np.isfinite(eps_perturbation)
                or eps_perturbation < 0):
            raise ValueError("eps_perturbation must be finite and nonnegative.")
    if isinstance(func.domain,MeasureSpaceFcts):
        shape = func.domain.shape
        # Coordinate-dependent bounds need not have a common intersection.
        lower = np.broadcast_to(np.asarray(func.dom_l, dtype=float), shape)
        upper = np.broadcast_to(np.asarray(func.dom_u, dtype=float), shape)
        if (np.any(np.isnan(lower)) or np.any(np.isnan(upper))
                or np.any(lower > upper) or np.any(lower == np.inf)
                or np.any(upper == -np.inf)):
            raise ValueError("Each coordinate must have a nonempty finite-valued interval.")
        finite_l, finite_u = np.isfinite(lower), np.isfinite(upper)
        bounded = finite_l & finite_u
        lower_only = finite_l & ~finite_u
        upper_only = ~finite_l & finite_u
        unbounded = ~finite_l & ~finite_u
        numel = int(np.prod(shape))
        if u is None:
            u = np.empty(shape, dtype=float)
            fractions = np.linspace(0., 1., numel + 2)[1:-1].reshape(shape)
            # Convex interpolation avoids subtracting potentially large bounds.
            u[bounded] = ((1. - fractions[bounded]) * lower[bounded]
                          + fractions[bounded] * upper[bounded])
            if np.any(lower_only | upper_only):
                offsets = (np.exp(np.linspace(-np.sqrt(numel), np.sqrt(numel), numel))
                           - 0.5 * np.exp(-np.sqrt(numel))).reshape(shape)
                u[lower_only] = lower[lower_only] + offsets[lower_only]
                u[upper_only] = upper[upper_only] - offsets[upper_only]
            if np.any(unbounded):
                values = np.tan(np.linspace(-np.pi/2 + 1/numel,
                                            np.pi/2 - 1/numel, numel)).reshape(shape)
                u[unbounded] = values[unbounded]
            # Preserve coordinates whose feasible interval is a singleton.
            fixed = lower == upper
            u[fixed] = lower[fixed]
        else:
            u = np.asarray(u).reshape(shape)
        if (not np.all(np.isfinite(u)) or np.any(u < lower)
                or np.any(u > upper)):
            raise ValueError("The sample must satisfy the coordinate domain bounds.")
        if eps_perturbation is None:
            return u
        h = np.ones(shape, dtype=float)
        h[upper_only] = -1.
        midpoint = 0.5 * lower[bounded] + 0.5 * upper[bounded]
        direction = np.where(u[bounded] <= midpoint, 1., -1.)
        half_width = 0.5 * upper[bounded] - 0.5 * lower[bounded]
        scale = np.ones_like(half_width)
        if eps_perturbation > 0:
            # Divide only where rescaling is needed; zero-width coordinates
            # get zero directions, without ever dividing by their width.
            narrow = half_width < eps_perturbation
            scale[narrow] = half_width[narrow] / eps_perturbation
        scale[half_width == 0.] = 0.
        h[bounded] = direction * scale
        return u, h
    elif isinstance(func.domain, DirectSum):
        from regpy.functionals.base import Conj, FunctionalOnDirectSum, HorizontalShiftDilation
        if not isinstance(func, (FunctionalOnDirectSum,Conj,HorizontalShiftDilation)):
            raise TypeError(Errors.type_error(f"domain of type {func.domain}, but functional of type {type(func)},{func}."))
        if isinstance(func, FunctionalOnDirectSum):
            funcs = func.funcs
        elif isinstance(func, Conj) and isinstance(func.func, FunctionalOnDirectSum):
            funcs = (f.conj for f in func.func.funcs)
        else: 
            if isinstance(func, HorizontalShiftDilation):
                SDfunc = func
            elif isinstance(func, Conj) and isinstance(func.func,HorizontalShiftDilation):
                SDfunc = func.func
            else: 
                raise TypeError
            if SDfunc.shift_val is None:
                funcs = (HorizontalShiftDilation(f, dilation=SDfunc.dilation) for f in SDfunc.func.funcs)
            else:
                funcs = (HorizontalShiftDilation(f, shift=sh, dilation=SDfunc.dilation) for f,sh in zip(SDfunc.func.funcs,func.domain.split(SDfunc.shift_val)))
            if isinstance(func,Conj):
                funcs = (f.conj for f in funcs)
        components = [None] * len(func.domain.summands) if u is None else func.domain.split(u)
        if eps_perturbation is None:
            return func.domain.join(*(sample_essential_domain(funci, u=ui)
                                      for funci, ui in zip(funcs, components)))
        else:
            xx,hh = zip(*(sample_essential_domain(funci, u=ui, eps_perturbation=eps_perturbation)
                         for funci, ui in zip(funcs, components)))
            return func.domain.join(*xx), func.domain.join(*hh)
    

def sample_vector_in_domain(func, dist = 1e-10):
    r"""
    Samples a vector in the domain of an VectorIntegralFunctional such that the taking the norm
    in the vector dimension :math:`lower<|v|<upper` satisfies with a given distance `dist` 
    to the upper bounds.

    Note that if lower is below zero it is treated as zero.

    Parameters
    ----------
    func : regpy.functionals.VectorIntegralFunctional
        The vector integral functional.
    dist : float, optional
        Distance to the upper and lower bound. Defaults to 1e-10.

    Returns
    -------
    array-like
        A vector in the domain satisfying the norm constraints.

    """
    from regpy.functionals.base import Conj
    if isinstance(func,Conj):
        v_axes = func.func._vaxes
        norm = func.func.vector_norm
        dom_u = np.min(func.func.scalar_func.conj_dom_u) - dist
        dom_l = max(np.max(func.func.scalar_func.conj_dom_l),0.) + dist
    else:
        v_axes = func._vaxes
        norm = func.vector_norm
        dom_u = np.min(func.scalar_func.dom_u) - dist
        dom_l = max(np.max(func.scalar_func.dom_l),0.) + dist

    if dom_l >= dom_u:
        raise ValueError("Cannot sample vector in domain: No feasible region. upper has to be larger than lower.")

    u = func.domain.randn()
    point_norms = norm(x=u, axis=v_axes)
    max_norm = np.max(point_norms)
    if max_norm == 0:
        raise RuntimeError("Cannot normalize a randomly generated zero vector.")

    # Interpolate the pointwise norms, not the vector components. Adding the
    # lower bound componentwise changes the vector norm and can move a sample
    # outside the requested interval through rounding and cross terms.
    if dom_u < np.inf:
        target_norms = dom_l + (dom_u - dom_l) * point_norms / max_norm
    elif dom_l > 0:
        target_norms = dom_l + point_norms
    else:
        return u

    expanded_norms = np.expand_dims(point_norms, axis=v_axes)
    expanded_targets = np.expand_dims(target_norms, axis=v_axes)
    result = np.zeros_like(u)
    np.divide(u, expanded_norms, out=result, where=expanded_norms > 0)
    result *= expanded_targets
    return result


def test_moreaus_identity(func,u=None,tau=1.0,tolerance=1e-10):
    r"""Numerically test the validity of Moreau's identity for a given functional

    Checks if:

    .. math::
        u=prox_{\tau F}(u)+\tau prox_{\frac{1}{\tau}F^*}(\frac{u}{\tau}).

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : any
        Element in domain of func, if None it is chosen at random. Defaults to None.
    tau : float, optional
        Positive number tau in prox
    tolerance : float, optional
        The maximum allowed error in norm. Defaults to 1e-10.

    Returns
    -------
    boolean
        False, if the test fails and True otherwise.
    """
    if(u is None):
        if func.is_separable:
            u=sample_essential_domain(func)
        else:
            u=func.domain.randn()
    prox = func.proximal(u,tau)
    gram = func.h_domain.gram
    proxstar = func.conj.proximal(gram(u/tau),1/tau)
    err=func.domain.norm(u-prox-tau*gram.inverse(proxstar))
    if err < tolerance:
        func.log.info(f"Passed Moreaus identity with: err={err}, tolerance={tolerance}")
        return True
    else:
        func.log.warning(f"Failed Moreaus identity with: err={err}, tolerance={tolerance}")
        return False

def test_prox_optimality_cond(func,tau=1,u=None,tol=1e-10):
    r"""Numerically test validity of the optimality condition characterizing the prox operator: 

    .. math::
        (u-prox_{\tau F}(u))/tau \in  \partial F(prox_{\tau F}(u))

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    tau : float, optional
        Positive number tau in prox. Defaults to 1.
    u : array-like
        Element in domain of func, if None it is chosen at random. Defaults to None.
    
    Returns
    -------
    boolean
        False, if the test fails and True otherwise.
    """
    if u is None:
        numel = np.prod(func.domain.shape)
        u = np.tan(np.linspace(-np.pi/2+1/numel,np.pi/2-1/numel,numel))
    prox = func.proximal(u,tau)
    vec = func.h_domain.gram((u-prox)/tau)
    if func.dist_subdiff(vec,prox)<=tol:
        func.log.info("Passed optimality condition characterizing the prox operator")
        return True
    else:
        func.log.warning("Failed optimality condition characterizing the prox operator")
        return False



def test_subgradient(func,u=None,h_length=1e-8,tol_smooth=1e-2,tol_convex=1e-3):
    r"""Numerically test validity of subgradient for a given functional

    Checks if:

    .. math::
        0\geq F(u)-F(v)+\langle \grad F(u),v-u \rangle

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : any, optional
        Element in essential domain of func, where gradient is computed. If None it is chosen at random. Defaults to None.
    h_length : float, optional
        Positive number determining the length of the perturbation vector. Defaults to 1e-8.
    tol_convex : float, optional
        The maximum allowed violation of the convexity condition. Defaults to 1e-3.
    tol_smooth: float, optional
        The maximum violation of the differentiability condition. Defaults to 1e-2.

    Returns
    -------
    boolean
        False, if the test fails and True otherwise.
    """
    if (not func.is_separable):
        if u is None:
            u=func.domain.randn()
        h=func.domain.randn()
        h*=h_length/func.domain.norm(h)
    else:
        u,h=sample_essential_domain(func,u=u,eps_perturbation=h_length)
            
    grad_u=func.subgradient(u)
    diffq = (func(u)-func(u+h_length*h))/h_length
    deriv = (func.domain.vdot(grad_u,h)).real
    err= diffq+deriv
    if err<=tol_convex*func.domain.norm(grad_u):
        if np.abs(err)<=tol_smooth*func.domain.norm(grad_u):
            func.log.info("Passed subgradient test! Both the convexity and differentiability condition!")
            return True
        else:
            func.log.warning(f"Failed subgradient test! The differentiability condition is violated: |err|={np.abs(err)}, tol_smooth={tol_smooth}, |grad u| = {func.domain.norm(grad_u)}")
            return False
    else:
        func.log.warning(f"Failed subgradient test! The convexity condition is violated: err={err}, tol_convex={tol_convex}, |grad u| = {func.domain.norm(grad_u)}")
        return False

def test_gradient_as_operator(
        func, u=None, h=None, eps=1e-8, tolerance=1e-2,
        abs_tol=1e-6, test_inplace=True, test_inplace_aliasing=True,
        inplace_tolerance=1e-10):
    r"""Test :meth:`Functional.gradient_as_operator` and its derivative.

    The operator evaluation is compared with :meth:`Functional.subgradient`.
    Its derivative is checked by a difference quotient of the gradient and
    must be marked as self-adjoint, since it represents the Hessian. When
    requested, the generic operator tests additionally exercise allocating,
    ``out=`` and aliasing paths of the gradient operator and its derivative.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : any, optional
        Element in the essential domain of ``func``. If ``None``, it is
        sampled automatically.
    h : any, optional
        Perturbation direction. If ``None``, it is sampled automatically.
    eps : float, optional
        Perturbation length. Defaults to 1e-8.
    tolerance : float, optional
        The maximum relative violation of the condition.
    abs_tol : float, optional
        The maximum absolute violation of the condition.
    test_inplace : bool, optional
        Whether to run the generic allocating/``out=`` operator tests.
    test_inplace_aliasing : bool, optional
        Whether the generic operator test should include ``out is input``.
    inplace_tolerance : float, optional
        Tolerance for allocating/``out=`` comparisons.
    
    Returns
    -------
    bool
        ``True`` if all selected tests pass, otherwise ``False``.
    """
    if func.is_separable:
        u, sampled_h = sample_essential_domain(
            func, u=u, eps_perturbation=eps
        )
        if h is None:
            h = sampled_h
    else:
        if u is None:
            u = func.domain.randn()

    gradient_operator = func.gradient_as_operator()
    gradient, derivative = gradient_operator.linearize(u)

    # For non-separable functionals the essential domain can be a proper
    # subset of the vector space (positive-semidefinite matrices are an
    # important example).  The negative gradient is the same default
    # direction used by the former second-derivative test and, for convex
    # constraints, keeps a sufficiently short step on a feasible segment.
    if h is None:
        h = -gradient

    expected_gradient = func.subgradient(u)
    gradient_error = func.domain.norm(gradient - expected_gradient)
    gradient_scale = max(1., func.domain.norm(expected_gradient))
    if gradient_error > inplace_tolerance * gradient_scale:
        func.log.warning(
            "gradient_as_operator evaluation differs from subgradient: "
            f"err={gradient_error}, tolerance="
            f"{inplace_tolerance * gradient_scale}."
        )
        return False

    if not derivative.self_adjoint or derivative.adjoint is not derivative:
        func.log.warning(
            "The derivative returned by gradient_as_operator is not marked "
            "as self-adjoint."
        )
        return False

    hessian_h = derivative(h)
    difference_quotient = (
        gradient_operator(u + eps * h) - gradient
    ) / eps
    difference = hessian_h - difference_quotient
    difference_norm = func.h_domain.norm(difference)
    hessian_norm = func.h_domain.norm(hessian_h)
    allowed_error = max(abs_tol, tolerance * hessian_norm)
    if difference_norm > allowed_error:
        relative_error = difference_norm / (1e-14 + hessian_norm)
        func.log.warning(
            "Failed the derivative test of gradient_as_operator: "
            f"relative error={relative_error}, tolerance={tolerance}, "
            f"Hessian action norm={hessian_norm}."
        )
        return False

    if test_inplace:
        from .operator_tests import test_inplace_operator_methods
        if not test_inplace_operator_methods(
                gradient_operator, x=u, tolerance=inplace_tolerance,
                test_aliasing=test_inplace_aliasing):
            func.log.warning(
                "Failed allocating/in-place tests for gradient_as_operator."
            )
            return False

    func.log.info("Passed gradient_as_operator test!")
    return True


def test_reference_vector(
        func, check_minimizer=True, tolerance=1e-10, test_inplace=True):
    r"""Test :meth:`Functional.reference_vector`.

    Every reference vector must belong to the functional's domain and be
    feasible, i.e. the functional value at that vector must be finite. If
    ``check_minimizer`` is true and :meth:`Functional.dist_subdiff` is
    available, this function additionally checks that zero belongs to the
    subdifferential at the reference vector. The latter is deliberately
    optional because minimization is stronger than the public
    reference-vector contract.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    check_minimizer : bool, optional
        Whether to check that zero belongs to the subdifferential at the
        reference vector. Defaults to ``True``.
    tolerance : float, optional
        Absolute tolerance for the distance to the subdifferential. Defaults
        to 1e-10.
    test_inplace : bool, optional
        Whether to test the ``out=`` variant. Defaults to ``True``.
    Returns
    -------
    bool
        ``True`` if all selected tests pass, otherwise ``False``.
    """
    if "eval" not in func.methods:
        func.log.info(
            "Skipping the reference-vector test because feasibility cannot "
            "be verified without an advertised eval method."
        )
        return True

    reference = func.reference_vector()
    if reference not in func.domain:
        func.log.warning("reference_vector returned a vector outside the domain.")
        return False

    value = func(reference)
    if not np.isfinite(value):
        func.log.warning(
            f"reference_vector returned an infeasible vector: F(x)={value}."
        )
        return False

    if test_inplace:
        out = func.domain.zeros()
        returned = func.reference_vector(out=out)
        if returned is not out:
            func.log.warning(
                "reference_vector(out=out) did not return the supplied buffer."
            )
            return False
        if out not in func.domain or not np.isfinite(func(out)):
            func.log.warning(
                "reference_vector(out=out) returned an infeasible vector."
            )
            return False

    if check_minimizer:
        if "dist_subdiff" not in func.methods:
            func.log.info(
                "Skipping the optional reference-vector minimizer check "
                "because dist_subdiff is not advertised."
            )
        else:
            zero = func.domain.zeros(read_only=True)
            distance = func.dist_subdiff(zero, reference)
            if not np.isfinite(distance) or distance > tolerance:
                func.log.warning(
                    "The reference vector is feasible but does not pass the "
                    "optional minimizer check: "
                    f"dist(0, subdiff F(reference))={distance}, "
                    f"tolerance={tolerance}."
                )
                return False

    func.log.info("Passed reference_vector test!")
    return True


def zero_in_essential_domain(func):
    r"""Return whether zero belongs to a functional's essential domain.

    The check is based on finiteness of the functional value and therefore
    also applies to nonseparable functionals for which pointwise domain bounds
    are unavailable. If evaluation is not advertised, membership cannot be
    verified and ``False`` is returned.

    Parameters
    ----------
    func : regpy.functionals.Functional
        Functional whose essential domain is checked.

    Returns
    -------
    bool
        Whether the functional is finite at zero.
    """
    if "eval" not in func.methods:
        return False

    from regpy.functionals.base import NotInEssentialDomainError

    zero = func.domain.zeros(read_only=True)
    try:
        return bool(np.isfinite(func(zero)))
    except NotInEssentialDomainError:
        return False

def test_Lipschitz_convexity(func,u=None,safety=1.5):
    """Tests the Lipschitz convexity of the Functional!

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : any, optional
        Element in essential domain of func. If None it is chosen at random. Defaults to None.
    safety : float, optional
    
    Returns
    -------
    boolean
        False, if the test fails and True otherwise.
    """
    from regpy.vecsps import MeasureSpaceFcts
    if not func.is_separable:
        raise ValueError(Errors.value_error("Cannot sample in the essential domain if the functional is not separable!"))
    if u is None :
        u=sample_essential_domain(func)
    fpp = func.h_domain.gram_inv(func.hessian(u)(func.domain.ones()))
    ub = np.max(fpp) if isinstance(func.domain,MeasureSpaceFcts) else np.max(np.max([np.max(fpps) for fpps in func.domain.split(fpp)]))
    lb = np.min(fpp) if isinstance(func.domain,MeasureSpaceFcts) else np.min(np.min([np.max(fpps) for fpps in func.domain.split(fpp)])) 
    if func.Lipschitz<(1-1e-10)*ub:
        func.log.warning(f"Failed Lipschitz test! Lipschitz constant {func.Lipschitz} is smaller than second derivative {np.max(fpp)}")
        return False
    # if (np.max(func.dom_u)< np.inf or np.min(func.dom_l)>-np.inf) and func.Lipschitz != np.inf:
    #     func.log.warning(f"Failed Lipschitz test! Lipschitz constant finite {func.Lipschitz}, but essential domain is constrained: \n\t dom_u = {func.dom_u} \n\t dom_l = {func.dom_l}")
    #     return False
    # if func.convexity_param>np.min(fpp)+1e-12:
    #     func.log.warning("Failed Lipschitz test! Convexity parameter {func.convexity_param} is larger than second derivative {np.min(fpp)}")
    #     return False
    if func.convexity_param > 0 and func.convexity_param>lb+1e-12:
        func.log.warning(f"Failed Lipschitz test! Convexity parameter {func.convexity_param} is larger than second derivative {lb}")
        return False
    func.log.info("Passed Lipschitz test!")
    return True

def test_subgradient_conj_subgradient_dist_subdiff(func,u=None,tol=1e-6):
    if(u is None):
        if func.is_separable:
            u=sample_essential_domain(func)
        else:
            u=func.domain.rand()
    grad_u=func.subgradient(u)
    if func.conj.dist_subdiff(u,grad_u)>tol:
        func.log.warning(f"Failed subgradient_conj_subgradient test {func}: dist = {func.conj.dist_subdiff(u,grad_u)}, tol = {tol}")
        return False
    func.log.info("Passed subgradient_conj_subgradient_dist_subdiff test!")
    return True

def test_young_equality(func,u=None,tolerance=1e-10):
    r"""Numerically test validity of Young's equality for a given functional

    Checks if:

    .. math::
        F(u)+F^\ast(u^\ast)=\langle u^\ast,u \rangle.
    
    where :math:`u^\ast` is in the subgradient of :math:`F` at :math:`u`.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : any, optional
        Element in essential domain of func, where gradient is computed. If None it is chosen at random. Defaults to None.
    tolerance : float, optional
        The maximum allowed error. Defaults to 1e-10.

    Returns
    -------
    boolean
        False, if the test fails and True otherwise.
    """
    if(u is None):
        if func.is_separable:
            u=sample_essential_domain(func)
        else:
            u=func.domain.randn()
    grad_u=func.subgradient(u)
    t1 = (func.domain.vdot(u,grad_u)).real
    t2 = func(u)
    t3 = func.conj(grad_u)
    err=abs(t1-t2-t3)/np.max(np.abs([1e-14,t1,t2,t3]))
    if err<tolerance:
        func.log.info("Passed Young inequality test!")
        return True
    else:
        func.log.warning(f"Failed Young inequality test! err={err}, F(u)={t2}, F^*(grad_u)={t3}, <u,grad_u>={t1}")
        return False


def _copy_domain_vector(domain, x):
    """Return an independent copy of ``x`` using the vector-space interface."""
    result = domain.empty()
    domain.copyto(result, x)
    return result


def _vector_difference_within_tolerance(domain, actual, expected, tolerance):
    """Return ``(passed, error, scale)`` for two vectors in ``domain``."""
    error = domain.norm(actual - expected)
    scale = max(1., domain.norm(expected))
    return error <= tolerance * scale, error, scale


def _test_allocating_vs_inplace_call(func, method_name, x, method_args,
                                     tolerance, test_aliasing=True):
    """Compare one allocating functional call with its ``out=`` variants."""
    method = getattr(func, method_name)
    domain = func.domain

    x_reference = _copy_domain_vector(domain, x)
    expected = method(x, *method_args)

    passed, input_error, input_scale = _vector_difference_within_tolerance(
        domain, x, x_reference, tolerance
    )
    if not passed:
        func.log.warning(
            f"Allocating {method_name} modified its input: "
            f"err={input_error}, tolerance={tolerance * input_scale}."
        )
        return False

    # Initialize with arbitrary values so that incomplete writes are visible.
    out = domain.randn()
    result = method(x, *method_args, out=out)
    if result is not out:
        func.log.warning(
            f"In-place {method_name} did not return the supplied output buffer."
        )
        return False

    passed, error, scale = _vector_difference_within_tolerance(
        domain, out, expected, tolerance
    )
    if not passed:
        func.log.warning(
            f"Allocating and in-place {method_name} differ: "
            f"err={error}, tolerance={tolerance * scale}."
        )
        return False

    passed, input_error, input_scale = _vector_difference_within_tolerance(
        domain, x, x_reference, tolerance
    )
    if not passed:
        func.log.warning(
            f"Non-aliasing in-place {method_name} modified its input: "
            f"err={input_error}, tolerance={tolerance * input_scale}."
        )
        return False

    if test_aliasing:
        alias = _copy_domain_vector(domain, x_reference)
        result = method(alias, *method_args, out=alias)
        if result is not alias:
            func.log.warning(
                f"Aliasing in-place {method_name} did not return its input/output buffer."
            )
            return False

        passed, error, scale = _vector_difference_within_tolerance(
            domain, alias, expected, tolerance
        )
        if not passed:
            func.log.warning(
                f"Allocating and aliasing in-place {method_name} differ: "
                f"err={error}, tolerance={tolerance * scale}."
            )
            return False

    return True


def test_inplace_functional_methods(func, u=None, ustar=None, tau=1.,
                                    tolerance=1e-10, test_primal=True,
                                    test_conjugate=True, test_aliasing=True):
    r"""Compare allocating and ``out=`` variants of functional methods.

    The following pairs are checked whenever the corresponding method is
    advertised by the functional:

    * ``subgradient(x)`` and ``subgradient(x, out=out)``
    * ``proximal(x, tau)`` and ``proximal(x, tau, out=out)``
    * ``conj_subgradient(xstar)`` and
      ``conj_subgradient(xstar, out=out)``
    * ``conj_proximal(xstar, tau)`` and
      ``conj_proximal(xstar, tau, out=out)``

    In addition, the test checks the supported aliasing case ``out is x`` and
    verifies that non-aliasing calls leave their input unchanged.

    Parameters
    ----------
    func : regpy.functionals.Functional
        Functional whose methods are tested.
    u, ustar : element of ``func.domain``, optional
        Test arguments for the functional and its conjugate. Missing arguments
        are sampled from the corresponding essential domain when separable,
        and randomly otherwise.
    tau : float, optional
        Positive proximal parameter. Defaults to 1.
    tolerance : float, optional
        Relative tolerance, with an absolute scale floor of one.
    test_primal, test_conjugate : bool, optional
        Select whether primal and/or conjugate methods are checked.
    test_aliasing : bool, optional
        Also check calls with the input itself as output buffer.

    Returns
    -------
    bool
        ``True`` if all available selected methods agree.
    """
    if tau <= 0:
        raise ValueError(Errors.value_error(
            f"tau must be positive in test_inplace_functional_methods. Got {tau}."
        ))
    if tolerance < 0:
        raise ValueError(Errors.value_error(
            f"tolerance must be non-negative. Got {tolerance}."
        ))

    tests = []

    if test_primal:
        if u is None:
            u = (sample_essential_domain(func) if func.is_separable
                 else func.domain.randn())
        if "subgradient" in func.methods:
            tests.append(("subgradient", u, ()))
        if "proximal" in func.methods:
            tests.append(("proximal", u, (tau,)))

    if test_conjugate:
        if ustar is None:
            conj = func.conj
            ustar = (sample_essential_domain(conj) if conj.is_separable
                     else func.domain.randn())
        if "subgradient" in func.conj.methods:
            tests.append(("conj_subgradient", ustar, ()))
        if "proximal" in func.conj.methods:
            tests.append(("conj_proximal", ustar, (tau,)))

    for method_name, argument, method_args in tests:
        if not _test_allocating_vs_inplace_call(
                func, method_name, argument, method_args, tolerance,
                test_aliasing=test_aliasing):
            return False

    func.log.info(
        "Allocating and in-place functional methods produce identical results."
    )
    return True

def test_functional(func,u_s=None,sample_N=5,
                    test_conj=True,
                    u_stars=None,sample_conj_N=5,
                    test_second_deriv=None, test_second_deriv_conj=None,
                    test_inplace=True, test_inplace_aliasing=True,
                    tolerance=1e-10, msg='',
                    test_gradient_operator=True,
                    test_gradient_operator_conj=True,
                    test_reference=True, test_reference_conj=True,
                    reference_is_minimizer=None,
                    reference_is_minimizer_conj=None):
    r"""Runs all implemented tests for a given functional. By default tests that cannot be verified because of 
    missing implementations are ignored.


    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u_s : list of any, optional
        List of elements in essential domain of func. 
        If None, one sample is chosen by sample_essential_domain. 
        Defaults to None.
    sample_N : int, optional
        If u_s i None this is the number of randomly generated elements in u_s. Defaults to 5.
    test_conj : bool, optional
        Determines wether the conjugate functional should be tested aswell. Defaults to True.
    u_stars : list of any, optional
        Same as u_s but for conjugate functional. Defaults to None.
    sample_conj_N : int, optional
        Same as sample_N but for conjugate functional. Defaults to 5.
    test_second_deriv, test_second_deriv_conj : bool or None, optional
        Deprecated aliases for ``test_gradient_operator`` and
        ``test_gradient_operator_conj``. They are retained so existing test
        suites can migrate without an immediate API break.
    test_inplace : bool, optional
        Compare allocating and ``out=`` variants of subgradient, proximal,
        conjugate subgradient, and conjugate proximal methods. Defaults to True.
    test_inplace_aliasing : bool, optional
        Also test the supported aliasing case where the output buffer is the
        input vector itself. Defaults to True.
    tolerance : float, optional
        The maximum allowed relative error. Defaults to 1e-10.
    test_gradient_operator, test_gradient_operator_conj : bool, optional
        Test :meth:`Functional.gradient_as_operator`, especially its
        derivative, for the functional and its conjugate. Defaults to True.
    test_reference, test_reference_conj : bool, optional
        Test :meth:`Functional.reference_vector` for the functional and its
        conjugate. Defaults to True.
    reference_is_minimizer : bool or None, optional
        Whether the primal reference vector must minimize the primal
        functional. If ``None`` (the default), minimality is required exactly
        when zero belongs to the essential domain of the conjugate functional.
        In that case a conjugate subgradient at zero, and hence a minimizer of
        the primal functional, should be available. Explicit ``True`` or
        ``False`` values override this automatic rule.
    reference_is_minimizer_conj : bool or None, optional
        Whether the conjugate reference vector must minimize the conjugate
        functional. If ``None`` (the default), minimality is required exactly
        when zero belongs to the essential domain of the primal functional.
        In that case a primal subgradient at zero, and hence a minimizer of the
        conjugate, should be available. Explicit ``True`` or ``False`` values
        override this automatic rule.

    Raises
    ------
    AssertionError
        If the test fails.

    Notes
    -----
    The tests results are logged with further details when the tests failed 

    """
    if test_second_deriv is not None:
        test_gradient_operator = test_second_deriv
    if test_second_deriv_conj is not None:
        test_gradient_operator_conj = test_second_deriv_conj

    if reference_is_minimizer is None:
        reference_is_minimizer = (
            test_reference
            and zero_in_essential_domain(func.conj)
        )

    if reference_is_minimizer_conj is None:
        reference_is_minimizer_conj = (
            test_conj
            and test_reference_conj
            and zero_in_essential_domain(func)
        )

    if test_reference:
        if not test_reference_vector(
                func, check_minimizer=reference_is_minimizer,
                tolerance=tolerance, test_inplace=test_inplace):
            raise AssertionError(f"{func} failed reference_vector test!" + msg)

    if (u_s is None):
        if func.is_separable:
            u_s= [None] # [sample_essential_domain(func)]
        else:
            u_s = [func.domain.randn() for _ in range(sample_N)]
    func.log.info(f'Starting tests for functional!')
    for u in u_s:
        if test_inplace:
            tau_inplace = uniform(max(tolerance, np.finfo(float).eps), 4)
            if not test_inplace_functional_methods(
                    func, u=u, tau=tau_inplace, tolerance=tolerance,
                    test_primal=True, test_conjugate=False,
                    test_aliasing=test_inplace_aliasing):
                raise AssertionError(
                    f"{func} failed allocating/in-place equivalence tests!"
                    + msg
                )
        if "proximal" in func.methods and "proximal" in func.conj.methods:
            tau=uniform(tolerance,4)
            if not test_moreaus_identity(func,u,tau=tau,tolerance=tolerance):
                raise AssertionError(f"{func} failed Moreaus identity!"+msg)        
        if {"eval","subgradient"} <= func.methods:
            if not test_subgradient(func,u):
                raise AssertionError(f"{func} failed Subgradient Test!"+msg)
        if {"eval","subgradient"} <= func.methods and "eval" in func.conj.methods:
            if not test_young_equality(func,u,tolerance=tolerance):
                raise AssertionError(f"{func} failed Young equality!"+msg)
        if {"subgradient"} <= func.methods and {"subgradient","is_subgradient"} <= func.conj.methods:
            if not test_subgradient_conj_subgradient_dist_subdiff(func,u):
                raise AssertionError(f"{func} failed subgradient_conj_subgradient_dist_subdiff test"+msg)
        if test_gradient_operator:
            if {"subgradient","hessian"} <= func.methods:
                if not test_gradient_as_operator(
                        func, u, test_inplace=test_inplace,
                        test_inplace_aliasing=test_inplace_aliasing,
                        inplace_tolerance=tolerance):
                    raise AssertionError(
                        f"{func} failed gradient_as_operator test!" + msg
                    )
            if func.is_separable:
                try:
                    if not test_Lipschitz_convexity(func):
                        raise AssertionError(f"{func} failed Lipschitz convexity test!"+msg)
                except (NotImplementedError):
                    func.log.info('Lipschitz constant and convexity parameter could not be checked because of missing implementation.')
        func.log.info(f'All tests passed!')
    if test_conj and test_inplace:
        if u_stars is None:
            if func.conj.is_separable:
                inplace_u_stars = [None]
            else:
                inplace_u_stars = [
                    func.domain.randn() for _ in range(sample_conj_N)
                ]
        else:
            inplace_u_stars = u_stars
        for ustar in inplace_u_stars:
            tau_inplace = uniform(max(tolerance, np.finfo(float).eps), 4)
            if not test_inplace_functional_methods(
                    func, ustar=ustar, tau=tau_inplace,
                    tolerance=tolerance, test_primal=False,
                    test_conjugate=True,
                    test_aliasing=test_inplace_aliasing):
                raise AssertionError(
                    f"{func} failed allocating/in-place equivalence tests "
                    "for conjugate methods!" + msg
                )
    if(test_conj):
        test_functional(
            func.conj, u_s=u_stars, sample_N=sample_conj_N,
            test_conj=False,
            test_inplace=False,
            test_inplace_aliasing=test_inplace_aliasing,
            tolerance=tolerance, msg=msg,
            test_gradient_operator=test_gradient_operator_conj,
            test_gradient_operator_conj=False,
            test_reference=test_reference_conj,
            test_reference_conj=False,
            reference_is_minimizer=reference_is_minimizer_conj,
            reference_is_minimizer_conj=False,
        )
                
