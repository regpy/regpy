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
    eps_perturbation: float or None [default: None]
        If not None, an additional vector h is returned  
        
    Returns
    -------
    array-like
        An element u of func.domain
        If eps_perturbation is not None, an additional vector h is returned such that u+eps_perturbation is also in the essential domain. 
    """
    from regpy.vecsps import MeasureSpaceFcts, DirectSum
    if not func.is_separable:
        raise ValueError(Errors.value_error("Cannot sample in the essential domain if the functional is not separable!"))
    if isinstance(func.domain,MeasureSpaceFcts):
        dom_l = np.max(func.dom_l)
        dom_u = np.min(func.dom_u)
        assert dom_l<=dom_u
        numel = np.prod(func.domain.shape)
        if dom_l>-np.inf:
            if dom_u<np.inf:
                if u is None:
                    #u = np.linspace(dom_l,dom_u,numel)
                    u = np.linspace(dom_l,dom_u,numel+2)
                    u = u[1:-1]
                if eps_perturbation is not None:
                    h = np.sign(0.5*dom_l+0.5*dom_u-u)
                    fac = 2 * eps_perturbation / np.min(dom_u-dom_l)
                    if fac >= 1.:
                        h /= fac
            else: 
                if u is None:
                    u = dom_l-0.5*np.exp(-np.sqrt(numel))+np.exp(np.linspace(-np.sqrt(numel),np.sqrt(numel),numel))
                if eps_perturbation is not None:
                    h = np.ones_like(u)
        else: # dom_l == np.inf
            if dom_u==np.inf:
                if u is None:
                    u = np.tan(np.linspace(-np.pi/2+1/numel,np.pi/2-1/numel,numel))
                if eps_perturbation is not None:
                    h = np.ones_like(u)            
            else:
                if u is None:
                    u = dom_u+0.5*np.exp(-np.sqrt(numel))-np.exp(np.linspace(-np.sqrt(numel),np.sqrt(numel),numel))
                if eps_perturbation is not None:
                    h = - np.ones_like(u)            
        if eps_perturbation is None:
            return np.reshape(u,func.domain.shape)
        else:
            return np.reshape(u,func.domain.shape), np.reshape(h,func.domain.shape)
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
        if eps_perturbation is None:
            return func.domain.join(*(sample_essential_domain(funci) for funci in funcs))
        else:
            xx,hh = zip(*(sample_essential_domain(funci, eps_perturbation=eps_perturbation) for funci in funcs))
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
        Distance to the upper and lower bound. Defaults to 1e10.

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
    # scaling u s.t. |u| <= 1
    u /= np.max(norm(x = u, axis = v_axes))
    if dom_u < np.inf and dom_l > 0:
        return u * (dom_u - dom_l) + dom_l
    elif dom_u < np.inf:
        return u * dom_u
    else:
        return u        


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

def test_second_derivative(func,u=None,h=None,eps=1e-8,tolerance = 1e-2,abs_tol=1e-6):
    """Tests the second derivative of the functional.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : any, optional
        Element in essential domain of func. If None it is chosen at random. Defaults to None.
    h : any, optional
        Perturbation to u. Defaults to None.
    eps : float, optional
        Perturbation scalar. Defaults to 1e-3.
    tolerance: float, optional
        The maximum relative violation of the condition.
    abs_tol: float, optional
        the maximum absolute violation of the condition.
    
    Returns
    -------
    boolean
        False, if the test fails and True otherwise.
    """
    if func.is_separable:
        u,h = sample_essential_domain(func,eps_perturbation=eps)
    else:
        if u is None:
            u = func.domain.randn()
        h = - func.subgradient(u)
    func_pp = func.hessian(u)(h)
    diffq = (func.subgradient(u+eps*h)-func.subgradient(u))/eps
    err = func.h_domain.norm(func_pp-diffq)/(1e-14+func.h_domain.norm(func_pp))

    if func.h_domain.norm(func_pp-diffq)<=np.max([abs_tol,tolerance * func.h_domain.norm(func_pp)]):
        func.log.info("Passed second derivative test!")
        return True
    else:
        func.log.warning(f"Failed the second derivative test! err: {err}, tol: {tolerance}, norm second deriv. {func.h_domain.norm(func_pp)}")
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
                    test_second_deriv = True, test_second_deriv_conj = True,
                    test_inplace=True, test_inplace_aliasing=True,
                    tolerance=1e-10, msg=''):
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
    test_inplace : bool, optional
        Compare allocating and ``out=`` variants of subgradient, proximal,
        conjugate subgradient, and conjugate proximal methods. Defaults to True.
    test_inplace_aliasing : bool, optional
        Also test the supported aliasing case where the output buffer is the
        input vector itself. Defaults to True.
    tolerance : float, optional
        The maximum allowed relative error. Defaults to 1e-10.

    Raises
    ------
    AssertionError
        If the test fails.

    Notes
    -----
    The tests results are logged with further details when the tests failed 

    """
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
        if test_second_deriv:
            if {"subgradient","hessian"} <= func.methods:
                if not test_second_derivative(func,u):
                    raise AssertionError(f"{func} failed second derivative test!"+msg)
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
            test_conj=False, test_second_deriv=test_second_deriv_conj,
            test_inplace=False,
            test_inplace_aliasing=test_inplace_aliasing,
            tolerance=tolerance, msg=msg
        )
                