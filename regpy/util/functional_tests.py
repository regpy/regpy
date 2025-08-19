import numpy as np

def test_moreaus_identity(func,u=None,tau=1.0,tolerance=1e-10):
    r"""Numerically test validity of moreaus identity for a given functional

    Checks if:

    .. math::
        u=prox_{\tau F}(u)+\tau prox_{\frac{1}{\tau}F(\frac{u}{\tau}).

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : np.ndarray
        Element in domain of func, if None it is chosen at random. Defaults to None.
    tau : float, optional
        Positive number tau in prox
    tolerance : float, optional
        The maximum allowed error in norm. Defaults to 1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if(u is None):
        u=func.domain.randn()
    prox = func.proximal(u,tau)
    gram = func.h_domain.gram
    proxstar = func.conj.proximal(gram(u/tau),1/tau)
    err=np.linalg.norm(u-prox-tau*gram.inverse(proxstar))
    assert err<tolerance,f'err={err}'

def test_subgradient(func,u=None,v=None,v_length=1e-5,tolerance=1e-10):
    r"""Numerically test validity of subgradient for a given functional

    Checks if:

    .. math::
        0\geq F(u)-F(v)+\langle \grad F(u),v-u \rangle

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : np.ndarray, optional
        Element in essential domain of func, where gradient is computed. If None it is chosen at random. Defaults to None.
    v : np.ndarray, optional
        Element in domain of func, where gradient is computed. If None it is chosen at random with length given by v_length. Defaults to None.
    v_length : float, optional
        Positive number determining the length of v if it is not given explicitly. Defaults to 1e-5.
    tolerance : float, optional
        The maximum allowed error. Defaults to 1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if(u is None):
        u=func.domain.randn()
    if(v is None):
        v=func.domain.randn()
        v*=v_length/np.linalg.norm(v)
    grad_u=func.subgradient(u)
    err=func(u)-func(v)+np.real(np.vdot(grad_u,v-u))
    assert err<tolerance,f'err={err}'
    
# def test_subgradient_and_conj(func,u=None,eps=1e-10):
#     if(u is None):
#         u=func.domain.randn()
#     grad_u=func.subgradient(u)
#     print(u)
#     print(grad_u)
#     print(np.abs(grad_u))
#     assert func.conj_is_subgradient(u,grad_u,eps=eps)

def test_young_equality(func,u=None,tolerance=1e-10):
    r"""Numerically test validity of young equality for a given functional

    Checks if:

    .. math::
        F(u)+F^\ast(u^\ast)=\langle u^\ast,u \rangle.
    
    where :math:`u^\ast` is in the subradient of :math:`F` at :math:`u`.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u : np.ndarray, optional
        Element in essential domain of func, where gradient is computed. If None it is chosen at random. Defaults to None.
    tolerance : float, optional
        The maximum allowed error. Defaults to 1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if(u is None):
        u=func.domain.randn()
    grad_u=func.subgradient(u)
    err=np.abs(np.real(np.vdot(u,grad_u))-func(u)-func.conj(grad_u))
    assert err<tolerance,f'err={err}'

def test_functional(func,u_s=None,sample_N=5,test_conj=True,u_stars=None,sample_conj_N=5,print_results=False,tolerance=1e-10):
    r"""Runs all implemented tests for a given functional. By default tests that cannot be verified because of 
    missing implementations are ignored.


    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    u_s : list of np.ndarray, optional
        List of elements in essential domain of func. If None it they chosen at random. Defaults to None.
    sample_N : int, optional
        If u_s i None this is the number of randomly generated elements in u_s. Defaults to 5.
    test_conj : bool, optional
        Determines wether the conjugate functional should be tested aswell. Defaluts to True.
    u_stars : list of np.ndarray, optional
        Same as u_s but for conjugate functional. Defaults to None.
    sample_conj_N : int, optional
        Same as sample_N but for conjugate functional. Defaults to 5.
    print_results : bool, optional
        If set to True, further information about test that could not be evaluated because of missing implementations
        are printed. Defaults to False.
    tolerance : float, optional
        The maximum allowed error. Defaults to 1e-10.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    if(u_s is None):
        u_s=[func.domain.randn() for _ in range(sample_N)]
    if(print_results):
        print(type(func))
    for u in u_s:
        try:
            tau=np.random.uniform(tolerance,4)
            test_moreaus_identity(func,u,tau=tau,tolerance=tolerance)
        except(NotImplementedError):
            if(print_results):
                print('Moreaus identity could not be checked because of missing implementation')
        try:
            test_subgradient(func,u,tolerance=tolerance)
        except(NotImplementedError):
            if(print_results):
                print('Subgradient could not be checked because of missing implementation')
        try:
            test_young_equality(func,u,tolerance=tolerance)
        except(NotImplementedError):
            if(print_results):
                print('Young equality could not be checked because of missing implementation')
    if(test_conj):
        test_functional(func.conj,u_s=u_stars,sample_N=sample_conj_N,test_conj=False,print_results=print_results,tolerance=tolerance)

        