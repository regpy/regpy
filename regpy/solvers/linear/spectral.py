from __future__ import annotations

import inspect
from numbers import Real
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from scipy.linalg import cholesky
from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator
from scipy.sparse.linalg import eigsh, svds

from regpy import vecsps
from regpy.vecsps import ReadOnlyVector
from regpy.operators.base import (
    Composition,
    Identity,
    LinearCombination,
    ObsDataT,
    Operator,
    PtwMultiplication,
)
from regpy.operators.numpy import MatrixMultiplication
from regpy.util import Errors

from ..sequences import Naturals
from .. import RegSolver, Setting, check_Hilbert_applicability_linear
#from ..solver import 
#from ..setting import 


__all__ = [
    "SVD",
    "Spectral",
    "Spectral_SelfAdjoint",
    "Spectral_Tikhonov",
    "Spectral_IteratedTikhonov",
    "Spectral_Showalter",
    "Spectral_Landweber",
    "Spectral_Lardy",
    "Spectral_CutOff",
    "Spectral_Seidman",
    "Spectral_TruncatedSVD2",
    "SpectralSelfAdjoint_Lavrentiev",
    "SpectralSelfAdjoint_Richardson",
]


def _positive_identity_factor(gram: Operator) -> float | None:
    """Return c for gram == c*I, or None if this form is not recognized."""
    if isinstance(gram, Identity):
        return 1.
    if isinstance(gram, LinearCombination):
        if len(gram.ops) != 1 or not isinstance(gram.ops[0], Identity):
            return None
        factor = gram.coeffs[0]
    elif isinstance(gram, PtwMultiplication):
        values = np.asarray(gram.factor)
        if values.size == 0:
            return None
        factor = values.reshape(-1)[0]
        if not np.allclose(values, factor):
            return None
    else:
        return None

    factor = np.real_if_close(factor)
    if np.ndim(factor) != 0:
        return None
    factor = factor.item()
    if isinstance(factor, complex) or not isinstance(factor, Real):
        return None
    if factor <= 0:
        return None
    return float(factor)


def _real_nonnegative_singular_values(
    multiplier: Any,
    *,
    name: str = "singular-value multiplier",
) -> np.ndarray:
    """Return a real copy of a numerically real, non-negative multiplier.

    A complex storage dtype is permitted.  Its imaginary part must vanish up
    to a scale-aware rounding tolerance consistent with the other randomized
    SVD checks in this module.
    """
    values = np.asarray(multiplier)
    if not np.all(np.isfinite(values)):
        raise ValueError(Errors.value_error(
            f"The {name} must contain only finite values."
        ))

    real_values = np.asarray(values.real)
    if np.iscomplexobj(values) and values.size:
        try:
            eps = np.finfo(real_values.dtype).eps
        except (TypeError, ValueError):
            eps = np.finfo(float).eps
        scale = max(1., float(np.max(np.abs(values))))
        imaginary_tolerance = 5000 * eps * scale
        if float(np.max(np.abs(values.imag))) > imaginary_tolerance:
            raise ValueError(Errors.value_error(
                f"The {name} must be real-valued. A complex storage dtype "
                "is accepted only when all imaginary parts vanish up to "
                f"rounding errors (tolerance={imaginary_tolerance})."
            ))

    if not np.all(real_values >= 0):
        raise ValueError(Errors.value_error(
            f"The {name} must contain only non-negative values."
        ))
    return np.array(real_values, copy=True)


def _gram_rescaling(
    old_gram: Operator,
    new_gram: Operator,
    name: str,
) -> float:
    old_factor = _positive_identity_factor(old_gram)
    new_factor = _positive_identity_factor(new_gram)
    if old_factor is not None and new_factor is not None:
        return float(np.sqrt(old_factor / new_factor))
    if old_gram is not new_gram:
        raise ValueError(Errors.value_error(
            f"The labelled SVD and setting use different non-scalar {name} "
            "Gram operators. Such Gram operators must be identical objects."
        ))
    return 1.


def _gram_operators_equal(first: Operator, second: Operator) -> bool:
    if first is second:
        return True
    first_factor = _positive_identity_factor(first)
    second_factor = _positive_identity_factor(second)
    return (
        first_factor is not None
        and second_factor is not None
        and np.isclose(first_factor, second_factor)
    )


def _as_scipy_operator(op: Operator) -> SciPyLinearOperator:
    """Return a SciPy view preserving a genuinely complex scalar field."""
    if not (op.domain.is_complex and op.codomain.is_complex):
        # The standard RegPy wrapper uses the real flatten/fromflat
        # coordinates. Numerical SVDs with mixed scalar fields are rejected
        # by SVD._init_numerical before this helper is reached.
        return op.as_linear_operator()
    domain = op.domain
    codomain = op.codomain
    dtype = np.result_type(
        getattr(domain, 'dtype', complex),
        getattr(codomain, 'dtype', complex),
        1j,
    )

    try:
        domain_coordinates = domain.complex_flatten(domain.zeros())
        codomain_coordinates = codomain.complex_flatten(codomain.zeros())
        domain.complex_fromflat(np.asarray(domain_coordinates))
        codomain.complex_fromflat(np.asarray(codomain_coordinates))
    except (AttributeError, NotImplementedError) as error:
        raise TypeError(Errors.type_error(
            "A genuinely complex numerical SVD requires both endpoint "
            "spaces to implement complex_flatten and complex_fromflat."
        )) from error

    def matvec(vector):
        argument = domain.complex_fromflat(
            np.asarray(vector, dtype=dtype).reshape(-1)
        )
        return np.asarray(
            codomain.complex_flatten(op(argument)), dtype=dtype
        ).reshape(-1)

    def rmatvec(vector):
        argument = codomain.complex_fromflat(
            np.asarray(vector, dtype=dtype).reshape(-1)
        )
        return np.asarray(
            domain.complex_flatten(op.adjoint(argument)), dtype=dtype
        ).reshape(-1)

    return SciPyLinearOperator(
        dtype=dtype,
        shape=(int(codomain.size), int(domain.size)),
        matvec=matvec,
        rmatvec=rmatvec,
    )


def _assemble_matrix(op: Operator) -> np.ndarray:
    linear_operator = _as_scipy_operator(op)
    identity = np.eye(
        linear_operator.shape[1], dtype=linear_operator.dtype
    )
    return np.asarray(linear_operator.matmat(identity))


def _cholesky_factor(hilbert_space) -> Operator:
    gram = hilbert_space.gram
    scalar_factor = _positive_identity_factor(gram)
    if scalar_factor is not None:
        if scalar_factor == 1:
            return hilbert_space.vecsp.identity
        return PtwMultiplication(
            hilbert_space.vecsp, np.sqrt(scalar_factor)
        )

    try:
        factor = hilbert_space.cholesky
    except NotImplementedError:
        factor = None
    if factor is not None:
        if not isinstance(factor, Operator) or not factor.linear:
            raise TypeError(Errors.type_error(
                "A Hilbert-space Cholesky factor must be a linear Operator."
            ))
        if factor.codomain != hilbert_space.vecsp:
            raise ValueError(Errors.value_error(
                "For the convention G=L L*, a Hilbert-space Cholesky "
                "factor must have the Hilbert vector space as codomain."
            ))
        if factor.domain.is_complex != hilbert_space.vecsp.is_complex:
            raise ValueError(Errors.value_error(
                "The Cholesky factor's auxiliary domain has the wrong real "
                "or complex scalar field."
            ))
        try:
            factor.adjoint.inverse
        except NotImplementedError:
            # Transforming singular vectors back requires L^{-*}. If an
            # otherwise valid factor has no known inverse, assemble the Gram
            # matrix below and build an explicitly invertible factor.
            factor = None
        else:
            return factor

    gram_matrix = _assemble_matrix(gram)
    lower_factor = cholesky(
        gram_matrix,
        lower=True,
        check_finite=True,
    )
    return MatrixMultiplication(
        lower_factor,
        inverse='inv',
        domain=hilbert_space.vecsp,
        codomain=hilbert_space.vecsp,
    )


class SVD(Composition):
    r"""Singular-value decomposition associated with a Hilbert setting.

    The resulting operator is represented as

    .. math::
        T = V M U^*,\qquad
        V^*G_YV=I,\quad U^*G_X^{-1}U=I.

    Here, :math:`U` maps the reduced coordinate (spectral) space into the dual Hilbert
    space :math:`X^*`. Its Gram operator is :math:`G_X^{-1}`, which explains
    the second normalization. The entries ``G_X`` and ``G_Y`` in
    :attr:`svd_data` store the Gram operators actually used for ``U`` and
    ``V``; hence the former is :attr:`setting.h_domain.gram_inv`.

    If ``setting.op`` is already a labelled :class:`Composition`, its factors
    are reused. Scalar identity Gram operators may differ; in that case ``U``,
    ``V``, and ``M`` are rescaled without recomputing an SVD. Non-scalar Gram
    operators must be identical objects to those stored in the label. The
    labelled multiplier must be finite, non-negative, and real up to rounding
    errors. A complex storage dtype is allowed; accepted imaginary parts are
    discarded explicitly.

    Otherwise a numerical generalized SVD is computed. If :math:`G=LL^*`
    denotes the Cholesky convention used by RegPy, the standard SVD is applied
    to

    .. math::
        B=L_Y^*T L_X^{-*}.

    Its Euclidean singular-vector operators :math:`U_0,V_0` are transformed
    back by :math:`U=L_XU_0` and
    :math:`V=L_Y^{-*}V_0`.

    If ``as_eig=True``, the operator must be self-adjoint and positive
    semidefinite. The codomain is interpreted as :math:`X^*`, and the
    transformed operator :math:`L_X^{-1}TL_X^{-*}` is diagonalized with a
    Hermitian eigensolver. This mode is also available for codomain-free
    settings.

    Genuinely complex-linear operators between complex vector spaces are
    passed to SciPy as complex operators, so their singular values are not
    duplicated by realification. Real operators use the generic
    :meth:`~regpy.vecsps.VectorSpaceBase.flatten` and ``fromflat`` coordinates.
    Domain and codomain must use the same scalar field. Complex backends must
    implement the analogous :meth:`complex_flatten` and
    :meth:`complex_fromflat` methods.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Hilbert setting with a linear operator. A codomain-free setting is
        accepted only when ``as_eig=True``.
    as_eig : bool [default: False]
        Compute or reuse an eigenvalue decomposition of a self-adjoint,
        positive-semidefinite operator instead of a general SVD.
    k : int or None [default: None]
        If ``None``, compute a dense full SVD and retain all positive singular
        values. Otherwise compute a matrix-free partial SVD with ``k`` values.
    which : {'LM', 'SM'} [default: 'LM']
        Largest or smallest singular values for a partial SVD.
    solver : {'arpack', 'lobpcg', 'propack'} [default: 'arpack']
        SciPy solver for a partial SVD.
    tol : float [default: 0]
        Iterative tolerance passed to ``scipy.sparse.linalg.svds``.
    maxiter, ncv, v0, rng, options
        Further options passed to ``svds`` in the partial case.
    singular_value_tolerance : float or None [default: None]
        Values not exceeding this threshold are omitted. By default a
        dimension- and precision-dependent threshold is used.
    use_buffers : bool [default: True]
        Whether the resulting composition should reuse intermediate buffers.

    Attributes
    ----------
    U, V : regpy.operators.Operator
        Right and left Hilbert-isometric singular-vector operators.
    M : regpy.operators.PtwMultiplication
        Positive singular-value multiplier on the reduced coordinate space.
    singular_values : numpy.ndarray
        Retained singular values as a read-only real array.
    numerical : bool
        Whether a numerical SVD was needed.
    """

    def __init__(
        self,
        setting: Setting,
        *,
        as_eig: bool = False,
        k: int | None = None,
        which: str = 'LM',
        solver: str = 'arpack',
        tol: float = 0.,
        maxiter: int | None = None,
        ncv: int | None = None,
        v0: np.ndarray | None = None,
        rng: Any = None,
        options: dict | None = None,
        singular_value_tolerance: float | None = None,
        use_buffers: bool = True,
    ) -> None:
        if not isinstance(setting, Setting):
            raise TypeError(Errors.not_instance(setting, Setting))
        if not setting.is_hilbert:
            raise ValueError(Errors.value_error(
                "SVD requires a Hilbert setting."
            ))
        if not setting.op.linear:
            raise ValueError(Errors.value_error(
                "SVD requires a linear operator."
            ))
        if not isinstance(as_eig, (bool, np.bool_)):
            raise TypeError(Errors.type_error("as_eig must be boolean."))
        self.as_eig = bool(as_eig)
        if setting.codomain_free and not self.as_eig:
            raise ValueError(Errors.value_error(
                "A general SVD requires an explicit codomain Hilbert space."
            ))
        labelled_eigenoperator = _is_labelled_svd(
            setting.op, require_eig=True
        )
        if self.as_eig and not (
            setting.op.self_adjoint or labelled_eigenoperator
        ):
            raise ValueError(Errors.value_error(
                "An eigenvalue decomposition requires a self-adjoint "
                "operator."
            ))

        self.original_operator = setting.op
        self.h_domain = setting.h_domain
        self.h_codomain = (
            setting.h_domain.dual_space()
            if self.as_eig else setting.h_codomain
        )

        if (
            isinstance(setting.op, Composition)
            and setting.op.is_svd
            and (not self.as_eig or setting.op.svd_data['is_eig'])
        ):
            self._init_from_labelled(
                setting.op,
                use_buffers=use_buffers,
            )
            self.numerical = False
        else:
            self._init_numerical(
                k=k,
                which=which,
                solver=solver,
                tol=tol,
                maxiter=maxiter,
                ncv=ncv,
                v0=v0,
                rng=rng,
                options=options,
                singular_value_tolerance=singular_value_tolerance,
                use_buffers=use_buffers,
            )
            self.numerical = True

        self.U = self.svd_data['U']
        self.V = self.svd_data['V']
        self.M = self.svd_data['M']
        self.singular_values = _real_nonnegative_singular_values(
            self.M.factor,
        ).reshape(-1)
        self.singular_values.flags.writeable = False

    def _init_from_labelled(
        self,
        source: Composition,
        *,
        use_buffers: bool,
    ) -> None:
        data = source.svd_data
        singular_values = _real_nonnegative_singular_values(
            data['M'].factor,
            name="labelled SVD multiplier",
        )
        same_domain_hilbert = getattr(source, 'h_domain', None) is self.h_domain
        same_codomain_hilbert = (
            getattr(source, 'h_codomain', None) is self.h_codomain
        )
        # Some Hilbert implementations construct a fresh inverse Operator on
        # every ``gram_inv`` access. Reusing an SVD produced from the very same
        # Hilbert-space object is nevertheless exact and should not fail an
        # object-identity comparison.
        G_X = data['G_X'] if same_domain_hilbert else self.h_domain.gram_inv
        G_Y = (
            data['G_Y']
            if same_codomain_hilbert or (self.as_eig and same_domain_hilbert)
            else self.h_codomain.gram
        )
        U_scale = (
            1. if same_domain_hilbert
            else _gram_rescaling(data['G_X'], G_X, 'domain')
        )
        V_scale = (
            1. if same_codomain_hilbert or (self.as_eig and same_domain_hilbert)
            else _gram_rescaling(data['G_Y'], G_Y, 'codomain')
        )

        middle_index = data['l'] - 1
        factors = list(source.ops[:middle_index])
        if not np.isclose(V_scale, 1.):
            factors.append(PtwMultiplication(data['M'].domain, V_scale))

        rescaled_multiplier = PtwMultiplication(
            data['M'].domain,
            singular_values / (U_scale * V_scale),
        )
        factors.append(rescaled_multiplier)
        new_l = len(factors)

        if not np.isclose(U_scale, 1.):
            factors.append(PtwMultiplication(data['M'].domain, U_scale))
        factors.extend(source.ops[middle_index + 1:])

        super().__init__(*factors, use_buffers=use_buffers)
        new_is_eig = bool(
            data['is_eig']
            and (
                _gram_operators_equal(G_X, G_Y)
                or (self.as_eig and same_domain_hilbert)
            )
            and np.isclose(U_scale, V_scale)
        )
        self.label_as_svd(
            new_l,
            G_X=G_X,
            G_Y=G_Y,
            is_eig=new_is_eig,
        )
        if new_is_eig:
            self.declare_as_self_adjoint(check=False)

    def _init_numerical(
        self,
        *,
        k: int | None,
        which: str,
        solver: str,
        tol: float,
        maxiter: int | None,
        ncv: int | None,
        v0: np.ndarray | None,
        rng: Any,
        options: dict | None,
        singular_value_tolerance: float | None,
        use_buffers: bool,
    ) -> None:
        domain_complex = self.original_operator.domain.is_complex
        codomain_complex = self.original_operator.codomain.is_complex
        if domain_complex != codomain_complex:
            raise ValueError(Errors.value_error(
                "A numerical SVD requires both endpoint vector spaces to be "
                "real or both to be complex."
            ))
        complex_svd = bool(domain_complex and codomain_complex)

        if complex_svd:
            for name, operator in (
                ('forward operator', self.original_operator),
                ('domain Gram operator', self.h_domain.gram),
                ('inverse domain Gram operator', self.h_domain.gram_inv),
                ('codomain Gram operator', self.h_codomain.gram),
            ):
                test_vector = operator.domain.randn()
                lhs = operator(1j * test_vector)
                rhs = 1j * operator(test_vector)
                error = operator.codomain.norm(lhs - rhs)
                scale = max(
                    1.,
                    operator.codomain.norm(lhs),
                    operator.codomain.norm(rhs),
                )
                if error > 5000 * np.finfo(float).eps * scale:
                    raise ValueError(Errors.value_error(
                        f"The {name} is not complex-linear and therefore "
                        "does not admit the requested complex SVD "
                        "representation."
                    ))

        L_X = _cholesky_factor(self.h_domain)
        if self.as_eig:
            transformed = Composition(
                L_X.inverse,
                self.original_operator,
                L_X.adjoint.inverse,
            )
        else:
            L_Y = _cholesky_factor(self.h_codomain)
            transformed = Composition(
                L_Y.adjoint,
                self.original_operator,
                L_X.adjoint.inverse,
            )
        scipy_operator = _as_scipy_operator(transformed)
        minimum_dimension = min(scipy_operator.shape)

        if singular_value_tolerance is not None:
            if singular_value_tolerance < 0:
                raise ValueError(Errors.value_error(
                    "singular_value_tolerance must be non-negative."
                ))

        if k is None:
            matrix = _assemble_matrix(transformed)
            if self.as_eig:
                eigenvalues, right_vectors = np.linalg.eigh(matrix)
                left_vectors = right_vectors
                singular_values = eigenvalues
                right_adjoint = right_vectors.conj().T
            else:
                left_vectors, singular_values, right_adjoint = np.linalg.svd(
                    matrix,
                    full_matrices=False,
                )
        else:
            if isinstance(k, (bool, np.bool_)) or not isinstance(
                    k, (int, np.integer)):
                raise TypeError(Errors.type_error(
                    "k must be a positive integer or None."
                ))
            k = int(k)
            maximum_k = (
                minimum_dimension
                if solver == 'propack' and not self.as_eig
                else minimum_dimension - 1
            )
            if not 1 <= k <= maximum_k:
                raise ValueError(Errors.value_error(
                    f"For solver={solver!r}, k must lie in [1, {maximum_k}]. "
                    f"Got k={k}."
                ))
            if self.as_eig:
                singular_values, right_vectors = eigsh(
                    scipy_operator,
                    k=k,
                    which=which,
                    tol=tol,
                    maxiter=maxiter,
                    ncv=ncv,
                    v0=v0,
                )
                left_vectors = right_vectors
                right_adjoint = right_vectors.conj().T
            else:
                svds_kwargs = {
                    'k': k,
                    'which': which,
                    'solver': solver,
                    'tol': tol,
                    'maxiter': maxiter,
                    'ncv': ncv,
                    'v0': v0,
                    'options': options,
                }
                if rng is not None:
                    svds_kwargs['rng'] = rng
                try:
                    left_vectors, singular_values, right_adjoint = svds(
                        scipy_operator,
                        **svds_kwargs,
                    )
                except TypeError as error:
                    # SciPy versions predating the SPEC-007 transition expose
                    # ``random_state`` instead of ``rng``.
                    if 'rng' not in svds_kwargs:
                        raise
                    svds_kwargs['random_state'] = svds_kwargs.pop('rng')
                    try:
                        left_vectors, singular_values, right_adjoint = svds(
                            scipy_operator,
                            **svds_kwargs,
                        )
                    except TypeError:
                        raise error

        if self.as_eig and singular_values.size:
            negative_tolerance = (
                max(scipy_operator.shape)
                * np.finfo(np.asarray(singular_values).dtype).eps
                * max(1., float(np.max(np.abs(singular_values))))
            )
            if np.min(singular_values) < -negative_tolerance:
                raise ValueError(Errors.value_error(
                    "Spectral self-adjoint regularization requires a "
                    "positive-semidefinite operator."
                ))

        order = np.argsort(singular_values)[::-1]
        singular_values = np.asarray(singular_values)[order]
        left_vectors = np.asarray(left_vectors)[:, order]
        right_adjoint = np.asarray(right_adjoint)[order, :]

        if singular_value_tolerance is None:
            largest = (
                float(singular_values[0])
                if singular_values.size else 0.
            )
            singular_value_tolerance = (
                max(scipy_operator.shape)
                * np.finfo(singular_values.dtype).eps
                * largest
            )
        positive = singular_values > singular_value_tolerance
        singular_values = singular_values[positive]
        left_vectors = left_vectors[:, positive]
        right_adjoint = right_adjoint[positive, :]

        reduced_space = vecsps.NumPyVectorSpace(
            int(singular_values.size),
            dtype=(complex if complex_svd else float),
        )
        V_0 = MatrixMultiplication(
            left_vectors,
            domain=reduced_space,
            codomain=transformed.codomain,
        )
        U_0 = MatrixMultiplication(
            right_adjoint.conj().T,
            domain=reduced_space,
            codomain=transformed.domain,
        )
        U = Composition(L_X, U_0)
        V = U if self.as_eig else Composition(L_Y.adjoint.inverse, V_0)
        M = PtwMultiplication(reduced_space, singular_values)

        super().__init__(V, M, U.adjoint, use_buffers=use_buffers)
        middle_index = self.ops.index(M) + 1
        self.label_as_svd(
            middle_index,
            G_X=self.h_domain.gram_inv,
            G_Y=self.h_codomain.gram,
            is_eig=self.as_eig,
        )
        if self.as_eig:
            self.declare_as_self_adjoint(check=False)

def _iteration_count(k: int) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(
            k, (int, np.integer)):
        raise TypeError(Errors.type_error("k must be a non-negative integer."))
    if k < 0:
        raise ValueError(Errors.value_error("k must be non-negative."))
    return int(k)


def _q_tikhonov(lam: float, alpha: float, **kwargs: Any) -> float:
    return 1. / (lam + alpha)


def _q_iterated_tikhonov(
    lam: float,
    alpha: float,
    *,
    k: int = 1,
    **kwargs: Any,
) -> float:
    k = _iteration_count(k)
    if k == 0:
        return 0.
    return (1. - (alpha / (lam + alpha)) ** k) / lam


def _q_showalter(lam: float, alpha: float, **kwargs: Any) -> float:
    return -np.expm1(-lam / alpha) / lam


def _q_landweber(
    lam: float,
    alpha: float,
    *,
    k: int = 1,
    mu: float = 1.,
    **kwargs: Any,
) -> float:
    k = _iteration_count(k)
    if k == 0:
        return 0.
    if isinstance(mu, (bool, np.bool_)) or not isinstance(mu, Real):
        raise TypeError(Errors.type_error("mu must be a positive real number."))
    if mu <= 0:
        raise ValueError(Errors.value_error("mu must be positive."))
    return (1. - (1. - float(mu) * lam) ** k) / lam


def _q_lardy(
    lam: float,
    alpha: float,
    *,
    k: int = 1,
    **kwargs: Any,
) -> float:
    k = _iteration_count(k)
    if k == 0:
        return 0.
    # Alpha is fixed; the iteration count k indexes the regularization path.
    base_alpha = alpha
    return (1. - (base_alpha / (lam + base_alpha)) ** k) / lam


def _q_spectral_cutoff(lam: float, alpha: float, **kwargs: Any) -> float:
    return 1. / lam if lam >= alpha else 0.


def _q_seidman(lam: float, alpha: float, **kwargs: Any) -> float:
    return min(1. / lam, 1. / np.sqrt(lam * alpha))


def _q_truncated_svd_2(lam: float, alpha: float, **kwargs: Any) -> float:
    return min(1. / lam, 1. / alpha)


def _filter_entry(
    q: Callable[..., float],
    C_r: float,
    C_q: float | str,
) -> dict[str, Any]:
    """Construct one filter entry and derive ``r`` from ``q``."""
    return {
        'q': q,
        'r': lambda lam, alpha, **parameters: (
            1. - lam * q(lam, alpha, **parameters)
        ),
        'C_r': C_r,
        'C_q': C_q,
    }


_FILTERS = {
    'Tikhonov': _filter_entry(_q_tikhonov, 1., .5),
    'Iterated_Tikhonov': _filter_entry(
        _q_iterated_tikhonov, 1., '<= sqrt(k)'
    ),
    'Showalter': _filter_entry(_q_showalter, 1., 1.),
    'Landweber': _filter_entry(_q_landweber, 1., 1.),
    'Lardy': _filter_entry(_q_lardy, 1., 1.),
    'Spectral_CutOff': _filter_entry(_q_spectral_cutoff, 1., 1.),
    'Seidman': _filter_entry(_q_seidman, 1., 1.),
    'Truncated_SVD_2': _filter_entry(_q_truncated_svd_2, 1., 1.),
}
# Common spellings are aliases, not separate filter definitions.
_FILTERS['TSVD'] = _FILTERS['Spectral_CutOff']
_FILTERS['Truncated_SVD'] = _FILTERS['Spectral_CutOff']
_FILTERS['Modified_TSVD'] = _FILTERS['Truncated_SVD_2']


def _is_labelled_svd(op: Operator, *, require_eig: bool = False) -> bool:
    return bool(
        isinstance(op, Composition)
        and op.is_svd
        and (not require_eig or op.svd_data['is_eig'])
    )


class _SpectralFilterMixin:
    """Shared filter resolution and coefficient validation."""

    filters = _FILTERS

    def _init_parameters(self, filter, regpar, filter_parameters):
        self._filter_spec = filter
        self._parameters = dict(filter_parameters or {})
        self._count_parameter = isinstance(filter, str) and filter in ('Landweber', 'Lardy')
        if self._count_parameter:
            source = self._parameters.pop('k', None)
            source = (2 ** Naturals()) if source is None else source
            self._fixed_alpha = 1. if regpar is None else regpar
        else:
            source = (1.0 * (0.5) ** Naturals()) if regpar is None else regpar
        self.regpars = iter((source,)) if isinstance(source, Real) else iter(source)
        try:
            self._pending_parameter = next(self.regpars)
        except StopIteration as error:
            raise ValueError("The regularization-parameter sequence must not be empty.") from error
        self._activate_parameter()

    def _activate_parameter(self):
        value = self._pending_parameter
        parameters = dict(self._parameters)
        if self._count_parameter:
            self.k = _iteration_count(value)
            parameters['k'] = self.k
            self._set_filter(self._filter_spec, self._fixed_alpha, parameters)
            self.regpar = self.k
        else:
            self._set_filter(self._filter_spec, value, parameters)

    def _advance_parameters(self):
        # Look ahead only after producing a solution, so the last finite
        # parameter terminates without emitting a duplicate iterate.
        try:
            self._pending_parameter = next(self.regpars)
        except StopIteration:
            self.converge()

    def _set_filter(
        self,
        filter: str | Callable[[float], float],
        regpar: float,
        filter_parameters: dict[str, Any] | None,
    ) -> None:
        if isinstance(regpar, (bool, np.bool_)) or not isinstance(regpar, Real):
            raise TypeError(Errors.type_error(
                "regpar must be a positive real number."
            ))
        if not np.isfinite(regpar) or regpar <= 0:
            raise ValueError(Errors.value_error("regpar must be positive."))
        self.regpar = float(regpar)
        self._filter_alpha = float(regpar)
        parameters = (
            {} if filter_parameters is None else dict(filter_parameters)
        )
        self.filter_parameters = parameters

        if isinstance(filter, str):
            if filter not in self.filters:
                raise ValueError(Errors.value_error(
                    f"Unknown spectral filter {filter!r}. Known filters are "
                    f"{tuple(self.filters)}."
                ))
            self.filter_name = filter
            filter_function = self.filters[filter]['q']
            self.q = lambda lam: filter_function(
                lam, self._filter_alpha, **self.filter_parameters
            )
        elif callable(filter):
            if parameters:
                raise ValueError(Errors.value_error(
                    "filter_parameters are only used with named filters; "
                    "close custom parameters over the callable instead."
                ))
            self.filter_name = getattr(filter, '__name__', '<callable>')
            if isinstance(filter, np.ufunc):
                if filter.nin == 1:
                    self.q = filter
                elif filter.nin == 2:
                    self.q = lambda lam: filter(lam, self._filter_alpha)
                else:
                    raise TypeError(Errors.type_error(
                        "A custom q-ufunc must have one or two inputs."
                    ))
            else:
                try:
                    signature = inspect.signature(filter)
                except (TypeError, ValueError):
                    signature = None
            if not isinstance(filter, np.ufunc) and signature is not None:
                try:
                    signature.bind(1., self.regpar)
                except TypeError:
                    try:
                        signature.bind(1.)
                    except TypeError as error:
                        raise TypeError(Errors.type_error(
                            "A custom q-function must accept either lambda "
                            "or (lambda, regpar)."
                        )) from error
                    self.q = filter
                else:
                    self.q = lambda lam: filter(lam, self._filter_alpha)
            elif not isinstance(filter, np.ufunc):
                # Some extension callables expose no inspectable signature.
                # Treat them as closed one-argument q-functions.
                self.q = filter
        else:
            raise TypeError(Errors.type_error(
                "filter must be a registered filter name or a callable q."
            ))

        # Residual functions are never independent input: r(t)=1-t*q(t).
        self.r = lambda lam: 1. - lam * self.q(lam)

    def _filter_coefficients(
        self,
        spectral_values: np.ndarray,
        prefactor: Callable[[float], float],
        q_argument: Callable[[float], float] = lambda value: value,
    ) -> np.ndarray:
        values = []
        for value in np.asarray(spectral_values).reshape(-1):
            coefficient = np.real_if_close(
                prefactor(float(value)) * self.q(q_argument(float(value)))
            )
            if np.ndim(coefficient) != 0:
                raise TypeError(Errors.type_error(
                    "A spectral q-function must return a scalar for a "
                    "scalar argument."
                ))
            coefficient = coefficient.item()
            if isinstance(coefficient, complex) or not np.isfinite(coefficient):
                raise ValueError(Errors.value_error(
                    "The spectral q-function returned a non-real or "
                    f"non-finite value at {value!r}."
                ))
            values.append(float(coefficient))
        return np.asarray(values)


class Spectral(_SpectralFilterMixin, RegSolver):
    r"""Spectral regularization of a linear Hilbert-space equation.

    For :math:`T=VMU^*`, :math:`\lambda=M^2`, and a filter function
    :math:`q_\alpha`, the regularized inverse is

    .. math::
        R_\alpha=G_X^{-1}U\,M q_\alpha(M^2)V^*G_Y.

    Starting at ``init``, each solver step returns
    :math:`x_0+R_\alpha(data-Tx_0)`. The residual function is always derived
    from the filter by :math:`r_\alpha(t)=1-tq_\alpha(t)`.

    For a codomain-free setting, ``data`` may be arbitrary observational data
    accepted by :meth:`Operator.adjoint_data`. The solver forms
    :math:`T^*data`, delegates the computation to
    :class:`Spectral_SelfAdjoint` with ``op.adjoint_eval`` as :math:`T^*T`,
    and never allocates a vector in the original codomain.

    ``filter`` may be a key in :attr:`filters` or a callable accepting either
    one scalar spectral value or ``(spectral_value, regpar)``. Further custom
    parameters should be closed over the callable.
    Named filters are ``Tikhonov``, ``Iterated_Tikhonov``, ``Showalter``,
    ``Landweber``, ``Lardy``, ``Spectral_CutOff`` (truncated SVD), ``Seidman``,
    and ``Truncated_SVD_2``. For alpha-based filters, ``regpar`` accepts
    a positive scalar (one solution), an iterable, or ``None`` (a fresh
    ``0.5 ** Naturals()``). Iterated Tikhonov keeps ``k`` fixed.
    Landweber and Lardy instead accept an integer or iterable through ``k``
    in ``filter_parameters``; the default is ``2 ** Naturals()``. Their
    ``regpar`` is a fixed alpha (default 1). Lardy uses this fixed alpha in
    every iteration; Landweber uses a fixed ``mu``, defaulting to ``1``.
    Choose ``mu`` within the usual Landweber stability range for the operator.
    ``self.regpar`` is the alpha or k of the current solution. Finite
    sequences terminate after their final solution. The SVD is computed once.

    The registered filters are

    .. math::
        q_\alpha^{\rm Tik}(\lambda)
            &= (\lambda+\alpha)^{-1},\\
        q_\alpha^{\rm itTik}(\lambda)
            &= \frac{(\lambda+\alpha)^k-\alpha^k}
                    {\lambda(\lambda+\alpha)^k},\\
        q_\alpha^{\rm Show}(\lambda)
            &= \frac{1-e^{-\lambda/\alpha}}{\lambda},\\
        q_{1/(\mu k)}^{\rm LW}(\lambda)
            &= \mu\sum_{j=0}^{k-1}(1-\mu\lambda)^j,\\
        q_{\alpha/k}^{\rm Lardy}(\lambda)
            &= \frac{(\lambda+\alpha)^k-\alpha^k}
                    {\lambda(\lambda+\alpha)^k}.

    The remaining filters use respectively
    :math:`\lambda^{-1}\mathbf 1_{[\alpha,\infty)}`,
    :math:`\min(\lambda^{-1},(\lambda\alpha)^{-1/2})`, and
    :math:`\min(\lambda^{-1},\alpha^{-1})`.
    """

    def __init__(
        self,
        setting: Setting,
        data: ReadOnlyVector | ObsDataT | None = None,
        init: ReadOnlyVector | None = None,
        logging_level: int | str = "INFO",
        filter: str | Callable[[float], float] = 'Spectral_CutOff',
        regpar: float | Iterable[float] | None = None,
        filter_parameters: dict[str, Any] | None = None,
        svd_options: dict[str, Any] | None = None,
    ) -> None:
        out, par = self.check_applicability(
            setting,
            user_data=data,
            user_init=init,
        )
        if not out['applicable']:
            raise ValueError(Errors.value_error(
                'Spectral methods are not applicable to this setting. '
                + out['info']
            ))
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.data = par['data']
        self.init = par['init']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Spectral regularization requires data."
            ))
        if not setting.codomain_free and self.data not in self.op.codomain:
            raise TypeError(Errors.not_in_vecsp(
                self.data, self.op.codomain, vec_name='data',
                space_name='codomain'
            ))

        if setting.codomain_free:
            self.backprop_data = self.op.domain.empty()
            self.op.adjoint_data(self.data, out=self.backprop_data)
            self._normal_setting = Setting(
                self.op.adjoint_eval,
                self.penalty,
                self.penalty,
                codomain_free=True,
                explicitDataWarning=False,
                logging_level=logging_level,
            )
            self._self_adjoint_method = Spectral_SelfAdjoint(
                self._normal_setting,
                data=self.backprop_data,
                init=self.init,
                logging_level=logging_level,
                filter=filter,
                regpar=regpar,
                filter_parameters=filter_parameters,
                svd_options=svd_options,
            )
            self.SVD = self._self_adjoint_method.SVD
            self.spec_domain = self._self_adjoint_method.spec_domain
            self.filtered_sv_op = self._self_adjoint_method.filtered_sv_op
            self.x = self._self_adjoint_method.x
            self.y = None
            self.regpar = self._self_adjoint_method.regpar
            self.regpars = self._self_adjoint_method.regpars
            return

        self._init_parameters(filter, regpar, filter_parameters)
        self.SVD = SVD(setting, **({} if svd_options is None else svd_options))
        self.spec_domain = self.SVD.U.domain
        self._update_inverse()

        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)

    def _update_inverse(self):
        coefficients = self._filter_coefficients(
            self.SVD.singular_values,
            lambda sigma: sigma,
            lambda sigma: sigma ** 2,
        )
        self.filtered_sv_op = PtwMultiplication(
            self.spec_domain, coefficients
        )
        self._regularized_inverse = (
            self.h_domain.gram_inv
            * self.SVD.U
            * self.filtered_sv_op
            * self.SVD.V.adjoint
            * self.h_codomain.gram
        )


    @staticmethod
    def check_applicability(
        setting: Setting,
        user_data: ReadOnlyVector | ObsDataT | None = None,
        user_init: ReadOnlyVector | None = None,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        out, par = check_Hilbert_applicability_linear(
            setting,
            user_data=user_data,
            user_init=user_init,
            **kwargs,
        )
        data = par.get('data')
        if (
            not setting.codomain_free
            and data is not None
            and data not in setting.op.codomain
        ):
            out['info'] += 'Data do not belong to the operator codomain. '
        out['applicable'] = out['info'] == ''
        decomposition_op = (
            setting.op.adjoint_eval
            if setting.codomain_free else setting.op
        )
        if out['applicable'] and not _is_labelled_svd(
            decomposition_op,
            require_eig=setting.codomain_free,
        ):
            out['caveats'] = 'Requires a numerical SVD.'
        return out, par

    def _next(self) -> None:
        if self.setting.codomain_free:
            self._self_adjoint_method.next()
            self.x = self._self_adjoint_method.x
            self.y = None
            self.filtered_sv_op = self._self_adjoint_method.filtered_sv_op
            self.regpar = self._self_adjoint_method.regpar
            if self._self_adjoint_method.is_converged():
                self.converge()
            return
        self._activate_parameter()
        self._update_inverse()
        residual = self.op.codomain.empty()
        self.op(self.init, out=residual)
        residual *= -1
        residual += self.data
        correction = self._regularized_inverse(residual)
        self.op.domain.copyto(self.x, self.init)
        self.x += correction
        self.op(self.x, out=self.y)
        self._advance_parameters()

    def regularized_inverse(self) -> Operator:
        """Return the regularized inverse.

        For a codomain-free setting this is the inverse filter for the normal
        operator, mapping the already backpropagated data to the solution.
        """
        if self.setting.codomain_free:
            return self._self_adjoint_method.regularized_inverse()
        return self._regularized_inverse

    def residual_op(self) -> Operator:
        """Return :math:`I-R_\alpha T`, the estimator's bias operator."""
        if self.setting.codomain_free:
            return self._self_adjoint_method.residual_op()
        return self.op.domain.identity - self._regularized_inverse * self.op


class Spectral_SelfAdjoint(_SpectralFilterMixin, RegSolver):
    r"""Spectral regularization for a self-adjoint equation :math:`Ax=b`.

    If :math:`A=U\operatorname{diag}(\lambda)U^*` and
    :math:`U^*G_X^{-1}U=I`, this solver uses

    .. math::
        R_\alpha=Uq_\alpha(M)U^*G_X^{-1},\qquad
        x=x_0+R_\alpha(b-Ax_0).

    The decomposition is reused when the operator carries an eigenvalue/SVD
    label. Otherwise a numerical positive-semidefinite eigenvalue
    decomposition is computed. Codomain-free settings are supported because
    the codomain Hilbert structure is canonically interpreted as :math:`X^*`.
    """

    def __init__(
        self,
        setting: Setting,
        data: ReadOnlyVector | None = None,
        init: ReadOnlyVector | None = None,
        logging_level: int | str = "INFO",
        filter: str | Callable[[float], float] = 'Tikhonov',
        regpar: float | Iterable[float] | None = None,
        filter_parameters: dict[str, Any] | None = None,
        svd_options: dict[str, Any] | None = None,
    ) -> None:
        out, par = self.check_applicability(
            setting,
            user_data=data,
            user_init=init,
        )
        if not out['applicable']:
            raise ValueError(Errors.value_error(
                'Self-adjoint spectral methods are not applicable to this '
                'setting. ' + out['info']
            ))
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.data = par['data']
        self.init = par['init']
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Self-adjoint spectral regularization requires data."
            ))
        if self.data not in self.op.codomain:
            raise TypeError(Errors.not_in_vecsp(
                self.data, self.op.codomain, vec_name='data',
                space_name='codomain'
            ))

        self._init_parameters(filter, regpar, filter_parameters)
        decomposition_options = (
            {} if svd_options is None else dict(svd_options)
        )
        if 'as_eig' in decomposition_options:
            raise ValueError(Errors.value_error(
                "svd_options must not override as_eig for a self-adjoint "
                "spectral method."
            ))
        self.SVD = SVD(setting, as_eig=True, **decomposition_options)
        self.spec_domain = self.SVD.U.domain
        self._update_inverse()

        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        self.y = self.op.codomain.empty()
        self.op(self.x, out=self.y)

    def _update_inverse(self):
        coefficients = self._filter_coefficients(
            self.SVD.singular_values,
            lambda eigenvalue: 1.,
        )
        self.filtered_sv_op = PtwMultiplication(
            self.spec_domain, coefficients
        )
        self._regularized_inverse = (
            self.h_domain.gram_inv
            * self.SVD.U
            * self.filtered_sv_op
            * self.SVD.U.adjoint
            * self.h_domain.gram_inv
        )


    @staticmethod
    def check_applicability(
        setting: Setting,
        user_data: ReadOnlyVector | None = None,
        user_init: ReadOnlyVector | None = None,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        out, par = check_Hilbert_applicability_linear(
            setting,
            user_data=user_data,
            user_init=user_init,
            **kwargs,
        )
        if not (
            setting.op.self_adjoint
            or _is_labelled_svd(setting.op, require_eig=True)
        ):
            out['info'] += 'Operator is not self-adjoint. '
        data = par.get('data')
        if data is not None and data not in setting.op.codomain:
            out['info'] += 'Data do not belong to the operator codomain. '
        out['applicable'] = out['info'] == ''
        if out['applicable'] and not _is_labelled_svd(
                setting.op, require_eig=True):
            out['caveats'] = 'Requires a numerical SVD.'
        return out, par

    def _next(self) -> None:
        self._activate_parameter()
        self._update_inverse()
        residual = self.op.codomain.empty()
        self.op(self.init, out=residual)
        residual *= -1
        residual += self.data
        correction = self._regularized_inverse(residual)
        self.op.domain.copyto(self.x, self.init)
        self.x += correction
        self.op(self.x, out=self.y)
        self._advance_parameters()

    def regularized_inverse(self) -> Operator:
        """Return :math:`Uq_\alpha(M)U^*G_X^{-1}`."""
        return self._regularized_inverse

    def residual_op(self) -> Operator:
        """Return :math:`I-R_\alpha A`."""
        return self.op.domain.identity - self._regularized_inverse * self.op


class Spectral_Tikhonov(Spectral):
    """Spectral Tikhonov regularization."""

    def __init__(self, setting: Setting, regpar: float | Iterable[float] | None = None, *, svd_options=None):
        super().__init__(
            setting, filter='Tikhonov', regpar=regpar,
            svd_options=svd_options
        )


class Spectral_IteratedTikhonov(Spectral):
    """Spectral iterated Tikhonov regularization."""

    def __init__(self, 
        setting: Setting, 
        regpar: float | Iterable[float] | None = None, 
        k: int = 1, *,
        svd_options=None
    ):
        super().__init__(
            setting, filter='Iterated_Tikhonov', regpar=regpar,
            filter_parameters={'k': k}, svd_options=svd_options
        )


class Spectral_Showalter(Spectral):
    """Showalter's spectral method."""

    def __init__(self, 
                 setting: Setting, 
                 regpar: float | Iterable[float] | None = None, 
                 *, 
                 svd_options=None):
        super().__init__(
            setting, filter='Showalter', regpar=regpar,
            svd_options=svd_options
        )


class Spectral_Landweber(Spectral):
    """Landweber filter with fixed stepsize and a sequence of counts ``k``.

    ``k=None`` uses 2 ** Naturals(), mimicing the standard sampling of regularization parameters alpha in other spectral methods; pass Naturals(first=1) to mimic the behavior of the non-spectral Landweber iterator
    Each scalar k produces one solution. 

    """

    def __init__(self, 
                 setting: Setting, 
                 k: int | Iterable[int] | None = None,
                stepsize: float | None = None, *, 
                svd_options=None
    ):
        super().__init__(
            setting, filter='Landweber', 
            filter_parameters={'k': k, 'mu': stepsize},
            svd_options=svd_options
        )


class Spectral_Lardy(Spectral):
    """Lardy filter with fixed alpha ``regpar`` (default 1) and varying ``k``.

    ``k=None`` uses 2 ** Naturals(); pass Naturals(first=1) for every count.
    A scalar k produces one solution.
    """

    def __init__(self, 
                 setting: Setting, 
                 regpar: float | Iterable[float] | None = None, 
                 k: int | Iterable[int] | None = None, 
                 *,
                svd_options=None
    ):
        super().__init__(
            setting, filter='Lardy', regpar=regpar,
            filter_parameters={'k': k}, svd_options=svd_options
        )


class Spectral_CutOff(Spectral):
    """Truncated singular-value decomposition (spectral cut-off)."""

    def __init__(self, setting: Setting, regpar: float | Iterable[float] | None = None, *, svd_options=None):
        super().__init__(
            setting, filter='Spectral_CutOff', regpar=regpar,
            svd_options=svd_options
        )


class Spectral_Seidman(Spectral):
    """Seidman's optimal spectral filtering."""

    def __init__(self, 
                 setting: Setting, 
                 regpar: float | Iterable[float] | None = None, 
                 *, svd_options=None):
        super().__init__(
            setting, filter='Seidman', regpar=regpar,
            svd_options=svd_options
        )


class Spectral_TruncatedSVD2(Spectral):
    """Continuous truncated-SVD filter ``min(1/lambda, 1/alpha)``."""

    def __init__(self, 
                 setting: Setting, 
                 regpar: float | Iterable[float] | None = None, 
                 *, svd_options=None):
        super().__init__(
            setting, filter='Truncated_SVD_2', regpar=regpar,
            svd_options=svd_options
        )


class SpectralSelfAdjoint_Lavrentiev(Spectral_SelfAdjoint):
    """Lavrentiev regularization of a positive self-adjoint operator."""

    def __init__(self, 
                 setting: Setting, 
                 regpar: float | Iterable[float] | None = None, 
                 *, svd_options=None):
        super().__init__(
            setting, filter='Tikhonov', regpar=regpar,
            svd_options=svd_options
        )


class SpectralSelfAdjoint_Richardson(Spectral_SelfAdjoint):
    """Richardson iteration represented by its spectral filter."""

    def __init__(self, 
                 setting: Setting, 
                 regpar: float | Iterable[float] | None = None, 
                 k: int | Iterable[int] | None = None,
                stepsize: float | None = None, *, svd_options=None
    ):
        super().__init__(
            setting, filter='Landweber', regpar=regpar,
            filter_parameters={'k': k, 'mu': stepsize},
            svd_options=svd_options
        )
