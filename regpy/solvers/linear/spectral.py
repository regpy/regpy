from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from scipy.linalg import cholesky
from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator
from scipy.sparse.linalg import svds

from regpy import vecsps
from regpy.vecsps import Vector
from regpy.operators.base import (
    Composition,
    Identity,
    LinearCombination,
    Operator,
    PtwMultiplication,
)
from regpy.operators.numpy import MatrixMultiplication
from regpy.util import Errors

from ..solver import RegSolver
from ..setting import Setting, check_Hilbert_applicability_linear


__all__ = ["SVD", "Spectral"]


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
        V^*G_YV=I,\quad U^*G_XU=I.

    If ``setting.op`` is already a labelled :class:`Composition`, its factors
    are reused. Scalar identity Gram operators may differ; in that case ``U``,
    ``V``, and ``M`` are rescaled without recomputing an SVD. Non-scalar Gram
    operators must be identical objects to those stored in the label.

    Otherwise a numerical generalized SVD is computed. If :math:`G=LL^*`
    denotes the Cholesky convention used by RegPy, the standard SVD is applied
    to

    .. math::
        B=L_Y^*T L_X.

    Its Euclidean singular-vector operators :math:`U_0,V_0` are transformed
    back by :math:`U=L_X^{-*}U_0` and
    :math:`V=L_Y^{-*}V_0`.

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
        Hilbert setting with a linear, non-codomain-free operator.
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
        Retained singular values in descending order.
    numerical : bool
        Whether a numerical SVD was needed.
    """

    def __init__(
        self,
        setting: Setting,
        *,
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
        if setting.codomain_free:
            raise ValueError(Errors.value_error(
                "SVD requires an explicit codomain Hilbert space."
            ))

        self.original_operator = setting.op
        self.h_domain = setting.h_domain
        self.h_codomain = setting.h_codomain

        if (
            isinstance(setting.op, Composition)
            and setting.op.is_svd
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
        self.singular_values = np.array(self.M.factor, copy=True).reshape(-1)
        self.singular_values.flags.writeable = False

    def _init_from_labelled(
        self,
        source: Composition,
        *,
        use_buffers: bool,
    ) -> None:
        data = source.svd_data
        G_X = self.h_domain.gram
        G_Y = self.h_codomain.gram
        U_scale = _gram_rescaling(data['G_X'], G_X, 'domain')
        V_scale = _gram_rescaling(data['G_Y'], G_Y, 'codomain')

        middle_index = data['l'] - 1
        factors = list(source.ops[:middle_index])
        if not np.isclose(V_scale, 1.):
            factors.append(PtwMultiplication(data['M'].domain, V_scale))

        rescaled_multiplier = PtwMultiplication(
            data['M'].domain,
            np.asarray(data['M'].factor) / (U_scale * V_scale),
        )
        factors.append(rescaled_multiplier)
        new_l = len(factors)

        if not np.isclose(U_scale, 1.):
            factors.append(PtwMultiplication(data['M'].domain, U_scale))
        factors.extend(source.ops[middle_index + 1:])

        super().__init__(*factors, use_buffers=use_buffers)
        new_is_eig = bool(
            data['is_eig']
            and _gram_operators_equal(G_X, G_Y)
            and np.isclose(U_scale, V_scale)
        )
        self.label_as_svd(
            new_l,
            G_X=G_X,
            G_Y=G_Y,
            is_eig=new_is_eig,
        )

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
        L_Y = _cholesky_factor(self.h_codomain)
        transformed = Composition(
            L_Y.adjoint,
            self.original_operator,
            L_X,
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
                if solver == 'propack'
                else minimum_dimension - 1
            )
            if not 1 <= k <= maximum_k:
                raise ValueError(Errors.value_error(
                    f"For solver={solver!r}, k must lie in [1, {maximum_k}]. "
                    f"Got k={k}."
                ))
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
        V = Composition(L_Y.adjoint.inverse, V_0)
        U = Composition(L_X.adjoint.inverse, U_0)
        M = PtwMultiplication(reduced_space, singular_values)

        super().__init__(V, M, U.adjoint, use_buffers=use_buffers)
        middle_index = self.ops.index(M) + 1
        self.label_as_svd(
            middle_index,
            G_X=self.h_domain.gram,
            G_Y=self.h_codomain.gram,
            is_eig=False,
        )

class Spectral(RegSolver):

    filters = {
                'Tikhonov': {'q': lambda lam,alpha : 1. / (lam + alpha),
                             'r': lambda lam,alpha : alpha / (lam + alpha)
                     },
                'Spectral_CutOff': {'q': lambda lam,alpha : (1./lam if lam>=alpha else 0.),
                                    'r': lambda lam,alpha : (1 if lam<alpha else 0.)
                                    }
              }

    def __init__(self,
                 setting: Setting,
                 data:Vector | None  = None,
                 init:Vector | None  = None,
                 logging_level:int | str = "INFO",
                 filter = 'Spectral_CutOff',
                 regpar = 1. 
            ) -> None:
        out, par = Spectral.check_applicability(
                    setting,
                    user_data=data,
                    user_init=init,
                )
        if not out['applicable']:
            raise ValueError('Spectral methods are not applicable to this setting. '+out['info'])        
        super().__init__(setting)
        self.log.setLevel(logging_level)
        
        self.data = par['data']
        self.init = par['init']

        self.SVD = SVD(setting)
        sv = self.SVD.singular_values
        self.spec_domain = self.SVD.U.domain 
        filtered_sv = self.spec_domain.empty()
        self.aux_spec = self.spec_domain.empty()
        self.q = Spectral.filters[filter]['q']
        self.spec_domain.set_coefficients(
            filtered_sv,
            (
                sigma * self.q(sigma**2, regpar) 
                for sigma in self.spec_domain.iter_coefficients(sv))
            )
        self.filtered_sv_op = PtwMultiplication(self.spec_domain,filtered_sv) 
        self.regpar = regpar
        self.x = self.op.domain.empty()
        self.y = self.op.codomain.empty()
        self.op.domain.copyto(self.x,self.init)
        setting.op(self.x, out=self.y)

    check_applicability = staticmethod(check_Hilbert_applicability_linear)

    def _next(self):
        """
        This method is included to allow to use spectral solvers in the same way as 
        iterative methods. The solver is marked as converged after one iteration.
        """
        self.y *= -1
        self.y += self.data
        self.h_codomain.gram(self.y, out=self.y)
        self.SVD.V.adjoint(self.y, out = self.aux_spec)
        self.filtered_sv_op(self.aux_spec, out = self.aux_spec)
        self.SVD.U(self.aux_spec, out = self.x)
        self.h_domain.gram(self.x, out=self.x)
        self.x += self.init
        self.op(self.x, out=self.y)
        self.converge()

    def regularized_inverse(self) -> Operator:
        """ The regularized inverse of the forward operator T
        """
        return (self.h_domain.gram_inv 
                * self.SVD.U 
                * self.filtered_sv_op 
                * self.SVD.V.adjoint 
                * self.h_codomain.gram
        )

    def residual_op(self) -> Operator:
        """ The operator Identity - regularized_inverse * T.
        Applied to a vector in the domain of the forward operator T it yields  
        the bias of the estimator implemented by the method. 
        """
        rlambda = self.spec_domain.empty()
        self.spec_domain.set_coefficients(
                                self.aux_spec,
                                (
                                    self.q(lam,self.regpar)
                                    for lam in self.spec_domain.iter_coefficients(self.aux_spec)
                                ),
                            )        
        return (self.h_domain.gram_inv 
                * self.SVD.U
                * PtwMultiplication(self.spec_domain,rlambda)
                * self.SVD.U.adjoint
                )