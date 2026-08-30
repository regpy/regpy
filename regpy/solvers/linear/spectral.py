from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from scipy.linalg import cholesky
from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator
from scipy.sparse.linalg import svds

from regpy import vecsps
from regpy.operators.base import (
    Composition,
    Identity,
    LinearCombination,
    Operator,
    PtwMultiplication,
)
from regpy.util import Errors
from regpy.vecsps import Vector

from ..setting import Setting

__all__ = ["SVD"]


class _CoordinateSystem:
    """Coordinate conversion for a real or genuinely complex vector space."""

    def __init__(self, space, complex_coordinates: bool) -> None:
        self.space = space
        self.complex_coordinates = complex_coordinates
        if complex_coordinates:
            if not space.is_complex:
                raise ValueError("Complex coordinates require a complex space.")
            self.dimension = int(space.size)
            self.dtype = np.dtype(complex)
            self._numpy_space = isinstance(space, vecsps.NumPyVectorSpace)
            self._basis = None if self._numpy_space else self._complex_basis()
        else:
            self.dimension = int(space.realsize)
            self.dtype = np.dtype(float)
            self._numpy_space = False
            self._basis = None

    def _complex_basis(self) -> list[Vector]:
        iterator = iter(self.space)
        basis = []
        for _ in range(self.dimension):
            real_basis_vector = next(iterator)
            stored = self.space.empty()
            self.space.copyto(stored, real_basis_vector)
            basis.append(stored)
            # Complex vector-space iterators return i*e_j after every e_j.
            next(iterator)
        return basis

    def to_coordinates(self, vector: Vector) -> np.ndarray:
        if self.complex_coordinates:
            if self._numpy_space:
                return np.asarray(vector).reshape(-1).astype(
                    complex, copy=False
                )
            return np.asarray([
                self.space.vdot(basis_vector, vector)
                for basis_vector in self._basis
            ], dtype=complex)
        return np.asarray(self.space.flatten(vector), dtype=float)

    def from_coordinates(self, coefficients: np.ndarray) -> Vector:
        coefficients = np.asarray(coefficients, dtype=self.dtype)
        if coefficients.size != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} coordinates, got "
                f"shape={coefficients.shape}."
            )
        coefficients = coefficients.reshape(self.dimension)
        if self.complex_coordinates:
            if self._numpy_space:
                return coefficients.reshape(self.space.shape).astype(
                    self.space.dtype, copy=False
                )
            result = self.space.zeros()
            for coefficient, basis_vector in zip(
                    coefficients, self._basis):
                result += coefficient * basis_vector
            return result
        return self.space.fromflat(coefficients)


class _CoordinateLinearOperator(SciPyLinearOperator):
    """Matrix-free SciPy view using RegPy vector-space coordinates."""

    def __init__(self, op: Operator, complex_coordinates: bool) -> None:
        self.op = op
        self.domain_coordinates = _CoordinateSystem(
            op.domain, complex_coordinates
        )
        self.codomain_coordinates = _CoordinateSystem(
            op.codomain, complex_coordinates
        )
        super().__init__(
            dtype=(np.complex128 if complex_coordinates else np.float64),
            shape=(
                self.codomain_coordinates.dimension,
                self.domain_coordinates.dimension,
            ),
        )

    def _matvec(self, vector):
        argument = self.domain_coordinates.from_coordinates(vector)
        return self.codomain_coordinates.to_coordinates(self.op(argument))

    def _rmatvec(self, vector):
        argument = self.codomain_coordinates.from_coordinates(vector)
        return self.domain_coordinates.to_coordinates(
            self.op.adjoint(argument)
        )


class _MatrixOperator(Operator):
    """RegPy operator represented by a matrix in chosen coordinates."""

    def __init__(
        self,
        matrix: np.ndarray,
        domain,
        codomain,
        complex_coordinates: bool,
    ) -> None:
        self.domain_coordinates = _CoordinateSystem(
            domain, complex_coordinates
        )
        self.codomain_coordinates = _CoordinateSystem(
            codomain, complex_coordinates
        )
        dtype = complex if complex_coordinates else float
        matrix = np.asarray(matrix, dtype=dtype)
        expected_shape = (
            self.codomain_coordinates.dimension,
            self.domain_coordinates.dimension,
        )
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match "
                f"operator shape {expected_shape}."
            )
        self.matrix = np.array(matrix, copy=True)
        self.matrix.flags.writeable = False
        self._complex_coordinates = complex_coordinates
        self_adjoint = (
            domain == codomain
            and self.matrix.shape[0] == self.matrix.shape[1]
            and np.allclose(self.matrix, self.matrix.conj().T)
        )
        super().__init__(
            domain,
            codomain,
            linear=True,
            self_adjoint=self_adjoint,
        )

    def _eval(self, vector):
        coordinates = self.domain_coordinates.to_coordinates(vector)
        return self.codomain_coordinates.from_coordinates(
            self.matrix @ coordinates
        )

    def _adjoint(self, vector):
        coordinates = self.codomain_coordinates.to_coordinates(vector)
        return self.domain_coordinates.from_coordinates(
            self.matrix.conj().T @ coordinates
        )

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise NotImplementedError(
                "Only a square coordinate matrix can be inverted."
            )
        inverse = _MatrixOperator(
            np.linalg.inv(self.matrix),
            self.codomain,
            self.domain,
            self._complex_coordinates,
        )
        self._inverse = inverse
        inverse._inverse = self
        return inverse


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


def _assemble_matrix(
    op: Operator,
    complex_coordinates: bool,
) -> np.ndarray:
    linear_operator = _CoordinateLinearOperator(op, complex_coordinates)
    identity = np.eye(
        linear_operator.shape[1], dtype=linear_operator.dtype
    )
    return np.asarray(linear_operator.matmat(identity))


def _cholesky_factor(hilbert_space, complex_coordinates: bool) -> Operator:
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
        if factor.domain.is_complex != complex_coordinates:
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

    gram_matrix = _assemble_matrix(gram, complex_coordinates)
    lower_factor = cholesky(
        gram_matrix,
        lower=True,
        check_finite=True,
    )
    return _MatrixOperator(
        lower_factor,
        hilbert_space.vecsp,
        hilbert_space.vecsp,
        complex_coordinates,
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
        complex_coordinates = bool(domain_complex)

        if complex_coordinates:
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

        L_X = _cholesky_factor(self.h_domain, complex_coordinates)
        L_Y = _cholesky_factor(self.h_codomain, complex_coordinates)
        transformed = Composition(
            L_Y.adjoint,
            self.original_operator,
            L_X,
        )
        scipy_operator = _CoordinateLinearOperator(
            transformed, complex_coordinates
        )
        minimum_dimension = min(scipy_operator.shape)

        if singular_value_tolerance is not None:
            if singular_value_tolerance < 0:
                raise ValueError(Errors.value_error(
                    "singular_value_tolerance must be non-negative."
                ))

        if k is None:
            matrix = _assemble_matrix(transformed, complex_coordinates)
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
            dtype=(complex if complex_coordinates else float),
        )
        V_0 = _MatrixOperator(
            left_vectors,
            reduced_space,
            transformed.codomain,
            complex_coordinates,
        )
        U_0 = _MatrixOperator(
            right_adjoint.conj().T,
            reduced_space,
            transformed.domain,
            complex_coordinates,
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
