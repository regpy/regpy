Operators in RegPy
==================

This tutorial expands on the :doc:`usage guide <usage>` and explains how to
define custom operators using :class:`regpy.operators.Operator`.

In RegPy, an operator represents a possibly nonlinear mapping between vector
spaces :math:`\mathbb{X}` and :math:`\mathbb{Y}`:

.. math::
    F\colon \mathbb{X}\to\mathbb{Y}.

The spaces :math:`\mathbb{X}` and :math:`\mathbb{Y}` are instances of
:class:`regpy.vecsps.VectorSpaceBase`. Their vectors are duck typed and need
not be NumPy arrays; see :doc:`Spaces`. Complex coefficient spaces are treated
as real vector spaces, so a finite-dimensional domain and codomain can be
identified with :math:`\mathbb{R}^N` and :math:`\mathbb{R}^M`, respectively.

A linear operator :math:`T\colon \mathbb{X}\to\mathbb{Y}` can be represented
by a matrix :math:`\underline{T}\in\mathbb{R}^{M\times N}`. Forming this
matrix may be inefficient or impossible, whereas applying it to a vector is
often feasible. RegPy represents this action through ``_eval`` and the action
of the transpose through ``_adjoint``.

Equivalently, ``_adjoint`` represents the dual operator
:math:`T'\colon\mathbb{Y}'\to\mathbb{X}'` after identifying each space with
its dual through the standard real pairing:

.. math::
    \langle u,v\rangle = \operatorname{Re}(\operatorname{vdot}(u,v)).

Thus ``_eval`` and ``_adjoint`` must satisfy

.. code-block:: python

    T.codomain.vdot(T(x), y).real == T.domain.vdot(x, T.adjoint(y)).real

If :math:`T` is :math:`\mathbb{C}`-linear and represented by a matrix
:math:`\underline{T}\in\mathbb{C}^{M\times N}`, then ``T.adjoint`` is
represented by the conjugate transpose of
:math:`\underline{T}`. In that case, the corresponding complex identity also
holds before taking real parts.

Regularization methods often equip the spaces with additional Hilbert-space
structures. If their inner products are represented by the Gram matrices
:math:`G_{\mathbb{X}}` and :math:`G_{\mathbb{Y}}`, then the corresponding
Hilbert-space adjoint is

.. math::
    T^{\ast} = G_{\mathbb{X}}^{-1} T^{\top} G_{\mathbb{Y}}.

This decomposition motivates RegPy's design: an operator adjoint is defined
with respect to the standard pairings supplied by its vector spaces. Additional
Hilbert-space geometry is introduced separately through Gram operators. The
operator representation therefore remains independent of the inner products,
data-fidelity terms, and penalties used by a particular inverse problem.

Using existing operators
~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to construct an operator is often to combine existing classes
from :mod:`regpy.operators`. The module includes multiplication, Fourier
transform, convolution, direct-sum, and composition operators.

.. code-block:: python

    op_1 = Some_Operator(...)
    op_2 = Some_other_Operator(...)

    my_op = op_1 * op_2  # requires op_2.codomain == op_1.domain
    shifted_op = my_op + my_op.codomain.rand()

Common operator operations include:

* ``a * op1 + b * op2``: linear combination;
* ``op1 * op2``: composition;
* ``op * factor``: composition with pointwise multiplication in the domain;
* ``op + offset``: shift in the codomain.


.. _Linear_Operator:

Linear operators
~~~~~~~~~~~~~~~~

Every operator requires ``domain`` and ``codomain`` instances derived from
:class:`regpy.vecsps.VectorSpaceBase`. They may be supplied to the constructor
or constructed from its parameters. These objects describe vector membership
and storage operations; they do not prescribe a Hilbert-space inner product.

The initialization
------------------

For example, an operator between two uniform two-dimensional grids may accept
axis specifications ``(start, end, number)`` and construct
:class:`regpy.vecsps.UniformGridFcts` instances:

.. code-block:: python

    def __init__(self, d_1, d_2, cd_1, cd_2):
        domain = UniformGridFcts(d_1, d_2)
        codomain = UniformGridFcts(cd_1, cd_2)
        super().__init__(domain=domain, codomain=codomain, linear=True)

For a linear operator, you need to implement two methods:

* ``_eval`` computes the action of the forward operator.
* ``_adjoint`` computes the action of its adjoint.

Example
^^^^^^^

Consider a two-dimensional Fourier transform on a centered square grid. Let
``d = (-1, 1, 100)`` specify each domain axis. The dual grid spacing determines
the codomain:

.. code-block:: python

    def __init__(self, d):
        domain = UniformGridFcts(d, d, dtype=complex)
        cd = (-1 / (2 * domain.spacing[0]),
              1 / (2 * domain.spacing[0]),
              domain.shape[0])
        codomain = UniformGridFcts(cd, cd, dtype=complex)
        super().__init__(
            domain=domain,
            codomain=codomain,
            linear=True,
        )


The evaluation method
---------------------

For a linear operator, ``_eval`` receives a single vector ``x``. Users invoke
the public interface as ``op(x)``; :class:`Operator` checks that ``x`` belongs
to the domain and that the returned vector belongs to the codomain. An
implementation of ``_eval`` may therefore assume a valid input, but it must
return a freely modifiable codomain vector.

Example
^^^^^^^

For the two-dimensional Fourier transform, the NumPy implementation is:

.. code-block:: python

    def _eval(self, x):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))

The adjoint evaluation method
-----------------------------

The ``_adjoint`` method follows the same convention. Users call
``op.adjoint(y)``; the public interface validates ``y`` against the codomain
and validates the result against the domain. The adjoint is always taken with
respect to the standard real pairings of the two vector spaces.

Use :func:`regpy.util.operator_tests.test_adjoint` to test this identity
numerically.

Example
^^^^^^^

For the example above, the inverse Fourier transform defines the adjoint:

.. code-block:: python

    def _adjoint(self, y):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y), norm='ortho'))


Defining the class
------------------

A custom linear operator typically has the following structure:

.. code-block:: python

    from regpy.operators import Operator

    class MyOperator(Operator):
        def __init__(self, par_1, par_2, ...):
            # Compute the domain and codomain here if they are not parameters.
            super().__init__(
                domain=my_domain,
                codomain=my_codomain,
                linear=True,
            )

        def _eval(self, x):
            # Compute y = T(x).
            return y

        def _adjoint(self, y):
            # Compute x = T.adjoint(y) for the standard pairings.
            return x

See the :doc:`Volterra example <notebooks/volterra_main_example>` for a compact
application and the :doc:`traction-force microscopy example <notebooks/tfm>`
for an NGSolve-based operator.

Example
^^^^^^^

Combining the preceding snippets yields the following Fourier-transform
operator:

.. code-block:: python

    import numpy as np
    from regpy.operators import Operator
    from regpy.vecsps import UniformGridFcts

    class SimpleFFTOnSquare(Operator):
        def __init__(self, d):
            domain = UniformGridFcts(d, d, dtype=complex)
            # Dual grid for the FFT approximation of the continuous transform.
            cd = (-1 / (2 * domain.spacing[0]),
                  1 / (2 * domain.spacing[0]),
                  domain.shape[0])
            codomain = UniformGridFcts(cd, cd, dtype=complex)
            super().__init__(
                domain=domain,
                codomain=codomain,
                linear=True,
            )

        def _eval(self, x):
            return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))

        def _adjoint(self, y):
            return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y), norm='ortho'))

This is a simplified version of the Fourier-transform implementation in
:mod:`regpy.operators`.

.. _Non-linear_operators:

Nonlinear operators
~~~~~~~~~~~~~~~~~~~

Nonlinear operators also define a domain, a codomain, and an ``_eval`` method.
Regularization methods for inverse problems usually also need the Fréchet derivative
:math:`F'[x]` and its adjoint.

RegPy deliberately couples a derivative to the evaluation at its linearization
point. Calling ``op.linearize(x)`` evaluates :math:`F(x)`, permits ``_eval`` to
cache data required at :math:`x`, and returns both the value and a linear
operator representing :math:`F'[x]`. Evaluating the original operator again
revokes the previous derivative, because its cached data may no longer be
valid. If derivatives at several points must coexist, make independent copies
of the nonlinear operator before linearizing them.

The methods for evaluation, derivative and adjoint
--------------------------------------------------

Implement the following methods:

* ``_eval(x, differentiate=False)`` computes :math:`F(x)`. If
  ``differentiate`` is true, it must also prepare any state needed by the next
  two methods.
* ``_derivative(h)`` computes :math:`F'[x]h` at the most recent linearization
  point.
* ``_adjoint(y)`` computes :math:`F'[x]^*y` at that same point.

These private methods are called through ``op(x)``, ``op.linearize(x)``, and the
linear operator returned by ``linearize``; users should not invoke them
directly. In-place variants may be implemented instead, as described by the
:class:`regpy.operators.Operator` API.

.. _eval_nonlinear:

The _eval method
^^^^^^^^^^^^^^^^

The ``differentiate`` flag avoids derivative-specific precomputation during an
ordinary evaluation. A typical allocating implementation has the form:

.. code-block:: python

    def _eval(self, x, differentiate=False):
        y = compute_value(x)
        if differentiate:
            self._linearization_data = prepare_derivative_data(x, y)
        return y

Store only the data needed by ``_derivative`` and ``_adjoint``. If a stored
vector must remain independent of its input, copy it through the corresponding
vector space rather than assuming that the vector provides ``copy()``.

The _derivative and _adjoint method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After linearization, ``_derivative`` acts as the evaluation method of the
linear operator :math:`F'[x]`, while ``_adjoint`` acts as its adjoint method.
The returned derivative retains a revocable reference to the nonlinear
operator and becomes invalid after that operator is evaluated again.

.. _Linearization_Method:

What happens when you linearize
-------------------------------

When RegPy calls ``op.linearize(x)``, the following steps occur:

1. The operator is evaluated with ``differentiate=True``.
2. The operator stores any intermediate quantities needed to apply the
   derivative and its adjoint.
3. ``linearize`` returns the value and an :class:`regpy.operators.Operator`
   representing the derivative.

.. code-block:: python

    y, derivative = my_op.linearize(x)

For operators whose codomain vectors are too expensive to construct,
``linearize(x, return_adjoint_eval=True)`` may return :math:`F'[x]^*F(x)` in
place of :math:`F(x)`. The second return value is still the derivative.

Thus a typical implementation would look like this:

.. code-block:: python

    from regpy.operators import Operator

    class MyOperator(Operator):
        def __init__(self, par_1, par_2, ...):
            # Compute the domain and codomain here if they are not parameters.
            super().__init__(
                domain=my_domain,
                codomain=my_codomain,
                linear=False,
            )

        def _eval(self, x, differentiate=False):
            y = compute_value(x)
            if differentiate:
                self._linearization_data = prepare_derivative_data(x, y)
            return y

        def _derivative(self, h):
            return apply_derivative(self._linearization_data, h)

        def _adjoint(self, y):
            return apply_adjoint(self._linearization_data, y)


Example
-------

As an example, consider the phase-retrieval observation operator obtained by
composing a Fourier transform with the pointwise squared modulus,
:math:`x\mapsto |\mathcal{F}(x)|^2`. Its domain is a centered complex uniform
grid, and its codomain is the corresponding real Fourier grid.

.. code-block:: python

    def __init__(self, domain):
        # Dual grid for the FFT approximation of the continuous transform.
        cd = (-1 / (2 * domain.spacing[0]),
              1 / (2 * domain.spacing[0]),
              domain.shape[0])
        codomain = UniformGridFcts(cd, cd)
        super().__init__(
            domain=domain,
            codomain=codomain,
            linear=False,
        )

The derivative of :math:`Sq\colon x\mapsto |x|^2` at :math:`f` is
:math:`h\mapsto 2\operatorname{Re}(\overline{f}h)`. Since the Fourier
transform is linear, the chain rule gives

.. math::
    (Sq\circ\mathcal{F})'[f]h=Sq'[\mathcal{F}(f)]\mathcal{F}'[f]h = Sq'[\mathcal{F}(f)]\mathcal{F}(h).

Moreover, for the adjoint we obtain

.. math::
    (Sq\circ\mathcal{F})'[f]^\ast y= \mathcal{F}^\ast Sq'[\mathcal{F}(f)]^\ast y.


Both the derivative and its adjoint require :math:`\mathcal{F}(f)`. The
evaluation method therefore stores this value only when called with
``differentiate=True``:

.. code-block:: python

    def _eval(self, x, differentiate=False):
        y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))
        if differentiate:
            self._factor = y
        return y.real**2 + y.imag**2

The ``_derivative`` method applies this derivative to a direction :math:`h`.
Because :math:`\mathcal{F}(f)` is already cached, the implementation is:

.. code-block:: python

    def _derivative(self, h):
        return 2*(self._factor.conj() * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(h), norm='ortho'))).real

The adjoint maps :math:`y` to
:math:`\mathcal{F}^\ast(2\mathcal{F}(f)y)` and reuses the same cached factor:

.. code-block:: python

    def _adjoint(self, y):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(2*self._factor * y), norm='ortho'))

Together, these methods define the phase-retrieval observation operator:

.. code-block:: python

    import numpy as np
    from regpy.vecsps import UniformGridFcts
    from regpy.operators import Operator

    class Observation(Operator):

        def __init__(self, domain):
            # Dual grid for the FFT approximation of the continuous transform.
            cd = (-1 / (2 * domain.spacing[0]),
                  1 / (2 * domain.spacing[0]),
                  domain.shape[0])
            codomain = UniformGridFcts(cd, cd)
            super().__init__(
                domain=domain,
                codomain=codomain,
                linear=False,
            )

        def _eval(self, x, differentiate=False):
            y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))
            if differentiate:
                self._factor = y
            return y.real**2 + y.imag**2

        def _derivative(self, h):
            return 2*(self._factor.conj() * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(h), norm='ortho'))).real

        def _adjoint(self, y):
            return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(2*self._factor * y), norm='ortho'))

In practice, this operator need not be implemented from scratch. RegPy can
compose the existing Fourier-transform and squared-modulus operators:

.. code-block:: python

    from regpy.operators import SquaredModulus, FourierTransform
    from regpy.vecsps import UniformGridFcts

    domain = UniformGridFcts((-1, 1, 100), (-1, 1, 100), dtype=complex)

    ft = FourierTransform(domain, centered=True)
    sqm = SquaredModulus(ft.codomain)

    observe = sqm * ft


Further examples
^^^^^^^^^^^^^^^^

See the :doc:`Volterra example <notebooks/volterra_main_example>` for another
nonlinear operator.


NGSolve operators
~~~~~~~~~~~~~~~~~

RegPy integrates with NGSolve through :mod:`regpy.operators.ngsolve`,
:mod:`regpy.hilbert.ngsolve`, and :mod:`regpy.functionals.ngsolve`.

As for every RegPy operator, adjoints are implemented with respect to the
standard vector-space pairings, not a separately chosen Hilbert-space inner
product.

Use :class:`regpy.operators.ngsolve.NgsOperator` as the base class for custom
NGSolve operators.

.. caution::
    The NGSolve interface is still evolving and may change.

For coefficient identification in second-order elliptic PDEs, RegPy provides
:class:`regpy.operators.ngsolve.SecondOrderEllipticCoefficientPDE`. Subclasses
specify the required bilinear and linear forms. See the
:doc:`diffusion-coefficient example <notebooks/diffusion_coefficient>`.

.. _Adjoint_Derivative:

Combined adjoint and derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some operators admit a more efficient implementation of
:math:`F'[x]^*F'[x]` than separate derivative and adjoint applications. This is
particularly useful when codomain vectors would be too large to store. Such an
operator may implement ``_adjoint_derivative`` directly:

.. code-block:: python

    def _adjoint_derivative(self, h):
        return apply_normal_operator(self._linearization_data, h)

.. warning::
    Support for combined adjoint-derivative evaluations is specialized. Check
    that the intended solver uses this interface before relying on it.
