Functionals
===========

A :class:`regpy.functionals.base.Functional` maps vectors to extended real
values. Typical examples are data-fidelity terms, constraints and
regularization penalties. Its ``domain`` is a RegPy vector space, while
``h_domain`` specifies the Hilbert-space geometry used for proximal maps and
norms; see :doc:`Spaces`. In particular, RegPy does **not** assume that all
vector coefficients are real. Real-valued convex functionals can also be
defined on complex vector spaces, using the underlying real duality.

The examples below concentrate on the public interface and on the much
simpler task of implementing a *pointwise* integral functional. For the
complete reference, consult the :mod:`regpy.functionals.base`,
:mod:`regpy.functionals.integral_functionals`,
:mod:`regpy.functionals.vector_integral_functionals` and
:mod:`regpy.functionals.matrix_functionals` API pages.

Basic operations and duality
---------------------------

For a functional ``F``, important public operations are:

* ``F(x)``: evaluation;
* ``F.linearize(x)``: evaluation and one subgradient;
* ``F.subgradient(x)``: a dual vector (not generally a Hilbert gradient);
* ``F.hessian(x)``: a linear operator, if available;
* ``F.proximal(x, tau)``: the proximal mapping in the metric of ``F.h_domain``;
* ``F.dist_subdiff(xstar, x)``: distance to the subdifferential;
* ``F.conj``: the Fenchel conjugate, itself exposed through the same interface.

If :math:`F` is convex, its subgradient :math:`x^*` at :math:`x` satisfies

.. math::

   F(z)\geq F(x)+\langle x^*,z-x\rangle\qquad\text{for every }z.

The dual pairing is the **real** part of the standard vector-space pairing,
including for arrays with complex entries:

.. math::

   \langle x^*,x\rangle
       =\operatorname{Re}\operatorname{vdot}_{\mathbb X}(x^*,x).

Thus in Python use ``F.domain.vdot(xstar, x).real``. The Gram operator of
``F.h_domain`` is used where the chosen Hilbert metric is required; it must
not silently be inserted into the standard dual pairing.

The proximal map is defined by

.. math::

   \operatorname{prox}_{\tau F}(x)
   =\operatorname*{argmin}_{z}\left\{
       F(z)+\frac{1}{2\tau}\|z-x\|_{F.h\_domain}^{2}\right\}.

The conjugate functional uses the Fenchel--Legendre transform

.. math::

   F^*(x^*)=\sup_x\bigl(\langle x^*,x\rangle-F(x)\bigr).

For example, ``F.conj.proximal(xstar, tau)`` evaluates the proximal map of
:math:`F^*`. The attributes ``F.methods`` and ``F.conj.methods`` report the
implemented operations, with names such as ``'eval'``, ``'subgradient'``,
``'hessian'``, ``'proximal'`` and ``'dist_subdiff'``. 

The constants ``F.convexity_param`` and ``F.Lipschitz`` provide strong-convexity and
gradient-Lipschitz information, respectively, relative to ``F.h_domain``. The fact that 
RegPy functionals usually ''know'' their Lipschitz and convexity constants is useful for 
the automatic choice of parameters such that step sizes and the choice of accelerated versions 
of convex optimization methods. 

Arithmetic, shifts and direct sums
----------------------------------

RegPy supports building functionals from existing ones without implementing
a new class. For example, ``2 * F + G + 3`` creates a linear combination
with an additive constant. Such combinations are represented by
:class:`regpy.functionals.base.LinearCombination` (and
:class:`regpy.functionals.base.VerticalShift` for a vertical offset).
Functional arguments must have compatible domains; a general sum need not
have a simple proximal map even when its individual summands do.

A horizontal translation is written ``F.shift(v)`` and represents
:math:`x\mapsto F(x-v)`. The methods ``F.as_data_func(data)`` and
``F.dilation(a)`` are further conveniences, implemented using
:class:`regpy.functionals.base.HorizontalShiftDilation` where appropriate.
These operations are useful for data fidelity terms and penalties centered
at a reference estimate. The exact behavior of combinations, offsets and
composition with an operator is documented under
:class:`regpy.functionals.base.LinearCombination`,
:class:`regpy.functionals.base.HorizontalShiftDilation` and
:class:`regpy.functionals.base.Composed`.

To combine functionals on *different* spaces, use
:class:`regpy.functionals.base.FunctionalOnDirectSum`:

.. math::

   F(x_1,\ldots,x_p)=\sum_{j=1}^{p}F_j(x_j).

Its domain is a :class:`regpy.vecsps.DirectSum`, and its proximal mapping,
when supported by the components, acts componentwise. For example:

.. code-block:: python

   from regpy.functionals.base import FunctionalOnDirectSum
   from regpy.functionals.integral_functionals import Quadratic

   F1 = Quadratic(grid1)
   F2 = Quadratic(grid2)
   F = FunctionalOnDirectSum([F1, F2])

Here ``grid1`` and ``grid2`` are scalar-valued measure spaces. See the
:doc:`Spaces` tutorial for direct sums of vector spaces.

Pointwise integral functionals
------------------------------

A particularly convenient way to implement convex functionals on real-valued
:class:`regpy.vecsps.numpy.MeasureSpaceFcts` is to subclass
:class:`regpy.functionals.integral_functionals.IntegralFunctionalBase`.
On a discrete measure space, such a functional has the form

.. math::

   F(u)=\sum_{i}w_i\,f_i(u_i),

where :math:`w_i` are the (possibly nonuniform) integration weights of the
vector space. Most concrete classes provide the same scalar integrand at
every point, possibly with position-dependent parameters. For example,
:class:`regpy.functionals.integral_functionals.Quadratic` implements

.. math::

   F(u)=\sum_i w_i\left(\tfrac12 a_i u_i^2+b_i u_i+c_i\right).

**The base class handles the integration weights**: a subclass implements
pointwise operations on :math:`f_i`, *without multiplying by* :math:`w_i`.
For instance, it multiplies the pointwise derivative by the weights to obtain
the dual-coordinate subgradient. Conversely, when evaluating the conjugate,
the base class converts dual coefficients :math:`p_i` to pointwise conjugate
arguments :math:`p_i/w_i`. It also handles the weighting of pointwise
Hessians and the proximal mapping in the :math:`L^2` metric.

The scalar ``IntegralFunctionalBase`` requires a real-valued, scalar
``MeasureSpaceFcts`` domain. For vector-valued integrands see the section
:ref:`vector_integral_functionals` below.

Constraints and Taylor extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to an integrand, the base class can impose pointwise bounds or
replace the integrand outside a prescribed interval. The relevant constructor
arguments are:

* ``constr_l`` / ``constr_u`` for hard lower and upper constraints;
* ``lin_taylor_l`` / ``lin_taylor_u`` for first-order Taylor extensions;
* ``quad_taylor_l`` / ``quad_taylor_u`` for second-order Taylor extensions.

For example, at a lower threshold :math:`a`, the three alternatives for
:math:`u<a` are

.. math::

   \begin{array}{ll}
   +\infty, &\text{hard constraint},\\[1mm]
   f(a)+f'(a)(u-a), &\text{linear Taylor extension},\\[1mm]
   f(a)+f'(a)(u-a)+\tfrac12 f''(a)(u-a)^2,
       &\text{quadratic Taylor extension}.
   \end{array}

The upper-threshold variants are analogous. On each side, only **one** of
constraint, linear extension or quadratic extension may be selected.
Thresholds may be scalars or vectors in the domain. A hard constraint makes
the value infinite outside its feasible interval. By contrast, Taylor
extensions preserve a finite continuation, potentially improving
regularity or strong-convexity properties used by optimization methods.

A particularly simple example is the nonnegative quadratic functional

.. math::

   F(u)=\begin{cases}
       \frac12\sum_i w_i u_i^2,&u_i\geq 0\text{ for all }i,\\
       +\infty,&\text{otherwise}.
   \end{cases}

There is **no need to subclass anything** for this example:

.. code-block:: python

   import numpy as np
   from regpy.vecsps import UniformGridFcts
   from regpy.functionals.integral_functionals import Quadratic

   grid = UniformGridFcts(np.linspace(0, 1, 64), dtype=float)
   F = Quadratic(grid, a=1.0, constr_l=0.0)

   u = np.linspace(-1, 1, 64)
   assert np.isinf(F(u))        # negative values violate the constraint
   assert np.isfinite(F(np.maximum(u, 0)))

The pre-existing :class:`regpy.functionals.integral_functionals.QuadraticNonneg`
implements this mathematical functional as a dedicated class. The
``QuadNonneg(grid)`` abstract-functional factory also selects it on supported
measure spaces; see :class:`regpy.functionals.base.AbstractFunctional`.

Implementing ``QuadraticNonneg`` by hand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a **new** pointwise functional, the main work is to provide the integrand
and the operations that are needed by the intended solvers. To make the
interface concrete, we can implement the nonnegative quadratic *again*, this
time explicitly. This is a teaching implementation, not a replacement for the
optimized class supplied with RegPy.

The pointwise integrand and its Fenchel conjugate are

.. math::

   f(u)=\tfrac12u^2+I_{[0,\infty)}(u),\qquad
   f^*(p)=\tfrac12\max(p,0)^2,

where :math:`I_C` is zero on :math:`C` and infinite outside. Their pointwise
proximal mappings, for :math:`\tau>0`, are

.. math::

   \operatorname{prox}_{\tau f}(u)
       =\max\!\left(\frac{u}{1+\tau},0\right),\qquad
   \operatorname{prox}_{\tau f^*}(p)
       =\begin{cases}p/(1+\tau),&p>0,\\p,&p\leq0.\end{cases}

The following class intentionally creates intermediate arrays instead of
introducing scratch-buffer optimizations. It nevertheless supports the
``out`` and ``mask`` arguments used internally by ``IntegralFunctionalBase``:

.. code-block:: python

   import numpy as np
   from regpy.functionals.integral_functionals import IntegralFunctionalBase
   from regpy.functionals.base import NotInEssentialDomainError

   class QuadraticNonnegExample(IntegralFunctionalBase):
       """Educational implementation of 0.5*||u||_L2^2 for u >= 0."""

       def __init__(self, domain):
           super().__init__(
               domain,
               nat_dom_l=0.,
               convexity_param=1.,
               methods={'eval', 'subgradient', 'hessian',
                        'proximal', 'dist_subdiff'},
               conj_methods={'eval', 'subgradient', 'hessian',
                             'proximal', 'dist_subdiff'},
           )

       @staticmethod
       def _write(values, u, out=None, mask=None):
           """Write selected entries, without changing unselected entries."""
           if out is None:
               out = np.empty_like(u)
           if mask is None:
               out[...] = values
           else:
               np.copyto(out, values, where=mask)
           return out

       def _f(self, u, out=None, mask=None, **kwargs):
           values = np.where(u >= 0., 0.5 * u**2, np.inf)
           return self._write(values, u, out, mask)

       def _f_deriv(self, u, out=None, mask=None, **kwargs):
           # Smooth part. At u=0, zero is a valid subgradient.
           return self._write(u.copy(), u, out, mask)

       def _f_second_deriv(self, u, out=None, mask=None, **kwargs):
           # Second derivative on the strictly positive interior.
           return self._write(np.ones_like(u), u, out, mask)

       def _f_prox(self, u, tau, out=None, mask=None, **kwargs):
           values = np.maximum(u / (1. + tau), 0.)
           return self._write(values, u, out, mask)

       def _f_conj(self, p, out=None, mask=None, **kwargs):
           values = 0.5 * np.maximum(p, 0.)**2
           return self._write(values, p, out, mask)

       def _f_conj_deriv(self, p, out=None, mask=None, **kwargs):
           return self._write(np.maximum(p, 0.), p, out, mask)

       def _f_conj_second_deriv(self, p, out=None, mask=None,
                                **kwargs):
           # f* is not twice differentiable at p=0.
           # The value 1 there is a convention, not a classical Hessian.
           values = np.where(p >= 0., 1., 0.)
           return self._write(values, p, out, mask)

       def _f_conj_prox(self, p, tau, out=None, mask=None, **kwargs):
           values = np.where(p > 0., p / (1. + tau), p)
           return self._write(values, p, out, mask)

       def _ptw_dist_subdiff(self, p, u, **kwargs):
           # The dual vector p already includes the integration measure.
           if np.any(u < 0.):
               raise NotInEssentialDomainError
           residual = p - self.measure * u
           # At u=0, subgradients include all nonpositive dual values.
           return np.where(u == 0., np.maximum(residual, 0.), residual)

       def _conj_ptw_dist_subdiff(self, u, p, **kwargs):
           # Dual coordinates p_i correspond to densities p_i / w_i.
           return u - np.maximum(p / self.measure, 0.)

   example = QuadraticNonnegExample(grid)
   u = np.maximum(np.linspace(-1, 1, grid.shape[0]), 0.)
   assert np.isclose(example(u), F(u))
   assert np.allclose(example.proximal(u, 0.5), F.proximal(u, 0.5))

The hook methods have distinct responsibilities:

``_f``
   Pointwise values :math:`f(u)`; the base class performs the weighted sum.

``_f_deriv`` and ``_f_second_deriv``
   Pointwise first and second derivatives on the smooth part of the natural
   domain. The base class forms dual-coordinate subgradients and Hessians.
   At the boundary of a nonsmooth constraint, a classical Hessian generally
   does **not** exist; numerical algorithms must take that into account.

``_f_prox``
   Pointwise primal proximal map with the supplied ``tau``. This ``tau``
   belongs to the :math:`L^2` proximal term; the subclass does not insert
   integration weights.

``_f_conj``, ``_f_conj_deriv`` and ``_f_conj_second_deriv``
   Corresponding operations for the *pointwise* Fenchel conjugate :math:`f^*`.
   They receive the pointwise dual density, not the weighted dual vector.

``_f_conj_prox``
   Pointwise proximal map of :math:`f^*`.

``_ptw_dist_subdiff`` and ``_conj_ptw_dist_subdiff``
   Signed pointwise residuals for testing the subdifferential relations,
   including the normal cone at a hard constraint. These are relevant if
   ``'dist_subdiff'`` is advertised in the corresponding ``methods`` set.
   **Unlike the eight integrand hooks**, these residual methods operate on
   *dual coordinates*: this is why the example uses ``self.measure``.
   The default implementations suffice for many everywhere-differentiable
   functionals; for constrained nonsmooth functionals they may need to be
   overridden, as above.

In the example, ``_write`` is only a small helper to respect ``out`` and
``mask``. Actual RegPy implementations often use in-place NumPy operations
and reusable buffers to reduce allocations. For a new functional, begin with
correct mathematical formulas and add such optimizations only after testing.
The base class offers numerical proximal implementations in some situations;
therefore not every new integrand needs all eight analytic hooks. Declare
only the methods that are actually implemented and appropriate for the
functional's smoothness. See the docstring and private hooks of
:class:`regpy.functionals.integral_functionals.IntegralFunctionalBase`.

Bregman distances
-----------------

For a convex :math:`F` and a specified subgradient
:math:`y^*\in\partial F(y)`, the Bregman distance is

.. math::

   D_F^{y^*}(x,y)
   =F(x)-F(y)-\langle y^*,x-y\rangle.

It is nonnegative but generally neither symmetric nor a metric. RegPy
provides ``F.bregman(x, y, ystar=None)`` and the corresponding functional
``F.bregman_func(y, ystar=None)``. If the subgradient is omitted, RegPy
obtains one via ``F.subgradient(y)``. The API also offers
``F.conj_bregman(...)`` and ``F.symm_bregman(...)``.

.. code-block:: python

   from regpy.functionals.integral_functionals import Quadratic

   Q = Quadratic(grid)
   y = np.ones(grid.shape)
   D = Q.bregman_func(y)
   assert np.isclose(D(y), 0.)
   assert np.isclose(Q.bregman(y, y), 0.)

For a nonsmooth functional, different admissible subgradients at the same
base point can lead to different Bregman distances; pass ``ystar`` explicitly
when a particular choice is important. The subgradient is represented in
standard dual coordinates, not in Hilbert-gradient coordinates.

The gradient as an operator
---------------------------

When both subgradient and Hessian are available, the method
``F.gradient_as_operator()`` creates a nonlinear RegPy
:class:`regpy.operators.Operator` mapping the functional's domain into
itself. Its value is the **dual-coordinate subgradient**
:math:`G(x)=\partial F(x)` (for differentiable :math:`F`, the gradient in
the standard pairing). Its derivative is the Hessian:

.. math::

   G'[x]h=F''[x]h.

This makes gradients usable inside algorithms that expect the standard
operator interface, including ``linearize`` and subsequent derivative or
adjoint operations.

.. code-block:: python

   G = Q.gradient_as_operator()
   g, DG = G.linearize(y)
   assert np.allclose(g, Q.subgradient(y))
   assert np.allclose(DG(y), Q.hessian(y)(y))

Be careful with terminology: ``G(x)`` is **not** automatically
``F.h_domain.gram_inv(F.subgradient(x))``. For nonsmooth functionals, a
single-valued, globally differentiable gradient operator need not exist,
and the method requires both ``'subgradient'`` and ``'hessian'`` in
``F.methods``.

.. _vector_integral_functionals:

Vector-valued integral functionals
---------------------------------------------

:class:`regpy.functionals.vector_integral_functionals.VectorIntegralFunctional`
handles integrands depending on the norm of a vector at each spatial
point, for example

.. math::

   J(v)=\sum_i w_i\,\varphi_i(\|v_i\|_2).

It combines a scalar separable functional with a norm on the component
axes. The classes
:class:`regpy.functionals.vector_integral_functionals.L1L2`,
:class:`regpy.functionals.vector_integral_functionals.LppL2` and
:class:`regpy.functionals.vector_integral_functionals.HuberL2`
provide useful special cases.

A prominent application is the discrete isotropic total-variation
functional

.. math::

   \operatorname{TV}(u)=\sum_i w_i\,\|(\nabla_hu)_i\|_2.

Here the finite-difference gradient is an operator into a vector-valued
space, and the pointwise Euclidean norms are combined by ``L1L2``. In
RegPy, :class:`regpy.functionals.vector_integral_functionals.TVUniformGridFcts`
implements this as a composed functional (for ``beta=0``):

.. code-block:: python

   from regpy.functionals import TV

   tv = TV(grid, beta=0.0)
   # Internally, tv.grad is a ForwardFDGradient operator and
   # tv.func is an L1L2 functional on the gradient's codomain.

For ``beta > 0``, it uses a Huber-type approximation through ``HuberL2``.
A TV proximal mapping generally requires an inner iterative solve; it is
not just a pointwise soft-thresholding step in the original image space.
See :doc:`Operators` for the gradient operator and operator compositions.


Further references
------------------

The documented API is implemented primarily in
:mod:`regpy.functionals.base`,
:mod:`regpy.functionals.integral_functionals`, and
:mod:`regpy.functionals.vector_integral_functionals`
in :doc:`Spaces`, :doc:`Operators` and :doc:`Solvers`.
