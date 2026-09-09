Functionals
===========

A :class:`regpy.functionals.Functional` maps vectors to extended real values.
Examples include norms, data-fidelity terms, and regularization penalties.
Each functional has a :class:`regpy.vecsps.VectorSpaceBase` as its ``domain``
and a :class:`regpy.hilbert.HilbertSpace` as its ``h_domain``. See
:doc:`Spaces` for the separation between vector representation and
Hilbert-space geometry.

Convex functionals and dual pairings
------------------------------------

Most functionals used in RegPy's optimization methods are convex. For a convex
functional :math:`F` and vectors :math:`x,z` in its domain, a vector
:math:`x^*` is a subgradient at :math:`x` if

.. math::

   F(z) \geq F(x) + \langle x^*, z-x\rangle
   \qquad\text{for every }z.

As for operator adjoints, RegPy identifies dual elements with vectors from the
same underlying vector space and uses the standard real pairing

.. math::

   \langle x^*,x\rangle
   = \operatorname{Re}\bigl(\operatorname{vdot}_{\mathbb{X}}(x^*,x)\bigr).

In code, this pairing is obtained from the vector space as
``functional.domain.vdot(xstar, x).real``. This convention also applies when
vectors have complex coefficients: the coefficient space is still interpreted
as a real vector space.

The same pairing is used throughout the functional API. In particular, it
determines the representation of subgradients, the action of Hessian
operators, and the optimality conditions for proximal mappings. It is also
used by every method of the conjugate functional. The Hilbert space
``functional.h_domain`` supplies the metric for norms and proximal terms, but
it does not change the underlying dual-pairing convention; its Gram operator
provides the corresponding Riesz identification when one is needed.

Public methods
--------------

The principal operations of a functional ``functional`` are:

* ``functional(x)`` evaluates :math:`F(x)`;
* ``functional.linearize(x)`` returns the value and a particular subgradient;
* ``functional.subgradient(x, out=None)`` returns a subgradient, optionally
  writing it into ``out``;
* ``functional.hessian(x)`` returns the Hessian at :math:`x` as a linear
  :class:`regpy.operators.Operator`, when it is available;
* ``functional.proximal(x, tau, out=None)`` evaluates the proximal mapping;
* ``functional.dist_subdiff(xstar, x)`` measures the distance of a dual vector
  to :math:`\partial F(x)` in the appropriate dual norm.

Writing :math:`\lVert\cdot\rVert_H` for the norm of
``functional.h_domain``, the proximal mapping is

.. math::

   \operatorname{prox}_{\tau F}(x)
   = \operatorname*{argmin}_{z}
     \left(F(z)+\frac{1}{2\tau}\lVert z-x\rVert_H^2\right).

Concrete classes implement the corresponding private methods, such as
``_eval``, ``_subgradient`` or ``_linearize``, ``_hessian``, and
``_proximal``. Not every operation needs a separate implementation: for
example, RegPy can combine evaluation and subgradient computation through
``linearize`` and can obtain one of the primal and conjugate proximal mappings
from the other using Moreau's identity. The set ``functional.methods`` records
which operations a concrete functional provides. It contains applicable names
from ``{'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}``.

Conjugate functionals
---------------------

For a convex functional :math:`F`, its Fenchel conjugate is

.. math::

   F^*(x^*)
   = \sup_x \bigl(\langle x^*,x\rangle-F(x)\bigr),

with the real dual pairing defined above. The conjugate is available through
the ``functional.conj`` property. It behaves like an ordinary functional, so
its operations are accessed consistently:

.. code-block:: python

   conjugate = functional.conj
   value = conjugate(xstar)
   value, subgradient = conjugate.linearize(xstar)
   subgradient = conjugate.subgradient(xstar)
   hessian = conjugate.hessian(xstar)
   proximal = conjugate.proximal(xstar, tau)

The conjugate uses the dual Hilbert space of ``functional.h_domain`` for its
metric-dependent operations. Its available operations are reported by
``functional.conj.methods``. Concrete implementations may provide the private
counterparts ``_conj``, ``_conj_subgradient`` or ``_conj_linearize``,
``_conj_hessian``, and ``_conj_proximal`` on the original functional; the
``conj`` proxy exposes them through the ordinary public interface. For convex
functionals, taking the conjugate twice recovers the original functional under
the usual properness and lower-semicontinuity assumptions.

Convexity and Lipschitz constants
---------------------------------

A convex RegPy functional carries quantitative information about its geometry
relative to ``functional.h_domain``:

* ``functional.convexity_param`` is its strong-convexity parameter
  :math:`\mu\geq0`. With :math:`x^*\in\partial F(x)`, this means

  .. math::

     F(z) \geq F(x)+\langle x^*,z-x\rangle
     +\frac{\mu}{2}\lVert z-x\rVert_H^2.

  A value of zero records convexity without a positive strong-convexity bound.

* ``functional.Lipschitz`` is a Lipschitz constant :math:`L` for the gradient
  with respect to the same Hilbert-space geometry. The value ``math.inf``
  indicates that no finite Lipschitz bound is available.

The conjugate proxy derives the corresponding constants from the primal
functional: positive strong convexity of :math:`F` gives a Lipschitz gradient
for :math:`F^*`, while a finite positive gradient-Lipschitz constant for
:math:`F` gives strong convexity of :math:`F^*`.

Having these constants attached to the functional is useful when it is passed
to a convex optimization method. Solvers can select admissible step sizes,
activate accelerated or strongly convex variants, estimate convergence rates,
and assess conditioning without requiring the caller to supply the same
geometric information separately. 

Abstract functionals
--------------------

RegPy provides abstract functionals that select a concrete implementation for
a given vector-space class. For example:

.. code-block:: python

   from regpy.functionals import TV
   tv_on_my_space = TV(my_domain)

This dispatch mechanism makes algorithms reusable across compatible vector
representations while allowing specialized implementations where necessary.
