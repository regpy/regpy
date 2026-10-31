Spaces
======

Vectors and vector spaces
-------------------------

RegPy uses duck typing for vectors. A vector may be any object that implements
vector addition and subtraction, scalar multiplication and division, unary
negation, and the corresponding in-place operations. Generic algorithms must
not assume that vectors are NumPy arrays or that they support indexing,
componentwise operations such as ``.real``, attributes such as ``shape`` or ``dtype``, or a
``copy`` method.

Every vector representation is accompanied by a subclass of
:class:`regpy.vecsps.VectorSpaceBase`. The vector space recognizes its vectors
and provides representation-dependent operations such as allocation, copying,
inner products, random sampling, flattening, and reconstruction. Generic code
should therefore use, for example, ``space.empty()`` together with
``space.copyto(target, source)`` instead of calling ``source.copy()``.

Vector operations and storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distinction between a vector and its vector space is deliberate. The
vector itself supplies only linear arithmetic. The accompanying
``VectorSpaceBase`` instance owns operations whose implementation depends on
the representation or storage layout. Its most important public operations
include:

* ``x in space`` tests whether ``x`` is a vector of the space;
* ``space.zeros()``, ``space.ones()``, and ``space.empty()`` allocate vectors;
* ``space.copyto(target, source)`` copies values without replacing ``target``;
* ``space.vdot(x, y)`` supplies the standard pairing used by RegPy;
* ``space.rand()`` and ``space.randn()`` construct sample vectors; and
* ``space.flatten(x)`` and ``space.fromflat(values)`` convert between the
  representation and real coordinate vectors when the concrete space supports
  those conversions.

Allocation with ``empty`` does not initialize the vector. Code using it must
completely overwrite the result, usually through ``copyto`` or an in-place
operator call. This avoids unnecessary initialization for vector types whose
storage is expensive.

For example, the following generic function uses no NumPy-specific operation:

.. code-block:: python

   def linear_combination(space, x, y, a, b):
       if x not in space or y not in space:
           raise ValueError("x and y must belong to space")
       result = space.empty()
       space.copyto(result, x)
       result *= a
       result += b * y
       return result

Methods such as ``flatten`` are intentionally supplied by the space rather
than required from vectors. An algorithm that converts to coordinates is no
longer purely representation-independent and should use these methods
explicitly.

Specialized vector spaces may provide additional operations. NumPy vector
spaces support array operations, while NGSolve vector spaces expose operations
appropriate for their finite-element representation. Algorithms that require
such capabilities should state and validate the corresponding restriction.

All vector spaces are interpreted as spaces over the real numbers, including
spaces whose vectors have complex coefficients. This convention determines the
real dimension and the standard inner product used to define operator
adjoints, subgradients, Hessians, and conjugate functionals. See
:doc:`Functionals` for the corresponding dual-pairing convention.

Dimensions and metadata
~~~~~~~~~~~~~~~~~~~~~~~

A vector space may expose representation metadata even when its vectors do
not. In particular, ``space.shape`` describes the coefficient layout,
``space.ndim`` its number of axes, and ``space.size`` its number of
coefficients. The property ``space.realsize`` gives the dimension after the
space is interpreted over the real numbers. Consequently, a space with
``space.is_complex`` set to true normally has twice as many real dimensions as
complex coefficients.

Generic algorithms should query this metadata from the vector space and only
when it is mathematically required. They must not infer it from attributes on
an arbitrary vector object.

Direct sums
~~~~~~~~~~~

Several vector spaces can be combined into a
:class:`regpy.vecsps.DirectSum`. Addition provides a convenient construction:

.. code-block:: python

   product_space = parameter_space + auxiliary_space
   combined = product_space.join(parameter, auxiliary)
   parameter, auxiliary = product_space.split(combined)

Allocation, copying, membership tests, pairings, and random sampling are then
performed componentwise by the corresponding summand. Direct sums allow
operators and solvers to work with coupled unknowns whose components may even
use different vector representations.

Hilbert spaces
--------------

A :class:`regpy.hilbert.HilbertSpace` equips a vector space with an inner
product. Its ``vecsp`` attribute is the underlying
:class:`regpy.vecsps.VectorSpaceBase`; the Hilbert space adds geometry without
changing the vectors or their storage. If its Gram operator is denoted by
:math:`G_X`, the inner product is

.. math::

   \langle x, y\rangle_X = \langle G_X x, y\rangle,

where the pairing on the right is supplied by the underlying vector space.
This separates the vector representation from the geometry used by an inverse
problem or optimization method.

Gram operators and norms
~~~~~~~~~~~~~~~~~~~~~~~~

The property ``space.gram`` is a linear operator from ``space.vecsp`` to
itself. It represents the inner product relative to the standard vector-space
pairing and is required to be positive and self-adjoint in the corresponding
sense. The main Hilbert-space operations are:

* ``space.inner(x, y)`` evaluates the inner product;
* ``space.norm(x)`` evaluates the induced norm;
* ``space.gram(x)`` applies the Riesz map represented by the Gram operator;
* ``space.gram_inv(xstar)`` applies its inverse; and
* ``space.norm_functional`` returns the corresponding one-half squared-norm
  functional.

These operations should be used instead of explicitly assembling a Gram
matrix. In many applications the Gram operator is a multiplication,
differential, integral, or fast-transform-based operator whose action is much
cheaper than forming a dense matrix.

The Gram inverse is obtained from ``space.gram.inverse`` when the Gram operator
provides an inverse; otherwise the concrete Hilbert space can implement
``gram_inv`` separately. Some spaces also provide ``space.cholesky``, a
Cholesky-type factor or square root of the Gram operator. This factor is useful,
for example, when generating white Gaussian noise in the Hilbert-space
geometry.

Adjoints and dual Hilbert spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An operator's ``adjoint`` is always defined with respect to the standard
pairings of its vector spaces. If an operator :math:`T\colon :X\to Y` is viewed
between Hilbert spaces with Gram operators :math:`G_X:X\to X'` 
and :math:`G_Y:Y\to Y'`, its
Hilbert-space adjoint is obtained by

.. math::

   T^*_{X,Y}=G_X^{-1}T'G_Y,

where :math:`T'` is the standard RegPy adjoint. 

The method ``space.dual_space()`` returns the Hilbert structure on the same
underlying vector space for which the roles of ``gram`` and ``gram_inv`` are
interchanged. This is useful for dual norms, subgradient methods, and conjugate
functionals. Direct sums of Hilbert spaces can be formed by adding them; their
inner products and Gram operators act componentwise, with optional positive
weights.

Abstract Hilbert spaces
-----------------------

RegPy provides abstract Hilbert spaces that select a concrete implementation
for a given vector-space class. For example, ``Sobolev`` dispatches to the
implementation registered for the type of ``my_domain``:

.. code-block:: python

   from regpy.hilbert import Sobolev

   h1_on_my_space = Sobolev(my_domain, index=1)

Arguments can also be stored in an abstract-space factory and applied later:

.. code-block:: python

   h2 = Sobolev(index=2)
   h2_on_my_space = h2(my_domain)

This dispatch mechanism makes algorithms reusable across compatible vector
representations while allowing specialized implementations where necessary.
If no implementation is registered for the vector-space type and requested
parameters, construction raises ``NotImplementedError``. The analogous
mechanism for abstract functionals is described in :doc:`Functionals`.
