Spaces
======

Vectors and vector spaces
-------------------------

RegPy uses duck typing for vectors. A vector may be any object that implements
vector addition and subtraction, scalar multiplication and division, unary
negation, and the corresponding in-place operations. Generic algorithms must
not assume that vectors are NumPy arrays or that they support indexing,
componentwise operations such as ``.real``, attributes such as ``shape`` or
``dtype``, or a ``copy`` method.

Every vector representation is accompanied by a vector-space object satisfying
the :class:`regpy.vecsps.VectorSpace` protocol. Concrete implementations
usually inherit from :class:`regpy.vecsps.VectorSpaceBase` to reuse its
defaults, but inheritance is not required by the protocol. The vector space
recognizes its vectors and provides representation-dependent operations such
as allocation, copying, pairings, random sampling, and coordinate access.
Generic code should therefore use, for example, ``space.empty()`` together
with ``space.copyto(target, source)`` instead of calling ``source.copy()``.

The types :class:`regpy.vecsps.ReadOnlyVector` and
:class:`regpy.vecsps.Vector` describe the non-mutating and writable vector
protocols, respectively. They are typing aids rather than base classes that
vector implementations must inherit from.

Vector operations and storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distinction between a vector and its vector space is deliberate. The
vector itself supplies only linear arithmetic. The accompanying vector-space
object owns operations whose implementation depends on the representation or
storage layout. Its most important public operations include:

* ``x in space`` tests whether ``x`` belongs to the space;
* ``space.zeros()``, ``space.ones()``, and ``space.empty()`` create vectors;
* ``space.copyto(target, source)`` copies values without replacing ``target``;
* ``space.fill(target, value)`` assigns a scalar to the coefficients of a
  writable vector (where supported);
* ``space.vdot(x, y)`` supplies the standard pairing used by RegPy;
* ``space.rand()`` and ``space.randn()`` construct sample vectors; and
* ``space.iter_coefficients(x)``, ``space.iter_basis()``, ``space.flatten(x)``,
  and ``space.fromflat(values)`` provide access to distinguished coordinates
  and their basis.

Allocation with ``empty`` does not initialize the vector. Code using it must
completely overwrite the result, usually through ``copyto`` or an in-place
operator call. This avoids unnecessary initialization for vector types whose
storage is expensive.

For example, this generic function has explicit types and uses no
NumPy-specific vector operation:

.. code-block:: python

   from regpy.vecsps import ReadOnlyVector, Scalar, Vector, VectorSpace

   def linear_combination(
       space: VectorSpace,
       x: ReadOnlyVector,
       y: ReadOnlyVector,
       a: Scalar,
       b: Scalar,
   ) -> Vector:
       if x not in space or y not in space:
           raise ValueError("x and y must belong to space")
       result = space.empty()
       space.copyto(result, x)
       result *= a
       result += b * y
       return result

The scalars ``a`` and ``b`` must belong to the scalar field over which
``space`` is being used; the ``Scalar`` typing alias itself allows both real
and complex numeric types.

Specialized vector spaces may provide additional operations. NumPy vector
spaces support array operations, while NGSolve vector spaces expose operations
appropriate for their finite-element representation. Algorithms that require
such capabilities should state and validate the corresponding restriction.

Real and complex scalar fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RegPy supports vector spaces and Hilbert spaces over either the real or the
complex numbers. The scalar field is not determined by the data type alone:
a representation with complex-valued coefficients can also be regarded as a
real vector space (realification).

Two properties of a vector space distinguish these cases:

* ``space.is_complex`` describes whether the representation uses complex
  coefficients;
* ``space.as_real_space`` determines the scalar field: ``True`` means
  :math:`\mathbb{R}`, and ``False`` means :math:`\mathbb{C}`. A representation
  that is not complex must be used over :math:`\mathbb{R}`.

For instance, a representation with :math:`n` complex coefficients has
complex dimension :math:`n` if ``as_real_space=False``, but real dimension
:math:`2n` if ``as_real_space=True``. In the latter case, the real and imaginary
parts are independent *real* coordinates; only real scalars are used in
real-linear algebra, even though the stored values are complex.

The distinction matters when specifying linear operators, adjoints, Hilbert
products, derivatives, and subgradients. In particular, an operator that is
real-linear on complex-valued data need not be complex-linear. Generic code
should respect the scalar-field interpretation of the domain and codomain
rather than assume that all spaces are real or that complex storage implies
complex linearity.

Dimensions and coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~

The generic vector-space interface does **not** assign an array shape or a
number of axes to a vector space. In particular, ``space.shape`` and
``space.ndim`` are not part of the :class:`regpy.vecsps.VectorSpace` protocol.
The public property ``space.dimension`` is the dimension over the selected
scalar field. The older names ``size`` and ``realsize`` are deprecated aliases
of ``dimension`` in :class:`regpy.vecsps.VectorSpaceBase`; neither describes
a storage shape or invariably a real dimension.

RegPy assumes a **distinguished ordered basis** for its finite-dimensional
vector spaces. This permits a consistent coordinate representation even when
the stored vectors have no common indexing or array layout. If
:math:`e_0,\ldots,e_{n-1}` is that basis, a vector has coordinates
:math:`(x_0,\ldots,x_{n-1})` determined by

.. math::

   x = \sum_{j=0}^{n-1} x_j e_j,
   \qquad n = \texttt{space.dimension}.

The following operations use exactly this basis order:

* ``space.iter_coefficients(x)`` yields the :math:`n` scalar coordinates of
  ``x`` in order;
* ``space.set_coefficients(x, coefficients)`` writes exactly :math:`n`
  coordinates into an existing writable vector;
* ``space.iter_basis()`` yields the corresponding basis vectors
  :math:`e_0,\ldots,e_{n-1}`;
* ``space.flatten(x)`` returns a one-dimensional NumPy array containing these
  coordinates (of length ``space.dimension``); and
* ``space.fromflat(values)`` reconstructs a vector from a one-dimensional
  coordinate array of that length.

For example, for any space providing these operations:

.. code-block:: python

   x = space.randn()
   coefficients = list(space.iter_coefficients(x))
   coords = space.flatten(x)
   reconstructed = space.fromflat(coords)

   assert len(coefficients) == space.dimension
   assert coords.shape == (space.dimension,)
   # reconstructed has the same coordinates as x.

This use of ``coords.shape`` is intentional: ``coords`` is a NumPy array;
``space`` itself has no generic shape. Likewise, ``flatten`` is a coordinate
conversion, **not** a promise that the backend stores vectors in a flat
array. Coordinate-dependent algorithms should call the vector-space methods
explicitly, rather than index arbitrary vector objects.

With complex storage interpreted over the reals, the coordinate order is
:math:`\operatorname{Re} z_0,\operatorname{Im} z_0,
\operatorname{Re} z_1,\operatorname{Im} z_1,\ldots`.
Thus the real basis contains the real and imaginary directions for each
native complex coefficient. With a complex scalar field, each complex
coefficient instead contributes one coordinate.

**Note:** ``iter_basis()`` may reuse a single mutable work vector for
successive basis elements. Process each result immediately or explicitly copy
it using ``space.empty()`` and ``space.copyto`` if it must be retained. Also,
prefer the named method ``iter_basis()`` to iterating over the space itself:
for a :class:`regpy.vecsps.DirectSum`, iteration over the space returns its
summand spaces instead.

Direct sums
~~~~~~~~~~~

Several vector spaces can be combined into a
:class:`regpy.vecsps.DirectSum`. Addition provides a convenient construction:

.. code-block:: python

   sum_space = parameter_space + auxiliary_space
   combined = sum_space.join(parameter, auxiliary)
   parameter, auxiliary = sum_space.split(combined)

Allocation, copying, membership tests, pairings, and random sampling are then
performed componentwise by the corresponding summand. The distinguished
coordinate order of a direct sum concatenates the coordinates of its
summands, and ``sum_space.dimension`` is the sum of their dimensions.
Direct sums allow operators and solvers to work with coupled unknowns whose
components may even use different vector representations. Their summands
must be interpreted over a common scalar field; a mixed-field direct sum is
interpreted over the reals.

Hilbert spaces
--------------

A :class:`regpy.hilbert.HilbertSpace` equips a vector space with an inner
product over its selected scalar field. Its ``vecsp`` attribute is the
underlying vector space; the Hilbert space adds geometry without changing
the vectors or their storage. Hilbert spaces may be real or complex. If the
Gram operator is denoted by :math:`G_X`, the inner product is

.. math::

   \langle x, y\rangle_X = \langle G_X x, y\rangle,

where the pairing on the right is supplied by the underlying vector space.
In the complex case these pairings are sesquilinear, with the corresponding
adjoint convention. This separates the vector representation from the
geometry used by an inverse problem or optimization method.

Gram operators and norms
~~~~~~~~~~~~~~~~~~~~~~~~

The property ``space.gram`` is a linear operator on ``space.vecsp``. It
represents the inner product relative to the standard vector-space pairing
and is required to be positive and self-adjoint in the corresponding scalar
field. The main Hilbert-space operations are:

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
Cholesky-type factor or square root of the Gram operator. This factor is
useful, for example, when generating white Gaussian noise in the Hilbert-space
geometry.

Adjoints and dual Hilbert spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An operator's ``adjoint`` is defined with respect to the standard pairings of
its vector spaces, for the scalar field in use. If a linear operator
:math:`T\colon X\to Y` is viewed between Hilbert spaces with Gram operators
:math:`G_X` and :math:`G_Y`, its Hilbert-space adjoint is obtained by

.. math::

   T^*_{X,Y} = G_X^{-1} T' G_Y,

where :math:`T'` denotes the standard RegPy adjoint. For complex vector
spaces this is the adjoint with respect to their complex inner products; for
realifications it is the real-linear adjoint instead. The same formula
applies provided the operators and Gram maps use compatible scalar fields.

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
