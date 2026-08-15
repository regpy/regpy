Spaces
======

Vectors and vector spaces
-------------------------

RegPy uses duck typing for vectors. A vector may be any object that implements
vector addition and subtraction, scalar multiplication and division, unary
negation, and the corresponding in-place operations. Generic algorithms must
not assume that vectors are NumPy arrays or that they support indexing,
componentwise operations, attributes such as ``shape`` or ``dtype``, or a
``copy`` method.

Every vector representation is accompanied by a subclass of
:class:`regpy.vecsps.VectorSpaceBase`. The vector space recognizes its vectors
and provides representation-dependent operations such as allocation, copying,
inner products, random sampling, flattening, and reconstruction. Generic code
should therefore use, for example, ``space.empty()`` together with
``space.copyto(target, source)`` instead of calling ``source.copy()``.

Specialized vector spaces may provide additional operations. NumPy vector
spaces support array operations, while NGSolve vector spaces expose operations
appropriate for their finite-element representation. Algorithms that require
such capabilities should state and validate the corresponding restriction.

All vector spaces are interpreted as spaces over the real numbers, including
spaces whose vectors have complex coefficients. This convention determines the
real dimension and the standard inner product used to define operator
adjoints, subgradients, Hessians, and conjugate functionals. See
:doc:`Functionals` for the corresponding dual-pairing convention.

Hilbert spaces
--------------

A :class:`regpy.hilbert.HilbertSpace` equips a vector space with an inner
product. If its Gram operator is denoted by :math:`G_X`, the inner product is

.. math::

   \langle x, y\rangle_X = \operatorname{Re}\langle G_X x, y\rangle,

where the pairing on the right is supplied by the underlying vector space.
This separates the vector representation from the geometry used by an inverse
problem or optimization method.

Abstract Hilbert spaces
-----------------------

RegPy provides abstract Hilbert spaces that select a concrete implementation
for a given vector-space class. For example:

.. code-block:: python

   from regpy.hilbert import H1
   h1_on_my_space = H1(my_domain)

This dispatch mechanism makes algorithms reusable across compatible vector
representations while allowing specialized implementations where necessary.
The analogous mechanism for abstract functionals is described in
:doc:`Functionals`.
