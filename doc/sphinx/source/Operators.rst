Operators
=========

A :class:`regpy.operators.Operator` represents a possibly nonlinear map
:math:`F:X\to Y` between RegPy vector spaces. Both real- and complex-valued
vectors occur in practice. In particular, an operator between complex array
spaces does not need to be complex-linear: differentiation and adjoints use the
underlying real duality whenever appropriate. See :doc:`Spaces` for vector
storage, scalar fields and Hilbert-space structures.

Users evaluate an operator with ``F(x)`` and can optionally supply a writable
result buffer using ``F(x, out=y)``. RegPy also supports compositions,
linear combinations, scalar and pointwise multiplication, and offsets in the
input or output. We first explain how to implement a simple operator; the
section :ref:`building_operators` then discusses combinations in detail.

The operator's adjoint is defined with respect to the *standard vector-space
pairings*, not with respect to arbitrary Hilbert-space norms chosen for an
inverse problem. For NumPy vectors the standard pairing is

.. math::

   \langle x,z\rangle= \operatorname{vdot}(x,z).

When Hilbert norms are induced by Gram operators :math:`G_X,G_Y`, the
corresponding Hilbert adjoint is instead :math:`G_X^{-1}T^*G_Y`. This
separation is important when implementing discrete adjoints.

.. _implementing_linear_operators:

Implementing a linear operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every linear operator defines its ``domain`` and ``codomain`` and supplies an
evaluation and its adjoint. The public ``op(x)`` and ``op.adjoint(y)``
interfaces dispatch to the implementation hooks ``_eval`` and ``_adjoint``
(or the in-place variants discussed below).

**Example: summation with constant extrapolation.** Let :math:`1\le n\le m`
and define :math:`T:\mathbb K^n\to\mathbb K^m`, for
:math:`\mathbb K\in\{\mathbb R,\mathbb C\}`, by

.. math::

   (Tf)_j=\sum_{k=1}^{j}\widetilde f_k,\qquad
   \widetilde f_k=\begin{cases}
     f_k & k\le n,\\
     f_n & n<k\le m.
   \end{cases}

Here indexing starts at one. Thus :math:`T(1,2,3)=(1,3,6,9,12)` for
:math:`n=3,m=5`. The operator does not need an explicit dense matrix. Let
:math:`s_k=\sum_{j=k}^{m}y_j`. Its standard adjoint is

.. math::

   (T^*y)_i=\begin{cases}
       s_i & i<n,\\
       \displaystyle\sum_{k=n}^{m}s_k & i=n.
   \end{cases}

The following complete implementation uses
:class:`regpy.vecsps.numpy.NumPyVectorSpace` for both spaces. The same code
works with real or complex NumPy arrays.

.. code-block:: python

   import numpy as np
   from regpy.operators import Operator
   from regpy.vecsps import NumPyVectorSpace

   class CumulativeSumExtrapolated(Operator):
       """Cumulative sum with constant extension of the last entry."""

       def __init__(self, n, m, dtype=float):
           if not (1 <= n <= m):
               raise ValueError("Require 1 <= n <= m.")
           self.n, self.m = n, m
           domain = NumPyVectorSpace(n, dtype=dtype)
           codomain = NumPyVectorSpace(m, dtype=dtype)
           super().__init__(domain, codomain, linear=True)

       def _eval(self, x):
           extended = np.concatenate((
               x, np.full(self.m - self.n, x[-1], dtype=x.dtype)
           ))
           return np.cumsum(extended)

       def _adjoint(self, y):
           suffix = np.cumsum(y[::-1])[::-1]
           result = self.domain.empty()
           result[:-1] = suffix[:self.n - 1]
           result[-1] = np.sum(suffix[self.n - 1:])
           return result

   T = CumulativeSumExtrapolated(3, 5)
   x = np.array([1., 2., 3.])
   assert np.array_equal(T(x), [1., 3., 6., 9., 12.])

The forward and adjoint actions both require only :math:`O(m)` work.
Different domain and codomain sizes also occur in discretizations using
staggered grids; unlike this abstract array example, physical grids may
require additional quadrature factors.

Complex storage does not by itself force complex-scalar semantics: the
``as_real_space`` argument of ``NumPyVectorSpace`` controls the scalar field
and defaults to ``True``. Use ``as_real_space=False`` for a complex-scalar
space if only one of the spaces :math:`X` and :math:`Y` is complex (otherwise 
the notion of linearity is not defined) or if your operator is real-linear, 
but not complex-linear.

In-place implementations and buffers
------------------------------------

For linear operators, RegPy supports pairs of implementation hooks:
``_eval(x)`` / ``_ieval(x, out)`` and ``_adjoint(y)`` /
``_iadjoint(y, out)``. Normally implementing one member of each pair is
sufficient; the base class provides the other. For repeated evaluations,
preallocated buffers can reduce allocation and copying:

.. code-block:: python

   out = T.codomain.empty()
   T(x, out=out)
   back = T.domain.empty()
   T.adjoint(out, out=back)

An in-place implementation must overwrite its entire output buffer and
return **the same** ``out`` object. If domain and codomain permit aliasing,
protect inputs when an output overlaps them. For specialized array operators,
NumPy's ``out=`` support often makes this easy.

.. _nonlinear_operator_tutorial:

Nonlinear operators and linearization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a nonlinear :math:`F:X\to Y`, implement ``_eval(x,
differentiate=False)``. If ``differentiate=True``, cache the quantities
needed for the Fréchet derivative. Implement ``_derivative(h)`` and
``_adjoint(y)`` to apply :math:`F'[x]` and its adjoint at the most recent
linearization point. The corresponding in-place hooks are ``_ieval``,
``_iderivative`` and ``_iadjoint``.

.. code-block:: python

   value, DF = F.linearize(x)
   directional_change = DF(h)
   backprojection = DF.adjoint(y)

The derivative object is tied to the current linearization state. A later
evaluation of the original nonlinear operator may invalidate it. If
linearizations at distinct points must coexist, use independent operator
instances or copies with independent state.

For example, the nonlinear mapping :math:`x\mapsto |\mathcal Fx|^2`
(phase-retrieval data) already has an implementation using existing RegPy
operators. There is no need to write its derivative or adjoint yourself:

.. code-block:: python

   from regpy.vecsps import UniformGridFcts
   from regpy.operators import FourierTransform, SquaredModulus

   grid = UniformGridFcts((-1, 1, 64), (-1, 1, 64), dtype=complex)
   fft = FourierTransform(grid, centered=True)
   intensity = SquaredModulus(fft.codomain) * fft

RegPy applies the chain rule automatically when such a composition is
linearized. 

.. _operator_tests_tutorial:

Testing operators
~~~~~~~~~~~~~~~~~

**Test every new operator.** For a linear :math:`T`, the first requirement
is the standard adjoint identity

.. math::

   \operatorname{Re}\langle Tx,y\rangle
     =\operatorname{Re}\langle x,T^*y\rangle.

For the cumulative-sum example, a direct check also works over complex
coefficients:

.. code-block:: python

   Tc = CumulativeSumExtrapolated(3, 5, dtype=complex)
   rng = np.random.default_rng(7)
   x = rng.normal(size=3) + 1j * rng.normal(size=3)
   y = rng.normal(size=5) + 1j * rng.normal(size=5)
   assert np.allclose(
       np.vdot(Tc(x), y),
       np.vdot(x, Tc.adjoint(y)),
   )

RegPy contains the dedicated utilities
:func:`regpy.util.operator_tests.test_adjoint` and
:func:`regpy.util.operator_tests.test_derivative`:

.. code-block:: python

   from regpy.util.operator_tests import test_adjoint, test_derivative

Use ``test_adjoint`` for **every newly implemented linear operator** and for
the derivatives of nonlinear operators. Use ``test_derivative`` for **every
new nonlinear operator**, preferably with several representative points and
step sizes. Also check ``out=`` buffers and the behaviour
at degenerate inputs. These tests check vector-space adjoints; Hilbert-space
adjoints with nontrivial Gram operators should be checked separately.

Existing operators in RegPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before implementing a new operator, consult the
:doc:`full API reference <autoapi/regpy/index>`. Examples include:

* **Elementary linear operators:** :class:`regpy.operators.base.Identity`,
  :class:`regpy.operators.base.PtwMultiplication`,
  :class:`regpy.operators.base.CoordinateProjection` and
  :class:`regpy.operators.base.CoordinateMask`.
* **Linear observations of complex data:**
  :class:`regpy.operators.base.RealPart` and
  :class:`regpy.operators.base.ImaginaryPart`.
* **Nonlinear pointwise maps:**
  :class:`regpy.operators.base.SquaredModulus`,
  :class:`regpy.operators.numpy.Exponential`, and
  :class:`regpy.operators.numpy.NemytskiiOperator`. A Nemytskii
  operator applies a supplied point function at each spatial location;
  its constructor also takes the pointwise (real) Jacobian. 
* **Linear algebra and transforms:**
  :class:`regpy.operators.numpy.MatrixMultiplication`,
  :class:`regpy.operators.numpy.FourierTransform`,
  :class:`regpy.operators.numpy.ForwardFDGradient`.
* **Convolution and differential operators:**
  :class:`regpy.operators.convolution.ConvolutionOperator`,
  :class:`regpy.operators.convolution.PaddingOperator`,
  :class:`regpy.operators.convolution.TruncationOperator`,
  :class:`regpy.operators.convolution.Derivative`, and
  :class:`regpy.operators.convolution.Laplacian`.
* **NGSolve integration:**
  :class:`regpy.operators.ngsolve.NgsOperator` and
  :class:`regpy.operators.ngsolve.SecondOrderEllipticCoefficientPDE`
  (requires NGSolve).

The classes above are imported through ``regpy.operators`` (subject to
optional dependencies), even though their implementations are organized
into submodules.

.. _building_operators:

Building complex operators from simple operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combining existing, tested operators is usually preferable to repeating
implementation and adjoint calculations. The operator algebra automatically
manages compatible domain/codomain spaces and propagates derivatives through
compositions.

.. _operator_compositions_tutorial:

Compositions
------------

If :math:`B:X\to Z` and :math:`A:Z\to Y`, then

.. math::

   (A\circ B)(x)=A(B(x)),\qquad (A\circ B)^*=B^*A^*.

Write ``A * B`` or use
:class:`regpy.operators.base.Composition` explicitly:

.. code-block:: python

   from regpy.operators import Composition

   chain = A * B                 # B is evaluated first
   chain = Composition(A, B, use_buffers=True)

``Composition`` can reuse intermediate buffers (``use_buffers=True`` by
default); set ``chain.use_buffers = False`` to disable this. Buffer reuse
is especially useful when each operator acts on large arrays. Because
buffers and nonlinear linearization caches are mutable, a buffered
composition should not be evaluated concurrently in multiple threads
without separate instances.

Compositions can also explicitly represent a known singular-value
decomposition :math:`T=V\Sigma U^*`. The method
:meth:`regpy.operators.base.Composition.label_as_svd` marks and (partly) verifies
such a decomposition using the position of the positive pointwise
singular-value multiplier:

.. code-block:: python

   # If C = Composition(V, Sigma, U_adjoint) is a known SVD:
   C.label_as_svd(l=2, G_X=1., G_Y=1.)
   assert C.is_svd

Here ``l`` is **one-based**; the method checks the isometry conditions on a
sample before labeling the factorization. It does not compute the SVD of
an arbitrary operator. When a composition is already labeled, its SVD
information is available through ``C.svd_data``.

.. _operator_vectors_tutorial:

VectorOfOperators, operator matrices, and parallel execution
------------------------------------------------------------

If :math:`T_i:X\to Y_i`, the class
:class:`regpy.operators.base.VectorOfOperators` forms

.. math::

   Vx=(T_1x,\ldots,T_px),\qquad
   V:X\to\bigoplus_{i=1}^{p}Y_i.

.. code-block:: python

   from regpy.operators import VectorOfOperators

   V = VectorOfOperators([T1, T2, T3])  # common input space
   y = V(x)                             # direct-sum output

The adjoint sums the contributions of all branches:
:math:`V^*(y_1,\ldots,y_p)=\sum_iT_i^*y_i`.

The class :class:`regpy.operators.base.MatrixOfOperators` handles block
operator matrices. For :math:`M_{ij}:X_j\to Y_i`, its action is
:math:`(Mx)_i=\sum_jM_{ij}x_j`; ``None`` denotes a zero block.
**Important for this RegPy version:** ``MatrixOfOperators`` receives
**columns as the outer sequence**, even though traditional mathematical
notation is row-first. For example:

.. code-block:: python

   from regpy.operators import MatrixOfOperators

   M = MatrixOfOperators([
       [A11, A21],  # blocks from input X1 to outputs Y1 and Y2
       [A12, A22],  # blocks from input X2 to outputs Y1 and Y2
   ])

For a block-diagonal mapping of independent input components, use
:class:`regpy.operators.base.DirectSum`.

**Actual parallel variants.** RegPy supplies the multiprocessing classes
:class:`regpy.operators.parallel_operators.ParallelVectorOfOperators`
and :class:`regpy.operators.parallel_operators.DistributedVectorOfOperators`.
``ParallelVectorOfOperators`` has the same mathematical action as
``VectorOfOperators`` but evaluates its components in separate worker
processes. ``DistributedVectorOfOperators`` additionally selects different
subsets of the inputs from a direct-sum domain, according to a
``distribution_mat`` (pass a NumPy array; the implementation accesses its
``shape``). Process resources should be managed using
:class:`regpy.operators.parallel_operators.ParallelExecutionManager`:

.. code-block:: python

   from regpy.operators import (
       ParallelVectorOfOperators, ParallelExecutionManager,
   )

   # In a normal Python script, inside if __name__ == "__main__":
   with ParallelExecutionManager():
       P = ParallelVectorOfOperators([T1, T2])
       result = P(x)

The operators must have the same domain; multiprocessing introduces
interprocess communication, so it is most helpful when each application
is computationally expensive. Use the usual multiprocessing main-module
guard, especially on platforms using the ``spawn`` start method.

One of the use cases of ``ParallelVectorOfOperators`` are forward operators 
involving PDE solutions for several right-hand sides
(e.g. incident waves in scattering). Another use case is the parallelization 
of the forward operator in parallel magnetic resonance imaging. 

.. _operator_graphs_tutorial:

Operator graphs
---------------

Compositions describe chains; :class:`regpy.operators.graph_operator.OperatorGraph`
represents a **directed acyclic graph** of operator instances. Its nodes
are managed by :class:`regpy.operators.graph_operator.OperatorNode`, and
:class:`regpy.operators.graph_operator.Edge` routes outputs to inputs.
Unlike two independent compositions, a graph can compute a common
intermediate value once and feed it to several branches. During adjoint
evaluation, contributions from branches are accumulated.

The constructor receives an operator list and an explicit edge list. Each
edge has the form ``((source, [source_output_indices]), (target,
target_input_index))``; ``None`` denotes the graph's external input or
output. The example below uses **two** external inputs, :math:`f` and
:math:`g`, and constructs

.. math::

   G(f,g)=\bigl(\operatorname{Re}(Tf),\,
                   \operatorname{Im}(Tf),\,Tg\bigr).

The first evaluation :math:`Tf` is shared by the real- and
imaginary-part branches, rather than recomputed for each output.

.. code-block:: python

   from regpy.operators import OperatorGraph, RealPart, ImaginaryPart
   from regpy.vecsps import DirectSum

   first = CumulativeSumExtrapolated(3, 5, dtype=complex)
   second = CumulativeSumExtrapolated(3, 5, dtype=complex)
   take_real = RealPart(first.codomain)
   take_imag = ImaginaryPart(first.codomain)

   graph = OperatorGraph(
       operators=[first, second, take_real, take_imag],
       edges=[
           ((None, [0]), (first, 0)),     # f -> first
           ((None, [1]), (second, 0)),    # g -> second
           ((first, [0]), (take_real, 0)),
           ((first, [0]), (take_imag, 0)),
           ((take_real, [0]), (None, 0)),
           ((take_imag, [0]), (None, 0)),
           ((second, [0]), (None, 0)),
       ],
   )

   f = np.array([1 + 1j, 2 + 2j, 3 + 3j])
   g = np.array([2 + 0j, 1 + 1j, 0 + 2j])
   y_real, y_imag, y_second = graph(DirectSum(
       first.domain, second.domain
   ).join(f, g))

   assert np.allclose(y_real, first(f).real)
   assert np.allclose(y_imag, first(f).imag)
   assert np.allclose(y_second, second(g))

Each operator in ``operators`` is a *distinct instance*: nodes are keyed
by operator identity, so adding the same instance twice does not create
two different nodes. For this version of ``OperatorGraph``, make sure each
external input is explicitly connected and the graph is acyclic.

For nonlinear nodes, ``OperatorGraph`` propagates derivatives along its
edges and adjoints in reverse order. As with all operator assemblies,
verify the derivative and adjoint of the **whole graph**, not just those of
its components. See the
:doc:`operator graph API <autoapi/regpy/operators/graph_operator/index>`.

.. _codomain_free_tutorial:

Codomain-free operators and efficient normal actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many iterative regularization methods need normal-equation expressions
without ever requiring an explicit forward vector in the codomain.
There are two major motivations:

* **Large codomain:** measurement vectors may be prohibitively large to
  allocate or materialize (e.g. many measurement directions or a large
  collection of PDE observations).
* **Lower cost:** a direct normal action can eliminate redundant transforms
  or PDE solves even when the codomain fits in memory.

For a linear :math:`T:X\to Y`, the typical operations are :math:`T^*d`
and :math:`T^*Tx`. For a nonlinear :math:`F`, one encounters
:math:`F'[x]^*F(x)` and :math:`F'[x]^*F'[x]h`. RegPy provides the following
interfaces:

.. list-table:: Codomain-free operator methods
   :header-rows: 1
   :widths: 37 38 25

   * - Public operation
     - Computation
     - Optional implementation hook
   * - ``op.adjoint_data(data)``
     - :math:`T^*d`; accepts specialized observation-data representations
     - ``_adjoint_data`` / ``_iadjoint_data``
   * - ``op.adjoint_eval(x)`` (linear operator)
     - :math:`T^*Tx`
     - ``_adjoint_eval`` / ``_iadjoint_eval``
   * - ``F.linearize(x, return_adjoint_eval=True)``
     - Returns :math:`F'[x]^*F(x)` and the derivative operator
     - ``_adjoint_eval`` / ``_iadjoint_eval`` on ``F``
   * - ``DF.adjoint_eval(h)`` for ``DF=F'[x]``
     - :math:`F'[x]^*F'[x]h`
     - ``_adjoint_derivative`` / ``_iadjoint_derivative`` on ``F``

The name ``adjoint_eval`` is a property returning a callable operator for
**linear** maps. The nonlinear combined action is obtained through the
``return_adjoint_eval=True`` flag of ``linearize``. These methods use
**vector-space adjoints**; a Hilbert-space normal operator may also include
Gram factors. The fallback implementation constructs codomain vectors, so
operators must implement optimized hooks if they want to avoid doing so.

**Example: two FFTs rather than four.** For periodic convolution on a
complex-valued uniform grid,

.. math::

   Tf=\mathcal F^{-1}\bigl(m\,\mathcal F f\bigr).

Using ``T.adjoint(T(f))`` typically requires two FFTs for :math:`T` and
two more for :math:`T^*`. When the Fourier transform is unitary in the
relevant discrete pairing, the same normal action is

.. math::

   T^*Tf=\mathcal F^{-1}\bigl(|m|^2\,\mathcal F f\bigr),

which requires just **one forward and one inverse FFT**. This is a
natural use of ``_adjoint_eval``. If the convolution uses padding,
Fourier resizing, or other nonunitary steps, the exact discrete normal
map also contains the adjoints of those steps; blindly replacing the
kernel multiplier by :math:`|m|^2` can give the wrong operator.

When implementing specialized codomain-free operations, first check them
against explicit ``adjoint(eval(...))`` and
``derivative.adjoint(derivative(...))`` evaluations on a small test
problem, including normalization and Gram conventions. The calling
solver must also support codomain-free mode; providing a fast
``_adjoint_eval`` alone does not automatically eliminate all codomain
allocations.

Further reading
~~~~~~~~~~~~~~~

* :doc:`Spaces` — vector-space representations and Hilbert structures.
* :doc:`Functionals` — functionals and their derivatives.
* :doc:`Solvers` — iterative regularization methods.
* :doc:`notebooks/diffusion_coefficient` — PDE-based example.
* :doc:`notebooks/parallel_mri` — parallel MRI example.
* :doc:`full RegPy API <autoapi/regpy/index>` — complete class and method
  documentation, including the operator algebra, operator graphs and
  multiprocessing wrappers.
