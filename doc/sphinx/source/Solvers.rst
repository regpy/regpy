.. _solvers-and-stoprules:

Solvers and stopping rules
==========================

This tutorial explains how to implement an iterative method and control it
with a stopping rule. We use the Jacobi method for a finite-dimensional linear
system as a complete, executable example. No regularization setting is needed
for this example; the connection to inverse problems is described at the end.

An iterative algorithm and its stopping criterion have different tasks.
A :class:`regpy.solvers.Solver` computes the next iterate. A
:class:`regpy.stoprules.StopRule` decides whether to stop and can also collect
information for monitoring. Keeping them separate allows one algorithm to be
used with different accuracy criteria, iteration limits, and parameter-choice
rules. Even a solver-specific stopping rule benefits from this separation.

Implementing a solver
---------------------

For a square matrix :math:`A` with nonzero diagonal, write :math:`A=D+R`,
where :math:`D` is diagonal. The Jacobi iteration for :math:`Ax=b` is

.. math::

   x_{n+1}=D^{-1}(b-Rx_n).

A solver stores its current iterate in ``self.x``. Its ``self.y`` attribute
stores an associated operator value when one is available. The Jacobi update
does not need :math:`Ax_n`, so we set ``self.y = None`` and avoid computing it.

Initialize the base class with the starting pair and implement one update in
``_next()``. This method modifies the solver state and does not return an
iterate; the base class supplies the iteration interface and step counter.

.. code-block:: python

   import numpy as np
   from regpy.solvers import Solver

   class Jacobi(Solver):
       def __init__(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray | None = None):
           super().__init__(x=x0 if x0 is not None else np.zeros_like(b), y=None)
           # As the operator values y_n= Ax_n are not needed for the Jacobi method, 
           # we do not compute them here and set self.y to None.
           self.b = b
           self.diagA = np.diag(A)
           self.R = A - np.diag(self.diagA)

       def _next(self) -> None:
           self.x = (self.b - self.R @ self.x) / self.diagA

This example uses NumPy arrays and assumes that ``A``, ``b``, and ``x0`` have
compatible shapes. General RegPy solvers can work with other vector backends;
see :doc:`Spaces` for their interfaces.

The initial state has ``iteration_step_nr == 0``. The public method
``solver.next()`` increments the counter and calls ``_next()``. A method that
recognizes convergence internally may call ``self.converge()``; later calls to
``solver.next()`` then return ``False`` without another update. Jacobi leaves
the stopping decision to an external rule.

Manual iteration
----------------

First construct a strictly diagonally dominant example. Its reference solution
is used only to illustrate and check the error; it is not required by either
the iteration or the stopping rule below.

.. code-block:: python

   A = np.array([[4, -1, 0, 0],
                 [-1, 4, -1, 0],   
                 [0, -1, 4, -1],
                 [0, 0, -1, 3]], dtype=float)
   b = np.array([15, 10, 10, 10], dtype=float)
   xref = np.linalg.solve(A, b)
   solver = Jacobi(A, b)

Calling ``iter(solver)`` creates a generator. Advancing it performs a step and
yields the current ``(x, y)`` pair. The initial guess is not yielded by this
interface; it is available as ``solver.x`` before the first step.

.. code-block:: python

   it = iter(solver)
   for _ in range(10):
       x_n, _ = next(it)
       error = np.linalg.norm(x_n - xref, ord=np.inf)
       print(f"Iteration {solver.iteration_step_nr}: error = {error:.6g}")

In a notebook, ``next(it)`` can also be evaluated repeatedly in a separate
cell to inspect individual updates. Creating another iterator for the same
solver continues from its current state; it does not restart the algorithm.
Create a new ``Jacobi(A, b)`` instance to restart from zero.

The yielded NumPy arrays are read-only views, not independent snapshots.
Another solver may update its arrays in place, so copy an iterate if it must
be retained unchanged. For this NumPy example, use ``x_n.copy()``.

Deriving a stopping criterion
----------------------------

The Jacobi fixed-point map has contraction factor

.. math::

   q=\lVert D^{-1}R\rVert_\infty
    =\max_j\sum_{\ell\ne j}\left|\frac{a_{j\ell}}{a_{jj}}\right|.

If :math:`q<1`, Banach's fixed-point theorem gives the a-posteriori bound

.. math::

   \lVert x_n-x\rVert_\infty
   \leq \frac{q}{1-q}\lVert x_n-x_{n-1}\rVert_\infty,
   \qquad n\geq 1.

We stop when the right-hand side is strictly smaller than a prescribed
absolute tolerance. This criterion requires no exact solution or additional
matrix-vector product. The estimate is an exact-arithmetic result; it does
not account for floating-point rounding errors.

If :math:`q\geq1`, this particular estimate cannot certify an error bound.
That does not by itself imply that Jacobi diverges: a different norm or
criterion may still be useful. The following rule rejects this case explicitly.

Implementing a StopRule
----------------------

A custom rule overrides three hooks:

* ``_complete_init_with_solver(solver)`` is called when the solver attaches the
  rule. Use it for solver-dependent initialization and call the base hook.
* ``_stop()`` reads the current solver state and returns a boolean. The public
  ``stop()`` method handles triggering and the base class's bookkeeping.
* ``reset()`` clears any additional state so the rule can be reused. Call the
  base method to reset its state too.

Initialization at attachment time lets a rule be constructed before its solver
is known. It does not automatically eliminate circular imports: this tutorial
can use ``isinstance(solver, Jacobi)`` because both classes are defined in the
same example.

.. code-block:: python

   from regpy.stoprules import StopRule


   class JacobiStopping(StopRule):
       """Stop when the a-posteriori bound for the current iterate is below tol.

       Parameters
       ----------
       tol: float [default: 1e-6]
           Positive absolute tolerance for the error in the supremum norm.
       logging_level: int | str [default: "INFO"]
           Logging level passed to StopRule.

       The attached solver must be a Jacobi solver with contraction factor q < 1.
       The rule must be checked at every iteration. It needs two consecutive
       iterates, so it cannot certify the initial guess. 
       """

       def __init__(self, tol: float = 1e-6,
                    logging_level: int | str = "INFO") -> None:
           super().__init__(logging_level=logging_level)
           if not np.isfinite(tol) or tol <= 0:
               raise ValueError("tol must be finite and strictly positive.")
           self.tol = float(tol)
           self.q: float | None = None
           self.error_bound = np.inf
           self._previous_x: np.ndarray | None = None
           self._previous_step: int | None = None

       def _complete_init_with_solver(self, solver: Jacobi) -> None:
           """Initialize the contraction factor and the first snapshot."""
           if not isinstance(solver, Jacobi):
               raise TypeError("JacobiStopping requires a Jacobi solver.")
           if np.any(solver.diagA == 0):
               raise ValueError("The Jacobi diagonal must be nonzero.")
           q = float(np.max(np.sum(
               np.abs(solver.R / solver.diagA[:, None]), axis=1
           )))
           if not np.isfinite(q) or q >= 1:
               raise ValueError(f"The error bound requires q < 1; got q={q}.")
           super()._complete_init_with_solver(solver)
           self.q = q
           self.error_bound = np.inf
           # A snapshot is essential: a solver may modify its iterate in place.
           self._previous_x = np.array(solver.x, copy=True)
           self._previous_step = solver.iteration_step_nr

       def _stop(self) -> bool:
           if self.solver is None or self._previous_x is None:
               raise RuntimeError("Attach the rule to a solver before checking it.")
           step = self.solver.iteration_step_nr
           if step == self._previous_step:
               # Includes the initial check; repeated checks do not create a
               # spurious zero difference. 
               return bool(self.error_bound < self.tol)
           if step != self._previous_step + 1:
               raise RuntimeError("JacobiStopping must be checked at every iteration.")

           change = np.linalg.norm(self.solver.x - self._previous_x, ord=np.inf)
           self.error_bound = float(self.q / (1 - self.q) * change)
           self._previous_x = np.array(self.solver.x, copy=True)
           self._previous_step = step
           return bool(self.error_bound < self.tol)

       def reset(self) -> None:
           super().reset()
           self.q = None
           self.error_bound = np.inf
           self._previous_x = None
           self._previous_step = None

Several details matter for a reliable criterion:

* At attachment, the rule copies the current iterate and remembers its step
  number. This also works when attaching it to a solver that has already run.
* At the initial check, no pair of consecutive iterates is available. The
  bound remains infinite, so the rule cannot stop prematurely.
* Rechecking the same step reuses its bound. Comparing the iterate with its
  saved copy again would otherwise produce a spurious zero error.
* The previous iterate is copied because solvers may modify their arrays in
  place. A reference alone would not preserve the previous values.
* The rule must be checked at every step. Sparse checking schedules would
  compare nonconsecutive iterates, for which the displayed bound is not the
  same estimate. The step-number check detects skipped updates.

There is no need to override ``best_iterate()`` here: the default selects the
iterate at which the rule triggers. Other rules, such as parameter-choice
rules, may select a different previously computed iterate.

Running and monitoring
----------------------

Use ``while_`` when intermediate iterates are of interest. It checks the
stopping rule at the current state before attempting another step, including
at the initial state. Each completed step is yielded before the next check.
The following example starts a fresh solver and records actual errors only
for demonstration:

.. code-block:: python

   from matplotlib import pyplot as plt

   solver = Jacobi(A, b)
   rule = JacobiStopping(tol=1e-6)
   steps, errors = [], []
   for x, _ in solver.while_(rule, max_it=1000):
       steps.append(solver.iteration_step_nr)
       errors.append(np.linalg.norm(x - xref, ord=np.inf))

   plt.semilogy(steps, errors)
   plt.xlabel("Iteration")
   plt.ylabel("Error in the supremum norm")
   plt.title("Convergence of the Jacobi method")
   plt.show()

Here the rule reaches its tolerance after 19 iterations. Its final error
bound is approximately :math:`6.70\times10^{-7}`, while the actual error is
approximately :math:`4.71\times10^{-7}`.

The ``max_it`` argument adds an independent iteration cap: the loop stops
when either criterion triggers. Reaching that cap alone does not certify
the requested accuracy. Check ``rule.error_bound < rule.tol`` if the guarantee
is required. Matplotlib is needed only for this plotting example.

If only the selected final iterate is needed, use ``run``:

.. code-block:: python

   solver = Jacobi(A, b)
   rule = JacobiStopping(tol=1e-6)
   x, _ = solver.run(rule, max_it=1000)

   print(f"Iterations: {solver.iteration_step_nr}")
   print(f"Error bound: {rule.error_bound:.6g}")
   assert rule.error_bound < rule.tol

For this rule, ``run`` returns the current ``(x, y)`` pair, with ``y=None``.
In general, it returns the result selected by ``best_iterate()``, rather than
necessarily the last pair yielded by the loop. Treat returned vectors as
read-only and copy them before modification.

The related ``until`` interface attempts a first step before checking the rule,
then continues like ``while_``. Both methods return generators; their execution
starts when the generator is advanced. Stopping consumption early also stops
the driver, without implying that the rule's tolerance has been reached.

To reuse the same rule in a new run, call ``rule.reset()`` before attaching it
to a fresh solver. Resetting the rule does not reset the solver.

Combining criteria and recording history
---------------------------------------

The ``+`` operator combines stopping rules with an OR condition. For example,
the following is the explicit counterpart of using ``max_it=1000``:

.. code-block:: python

   from regpy.stoprules import CountIterations

   solver = Jacobi(A, b)
   accuracy_rule = JacobiStopping(tol=1e-6)
   rule = accuracy_rule + CountIterations(max_iterations=1000)
   x, _ = solver.run(rule)

Rules can also monitor quantities. To record the bound inside
``JacobiStopping``, initialize a history entry after calling the base
constructor:

.. code-block:: python

   self.history_dict["error bound"] = []

Then append each newly computed bound in ``_stop()``, after assigning
``self.error_bound`` and before returning:

.. code-block:: python

   self.history_dict["error bound"].append(self.error_bound)

Place this after the repeated-step guard so each bound is recorded only once.
The base class associates appended values with iteration numbers in
``history_step_dict`` and clears histories on ``reset()``. This permits
monitoring without storing all solution vectors.

Landweber iteration as a RegSolver
---------------------------------

A :class:`regpy.solvers.RegSolver` provides the same iteration interface as
``Solver``, but additionally receives a :class:`regpy.solvers.Setting`. Its
base constructor makes the operator and Hilbert spaces available as
``self.op``, ``self.h_domain``, and ``self.h_codomain``. The separate
:doc:`Settings` tutorial describes how to construct settings.

As a simple example, consider Landweber iteration for a linear equation
:math:`Tx=g` in Hilbert spaces :math:`X` and :math:`Y`:

.. math::

   x_{n+1}=x_n-\tau T^\ast(Tx_n-g).

Here :math:`T^\ast` is the Hilbert-space adjoint. RegPy's ``op.adjoint``
represents the adjoint with respect to the underlying vector-space pairings,
so the Hilbert-space gradient is computed as

.. math::

   T^\ast(Tx_n-g)=G_X^{-1}T'G_Y(Tx_n-g).

The step length :math:`\tau` is supplied by the user. For a nonzero bounded
linear operator, the standard convergence condition is
:math:`0<\tau<2/\lVert T\rVert_{X\to Y}^2`. The following example performs
no norm estimation or automatic step-length selection.

.. code-block:: python

   from regpy.solvers import RegSolver, Setting
   from regpy.vecsps import ReadOnlyVector


   class SimpleLandweber(RegSolver):
       """Landweber iteration for a linear operator in Hilbert spaces.

       Parameters
       ----------
       setting: Setting
           Hilbert-space least-squares problem with explicit data and codomain,
           without a fixed Tikhonov regularization parameter.
       stepsize: float
           User-provided positive step length satisfying the convergence bound.
       init: ReadOnlyVector | None [default: None]
           Initial guess. If omitted, start at zero.
       """

       def __init__(self, setting: Setting, stepsize: float,
                    init: ReadOnlyVector | None = None) -> None:
           x = setting.op.domain.zeros() if init is None else init
           super().__init__(setting, x=x, y=setting.op(x))
           self.stepsize = stepsize
           self.data = setting.data

       def _next(self) -> None:
           residual = self.y - self.data
           gradient = self.h_domain.gram_inv(
               self.op.adjoint(self.h_codomain.gram(residual))
           )
           self.x = self.x - self.stepsize * gradient
           self.y = self.op(self.x)

This intentionally small implementation assumes a linear operator, explicit
vector-valued data, and a squared Hilbert-norm data fidelity. The penalty
supplies the domain geometry; its gradient is not added to the update. Thus
this is an iterative regularization method, with regularization controlled by
stopping the iteration, rather than a solver for a fixed Tikhonov objective.

Unlike the Jacobi example, it maintains ``self.y = self.op(self.x)`` so that
stopping rules can use the current predicted data. Every update creates new
vectors for clarity; there are no preallocated work arrays. The initial guess
is only read, and the first update assigns a new vector to ``self.x``.

For a runnable example, reuse the matrix ``A`` and right-hand side ``b`` above:

.. code-block:: python

   from regpy.hilbert import L2
   from regpy.operators import MatrixMultiplication

   setting = Setting(op=MatrixMultiplication(A), penalty=L2, data_fid=L2, data=b)
   stepsize = 1 / np.linalg.norm(A, ord=2)**2
   solver = SimpleLandweber(setting, stepsize=stepsize)
   x, y = solver.run(max_it=200)

   print(f"Solution error: {np.linalg.norm(x - xref):.6g}")
   print(f"Residual norm: {np.linalg.norm(y - b):.6g}")

The norm in this example is the matrix spectral norm, appropriate for these
Euclidean ``L2`` spaces. For other Hilbert geometries, use the corresponding
operator norm when choosing the step length. ``max_it`` supplies an iteration
cap, not an accuracy guarantee. The ``while_`` interface and combinations of
stopping rules work exactly as in the Jacobi example; for noisy inverse
problems, a discrepancy principle is a typical choice.

Logging
-------

Solvers use Python's :mod:`logging` module. Configure log levels, handlers, and
formatters through the standard logging API; see the `Python logging
documentation <https://docs.python.org/3/library/logging.html>`_.

Further reading
---------------

For operator-based inverse problems, :class:`regpy.solvers.RegSolver` extends
this solver interface. The separate :doc:`Settings` tutorial explains how to
assemble the underlying regularization problem.
