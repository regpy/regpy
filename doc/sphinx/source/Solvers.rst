Regularization and solvers
==========================

The :ref:`usage guide <regularisation-and-solvers>` introduces RegPy's solver
interface. This section describes the underlying regularization setting in
more detail.

Solvers inherit from :class:`regpy.solvers.Solver`, which provides a common
iteration interface. Most regularization methods inherit from
:class:`regpy.solvers.RegSolver` and operate on a
:class:`regpy.solvers.Setting`. A setting combines

* a forward operator,
* a penalty functional, and
* a data-fidelity functional.

A typical construction is:

.. code-block:: python

   from regpy.solvers import Setting

   setting = Setting(
       op=operator,
       penalty=penalty_functional,
       data_fid=data_fidelity_functional,
   )

Both ``penalty`` and ``data_fid`` may be concrete functionals. A Hilbert space
may be supplied instead, in which case RegPy uses the corresponding squared
Hilbert-space norm. Abstract functionals and Hilbert spaces are often more
convenient because they select an implementation appropriate for the operator's
domain or codomain:

.. code-block:: python

   from regpy.hilbert import H1, L2
   from regpy.solvers import Setting

   setting = Setting(
       op=operator,
       penalty=L2,
       data_fid=H1,
   )

Likewise, abstract functionals can be used directly:

.. code-block:: python

   from regpy.functionals import L1, TV
   from regpy.solvers import Setting

   setting = Setting(
       op=operator,
       penalty=TV,
       data_fid=L1,
   )

The chosen abstract objects must have concrete implementations for the
operator's domain and codomain. Construction of the setting raises an error if
no suitable implementation is registered.

Logging
-------

Solvers use Python's :mod:`logging` module. Configure log levels, handlers, and
formatters through the standard logging API; see the `Python logging
documentation <https://docs.python.org/3/library/logging.html>`_.
