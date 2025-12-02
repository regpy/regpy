Regularization
==============

We have slightly touched on the topic how to use the solvers of `RegPy` in the section :ref:`/usage.rst#regularisation-and-solvers`. In this section we will go into more detail about the regularization methods implemented in `RegPy`.

The solvers in `RegPy` are implemented as classes that inherit from the base class :class:`Solver`. The base class provides a common interface for all solvers, allowing you to easily switch between different methods without changing your code. In particular, can you define one :class:`regpy.solvers.Setting` and reuse it for different solvers.

First we have the base class :class:`Solver` and its derivate :class:`RegSolver` which is used for many explicit regularization methods. Since the :class:`RegSolver` is a subclass that explicitly incorporates the regularization setting :class:`Setting`, it is the most commonly used solver in `RegPy`. However, there are two solvers which do not use the regularization setting that is the Richardson-Lucy solver :class:`RichardsonLucy` and the :math:`L^1` type regularization in :class:`IrgnmL1Fid`. Thus for almost all regularization methods your are required to define a regularization setting, which is composed of the following three components:

- the Operator
- the penalty functional
- the data fidelity functional

.. code-block:: python

    from regpy.solvers import Setting
    setting = Setting(
        operator=operator,
        penalty=penalty_functional,
        data_fid=data_fidelity_functional
    )

Note that both the penalty functional and the data fidelity functional are functionals in the sense of :class:`Functional`. However, if you want to use any squared Hilbert space norm for one of these functionals, you may simply pass the respective Hilbert space when initializing the regularization setting. The initialization then takes care of the treatment. As a general suggestion, you should use abstract functionals or Hilbert spaces in the definition of the regularization setting, since this allows `RegPy` to automatically choose the correct implementation for your domain and codomain respectively.

.. code-block:: python

    from regpy.hilbert import L2, H1
    from regpy.solvers import Setting
    setting = Setting(
        operator=operator,
        penalty=L2,
        data_fid=H1
    )

.. code-block:: python

    from regpy.functionals import L1, TV
    from regpy.solvers import Setting
    setting = Setting(
        operator=operator,
        penalty=TV,
        data_fid=L1
    )

However, you may want to check whether there exists an implementation for the respective functional and ore Hilbert space for your domain or codomain.

Logging of the Regularization
-----------------------------

There is a native logging of the regularization of any solver. This is done by the `logging` and the output can be modified as usual using [logging](https://docs.python.org/3/library/logging.html).
