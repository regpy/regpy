itreg — Iterative solvers for ill-posed problems
================================================
Introduction
------------
The toolbox provides methods to solve ill-conditioned nonlinear systems of 
equations

.. math :: T(x) = y.

Here  :math:`T:~X \rightarrow Y` is a nonlinear mapping between finite 
dimensional spaces 

.. math :: X = \mathbb{K}^n

and

.. math :: Y = \mathbb{K}^m

where  :math:`\mathbb{K}=\mathbb{R}` or  :math:`\mathbb{K}=\mathbb{C}`.
:math:`X` and  :math:`Y` are equipped with inner products described by Gram
matrices  :math:`G_X` and  :math:`G_Y`:
   
.. math :: \langle x_1,x_2 \rangle_X = x_1^\ast G_X x_2

and

.. math :: \langle y_1,y_2 \rangle_Y = y_1^\ast G_Y y_2.


Usage
-----
We will translate the mathematical expressions from the introduction into the 
respective python expressions.

Space: :class:`Space <itreg.spaces.Space>`
    Spaces like :math:`X` and  :math:`Y` for example are stored in the variable
    Space of class :class:`Space <itreg.spaces.Space>`. It has attributes like
    inner product, norm, Gram matrix product and the inverse Gram matrix
    product stored as attributes. They are called by 
    :attr:`inner <itreg.spaces.Space.inner>`,
    :attr:`norm <itreg.spaces.Space.norm>`, 
    :attr:`gram <itreg.spaces.Space.gram>` and 
    :attr:`gram_inv <itreg.spaces.Space.gram_inv>`, respectively.
    
Op: :class:`Operator <itreg.operators.Operator>`
    This is the nonlinear mapping  :math:`T` between the spaces  :math:`X` and
    :math:`Y`. The spaces are saved in Op as attributes 
    :attr:`domx <itreg.operators.Operator.domx>` and 
    :attr:`domy <itreg.operators.Operator.domy>`, respectively. The evaluation 
    of the operator can be called by Op(x), where  :math:`x\in X`. Furthermore,
    the attribute :attr:`derivative <itreg.operators.Operator.derivative>` of 
    Op is implemented. It returns an object of class
    :class:`Operator <itreg.operators.Operator> again. There is a subclass 
    :class:`LinearOperator <itreg.operators.LinearOperator>` that
    provides more functions for linear operators. It has additional attributes
    like :attr:`adjoint <itreg.operators.LinearOperator.adjoint>`,
    :attr:`abs_squared <itreg.operators.LinearOperator.abs_squared>` and 
    :attr:`norm <itreg.operators.LinearOperator.norm>`.

Solver: :class:`Solver <itreg.solvers.Solver>`
    An object of type Solver is used to solve the above equation in the
    introduction. To solve such an equation, one has to define several things:
    One has to define the object of type :class:`Solver <itreg.solvers.Solver>`
    with all its parameters. Then one can use the attribute :attr:`run` to
    compute the solution.
    
Stoprule: :class:`Stoprule <itreg.stoprules.Stoprule>`
    Some solvers cannot stop on their own and need stoprules as arguments in
    Solver.run(stoprule=stoprule).
    
For detailed examples for every solver see the examples in
/itreg/examples.

Submodules
~~~~~~~~~~
Solvers of class :class:`Solver <itreg.solvers.Solver>`:
    :class:`Landweber <itreg.solvers.Landweber>`,
    :class:`IRGNM_CG <itreg.solvers.IRGNM_CG>`,
    :class:`IRGNM_L1_fid <itreg.solvers.IRGNM_L1_fid>`,
    :class:`IRNM_KL <itreg.solvers.IRNM_KL>`,
    :class:`IRNM_KL_Newton <itreg.solvers.IRNM_KL_Newton>`,
    :class:`Newton_CG <itreg.solvers.Newton_CG>`.

Spaces of class :class:`Space <itreg.spaces.Space>`:
    :class:`L2 <itreg.spaces.L2>`.

Operators of class :class:`Operator <itreg.operators.Operator>`:
    :class:`Volterra <itreg.operators.Volterra>`,
    :class:`WeightedOp <itreg.operators.WeightedOp>`.

Inner Solvers of class :class:`Inner Solver <itreg.innersolvers.Inner_Solver>`:
    :class:`SQP <itreg.innersolvers.SQP>`.

Stoprules of class :class:`StopRule <itreg.stoprules.StopRule>`:
    :class:`CountIterations <itreg.stoprules.CountIterations>`,
    :class:`Discrepancy <itreg.stoprules.Discrepancy>`.
    :class:`CombineRules <itreg.stoprules.CombineRules>`.

Utilities:
    :class:`CGNE_reg <itreg.util.CGNE_reg>`,
    :class:`CG <itreg.util.CG>`,
    :class:`test_adjoint <itreg.util.test_adjoint>`.

Logging
~~~~~~~
The standard python :mod:`logging` framework is used. Most classes and utility
functions use this. Abstract base classes usually take a `log` parameter and
expose the loggers as attributes, while derived classes default to using a
logger named by their module. E.g. :class:`Landweber <itreg.solvers.Landweber>`
has an attribute :attr:`log` that defaults to the `itreg.solvers.landweber`
logger. These defaults are not always indicated in the docstrings.

Most diagnostic messages, i.e. status information about solver iterations, are
emitted at log level :const:`logging.INFO`.

So far this feature is only implemented in 
:class:`Landweber <itreg.solvers.Landweber>`.

Coding style
------------

Please follow the `Official NumPy documentation guidelines`_, with the follwing
deviations:

- Write the constructor documentation in the `__init__` docstring, not in the
  class docstring.

http://docutils.sourceforge.net/docs/user/rst/demo.txt
https://www.python.org/dev/peps/pep-0008/


.. _Official NumPy documentation guidelines: 
    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt