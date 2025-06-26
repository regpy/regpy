Space Structures
================

**The current vector space architecture is undergoing a significant redesign to support a more flexible and extensible framework.**

At the core of this framework are vector space structures, represented by the base class \:class:`VectorSpace`. Currently, vectors are assumed to be instances of `numpy.ndarray`. While complex-valued vectors are supported, they are interpreted as vector spaces over the **real numbers**.

The inner product, as defined in the \:class:`HilbertSpace` structure, is given by:

.. math::
    \Re\left(\overline{x}^T G_{\underline{\mathbb{X}}} y\right)

This defines a real-valued inner product, making Hilbert spaces the first level of structure applied to vector spaces in this framework.

Functionals
~~~~~~~~~~~

In addition, `regpy` introduces the notion of convex real **functionals** via the class \:class:`Functional`. These include, for example, norms or regularization terms. Functionals provide a second layer of structure by mapping vectors to real values, typically in the context of optimization.

Abstract Spaces
~~~~~~~~~~~~~~~

`Regpy` provides abstract base classes for Hilbert spaces and functionals. These abstract definitions can then be instantiated with concrete vector spaces to produce problem-specific implementations.

.. code-block:: python

    from regpy.hilbert import H1
    h1_on_my_space = H1(my_domain)

    from regpy.functionals import TV 
    tv_on_my_space = TV(my_domain)

These components enable the composition of rich mathematical structures on top of simple numerical vector spaces, paving the way for general and reusable algorithms in inverse problems and variational methods.
