itreg — Iterative solvers for ill-posed problems
=================================================

Introduction
------------

Usage
-----

Submodules
~~~~~~~~~~

Logging
~~~~~~~

The standard python :mod:`logging` framework is used. Most classes and utility
functions use this. Abstract base classes usually take a `log` parameter and
expose the loggers as attributes, while derived classes default to using a
logger named by their module. E.g. :class:`itreg.solvers.landweber.Landweber`
has an attribute :attr:`log` that defaults to the `itreg.solvers.landweber`
logger. These defaults are not always indicated in the docstrings.

Most diagnostic messages, i.e. status information about solver iterations, are
emitted at log level :const:`logging.INFO`.

Coding style
------------

Please follow the `Official NumPy documentation guidelines`_, with the follwing deviations:

- Write the constructor documentation in the `__init__` docstring, not in the
  class docstring.

http://docutils.sourceforge.net/docs/user/rst/demo.txt
https://www.python.org/dev/peps/pep-0008/


.. _Official NumPy documentation guidelines: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
