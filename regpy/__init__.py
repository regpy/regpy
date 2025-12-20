r"""RegPy: Python tools for regularization methods
==============================================

.. image:: https://img.shields.io/github/v/release/regpy/regpy?label=latest%20release&logo=github
   :target: https://github.com/regpy/regpy
   :alt: GitHub release

.. image:: https://zenodo.org/badge/215324707.svg
   :target: https://doi.org/10.5281/zenodo.16837824
   :alt: DOI

.. image:: https://img.shields.io/readthedocs/latest?logo=read-the-docs&logoColor=white
   :target: https://num.math.uni-goettingen.de/regpy/
   :alt: Read the Docs


.. image:: https://img.shields.io/pypi/v/regpy?color=blue&label=latest%20PyPI%20version&logo=pypi&logoColor=white
   :target: https://pypi.org/project/regpy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/implementation/regpy?logo=pypi&logoColor=white
   :target: https://pypi.org/project/regpy/
   :alt: PyPI implementation


.. image:: https://img.shields.io/github/actions/workflow/status/regpy/regpy/pypi.yml?branch=release&label=build%20PyPI%20Release&logo=github
   :target: https://github.com/regpy/regpy/actions/workflows/pypi.yml
   :alt: GitHub Actions PyPI build

.. image:: https://img.shields.io/github/actions/workflow/status/regpy/regpy/docker-deploy.yml?branch=release&label=build%20Docker%20Image&logo=github
   :target: https://github.com/regpy/regpy/actions/workflows/docker-deploy.yml
   :alt: GitHub Actions Docker build


.. image:: https://img.shields.io/pypi/dm/regpy?label=PyPI%20downloads&logo=pypi&logoColor=white
   :target: https://pypi.org/project/regpy/
   :alt: PyPI downloads

.. image:: https://img.shields.io/docker/pulls/regpy/regpy?logo=docker&logoColor=white
   :target: https://hub.docker.com/repository/docker/regpy/regpy
   :alt: Docker pulls


``RegPy`` is a Python library for implementing and solving ill-posed inverse problems,
developed at the
`Institute for Numerical and Applied Mathematics Goettingen <https://num.math.uni-goettingen.de>`_.
It provides tools to implement your own forward models, both linear and non-linear,
and a variety of regularization methods that can be stopped using common stopping rules.

This project is currently in an almost beta-quality state. However, it is still under
intensive development. Therefore, expect bugs and partially undocumented tools.
If you encounter any issues, we welcome reports on our
`GitHub issue tracker <https://github.com/regpy/regpy/issues>`_.

For the current version, we provide information and detailed documentation at

- https://num.math.uni-goettingen.de/regpy/


Usage examples
--------------

We provide an explanation of how to use ``RegPy`` in

- ``USAGE.md`` (see ``./USAGE.md``)

On our website, we also provide some
`usage examples <https://num.math.uni-goettingen.de/regpy/examples>`_.
These examples are Jupyter notebooks that give a tutorial-style introduction to the
usage of ``RegPy``.

To get a full impression of how ``RegPy`` is used, we provide many examples in the
`examples folder on GitHub <https://github.com/regpy/regpy/tree/release/examples>`_,
as well as inside the release tarballs.
Most examples include both a commented Python script and a Python notebook with
more detailed explanations.


Installation
------------

We provide different installation methods, such as installation using ``pip``,
which are listed and explained in ``INSTALLATION.md`` (see ``./INSTALLATION.md``).


Dependencies
------------

The required dependencies are:

- ``numpy >= 1.14``
- ``scipy >= 1.1``


Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

- `ngsolve <https://ngsolve.org/>`_, for some forward operators that require solving PDEs.
  An optional installation tag ``ngsolve`` is provided when installing with ``pip``.
- `bart <https://mrirecon.github.io/bart/>`_, for the MRI operator
- ``matplotlib``, for some of the examples
- `sphinx <https://www.sphinx-doc.org/en/master/>`_, for generating the documentation
  (additional requirements are listed in ``doc/sphinx/requirements.txt``)
"""
from regpy import util, stoprules, vecsps, operators, functionals, hilbert, solvers

__all__ = ["util","stoprules","vecsps","operators","functionals","hilbert","solvers"]

hilbert._register_spaces()
functionals._register_functionals()