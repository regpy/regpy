r"""# `RegPy`: Python tools for regularization methods

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/regpy/regpy?label=latest%20release&logo=github)](https://github.com/regpy/regpy)

[![PyPI](https://img.shields.io/pypi/v/regpy?color=blue&label=latest%20PyPI%20version&logo=pypi&logoColor=white)](https://pypi.org/project/regpy/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/regpy?logo=pypi&logoColor=white)](https://pypi.org/project/regpy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/regpy?label=PyPI%20downloads&logo=pypi&logoColor=white)](https://pypi.org/project/regpy/)

[![Docker Pulls](https://img.shields.io/docker/pulls/regpy/regpy?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/regpy/regpy)

`RegPy` is a python library for implementing and solving ill-posed inverse problems developed at the [Institute for Numerical and Applied Mathematics Goettingen](https://num.math.uni-goettingen.de). It provides tolls to implement your own forward model both linear and non-linear and a variety of regularization methods that can be stopped using common stopping rules.

This project is currently in an almost beta quality state. However, the project is still under intensive development. Therefore, expect bugs and partially undocumented tools. If you encounter any issues we welcome any information on our [github issue tracker](https://github.com/regpy/regpy/issues).

For the current version we provide information and a detailed documentation under <https://num.math.uni-goettingen.de/regpy/>.

## Usage examples

We provide a explanation on how to use `RegPy` [here](./USAGE.md). On our website we provide some [usage examples](https://num.math.uni-goettingen.de/regpy/examples). These examples are jupyter notebooks that should provide a tutorial kind of introduction to the usage of `RegPy`.

To get an full impression of the usage of `RegPy`, we provide many examples in the [`examples`
folder on GitHub](https://github.com/regpy/regpy/tree/release/examples), as well as inside the release tarballs (see below). Most of the examples supply both a commented python script and a python notebook with more detailed explanation.

## Installation

We provide different installation methods, such installation using `pip`, listed and explained in [INSTALLATION.md](./INSTALLATION.md).

### Dependencies

- `numpy >= 1.14`
- `scipy >= 1.1`

#### Optional dependencies

- [`ngsolve`](https://ngsolve.org/), for some forward operators that require solving PDEs. We provide an optional installation tag `ngsolve` when installing with `pip`.
- [`bart`](https://mrirecon.github.io/bart/) (for the MRI operator)
- `matplotlib` (for some of the examples)
- [`sphinx`](https://www.sphinx-doc.org/en/master/) (for generating the documentation) further requirements in `doc/sphinx/requirements.txt`
"""
from regpy import util, stoprules, vecsps, operators, functionals, hilbert, solvers

__all__ = ["util","stoprules","vecsps","operators","functionals","hilbert","solvers"]

hilbert._register_spaces()
functionals._register_functionals()