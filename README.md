# `RegPy`: Python tools for regularization methods

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/regpy/regpy?label=latest%20release&logo=github)](https://github.com/regpy/regpy)
[![DOI](https://zenodo.org/badge/215324707.svg)](https://doi.org/10.5281/zenodo.16837824)
[![Read the Docs](https://img.shields.io/readthedocs/latest?logo=read-the-docs&logoColor=white)](https://num.math.uni-goettingen.de/regpy/)


[![PyPI](https://img.shields.io/pypi/v/regpy?color=blue&label=latest%20PyPI%20version&logo=pypi&logoColor=white)](https://pypi.org/project/regpy/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/regpy?logo=pypi&logoColor=white)](https://pypi.org/project/regpy/)

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/regpy/regpy/pypi.yml?branch=release&label=build%20PyPI%20Release&logo=github)](https://github.com/regpy/regpy/actions/workflows/pypi.yml)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/regpy/regpy/docker-deploy.yml?branch=release&label=build%20Docker%20Image&logo=github)](https://github.com/regpy/regpy/actions/workflows/docker-deploy.yml)


[![PyPI - Downloads](https://img.shields.io/pypi/dm/regpy?label=PyPI%20downloads&logo=pypi&logoColor=white)](https://pypi.org/project/regpy/)
[![Docker Pulls](https://img.shields.io/docker/pulls/regpy/regpy?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/regpy/regpy)

`RegPy` is a Python library developed at the [Institute for Numerical and Applied Mathematics](https://num.math.uni-goettingen.de) at the University of Göttingen. It provides tools to implement custom forward models -- both linear and non-linear -- as well as a variety of regularization methods and stopping rules.

This project is currently approaching beta quality state, but remains under active development. As a result, you may run into bugs or partially undocumented features. If you run into any issues we welcome any information on our [GitHub issue tracker](https://github.com/regpy/regpy/issues).

Detailed information and documentation of the current version can be found at <https://num.math.uni-goettingen.de/regpy/>.

## Usage examples

We offer an explanation on how to use `RegPy` [here](./USAGE.md) and our website features several detailed [usage examples](https://num.math.uni-goettingen.de/regpy/examples). These examples are provided as Jupyter notebooks that serve as a tutorial-style introduction to `RegPy`.

For a more comprehensive overview of `RegPy`'s capabilities, we provide numerous examples in the [examples GitHub repository](https://github.com/regpy/regpy-examples). These examples are also part of the docker image provided on [DockerHub](https://hub.docker.com/repository/docker/regpy/regpy) (see in the installation instructions for details). Most examples include both a commented Python script and a Jupyter notebook with more detailed explanations.

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
