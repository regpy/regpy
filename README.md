# regpy: Python tools for regularization methods

For information and detailed documentation, please visit <https://num.math.uni-goettingen.de/regpy/>.

A library for implementing and solving ill-posed inverse problems developed at the [Institute for
Numerical and Applied Mathematics Goettingen](https://num.math.uni-goettingen.de).

This project is now low beta quality software and still under under heavy development. 
Therefore, expect bugs and partially undocumented tools.

## Usage examples

To get an impression of how using `regpy` looks, there are some examples in the [`examples`
folder on GitHub](https://github.com/regpy/regpy/tree/release/examples), as well as inside the release tarballs (see below). Most of the examples supply both a commented python script and a python notebook with more detailed explanation. 

## Installation

### Obtaining the source code

- The source code is on GitHub ([regpy/regpy](https://github.com/regpy/regpy)).
- Releases are at the corresponding [release page](https://github.com/regpy/regpy/releases). The
  current version is 0.3.

### Dependencies

- `numpy >= 1.14`
- `scipy >= 1.1`

#### Optional dependencies

- [`ngsolve`](https://ngsolve.org/) (for some forward operators that require solving PDEs)
- [`bart`](https://mrirecon.github.io/bart/) (for the MRI operator)
- `matplotlib` (for some of the examples)
- [`pdoc3`](https://pdoc3.github.io/pdoc) (for generating the documentation)

### Installation with `pip`

A basic `setup.py` is provided, but the package is not on PyPI yet. To install it, clone this
repository or download the release tarball, and run

~~~ bash
pip install .
~~~

from the project's root folder. If you want to modify `regpy` itself, you can use

~~~ bash
pip install --editable .
~~~

to have Python load `regpy` from your current directory rather than copying it to its library
folder.

In both cases, you can add `--no-deps` to prevent installing dependencies automatically,
in case you want to install them via some other package manager.
