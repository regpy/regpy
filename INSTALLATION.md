# Installation instructions

## Installation of `RegPy`

We provide currently a setup of `RegPy` on your machine by:

* Building/Installing using pip
* Building/Installing from sources
* Running in a docker image

In the following we discuss the installation steps for each method in more detail. If you observe any problems during the installation, please create an issue on github with detailed information so that we may reproduce the problem [github issue tracker](https://github.com/regpy/regpy/issues).

Regpy in its cor version depends only on `numpy` and `scipy`. However, we provide a woking interface for `ngsolve`. This can be installed as an optional dependency and is by default included in the docker image.

## Installation using pip

We publish `RegPy` on the PyPi, which can then be installed running the command:

~~~ bash
pip3 install regpy
~~~

Note that you can add standard options to `pip3` influencing the installation, as well as specify a specific version. If you wish to install with the optional dependency for ngsolve use can use

~~~ bash
pip3 install regpy[ngsolve]
~~~

## Installation from source

You can directly install (through `pip`) the latest version of `RegPy` from github using

~~~ bash
pip3 install git+https://github.com/regpy/regpy.git@master
~~~

or clone and install it as an editable library if you wish to make modifications

~~~ bash
git clone https://github.com/regpy/regpy.git@master
cd regpy
pip install --editable .
~~~

Again you may add the optional dependency of `ngsolve`

~~~ bash
pip3 install git+https://github.com/regpy/regpy.git@master[ngsolve]
~~~

or

~~~ bash
git clone https://github.com/regpy/regpy.git@master
cd regpy
pip install --editable .[ngsolve]
~~~

## Using Docker image

For convenience we provide a [docker image](https://hub.docker.com) of `RegPy`. How to install and use docker daemon can be found in the [dockerdocs](https://docs.docker.com).

Assuming you have a working docker daemon you can simply run the latest `RegPy` docker with

~~~ bash
docker run -i -t regpy/regpy:latest /bin/bash
~~~

The image is also stuffed with a jupyter server. Thus running:

~~~ bash
docker run --name regpy-jupyter -p 8000:8000 regpy/regpy:latest
~~~

This will lunch a docker container with a jupyter server having `RegPy` already installed. To access the jupyter server just open the link that is displayed when running the above command.
