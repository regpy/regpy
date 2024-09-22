# Installation instructions
## Installation of `regpy`
We provide currently only a setup of `regpy` on your machine by:
* [Building/Installing from sources](#installation-from-source)

Below we discuss these installation steps in more detail. If you observe any problems with the installation, you can contact us through the [github issue tracker](https://github.com/regpy/regpy/issues).

Note that part of the provided library depend on NGSolve, which requires you to have a working NGSolve installation.  

## Installation from source

You can also directly install (through `pip`) the latest version of `regpy` which is the development version in the master branch on github using
~~~ bash
pip3 install git+https://github.com/regpy/regpy.git@master
~~~ 
or clone and install it as an editable library if you wish to make modifications
~~~ bash
git clone https://github.com/regpy/regpy.git@master
cd regpy
pip install --editable .
~~~

If you want to use the on NGSolve depending parts you will require an installed version of NGSolve. 