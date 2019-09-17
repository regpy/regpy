import setuptools

setuptools.setup(
    name='itreg',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib>=2.2,<3.0',
        'numpy>=1.14,<2.0',
        'scipy>=1.1,<2.0',
        'pyNFFT>=1.3,<2.0',
    ],
    python_requires='>=3.6,<4.0',
)
