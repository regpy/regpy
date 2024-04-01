import setuptools

setuptools.setup(
    name='regpy',
    version='0.2',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.14,<2.0',
        'scipy>=1.12,<2.0',
        'pooch>=1.8,<2.0',
        'pytest>=8.0,<9.0',
    ],
    python_requires='>=3.6,<4.0',
)
