import os
from setuptools import setup, find_packages
from poisson.version import get_version

setup(
    name='poisson',
    version=get_version(),
    description='Poisson equation solver for n-dimensional arrays',
    url='https://github.com/mfkasim91/poisson',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=2.7,>=3.5",
    install_requires=[
        "numpy>=1.12.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6"
    ],
    keywords="poisson solver mathematics engineering",
    zip_safe=False
)
