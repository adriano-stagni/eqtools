#!/usr/bin/env python3
import os
import numpy as np
from setuptools import setup, Extension

# Get NumPy's include directory for core headers.
numpy_include = np.get_include()

# Compute the path to the f2py headers (for fortranobject.h) and sources.
numpy_f2py_include = os.path.join(os.path.dirname(np.__file__), 'f2py', 'src')
numpy_f2py_src = os.path.join(numpy_f2py_include, 'fortranobject.c')

# Read the long description from the README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the extension module using the pre-generated C files and the fortranobject source.
tricub_ext = Extension(
    name="eqtools._tricub",
    sources=[
        os.path.join("eqtools", "_tricub.c"),
        os.path.join("eqtools", "_tricubmodule.c"),
        numpy_f2py_src,  # include fortranobject.c to define PyFortran_Type
    ],
    include_dirs=[numpy_include, numpy_f2py_include],
    extra_link_args=["-lgfortran"],   # adjust if needed for your system
)

setup(
    name="eqtools",
    version="1.3.3",
    author="Mark Chilenski, Ian Faust, John Walk, Nicola Vianello",
    author_email="psfcplasmatools@mit.edu",
    description="Python tools for magnetic equilibria in tokamak plasmas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicolavianello/eqtools/",
    packages=["eqtools"],
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib"
    ],
    license="GPL",
    ext_modules=[tricub_ext],
)