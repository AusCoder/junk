"""
Useful build commands:
    # build extensions
    python setup.py build_ext

    # build extensions and copy so we can do editable install
    python setup.py build_ext --inplace

    # clean
    python setup.py clean
"""

from setuptools import setup, find_packages, Extension

import numpy as np
from Cython.Build import cythonize

spam_extension = Extension("junk.spam", sources=["src/junk/spammodule.c"])
cython_modules = cythonize(
    [
        Extension(
            "junk.mandel",
            sources=["src/junk/mandel.pyx"],
            include_dirs=[np.get_include()],
        )
    ]
)

setup(
    name="junk",
    version="0.0.1",
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["black", "pytest", "pytest-cov", "pylint"],
    ext_modules=[spam_extension, *cython_modules],
)
