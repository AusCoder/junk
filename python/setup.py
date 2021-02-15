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
from setuptools.command.build_ext import build_ext as _build_ext

# Something is broken in the numpy import when installing
# this with pip :(. I am digging down a rabbit hole I don't
# really want to go down right now, so I'm leaving it for later.

# import numpy as np
# from Cython.Build import cythonize

# cython_extensions = cythonize(
#     [
#         Extension(
#             "junk.mandel",
#             sources=["src/junk/mandel.pyx"],
#             include_dirs=[np.get_include()],
#         )
#     ]
# )

spam_extension = Extension("junk.spam", sources=["src/junk/spammodule.c"])

setup(
    name="junk",
    version="0.0.1",
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=["numpy", "Cython"],
    install_requires=["numpy", "black", "pytest", "pytest-cov", "pylint"],
    ext_modules=[spam_extension],
    # ext_modules=[spam_extension, *cython_extensions],
)
