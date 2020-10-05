from setuptools import setup, find_packages, Extension

spam_extension = Extension("spam", sources=["src/junk/spammodule.c"])

setup(
    name="junk",
    version="0.0.1",
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["black", "pytest", "pytest-cov", "pylint"],
    ext_modules=[spam_extension],
)
