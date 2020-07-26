from setuptools import setup, find_packages

setup(
    name="problems",
    version="0.0.1",
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["black", "pytest", "pytest-cov", "pylint"],
)
