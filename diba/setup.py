#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

PYTORCH_VERSION = "1.7.0"

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


# Check that torch is installed before continuing the installation
pytorch_error = RuntimeError(f"Installation of this package requires a pytorch installation (>={PYTORCH_VERSION})")
try:
    import torch

    torch.zeros(1)
except ModuleNotFoundError:
    raise pytorch_error

requirements = ["Click>=7.0", "tqdm>=4.0", "numpy>=1.20.0", "transformers>=4.19.2"]

test_requirements = ["pytest>=3"]

setup(
    author="Giorgio Mariani",
    author_email="giorgiomariani94@gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Discrete bayesian signal separation library for general models and signals.",
    entry_points={
        "console_scripts": [
            "diba=diba.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="diba",
    name="diba",
    packages=find_packages(include=["diba", "diba.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/giorgio-mariani/diba",
    version="0.1.0",
    zip_safe=False,
)
