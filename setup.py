#!/usr/bin/env python

from setuptools import setup, find_packages

required = ["numpy", "gym", "matplotlib", "scipy", "networkx"]

extras_required = {
    "experiments_local": ["jupyterlab", "sklearn", "torch"],
    "experiments_remote": ["sklearn", "torch"],
}

setup(
    name="neuronav",
    version="0.4.1",
    description="Neuro-Nav",
    license="Apache License 2.0",
    author="Arthur Juliani",
    author_email="awjuliani@gmail.com",
    url="https://github.com/awjuliani/neuro-nav",
    packages=find_packages(),
    install_requires=required,
    extras_require=extras_required,
)
