#!/usr/bin/env python

from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="neuronav",
    version="0.1.0",
    description="Neuro-Nav",
    license="Apache License 2.0",
    author="Arthur Juliani",
    author_email="awjuliani@gmail.com",
    url="https://github.com/awjuliani/neuro-nav",
    packages=find_packages(),
    install_requires=required,
)
