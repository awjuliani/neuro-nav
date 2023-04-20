#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    "numpy",
    "gym",
    "matplotlib",
    "scipy",
    "networkx",
    "opencv-python",
    "PyOpenGL",
    "PyVirtualDisplay",
]

extras_required = {
    "experiments_local": ["jupyterlab", "sklearn", "torch"],
    "experiments_remote": ["scikit-learn", "torch"],
}

setup(
    name="neuronav",
    version="1.5.4",
    description="Neuro-Nav",
    license="Apache License 2.0",
    author="Arthur Juliani",
    author_email="awjuliani@gmail.com",
    url="https://github.com/awjuliani/neuro-nav",
    packages=find_packages(),
    install_requires=required,
    extras_require=extras_required,
    include_package_data=True,
    package_data={"neuronav": ["envs/textures/*.png"]},
)
