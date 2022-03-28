# Neuro-Nav (Beta)

![Example environments](/images/banner.png)

Neuro-Nav is an open-source library for neurally plausible reinforcement learning (RL). It offers a set of standardized environments and RL algorithms drawn from canonical behavioral and neural studies in rodents and humans. 

*(Note: Neuro-nav is currently in pre-release form. Additional documentation, environments, algorithms, and study replications plan to be added over the coming months. Please feel free to report issues or feature requests in the repository "Issues" section.)*

## Requirements

* Python 3.7+
* NumPy
* Gym
* PyPlot
* Scipy
* NetworkX
* Sklearn (optional)
* Jupyter (optional)

## Installation

The `neuronav` package can be installed by running `pip install -e ./` from the root of this directory.

## Benchmark Environments

Contains a set of Grid and Graph environments with various topographies and structures.

See [neuronav/envs](./neuronav/envs) for more information.

## Algorithm Toolkit

Contains artifical agents which implement Temporal Difference (TD) and Dyna versions of Q-Learning, Successor Representation, and Actor-Critic algorithms.

See [neuronav/agents](./neuronav/agents) for more information.

## Jupyter Notebooks

We include a number of interactive jupyter notebooks demonstrating various feature of the library, as well as reproducing known results.

See [notebooks](./notebooks) for more information.
