# Neuro-Nav (Beta)

![Example environments](/images/banner.png)

Neuro-Nav is an open-source library for neurally plausible reinforcement learning (RL). It offers a set of standardized environments and RL algorithms drawn from canonical behavioral and neural studies in rodents and humans. 

*(Note: Neuro-Nav is currently in pre-release form. Additional documentation, environments, algorithms, and study replications plan to be added over the coming months. Please feel free to report issues or feature requests in the repository "Issues" section.)*

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

## Experiment Notebooks

Neuro-nav includes a number of interactive jupyter notebooks, featuring different experimental environments, tasks, and RL agent algorithms. You can use these notebooks to replicate various experiments across the literature, or to simply learn about what's possible to do with the library.

See [notebooks](./notebooks) for more information.

## Contributing

Neuro-Nav is an open source project, and we encourage community contributions. 
These can take various forms, such as feature requests, bug fixes, additional documentation, or even additional notebooks. 
If there is a small contribution you would like to make, please feel free to open a pull request. 
If there is a larger contribution you are considering, please open an issue, where it can be discussed, and potential support can be provided. 
We are especially interested in novel algorithms and tasks relevant to the intersection of the neuroscience machine learning communities.

## Citing

If you use Neuro-Nav in your research, please cite the work as follows:

```
@inproceedings{neuronav2022,
  Author = {Juliani, Arthur and Barnett, Samuel and Davis, Brandon and Sereno, Margaret and Momennejad, Ida},
  Title = {Neuro-Nav: A Library for Neurally-Plausible Reinforcement Learning},
  Year = {2022},
  BookTitle = {The 5th Multidisciplinary Conference on Reinforcement Learning and Decision Making},
}
```

## License

[Apache License 2.0](./LICENSE.md)
