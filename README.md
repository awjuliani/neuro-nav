# Neuro-Nav

![Example environments](/images/banner.png)

Neuro-Nav is an open-source library for neurally plausible reinforcement learning (RL). It offers a set of standardized environments and RL algorithms drawn from canonical behavioral and neural studies in rodents and humans. In addition, this repository also contains a set of jupyter notebooks which reproduce various experimental results from the literature.

## Benchmark Environments

Contains two highly parameterizable environments: `GridEnv` and `GraphEnv`. Each comes with a variety of task templates, observation spaces, and other settings useful for research.

See [neuronav/envs](./neuronav/envs) for more information.

## Algorithm Toolkit

Contains artifical agents which implement over a dozen cannonical reinforcement learning algorithms, including: Temporal Difference (TD) and Dyna versions of Q-Learning, Successor Representation, and Actor-Critic algorithms.

See [neuronav/agents](./neuronav/agents) for more information.

## Experiment Notebooks

Neuro-nav includes a number of interactive jupyter notebooks, featuring different experimental environments, tasks, and RL agent algorithms. You can use these notebooks to replicate various experiments across the literature, or to simply learn about what's possible to do with the library. Notebooks include those that replicate results from computational neuroscience, psychiatry, and machine learning.

See [notebooks](./notebooks) for more information.

## Installation

The easiest way to install the `neuronav` package is by running the following command:

```
pip install 'git+https://github.com/awjuliani/neuro-nav'
```

This will provide the environments and algorithms, but not the jupyter notebooks. If you would like access to the notebooks as well, you can locally download the repository, and then install `neuronav` by running the following command from the root of the directory where the repository is downloaded to:

```
pip install -e .
```

If you would like to use the experiment notebooks as well as the core library, please run `pip install -e .[experiments_local]` from the root of this directory to install the additional dependencies.

It is also possible to access all notebooks using google colab. The links to the colab notebooks can be found [here](./notebooks/README.md).

## Requirements

Requirements for the `neuronav` library can be found [here](./setup.py).

## Contributing

Neuro-Nav is an open source project, and we actively encourage community contributions. 
These can take various forms, such as new environments, tasks, algorithms, bug fixes, documentation, citations of relevant work, or additional experiment notebooks. 
If there is a small contribution you would like to make, please feel free to open a pull request, and we can review it. 
If there is a larger contribution you are considering, please open a github issue. This way, the contribution can be discussed, and potential support can be provided if needed. 
If you have ideas for changes or features you would like to see in the library in the future, but don't have the resources to contribute yourself, please feel free to open a github issue describing the request.

## Citing

If you use Neuro-Nav in your research or educational material, please cite the work as follows:

```
@inproceedings{neuronav2022,
  Author = {Juliani, Arthur and Barnett, Samuel and Davis, Brandon and Sereno, Margaret and Momennejad, Ida},
  Title = {Neuro-Nav: A Library for Neurally-Plausible Reinforcement Learning},
  Year = {2022},
  BookTitle = {The 5th Multidisciplinary Conference on Reinforcement Learning and Decision Making},
}
```

The research paper corresponding to the above citation can be found [here](https://arxiv.org/abs/2206.03312).

## License

[Apache License 2.0](./LICENSE.md)
