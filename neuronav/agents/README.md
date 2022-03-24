# Algorithm Toolkit

A set of cognitive neuroscience inspired agents and learning algorithms.

These consist of implementations of the canonical Q-Learning, Actor-Critic, Value-Iteration, and Successor Representation algorithms.

All algorithms are "tabular" and work with observations that consist of integer representations of the state of the agent. This corresponds to the `index` observation type.

## Temporal Difference Algorithms

The implementations of the TD algorithms can be found [here](./td_agents.py).

* TD-Q
* TD-SR
* TD-AC

## Dyna Algorithms

The implementations of the Dyna algorithms can be found [here](./dyna_agents.py).

* Dyna-Q
* Dyna-SR
* Dyna-AC

## Model Based Algorithms

The implementations of the model-based algorithms can be found [here](./mb_agents.py).

* Value Iteration (MBV)
* TDSR / Value Iteration Hybrid (SRMB)
