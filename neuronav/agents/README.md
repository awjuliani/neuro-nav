# Algorithm Toolkit

A set of cognitive neuroscience inspired agents and learning algorithms.

These consist of implementations of the canonical Q-Learning, Actor-Critic, Value-Iteration, Successor Representation algorithms and more.

The algorithms included here are all tabular. Tabular algorithms work with observations that are integer representations of the state of the agent (e.g., which grid the agent is in a grid world). This corresponds to the `index` observation type. 

*(In order to support more rich observation spaces, future releases will include additional linear and non-linear policy and value functions.)*

## Included algorithms

| Algorithm | Function(s) | Update Rule(s) | Reference | Description | Expressivity | Code Link |
| --- | --- | --- | --- | --- | --- | --- |
| TD-Q | Q(s, a) | one-step temporal difference | [Watkins & Dayan, 1992](https://link.springer.com/article/10.1007/BF00992698) | A basic q-learning algorithm | Tabular | [Code](./td_agents.py) |
| TD-SR | ψ(s, a), ω(s) | one-step temporal difference | [Dayan, 1993](https://ieeexplore.ieee.org/abstract/document/6795455) | A basic successor representation algorithm | Tabular | [Code](./td_agents.py) |
| TD-AC | V(s), π(a \| s) | one-step temporal difference | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic actor-critic algorithm | Tabular | [Code](./td_agents.py) |
| Dyna-Q | Q(s, a) | one-step temporal difference, replay-based dyna | [Sutton, 1990](https://www.sciencedirect.com/science/article/pii/B9781558601413500304) | A dyna q-learning algorithm | Tabular | [Code](./dyna_agents.py) |
| Dyna-SR | ψ(s, a), ω(s) | one-step temporal difference, replay-base dyna | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) | A dyna successor representation algorithm | Tabular | [Code](./dyna_agents.py) |
| Dyna-AC | V(s), π(a \| s) | one-step temporal difference, replay-based dyna | [Sutton, 1990](https://www.sciencedirect.com/science/article/pii/B9781558601413500304) | A dyna actor-critic algorithm | Tabular | [Code](./dyna_agents.py) |
| MBV | Q(s, a), T(s' \| s, a) | value-iteration | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic value iteration algorithm | Tabular | [Code](./mb_agents.py) |
| SRMB | Q(s, a), T(s' \| s, a), ψ(s, a), ω(s) | value-iteration, one-step temporal difference | [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) | A hybrid of value iteration and temporal-difference successor algorithms | Tabular | [Code](./mb_agents.py) |
| QET | Q(s, a), e(s, a) | eligibility trace | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A q-learning algorithm using online eligibility traces | Tabular | [Code](./td_agents.py) |
| DistQ | Q(s, a, c) | one-step temporal difference | [Dabney et al., 2020](https://www.nature.com/articles/s41586-019-1924-6) | A distributional q-learning algorithm which uses separate learning rates for optimistic and pessimistic units. | Tabular | [Code](./dist_agents.py) |
| QEC | Q(s, a) | episodic control | [Lengyel & Dayan, 2007](https://proceedings.neurips.cc/paper/2007/hash/1f4477bad7af3616c1f933a02bfabe4e-Abstract.html) | An episodic control algorithm that uses return targets from monte-carlo rollouts | Tabular | [Code](./mc_agents.py) |
| QMC | Q(s, a) | monte-carlo | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A q-learning algorithm that uses return targets from monte-carlo rollouts | Tabular | [Code](./mc_agents.py) |

## Algorithm hyperparameters

Below is a list of the common hyperparameters shared between all algorithms and agent types. Typical value ranges provided are meant as rough guidelines for generally appropriate learning behavior. Depending on the nature of the specific algorithm or task, other values may be more desirable.

* `lr` - Learning rate of algorithm. Typical value range: `0` - `0.1`.
* `gamma` - Discount factor for bootstrapping. Typical value range: `0.5` - `0.99`.
* `poltype` - Policy type. Can be either `softmax` to sample actions proportional to action value estimates, or `egreedy` to sample either the most valuable action or random action stochastically.
* `beta` - The temperate parameter used with the `softmax` poltype. Typical value range: `1` - `1000`.
* `epsilon` - The probablility of randomly acting using with the `egreedy` poltype. Typical value range: `0.1` - `0.5`.
