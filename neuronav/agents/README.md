# Algorithm Toolkit

A set of cognitive neuroscience inspired agents and learning algorithms.

These consist of implementations of the canonical Q-Learning, Actor-Critic, Value-Iteration, and Successor Representation algorithms.

The algorithms included the beta release are "tabular". Tabular algorithms work with observations that are integer representations of the state of the agent (e.g., which grid the agent is in a grid world). This corresponds to the `index` observation type. 

*(In order to support more rich observation spaces, future releases will include additional linear and non-linear policy and value functions.)*

| Algorithm | Function(s) | Update Rule(s) | Reference | Description | Code Link |
| --- | --- | --- | --- | --- | --- |
| TD-Q | Q(s, a) | one-step temporal difference | [Watkins & Dayan, 1992](https://link.springer.com/article/10.1007/BF00992698) | A basic q-learning algorithm | [Code](./td_agents.py) |
| TD-SR | ψ(s, a), ω(s) | one-step temporal difference | [Dayan, 1993](https://ieeexplore.ieee.org/abstract/document/6795455) | A basic successor representation algorithm | [Code](./td_agents.py) |
| TD-AC | V(s), π(a \| s) | one-step temporal difference | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic actor-critic algorithm | [Code](./td_agents.py) |
| Dyna-Q | Q(s, a) | one-step temporal difference, replay-based dyna | [Sutton, 1990](https://www.sciencedirect.com/science/article/pii/B9781558601413500304) | A dyna q-learning algorithm | [Code](./dyna_agents.py) |
| Dyna-SR | ψ(s, a), ω(s) | one-step temporal difference, replay-base dyna | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) | A dyna successor representation algorithm |[Code](./dyna_agents.py) |
| Dyna-AC | V(s), π(a \| s) | one-step temporal difference, replay-based dyna | N/A | A dyna actor-critic algorithm |[Code](./dyna_agents.py) |
| MBV | Q(s, a), T(s' \| s, a) | value-iteration | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic value iteration algorithm | [Code](./mb_agents.py) |
| SRMB | Q(s, a), T(s' \| s, a), ψ(s, a), ω(s) | value-iteration, one-step temporal difference | [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) | A hybrid of value iteration and temporal-difference successor algorithms | [Code](./mb_agents.py) |
