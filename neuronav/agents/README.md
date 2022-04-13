# Algorithm Toolkit

A set of cognitive neuroscience inspired agents and learning algorithms.

These consist of implementations of the canonical Q-Learning, Actor-Critic, Value-Iteration, and Successor Representation algorithms.

All algorithms are "tabular" and work with observations that consist of integer representations of the state of the agent. This corresponds to the `index` observation type.

| Algorithm | Function(s) | Update Rule(s) | Reference | Description | Code Link |
| --- | --- | --- | --- | --- | --- |
| TD-Q | Q(s, a) | one-step temporal difference | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic q-learning algorithm | [Code](./td_agents.py) |
| TD-SR | V(s), π(a \| s) | one-step temporal difference | [Dayan, 1993](https://ieeexplore.ieee.org/abstract/document/6795455) | A basic successor representation algorithm | [Code](./td_agents.py) |
| TD-AC | ψ(s, a), ω(s) | one-step temporal difference | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic actor-critic algorithm | [Code](./td_agents.py) |
| Dyna-Q | Q(s, a) | one-step temporal difference, replay-based dyna | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A dyna q-learning algorithm | [Code](./dyna_agents.py) |
| Dyna-SR | V(s), π(a \| s) | one-step temporal difference, replayd-base dyna | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) | A dyna successor representation algorithm |[Code](./dyna_agents.py) |
| Dyna-AC | ψ(s, a), ω(s) | one-step temporal difference, replay-based dyna | N/A | A dyna actor-critic algorithm |[Code](./dyna_agents.py) |
| MBV | Q(s, a), T(s' \| s, a) | value-iteration | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html) | A basic value iteration algorithm | [Code](./mb_agents.py) |
| SRMB | Q(s, a), T(s' \| s, a), ψ(s, a), ω(s) | value-iteration, one-step temporal difference | [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) | A hybrid of value iteration and temporal-difference successor algorithms | [Code](./mb_agents.py) |
