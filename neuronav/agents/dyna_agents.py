import numpy as np
import numpy.random as npr
from neuronav.agents.td_agents import TDAC, TDQ, TDSR


class DynaModule:
    """
    Class which contains logic to enable Dyna algorithms.
    """

    def __init__(self, state_size, num_recall=3, recency="exponential", **kwargs):
        self.num_recall = num_recall
        self.recency = recency
        self.model = {}
        self.prioritized_states = np.zeros(state_size, dtype=int)

    def _sample_model(self):
        # sample state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[npr.choice(len(past_states))]
        # sample action previously taken from sampled state
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = past_actions[npr.choice(len(past_actions))]
        key = (sampled_state, sampled_action)
        # get reward, state_next, done, and make exp
        if self.recency == "exponential":
            successors = self.model[key][1]
            idx = np.minimum(len(successors) - 1, int(npr.exponential(scale=5)))
            successor = successors[::-1][idx]
        else:
            successor = self.model[key][1]
        exp = key + successor
        return exp

    def update(self, base_agent, current_exp, **kwargs):

        state, action, next_state, reward, done = current_exp

        # update (deterministic) model
        key = (state, action)
        value = (next_state, reward, done)
        if self.recency == "exponential":
            if key in self.model:
                successors = self.model[key][1]
                successors.append(value)
                # shorten successors to capture >99% of probability mass
                successors = successors[-25:]
                self.model[key] = base_agent.num_updates, successors
            else:
                self.model[key] = base_agent.num_updates, [value]
        else:
            self.model[key] = base_agent.num_updates, value

        for i in range(self.num_recall):
            exp = self._sample_model()
            self.prioritized_states[exp[0]] += 1
            base_agent._update(exp)

        return base_agent


class DynaQ(TDQ):
    """
    Dyna-enabled version of Temporal Difference Q-learning algorithm.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        w_value: float = 1.0,
    ):
        super(DynaQ, self).__init__(
            state_size,
            action_size,
            lr=lr,
            gamma=gamma,
            poltype=poltype,
            beta=beta,
            epsilon=epsilon,
            w_value=w_value,
        )
        self.dyna = DynaModule(state_size)

    def update(self, current_exp):
        _ = super().update(current_exp)
        self = self.dyna.update(self, current_exp)


class DynaAC(TDAC):
    """
    Dyna-enabled version of Temporal Difference Actor Critic algorithm.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
    ):
        super(DynaAC, self).__init__(
            state_size,
            action_size,
            lr=lr,
            gamma=gamma,
            poltype=poltype,
            beta=beta,
            epsilon=epsilon,
        )
        self.dyna = DynaModule(state_size)

    def update(self, current_exp):
        _ = super().update(current_exp)
        self = self.dyna.update(self, current_exp)


class DynaSR(TDSR):
    """
    Dyna-enabled version of Temporal Difference Successor Representation algorithm.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
    ):
        super(DynaSR, self).__init__(
            state_size,
            action_size,
            lr=lr,
            gamma=gamma,
            poltype=poltype,
            beta=beta,
            epsilon=epsilon,
        )
        self.dyna = DynaModule(state_size)

    def update(self, current_exp):
        _ = super().update(current_exp)
        self = self.dyna.update(self, current_exp)
