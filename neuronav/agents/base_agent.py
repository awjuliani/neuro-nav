import numpy as np
import numpy.random as npr
import neuronav.utils as utils


class BaseAgent:
    """
    Parent class for Agents which concrete implementations inherit from.
    """

    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-1,
        gamma=0.99,
        poltype="softmax",
        beta=1e4,
        epsilon=1e-1,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.poltype = poltype
        self.num_updates = 0
        self.epsilon = epsilon

    def base_sample_action(self, policy_logits):
        if self.poltype == "softmax":
            action = npr.choice(
                self.action_size, p=utils.softmax(self.beta * policy_logits)
            )
        else:
            if npr.rand() < self.epsilon:
                action = npr.choice(self.action_size)
            else:
                action = npr.choice(
                    np.flatnonzero(np.isclose(policy_logits, policy_logits.max()))
                )
        return action

    def update(self, current_exp):
        self.num_updates += 1
        self._update(current_exp)

    def base_get_policy(self, policy_logits):
        if self.poltype == "softmax":
            policy = utils.softmax(self.beta * policy_logits, axis=0)
        else:
            mask = policy_logits == policy_logits.max(0)
            greedy = mask / mask.sum(0)
            policy = (1 - self.epsilon) * greedy + (
                1 / self.action_size
            ) * self.epsilon * np.ones((self.action_size, self.state_size))
        return policy

    def _update(self, current_exp):
        return None

    def reset(self):
        return None
