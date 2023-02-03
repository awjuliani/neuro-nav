import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.base_agent import BaseAgent


class QEC(BaseAgent):
    """
    Implementation of episodic control Q-learning algorithm.
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
        Q_init=None,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        if Q_init is None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init

    def sample_action(self, state):
        Qs = self.Q[:, state]
        return self.base_sample_action(Qs)

    def _update(self, current_exp, **kwargs):
        s, sa, s_1, r, d = current_exp
        self.exp_list.append(current_exp)
        if d:
            self.backup(self.exp_list)
        return None

    def q_estimate(self, state):
        return self.Q[:, state]

    def backup(self, exp_list):
        # update Q values
        rewards = [exp[3] for exp in exp_list]
        returns = self.discount(rewards, self.gamma)
        for i, exp in enumerate(exp_list):
            s, sa, s_1, r, d = exp
            self.Q[sa, s] = np.max([self.Q[sa, s], returns[i]])
        self.exp_list = []

    def get_policy(self):
        return self.base_get_policy(self.Q)

    def reset(self):
        self.exp_list = []


class QMC(BaseAgent):
    """
    Implementation of Monte Carlo Q-learning
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
        Q_init=None,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        if Q_init is None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init

    def sample_action(self, state):
        Qs = self.Q[:, state]
        return self.base_sample_action(Qs)

    def q_estimate(self, state):
        return self.Q[:, state]

    def _update(self, current_exp, **kwargs):
        s, sa, s_1, r, d = current_exp
        self.exp_list.append(current_exp)
        if d:
            self.backup(self.exp_list)
        return None

    def backup(self, exp_list):
        # update Q values
        rewards = [exp[3] for exp in exp_list]
        returns = self.discount(rewards, self.gamma)
        for i, exp in enumerate(exp_list):
            s, sa, s_1, r, d = exp
            delta = returns[i] - self.Q[sa, s]
            self.Q[sa, s] += self.lr * delta
        self.exp_list = []

    def get_policy(self):
        return self.base_get_policy(self.Q)

    def reset(self):
        self.exp_list = []
