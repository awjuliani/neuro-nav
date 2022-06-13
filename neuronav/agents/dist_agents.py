from curses.panel import new_panel
import numpy as np
import numpy.random as npr
from neuronav.agents.base_agent import BaseAgent
import neuronav.utils as utils


class DistQ(BaseAgent):
    """
    Implementation of Temporal Difference Q-Learning Algorithm.
    """

    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-1,
        gamma=0.99,
        beta=1e4,
        poltype="softmax",
        Q_init=None,
        epsilon=1e-1,
        dist_cells=16,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta)

        if Q_init is None:
            self.Q = np.zeros((action_size, state_size, dist_cells))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size, dist_cells)
        else:
            self.Q = Q_init
        self.dist_cells = dist_cells
        self.lrs_pos = npr.uniform(0.001, 0.02, dist_cells)
        self.lrs_neg = npr.uniform(0.001, 0.02, dist_cells)

    def sample_action(self, state):
        Qs = self.Q[:, state]
        if self.poltype == "softmax":
            action = npr.choice(
                self.action_size,
                p=utils.softmax(self.beta * Qs[:, npr.randint(0, self.dist_cells)]),
            )
        else:
            if npr.rand() < self.epsilon:
                action = npr.choice(self.action_size)
            else:
                action = npr.choice(np.flatnonzero(np.isclose(Qs, Qs.max())))
        return action

    def update_q(self, current_exp, next_exp=None, prospective=False):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]

        # determines whether update is on-policy or off-policy
        if next_exp is None:
            s_a_1 = np.argmax(self.Q[:, s_1])
        else:
            s_a_1 = next_exp[1]

        r = current_exp[3]

        # next_q = self.Q[s_a_1, s_1]
        next_q = self.Q[s_a_1, s_1, npr.randint(0, self.dist_cells)]

        q_error = r + self.gamma * next_q - self.Q[s_a, s]
        qep = (q_error > 0.0) * 1.0

        if not prospective:
            # actually perform update to Q if not prospective
            self.Q[s_a, s] += (self.lrs_pos * qep + self.lrs_neg * (1 - qep)) * q_error
        return q_error

    def _update(self, current_exp, **kwargs):
        q_error = self.update_q(current_exp, **kwargs)
        td_error = {"q": np.linalg.norm(q_error)}
        return td_error

    def get_policy(self):
        if self.poltype == "softmax":
            policy = utils.softmax(self.beta * self.Q, axis=0)
        else:
            mask = self.Q == self.Q.max(0)
            greedy = mask / mask.sum(0)
            policy = (1 - self.epsilon) * greedy + (
                1 / self.action_size
            ) * self.epsilon * np.ones((self.action_size, self.state_size))
        return policy
