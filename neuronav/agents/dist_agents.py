import numpy as np
import numpy.random as npr
from neuronav.agents.base_agent import BaseAgent


class DistQ(BaseAgent):
    """
    Implementation of the distributional Q-Learning algorithm
    found in Dabney et al., 2019.
    `mirror` determines whether same learning rates are used for
    positive and negative td errors.
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
        mirror=False,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)

        if Q_init is None:
            self.Q = np.zeros((action_size, state_size, dist_cells))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size, dist_cells)
        else:
            self.Q = Q_init
        self.dist_cells = dist_cells
        self.lrs_pos = npr.uniform(0.001, 0.02, dist_cells)
        if mirror:
            self.lrs_neg = self.lrs_pos
        else:
            self.lrs_neg = npr.uniform(0.001, 0.02, dist_cells)

    def sample_action(self, state):
        Qs = self.Q[:, state, npr.randint(0, self.dist_cells)]
        return self.base_sample_action(Qs)

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
        Qs = self.Q[:, :, npr.randint(0, self.dist_cells)]
        return self.base_get_policy(Qs)
