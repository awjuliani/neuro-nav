from os import stat
import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.base_agent import BaseAgent
from neuronav.agents.td_agents import TDSR


class MBV(BaseAgent):
    """
    Implementation of Model-Based Value Iteration Algorithm
    """

    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lr=1e-1,
        beta=1e4,
        poltype="softmax",
        weights="direct",
        epsilon=1e-1,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        self.weights = weights
        self.T = np.stack(
            [
                np.zeros([self.state_size, self.state_size])
                for _ in range(self.action_size)
            ]
        )
        self.w = np.zeros(state_size)
        self.base_Q = np.zeros([self.action_size, self.state_size])

    def q_estimate(self, state):
        Q = self.Q
        return Q[:, state]

    def sample_action(self, state):
        return self.base_sample_action(self.q_estimate(state))

    def update_w(self, current_exp):
        s, a, s_1, r, _ = current_exp
        if self.weights == "direct":
            error = r - self.w[s_1]
            self.w[s_1] += self.lr * error
        return np.linalg.norm(error)

    def update_t(self, current_exp, next_exp=None, prospective=False):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]

        self.T[s_a, s] = utils.onehot(s_1, self.state_size)

        return 0.0

    def update_q(self, iters=1):
        for h in range(iters):
            for i in range(self.state_size):
                for j in range(self.action_size):
                    if np.sum(self.T[j][i]) > 0:
                        v_next = np.max(self.base_Q[:, np.argmax(self.T[j][i])])
                    else:
                        v_next = 0
                    self.base_Q[j, i] = self.w[i] + self.gamma * v_next

    def _update(self, current_exp, **kwargs):
        self.update_t(current_exp, **kwargs)
        w_error = self.update_w(current_exp)
        self.update_q()
        td_error = {"w": np.linalg.norm(w_error)}
        return td_error

    def get_policy(self):
        Q = self.Q
        return self.base_get_policy(Q)

    @property
    def Q(self):
        return self.base_Q.copy()


class SRMB(BaseAgent):
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lr=1e-1,
        beta=1e4,
        mix=0.1,
        poltype="softmax",
        weights="direct",
        epsilon=1e-1,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        self.mix = mix
        self.MB_agent = MBV(
            state_size, action_size, gamma, lr, beta, poltype, weights, epsilon
        )
        self.SR_agent = TDSR(
            state_size, action_size, lr, gamma, beta, poltype, None, weights, epsilon
        )

    @property
    def Q(self):
        return self.MB_agent.Q * self.mix + self.SR_agent.Q * (1 - self.mix)

    def update_w(self, current_exp):
        self.MB_agent.update_w(current_exp)
        self.SR_agent.update_w(current_exp)

    def _update(self, current_exp):
        self.MB_agent._update(current_exp)
        self.SR_agent._update(current_exp)

    def q_estimates(self, state):
        mb_q = self.MB_agent.q_estimate(state)
        sr_q = self.SR_agent.q_estimate(self.prepare_state(state))
        return mb_q * self.mix + sr_q * (1 - self.mix)

    def sample_action(self, state):
        return self.base_sample_action(self.q_estimates(state))
