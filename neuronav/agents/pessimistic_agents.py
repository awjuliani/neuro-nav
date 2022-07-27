from neuronav.agents.td_agents import TDQ
from neuronav.agents.mb_agents import MBV
import numpy as np
import neuronav.utils as utils


class PessimisticTDQ(TDQ):
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
        w_value=1.0,
        **kwargs
    ):
        super().__init__(
            state_size, action_size, lr, gamma, beta, poltype, Q_init, epsilon
        )
        self.w_value = w_value

    def update_q(self, current_exp, next_exp=None, prospective=False):
        s = self.linear_prepare_state(current_exp[0])
        s_a = current_exp[1]
        s_1 = self.linear_prepare_state(current_exp[2])
        r = current_exp[3]

        s_a_1_optim = np.argmax(self.q_estimate(s_1))
        s_a_1_pessim = np.argmin(self.q_estimate(s_1))

        target = r + self.gamma * (
            self.w_value * self.q_estimate(s_1)[s_a_1_optim]
            + (1 - self.w_value) * self.q_estimate(s_1)[s_a_1_pessim]
        )
        q_error = target - self.q_estimate(s)[s_a]

        if not prospective:
            # actually perform update to Q if not prospective
            self.Q[s_a, :] += self.lr * q_error * s
        return q_error


class PessimisticMBV(MBV):
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
        w_value=1.0,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        self.w_value = w_value

    def update_q(self, iters=1):
        for h in range(iters):
            for i in range(self.state_size):
                for j in range(self.action_size):
                    if np.sum(self.T[j][i]) > 0:
                        v_next = self.w_value * np.max(
                            self.base_Q[:, np.argmax(self.T[j][i])]
                        ) + (1 - self.w_value) * np.min(
                            self.base_Q[:, np.argmax(self.T[j][i])]
                        )
                    else:
                        v_next = 0
                    self.base_Q[j, i] = self.w[i] + self.gamma * v_next
