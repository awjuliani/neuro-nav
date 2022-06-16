import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.base_agent import BaseAgent


class TDQ(BaseAgent):
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
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)

        if Q_init is None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init

    def q_estimate(self, state):
        return state @ self.Q.T

    def sample_action(self, state):
        state = self.prepare_state(state)
        return self.base_sample_action(self.q_estimate(state))

    def update_q(self, current_exp, next_exp=None, prospective=False):
        s = self.prepare_state(current_exp[0])
        s_a = current_exp[1]
        s_1 = self.prepare_state(current_exp[2])
        r = current_exp[3]

        # determines whether update is on-policy or off-policy
        if next_exp is None:
            s_a_1 = np.argmax(self.q_estimate(s_1))
        else:
            s_a_1 = next_exp[1]

        q_error = r + self.gamma * self.q_estimate(s_1)[s_a_1] - self.q_estimate(s)[s_a]

        if not prospective:
            # actually perform update to Q if not prospective
            self.Q[s_a, :] += self.lr * q_error * s
        return q_error

    def _update(self, current_exp, **kwargs):
        q_error = self.update_q(current_exp, **kwargs)
        td_error = {"q": np.linalg.norm(q_error)}
        return td_error

    def get_policy(self):
        return self.base_get_policy(self.Q)


class TDAC(BaseAgent):
    """
    Implementation of Temporal Difference Actor Critic Algorithm
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
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        self.c_w = np.zeros([state_size])
        self.a_w = np.zeros([state_size, action_size])

    def critic(self, state):
        return np.matmul(state, self.c_w)

    def actor(self, state):
        return np.matmul(state, self.a_w)

    def sample_action(self, state):
        state = self.prepare_state(state)
        logits = self.actor(state)
        return self.base_sample_action(logits)

    def _update(self, current_exp):
        state, action, state_next, reward, done = current_exp
        state = self.prepare_state(state)
        state_next = self.prepare_state(state_next)
        if not done:
            td_target = reward + self.gamma * self.critic(state_next)
            td_estimate = self.critic(state)
            td_error = td_target - td_estimate
        else:
            td_error = reward - self.critic(state)
        self.c_w += self.lr * td_error * state
        self.a_w[:, action] += self.lr * td_error * state

    def get_policy(self):
        return self.base_get_policy(self.a_w)


class TDSR(BaseAgent):
    """
    Implementation of Temporal Difference Successor Representation Algorithm
    """

    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-1,
        gamma=0.99,
        beta=1e4,
        poltype="softmax",
        M_init=None,
        weights="direct",
        epsilon=1e-1,
        goal_biased_sr=True,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        self.weights = weights
        self.goal_biased_sr = goal_biased_sr

        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init

        self.w = np.zeros(state_size)

    def m_estimate(self, state):
        return state @ self.M

    def q_estimate(self, state):
        ms = self.m_estimate(state)
        return ms @ self.w

    def w_estimate(self, state):
        return state @ self.w

    def sample_action(self, state):
        state = self.prepare_state(state)
        return self.base_sample_action(self.q_estimate(state))

    def update_w(self, current_exp):
        s, a, s_1, r, _ = current_exp
        s = self.prepare_state(s)
        s_1 = self.prepare_state(s_1)
        if self.weights == "direct":
            error = r - self.w_estimate(s_1)
            self.w += self.lr * error * s_1
        elif self.weights == "td":
            Vs = self.q_estimate(s).max()
            Vs_1 = self.q_estimate(s_1).max()
            delta = r + self.gamma * Vs_1 - Vs
            # epsilon and beta are hard-coded, need to improve this
            M = self.get_M_states(epsilon=1e-1, beta=5)
            error = delta * M[s]
            self.w += self.lr * error
        return np.linalg.norm(error)

    def update_sr(self, current_exp, next_exp=None, prospective=False):
        s = self.prepare_state(current_exp[0])
        s_a = current_exp[1]
        s_1 = self.prepare_state(current_exp[2])

        # determines whether update is on-policy or off-policy
        if next_exp is None:
            s_a_1 = np.argmax(self.q_estimate(s_1))
        else:
            s_a_1 = next_exp[1]

        r = current_exp[3]
        d = current_exp[4]

        if d:
            m_error = s + self.gamma * s_1 - self.m_estimate(s)[s_a]
        else:
            if self.goal_biased_sr:
                m_error = (
                    s
                    + self.gamma * self.m_estimate(s_1)[s_a_1]
                    - self.m_estimate(s)[s_a]
                )
            else:
                m_error = (
                    s
                    + self.gamma * self.m_estimate(s_1).mean(0)
                    - self.m_estimate(s)[s_a]
                )

        if not prospective:
            # actually perform update to SR if not prospective
            self.M[s_a, :, :] += self.lr * (
                np.expand_dims(s, 1) @ np.expand_dims(m_error, 0)
            )
        return m_error

    def _update(self, current_exp, **kwargs):
        m_error = self.update_sr(current_exp, **kwargs)
        w_error = self.update_w(current_exp)
        td_error = {"m": np.linalg.norm(m_error), "w": np.linalg.norm(w_error)}
        return td_error

    def get_policy(self, M=None, goal=None):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = M @ goal
        return self.base_get_policy(Q)

    def get_M_states(self):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy(beta=self.beta)
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

    @property
    def Q(self):
        return self.M @ self.w


class QET(BaseAgent):
    """
    Implementation of Q-learning with eligibility traces.
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
        lamb=0.95,
        **kwargs
    ):
        super().__init__(state_size, action_size, lr, gamma, poltype, beta, epsilon)
        self.lamb = lamb

        if Q_init is None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init
        self.et = np.zeros([action_size, state_size])

    def sample_action(self, state):
        Qs = self.Q[:, state]
        return self.base_sample_action(Qs)

    def update_et(self, current_exp):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        r = current_exp[3]

        s_a_1 = np.argmax(self.Q[:, s_1])

        self.et[s_a, s] += 1.0
        td_error = r + self.gamma * self.Q[s_a_1, s_1] - self.Q[s_a, s]
        self.Q += self.lr * td_error * self.et
        self.et *= self.lamb * self.gamma
        return td_error

    def _update(self, current_exp, **kwargs):
        q_error = self.update_et(current_exp)
        td_error = {"q": np.linalg.norm(q_error)}
        return td_error

    def get_policy(self):
        return self.base_get_policy(self.Q)

    def reset(self):
        self.et *= 0.0
