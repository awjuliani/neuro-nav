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
        super().__init__(state_size, action_size, lr, gamma, poltype, beta)

        if Q_init is None:
            self.Q = np.zeros((action_size, state_size))
        elif np.isscalar(Q_init):
            self.Q = Q_init * npr.randn(action_size, state_size)
        else:
            self.Q = Q_init

    def sample_action(self, state):
        Qs = self.Q[:, state]
        if self.poltype == "softmax":
            action = npr.choice(self.action_size, p=utils.softmax(self.beta * Qs))
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

        q_error = r + self.gamma * self.Q[s_a_1, s_1] - self.Q[s_a, s]

        if not prospective:
            # actually perform update to Q if not prospective
            self.Q[s_a, s] += self.lr * q_error
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
        super().__init__(state_size, action_size, lr, gamma, poltype, beta)
        self.c_w = np.zeros([state_size])
        self.a_w = np.zeros([state_size, action_size])

    def critic(self, state):
        return np.matmul(state, self.c_w)

    def actor(self, state):
        return np.matmul(state, self.a_w)

    def sample_action(self, state):
        if type(state) != np.array:
            state = utils.onehot(state, self.state_size)
        logits = self.actor(state)
        if self.poltype == "softmax":
            probs = utils.softmax(logits * self.beta, axis=-1)
            action = np.random.choice(np.arange(0, self.action_size), p=probs)
        else:
            if npr.rand() < self.epsilon:
                action = npr.choice(self.action_size)
            else:
                action = npr.choice(np.flatnonzero(np.isclose(logits, logits.max())))
        return action

    def _update(self, current_exp):
        state, action, state_next, reward, done = current_exp
        if type(state) != np.array:
            state = utils.onehot(state, self.state_size)
            state_next = utils.onehot(state_next, self.state_size)
        if not done:
            td_target = reward + self.gamma * self.critic(state_next)
            td_estimate = self.critic(state)
            td_error = td_target - td_estimate
        else:
            td_error = reward - self.critic(state)
        self.c_w += self.lr * td_error * state
        self.a_w[:, action] += self.lr * td_error * state


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
        super().__init__(state_size, action_size, lr, gamma, poltype, beta)
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

    def Q_estimates(self, state):
        return self.M[:, state, :] @ self.w

    def sample_action(self, state):
        if self.poltype == "softmax":
            Qs = self.Q_estimates(state)
            action = npr.choice(self.action_size, p=utils.softmax(self.beta * Qs))
        else:
            if npr.rand() < self.epsilon:
                action = npr.choice(self.action_size)
            else:
                Qs = self.Q_estimates(state)
                action = npr.choice(np.flatnonzero(np.isclose(Qs, Qs.max())))
        return action

    def update_w(self, current_exp):
        s, a, s_1, r, _ = current_exp
        if self.weights == "direct":
            error = r - self.w[s_1]
            self.w[s_1] += self.lr * error
        elif self.weights == "td":
            Vs = self.Q_estimates(s).max()
            Vs_1 = self.Q_estimates(s_1).max()
            delta = r + self.gamma * Vs_1 - Vs
            # epsilon and beta are hard-coded, need to improve this
            M = self.get_M_states(epsilon=1e-1, beta=5)
            error = delta * M[s]
            self.w += self.lr * error
        return np.linalg.norm(error)

    def update_sr(self, current_exp, next_exp=None, prospective=False):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]

        # determines whether update is on-policy or off-policy
        if next_exp is None:
            s_a_1 = np.argmax(self.Q_estimates(s_1))
        else:
            s_a_1 = next_exp[1]

        r = current_exp[3]
        d = current_exp[4]
        I = utils.onehot(s, self.state_size)

        if d:
            m_error = (
                I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
            )
        else:
            if self.goal_biased_sr:
                m_error = I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :]
            else:
                m_error = I + self.gamma * self.M[:, s_1, :].mean(0) - self.M[s_a, s, :]

        if not prospective:
            # actually perform update to SR if not prospective
            self.M[s_a, s, :] += self.lr * m_error
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
        if self.poltype == "softmax":
            policy = utils.softmax(self.beta * Q, axis=0)
        else:
            mask = Q == Q.max(0)
            greedy = mask / mask.sum(0)
            policy = (1 - self.epsilon) * greedy + (
                1 / self.action_size
            ) * self.epsilon * np.ones((self.action_size, self.state_size))
        return policy

    def get_M_states(self):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy(beta=self.beta)
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

    @property
    def Q(self):
        return self.M @ self.w
