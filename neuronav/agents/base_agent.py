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

    def sample_action(self):
        return None

    def update(self, current_exp):
        self.num_updates += 1
        self._update(current_exp)

    def get_policy(self):
        return None

    def _update(self, current_exp):
        return None

    def reset(self):
        return None
