import torch


class BaseAgent(object):
    def __init__(
        self,
        model,
        agent_params,
    ):
        self.model = model
        self.batch_size = agent_params["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def update(self, obs, action, reward, done):
        pass

    def reset(self):
        pass
