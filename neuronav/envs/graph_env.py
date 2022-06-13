import networkx as nx
import neuronav.utils as utils
import enum
import numpy as np
from gym import Env, spaces
from neuronav.envs.graph_structures import GraphStructure, structure_map


class GraphObsType(enum.Enum):
    onehot = "onehot"
    index = "index"
    images = "images"


class GraphEnv(Env):
    """
    Graph Environment.
    """

    def __init__(
        self, graph_structure=GraphStructure.linear, obs_type=GraphObsType.index
    ):
        if isinstance(graph_structure, str):
            graph_structure = GraphStructure(graph_structure)
        if isinstance(obs_type, str):
            obs_type = GraphObsType(obs_type)
        self.generate_graph(graph_structure)
        self.running = False
        self.obs_mode = obs_type
        if obs_type == GraphObsType.onehot:
            self.observation_space = spaces.Box(
                0, 1, shape=(self.state_size,), dtype=np.int32
            )
        elif obs_type == GraphObsType.index:
            self.observation_space = spaces.Box(
                0, self.state_size, shape=(1,), dtype=np.int32
            )
        elif obs_type == GraphObsType.images:
            self.observation_space = spaces.Box(0, 1, shape=(32, 32, 3))
            self.images = utils.cifar10()

    def generate_graph(self, structure):
        self.rewarding_states, self.edges = structure_map[structure]()
        self.agent_start_pos = 0
        action_size = 0
        for edge in self.edges:
            if len(edge) > action_size:
                action_size = len(edge)
        self.action_space = spaces.Discrete(action_size)
        self.state_size = len(self.edges)
        self.reward_nodes = [0 for _ in range(self.state_size)]
        for state in self.rewarding_states:
            self.reward_nodes[state] = self.rewarding_states[state]

    @property
    def observation(self):
        if self.obs_mode == GraphObsType.onehot:
            return utils.onehot(self.agent_pos, self.state_size)
        elif self.obs_mode == GraphObsType.index:
            return self.agent_pos
        elif self.obs_mode == GraphObsType.images:
            return np.rot90(self.images[self.agent_pos], k=3)
        else:
            return None

    def get_free_spot(self):
        return np.random.randint(0, self.state_size)

    def reset(self, agent_pos=None, reward_locs=None, random_start=False):
        self.running = True
        if agent_pos != None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos
        self.done = False
        if reward_locs != None:
            self.reward_nodes = reward_locs
        else:
            for state in self.rewarding_states:
                self.reward_nodes[state] = self.rewarding_states[state]
        return self.observation

    def render(self):
        graph = nx.DiGraph()
        color_map = []
        for idx, edge in enumerate(self.edges):
            graph.add_node(idx)
            if idx == self.agent_pos:
                color_map.append("cornflowerblue")
            elif self.reward_nodes[idx] > 0:
                color_map.append("green")
            elif self.reward_nodes[idx] < 0:
                color_map.append("red")
            else:
                color_map.append("silver")
        for idx, edge in enumerate(self.edges):
            for target in edge:
                if type(target) == tuple:
                    for subtarget in target[0]:
                        graph.add_edge(idx, subtarget)
                else:
                    graph.add_edge(idx, target)
        nx.draw(
            graph,
            with_labels=True,
            node_color=color_map,
            node_size=750,
            pos=nx.spring_layout(graph, k=1, pos=nx.kamada_kawai_layout(graph)),
        )

    def step(self, action):
        if self.running is False:
            print("Please call env.reset() before env.step().")
            return None, None, None, None
        elif self.done:
            print("Episode fininshed. Please reset the environment.")
            return None, None, None, None
        else:
            candidate_positions = self.edges[self.agent_pos][action]
            if type(candidate_positions) == tuple:
                candidate_position = np.random.choice(
                    candidate_positions[0], p=candidate_positions[1]
                )
            else:
                candidate_position = candidate_positions
            self.agent_pos = candidate_position
            reward = self.reward_nodes[self.agent_pos]
            if np.abs(reward) == 1 or len(self.edges[self.agent_pos]) == 0:
                self.done = True
            return self.observation, reward, self.done, {}
