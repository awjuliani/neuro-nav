from typing import Dict
import networkx as nx
import neuronav.utils as utils
import enum
import copy
import numpy as np
from gym import Env, spaces
from neuronav.envs.graph_structures import GraphStructure, structure_map
import random


class GraphObsType(enum.Enum):
    onehot = "onehot"
    index = "index"
    images = "images"


class GraphEnv(Env):
    """
    Graph Environment.
    """

    def __init__(
        self,
        graph_structure: GraphStructure = GraphStructure.linear,
        obs_type: GraphObsType = GraphObsType.index,
        seed: int = None,
        use_noop: bool = False,
    ):
        self.use_noop = use_noop
        self.rng = np.random.RandomState(seed)
        if isinstance(graph_structure, str):
            graph_structure = GraphStructure(graph_structure)
        if isinstance(obs_type, str):
            obs_type = GraphObsType(obs_type)
        self.generate_graph(graph_structure)
        self.running = False
        self.obs_mode = obs_type
        self.base_objects = {"rewards": {}}
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

    def generate_graph(self, structure: GraphStructure):
        self.struct_objects, self.edges = structure_map[structure]()
        self.agent_start_pos = 0
        action_size = 0
        for edge in self.edges:
            if len(edge) > action_size:
                action_size = len(edge)
        self.action_space = spaces.Discrete(action_size + self.use_noop)
        self.state_size = len(self.edges)

    @property
    def observation(self):
        """
        Returns an observation corresponding to the current state.
        """
        if self.obs_mode == GraphObsType.onehot:
            return utils.onehot(self.agent_pos, self.state_size)
        elif self.obs_mode == GraphObsType.index:
            return self.agent_pos
        elif self.obs_mode == GraphObsType.images:
            return np.rot90(self.images[self.agent_pos], k=3)
        else:
            return None

    def get_free_spot(self):
        return self.rng.randint(0, self.state_size)

    def reset(
        self,
        agent_pos: int = None,
        objects: Dict = None,
        random_start: bool = False,
        time_penalty: float = 0.0,
        stochasticity: float = 0.0,
    ):
        """
        Resets the environment to initial configuration.
        """
        self.running = True
        self.stochasticity = stochasticity
        self.time_penalty = time_penalty
        if agent_pos != None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos
        self.done = False
        if objects != None:
            use_objects = copy.deepcopy(self.base_objects)
            for key in objects.keys():
                if key in use_objects.keys():
                    use_objects[key] = objects[key]
            self.objects = use_objects
        else:
            self.objects = self.struct_objects
        return self.observation

    def render(self):
        """
        Renders the graph environment to a pyplot figure.
        """
        graph = nx.DiGraph()
        color_map = []
        for idx, edge in enumerate(self.edges):
            graph.add_node(idx)
            if idx == self.agent_pos:
                color_map.append("cornflowerblue")
            elif idx in self.objects["rewards"]:
                if self.objects["rewards"][idx] > 0:
                    color_map.append(
                        [0, np.clip(self.objects["rewards"][idx], 0, 1), 0]
                    )
                elif self.objects["rewards"][idx] < 0:
                    color_map.append(
                        [-np.clip(self.objects["rewards"][idx], -1, 0), 0, 0]
                    )
                else:
                    color_map.append("silver")
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
            font_color="white",
            pos=nx.spring_layout(graph, k=3, pos=nx.kamada_kawai_layout(graph)),
        )

    def step(self, action: int):
        """
        Takes a step in the environment given an action.
        """
        if self.running is False:
            print("Please call env.reset() before env.step().")
            return None, None, None, None
        elif self.done:
            print("Episode fininshed. Please reset the environment.")
            return None, None, None, None
        else:
            if self.use_noop and action == self.action_space.n - 1:
                pass
            else:
                if self.stochasticity > self.rng.rand():
                    action = self.rng.randint(0, len(self.edges[self.agent_pos]))
                candidate_positions = self.edges[self.agent_pos][action]
                if type(candidate_positions) == tuple:
                    candidate_position = self.rng.choice(
                        candidate_positions[0], p=candidate_positions[1]
                    )
                else:
                    candidate_position = candidate_positions
                self.agent_pos = candidate_position
                reward = 0
                if self.agent_pos in self.objects["rewards"]:
                    reward += self.objects["rewards"][self.agent_pos]
                reward -= self.time_penalty
                if len(self.edges[self.agent_pos]) == 0:
                    self.done = True
            return self.observation, reward, self.done, {}
