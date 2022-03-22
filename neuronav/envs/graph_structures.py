import enum


class GraphStructure(enum.Enum):
    two_step = "two_step"
    ring = "ring"
    two_way_linear = "two_way_linear"
    two_step_flip = "two_step_flip"
    linear = "linear"
    t_graph = "t_graph"
    neighborhood = "neighborhood"
    human_a = "human_a"
    human_b = "human_b"
    t_loop = "t_loop"


def two_step():
    rewarding_states = {5: 1}
    edges = [[1, 2], [3, 4], [5, 6], [], [], [], []]
    return rewarding_states, edges


def two_way_linear():
    rewarding_states = {4: 1}
    edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]]
    return rewarding_states, edges


def ring():
    rewarding_states = {4: 1}
    edges = [[1, 5], [0, 2], [1, 3], [2, 4], [3, 5], [4, 0]]
    return rewarding_states, edges


def linear():
    rewarding_states = {5: 1}
    edges = [[1], [2], [3], [4], [5], []]
    return rewarding_states, edges


def t_graph():
    rewarding_states = {5: 1}
    edges = [[1, 0], [2, 1], [3, 4], [5, 3], [6, 4], [], []]
    return rewarding_states, edges


def two_step_flip():
    rewarding_states = {6: 1}
    edges = [[2, 1], [3, 4], [5, 6], [], [], [], []]
    return rewarding_states, edges


def neighborhood():
    rewarding_states = {14: 1}
    edges = [
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [5, 1, 0, 4],
        [10, 4, 0, 1],
        [0, 1, 2, 3],
        [2, 6, 8, 9],
        [5, 8, 7, 9],
        [5, 7, 6, 8],
        [6, 8, 9, 11],
        [5, 6, 7, 9],
        [3, 12, 13, 14],
        [7, 12, 13, 14],
        [10, 11, 13, 14],
        [11, 10, 12, 14],
        [10, 12, 11, 13],
    ]
    return rewarding_states, edges


def human_a():
    rewarding_states = {4: 10, 5: 1}
    edges = [[2], [3], [4], [5], [], []]
    return rewarding_states, edges


def human_b():
    rewarding_states = {3: 15, 5: 30}
    edges = [[1, 2], [3, 4], [4, 5], [3, 3], [4, 4], [5, 5]]
    return rewarding_states, edges


def t_loop():
    rewarding_states = {12: 1, 11: 1}
    edges = [
        [1, 0],
        [2, 1],
        [3, 4],
        [5, 3],
        [6, 4],
        [7, 5],
        [8, 6],
        [9, 7],
        [10, 8],
        [11, 9],
        [12, 10],
        [0, 11],
        [0, 12],
    ]
    return rewarding_states, edges


structure_map = {
    GraphStructure.two_step: two_step,
    GraphStructure.two_way_linear: two_way_linear,
    GraphStructure.ring: ring,
    GraphStructure.linear: linear,
    GraphStructure.t_graph: t_graph,
    GraphStructure.two_step_flip: two_step_flip,
    GraphStructure.neighborhood: neighborhood,
    GraphStructure.human_a: human_a,
    GraphStructure.human_b: human_b,
    GraphStructure.t_loop: t_loop,
}
