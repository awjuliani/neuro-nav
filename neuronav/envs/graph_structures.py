import enum
import numpy as np


class GraphStructure(enum.Enum):
    two_step = "two_step"
    ring = "ring"
    two_way_linear = "two_way_linear"
    linear = "linear"
    t_graph = "t_graph"
    neighborhood = "neighborhood"
    human_a = "human_a"
    human_b = "human_b"
    t_loop = "t_loop"
    variable_magnitude = "variable_magnitude"
    three_arm_bandit = "three_arm_bandit"


def two_step():
    rewarding_states = {3: 1, 4: -1, 5: 0.5, 6: 0.5}
    edges = [[1, 2], [3, 4], [5, 6], [], [], [], []]
    return rewarding_states, edges


def three_arm_bandit():
    rewarding_states = {1: 1, 2: 0.5, 3: -0.5}
    edges = [[1, 2, 3], [], [], []]
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
        [5, 9, 6, 8],
        [7, 6, 5, 9],
        [6, 7, 8, 11],
        [3, 12, 13, 14],
        [9, 12, 13, 14],
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


def variable_magnitude():
    # Values taken from original author's code availabe here: https://osf.io/ux5rg/
    fmax = 10.0
    sigma = 200
    utility_func = lambda r: (fmax * np.sign(r) * np.abs(r) ** (0.5)) / (
        np.abs(r) ** (0.5) + sigma ** (0.5)
    )
    rewarding_states = {
        1: utility_func(0.1),
        2: utility_func(0.3),
        3: utility_func(1.2),
        4: utility_func(2.5),
        5: utility_func(5),
        6: utility_func(10),
        7: utility_func(20),
    }
    edges = [
        [((1, 2, 3, 4, 5, 6, 7), (0.067, 0.090, 0.148, 0.154, 0.313, 0.151, 0.077))],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    return rewarding_states, edges


structure_map = {
    GraphStructure.two_step: two_step,
    GraphStructure.two_way_linear: two_way_linear,
    GraphStructure.ring: ring,
    GraphStructure.linear: linear,
    GraphStructure.t_graph: t_graph,
    GraphStructure.neighborhood: neighborhood,
    GraphStructure.human_a: human_a,
    GraphStructure.human_b: human_b,
    GraphStructure.t_loop: t_loop,
    GraphStructure.variable_magnitude: variable_magnitude,
    GraphStructure.three_arm_bandit: three_arm_bandit,
}
