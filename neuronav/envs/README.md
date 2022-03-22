# Navigation Environments

A set of neuroscience inspired navigation and decision making tasks in environments with either graph or grid structures.

Both of these environments support the default `gym` interface. You can learn more about gym [here](https://github.com/openai/gym).

## GridEnv

`GridEnv` consists of a simple 2D grid environment with various topographies and observation types.

### Topographies

The `GridEnv` class can generate a variety of different maze layouts by setting the topography. Below is a list of the topographies which are supported. To add your own, edit [grid_topographies.py](./grid_topographies.py).

| Topography | Image Small (11x11) | Image Large (17 x 17)
| --- | --- | --- |
| empty | ![empty](/imgs/empty.png) | ![empty_large](/imgs/empty_large.png) |
| four_rooms | ![four_rooms](/imgs/four_rooms.png) | ![four_rooms_large](/imgs/four_rooms_large.png) |
| outer_ring | ![outer_ring](/imgs/outer_ring.png) | ![outer_ring_large](/imgs/outer_ring_large.png) |
| two_rooms| ![two_rooms](/imgs/two_rooms.png) | ![two_rooms_large](/imgs/two_rooms_large.png) |
| u_maze | ![u_maze](/imgs/u_maze.png) | ![u_maze_large](/imgs/u_maze_large.png) |
| t_maze | ![t_maze](/imgs/t_maze.png) | ![t_maze_large](/imgs/t_maze_large.png) |
| hallways | ![hallways](/imgs/hallways.png) | ![hallways_large](/imgs/hallways_large.png) |
| ring | ![ring](/imgs/ring.png) | ![ring_large](/imgs/ring_large.png) |
| s_maze | ![s_maze](/imgs/s_maze.png) | ![s_maze_large](/imgs/s_maze_large.png) |
| circle | ![circle](/imgs/circle.png) | ![circle_large](/imgs/circle_large.png) |
| hairpin | ![hairpin](/imgs/hairpin.png) | ![hairpin_large](/imgs/hairpin_large.png) |
| i_maze | ![i_maze](/imgs/i_maze.png) | ![i_maze_large](/imgs/i_maze_large.png) |
| detour | ![detour](/imgs/detour.png) | ![detour_large](/imgs/detour_large.png) |
| detour_block | ![detour_block](/imgs/detour_block.png) | ![detour_block_large](/imgs/detour_block_large.png) |

### Observation Types

The `GridEnv` class also supports a variety of observation types for the agent. These vary in the amount of information provided to the agent, and in their format. To add your own, edit [grid_env.py](./grid_env.py).

| Observation | Type | Shape (Fixed Orientation) | Shape (Dynamic Orientation) | Description |
| --- | --- | --- | --- | --- |
| index | Tabular | `[1]` | `[1]` | An integer number representing the current state of the agent in the environment. |
| onehot | Function-Approx | `[n * n]` | `[n * n * 4]` | A one-hot encoding of the current state of the agent in the environment. |
| twohot | Function-Approx | `[n + n]` | `[n + n + 4]` | Two concatenated one-hot encodings of the agent's x and y coordinates in the environment. |
| geometric | Function-Approx | `[2]` | `[3]` | Two real-valued numbers between 0 and 1 representing the x and y coordinates of the agent in the environment. |
| boundary | Function-Approx | `[n * 4]` | `[n * 4 + 4]` | A matrix corresponding to the one-hot encodings of the distances of the agent from the nearest wall in the four cardinal directions |
| visual | Function-Approx | `[n, n, 3]` | `[n, n, 3` | A 3D tensor corresponding to the RGB image of the environment. |
| images | Function-Approx | `[32, 32, 3]` | `[32, 32, 3` | A 3D tensor corresponding to a unique CIFAR10 image per state. |

* Where `n` is the length of the grid.

## GraphEnv

`GraphEnv` consists of a simple graph environment with various layout structures and observation types. 

### Structures

The graph structures can be added to by editing [graph_structures.py](./graph_structures.py)

| Structure | Image |
| --- | --- |
| two_step | ![two_step](/imgs/two_step.png) |
| linear | ![linear](/imgs/linear.png) |
| t_graph | ![t_graph](/imgs/t_graph.png) |
| neighborhood | ![neighborhood](/imgs/neighborhood.png) |
| ring | ![ring](/imgs/ring_graph.png) |
| two_way_linear | ![two_way_linear](/imgs/two_way_linear.png) |

### Observation Types

The `GraphEnv` class also supports a variety of observation types for the agent. These vary in the amount of information provided to the agent, and in their format. To add your own, edit [graph_env.py](./graph_env.py).

| Observation | Type | Shape | Description |
| --- | --- | --- | --- |
| index | Tabular | `[1]` | An integer number representing the current state of the agent in the environment. |
| onehot | Function-Approx | `[n]` | A one-hot encoding of the current state of the agent in the environment. |
| images | Function-Approx | `[32, 32, 3]` | `[32, 32, 3` | A 3D tensor corresponding to a unique CIFAR10 image per state. |

* Where `n` is the number of nodes in the graph.
