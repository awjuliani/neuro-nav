# Benchmark Environments

A set of cognitive neuroscience inspired navigation and decision making tasks in environments that have either a grid or graph structure.

These environments support the default `gym` interface which is commonly used in open source reinforcement learning packages. You can learn more about gym [here](https://github.com/openai/gym).

## GridEnv

`GridEnv` consists of a simple 2D grid environment with various topographies and observation types.

### Templates

The `GridEnv` class can generate a variety of different maze layouts by selecting one of the predefined template layouts. Below is a list of the templates which are included. Templates can be set as an enum when initializing the environment. For example: 

```env = GridEnv(template=GridTemplate.empty)```

 To add your own templates, edit [grid_templates.py](./grid_templates.py).

| Template | Image Small (11x11) | Image Large (17 x 17) | Reference |
| --- | --- | --- | --- |
| empty | ![empty](/images/grid_small/empty.png) | ![empty](/images/grid_large/empty.png) |
| four_rooms | ![four_rooms](/images/grid_small/four_rooms.png) | ![four_rooms](/images/grid_large/four_rooms.png) |
| outer_ring | ![outer_ring](/images/grid_small/outer_ring.png) | ![outer_ring](/images/grid_large/outer_ring.png) |
| two_rooms| ![two_rooms](/images/grid_small/two_rooms.png) | ![two_rooms](/images/grid_large/two_rooms.png) |
| u_maze | ![u_maze](/images/grid_small/u_maze.png) | ![u_maze](/images/grid_large/u_maze.png) |
| t_maze | ![t_maze](/images/grid_small/t_maze.png) | ![t_maze](/images/grid_large/t_maze.png) |
| hallways | ![hallways](/images/grid_small/hallways.png) | ![hallways](/images/grid_large/hallways.png) |
| ring | ![ring](/images/grid_small/ring.png) | ![ring](/images/grid_large/ring.png) |
| s_maze | ![s_maze](/images/grid_small/s_maze.png) | ![s_maze](/images/grid_large/s_maze.png) |
| circle | ![circle](/images/grid_small/circle.png) | ![circle](/images/grid_large/circle.png) |
| hairpin | ![hairpin](/images/grid_small/hairpin.png) | ![hairpin](/images/grid_large/hairpin.png) |
| i_maze | ![i_maze](/images/grid_small/i_maze.png) | ![i_maze](/images/grid_large/i_maze.png) |
| detour | ![detour](/images/grid_small/detour.png) | ![detour](/images/grid_large/detour.png) | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) |
| detour_block | ![detour_block](/images/grid_small/detour_block.png) | ![detour_block](/images/grid_large/detour_block.png) | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) |
| four_rooms_split | ![four_rooms_split](/images/grid_small/four_rooms_split.png) | ![four_rooms_split](/images/grid_large/four_rooms_split.png) |
| obstacle | ![obstacle](/images/grid_small/obstacle.png) | ![obstacle](/images/grid_large/obstacle.png) | [Stachenfeld et al., 2017](https://www.nature.com/articles/nn.4650) |
| two_step | ![two_step](/images/grid_small/two_step.png) | ![two_step](/images/grid_large/two_step.png) | |
| narrow | ![narrow](/images/grid_small/narrow.png) | ![narrow](/images/grid_large/narrow.png) | [Zorowitz et al., 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8143038/) |

### Objects

There are a number of possible objects which can be placed at various locations in a grid environment by utilizing an `objects` dictionary. They are as follows:

| Object | Usage | Description | Color | Image |
| --- | --- | --- | --- | --- |
| reward | `'rewards': {(x, y): [v, o, t]}`, where `x, y` is the location of the reward, `o` is whether the reward should be visible, `v` is the value of the reward, and `t` is whether the episode should terminate when the agent reaches the reward. | A reward object. Provides the agent with the reward value if it occupies the location. | Blue (Pos) / Red (Neg) | ![reward_pos](/images/objects/reward_pos.png) ![reward_neg](/images/objects/reward_neg.png) |
| marker | `'markers': {(x, y): (r, g, b)}`, where `x, y` is the location of the marker, and `r, g, b` are the color values for the marker. | A marker object. Used to provide contextual information to the agent. | Variable | N/A |
| key | `'keys': [(x, y)]`, where `x, y` is the location of the key. | A consumable key object. Allows the agent to open a door. | Yellow | ![key](/images/objects/key.png) |
| door | `'doors': {(x, y): o}`, where `x, y` is the location of the door, and `o` is the orientation of the door (either 'h' or 'v'). | A door object. Agent cannot enter a location with a door unless it posesses a key, which is consumed upon entry. | Green | ![door](/images/objects/door.png) |
| warp | `'warps': {(x, y): (a, b)}`, where `x, y` is the location of the warp, and `a, b` is the location of the warp target. | A warp object. Transports the agent from the location of the warp to a specificed other location in the environment. | Purple | ![warp](/images/objects/warp.png) |

### Observation Types

The `GridEnv` class also supports a variety of observation types for the agent. These vary in the amount of information provided to the agent, and in their format. 

Observation types can be set as an enum when initializing the environment. For example: 

```env = GridEnv(obs_type=GridObservation.index)```

To add your own, edit [grid_env.py](./grid_env.py).

| Observation | Type | Shape (Fixed Orientation) | Shape (Dynamic Orientation) | Description |
| --- | --- | --- | --- | --- |
| index | Tabular | `[1]` | `[1]` | An integer number representing the current state of the agent in the environment. |
| onehot | Function-Approx | `[n * n]` | `[n * n * 4]` | A one-hot encoding of the current state of the agent in the environment. |
| twohot | Function-Approx | `[n + n]` | `[n + n + 4]` | Two concatenated one-hot encodings of the agent's x and y coordinates in the environment. |
| geometric | Function-Approx | `[2]` | `[3]` | Two real-valued numbers between 0 and 1 representing the x and y coordinates of the agent in the environment. |
| boundary | Function-Approx | `[n * 4]` | `[n * 4 + 4]` | A matrix corresponding to the one-hot encodings of the distances of the agent from the nearest wall in the four cardinal directions |
| visual | Function-Approx | `[110, 110, 3]` | `[110, 110, 3` | A 3D tensor corresponding to the RGB image of the environment. |
| images | Function-Approx | `[32, 32, 3]` | `[32, 32, 3` | A 3D tensor corresponding to a unique CIFAR10 image per state. |
| window | Function-Approx | `[64, 64, 3]` | `[64, 64, 3]` | A 3D tensor corresponding to the 5x5 local window around the agent. |
| symbolic | Function-Approx | `[n, n, 6]` | `[n, n, 6]` | A 5D tensor corresponding to a symbolic representation of the environment state.
| symbolic_window | Function-Approx | `[5, 5, 6]` | `[5, 5, 6]` | A 5D tensor corresponding to a symbolic representation of the 5x5 environment state around the agent.
| window_tight | Function-Approx | `[64, 64, 3]` | `[64, 64, 3]` | A 3D tensor corresponding to the 3x3 local window around the agent. |
| symbolic_window_tight | Function-Approx | `[3, 3, 6]` | `[3, 3, 6]` | A 5D tensor corresponding to a symbolic representation of the 3x3 environment state around the agent.

* Where `n` is the length of the grid.

## GraphEnv

`GraphEnv` consists of a simple graph environment with various layout structures and observation types. 

### Templates

Graph templates can be set as an enum when initializing the environment. For example: 

```env = GraphEnv(template=GraphTemplate.neighborhood)```

The graph templates can be added to by editing [graph_templates.py](./graph_templates.py)

| Template | Image | Reference |
| --- | --- | --- |
| two_step | ![two_step](/images/graph/two_step.png) |
| linear | ![linear](/images/graph/linear.png) |
| t_graph | ![t_graph](/images/graph/t_graph.png) |
| neighborhood | ![neighborhood](/images/graph/neighborhood.png) | [Schapiro et al., 2013](https://www.nature.com/articles/nn.3331) |
| ring | ![ring](/images/graph/ring_graph.png) |
| two_way_linear | ![two_way_linear](/images/graph/two_way_linear.png) |
| human_a | ![human_a](/images/graph/human_a.png) | [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) |
| human_b | ![human_b](/images/graph/human_b.png) | [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) |
| t_loop | ![t_loop](/images/graph/t_loop.png) |
| variable_magnitude | ![variable_magnitude](/images/graph/variable_magnitude.png) | [Dabney et al., 2020](https://www.nature.com/articles/s41586-019-1924-6) |


### Observation Types

The `GraphEnv` class also supports a variety of observation types for the agent. These vary in the amount of information provided to the agent, and in their format. 

Observation types can be set as an enum when initializing the environment. For example: 

```env = GraphEnv(obs_type=GraphObservation.index)```

To add your own, edit [graph_env.py](./graph_env.py).

| Observation | Type | Shape | Description |
| --- | --- | --- | --- |
| index | Tabular | `[1]` | An integer number representing the current state of the agent in the environment. |
| onehot | Function-Approx | `[n]` | A one-hot encoding of the current state of the agent in the environment. |
| images | Function-Approx | `[32, 32, 3]` | `[32, 32, 3` | A 3D tensor corresponding to a unique CIFAR10 image per state. |

* Where `n` is the number of nodes in the graph.
