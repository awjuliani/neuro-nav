import neuronav.utils as utils
import enum


def four_rooms(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    mid = int(grid_size // 2)
    earl_mid = int(mid // 2) + 1
    if grid_size == 11:
        late_mid = mid + earl_mid - 1
    else:
        late_mid = mid + earl_mid - 2
    blocks_a = [[mid, i] for i in range(grid_size)]
    blocks_b = [[i, mid] for i in range(grid_size)]
    blocks = blocks_a + blocks_b
    bottlenecks = [
        [mid, earl_mid],
        [mid, late_mid],
        [earl_mid, mid],
        [late_mid, mid],
    ]
    for bottleneck in bottlenecks:
        blocks.remove(bottleneck)
    return blocks, agent_start, reward_locs


def four_rooms_split(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    mid = int(grid_size // 2)
    earl_mid = int(mid // 2)
    if grid_size == 11:
        late_mid = mid + earl_mid + 1
    else:
        late_mid = mid + earl_mid
    blocks_a = [[mid, i] for i in range(grid_size)]
    blocks_b = [[i, mid] for i in range(grid_size)]
    blocks_c = [[mid - 1, i] for i in range(grid_size)]
    blocks_d = [[i, mid - 1] for i in range(grid_size)]
    blocks_e = [[mid + 1, i] for i in range(grid_size)]
    blocks_f = [[i, mid + 1] for i in range(grid_size)]
    blocks = blocks_a + blocks_b + blocks_c + blocks_d + blocks_e + blocks_f
    bottlenecks = [
        [earl_mid, mid - 1],
        [earl_mid, mid],
        [earl_mid, mid + 1],
        [late_mid, mid - 1],
        [late_mid, mid],
        [late_mid, mid + 1],
    ]
    for bottleneck in bottlenecks:
        blocks.remove(bottleneck)
    return blocks, agent_start, reward_locs


def empty(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    blocks = []
    reward_locs = {(1, 1): 1.0}
    return blocks, agent_start, reward_locs


def outer_ring(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    blocks = []
    extra_depth = 2
    for i in range(grid_size):
        for j in range(grid_size):
            if (
                extra_depth < i < grid_size - 1 - extra_depth
                and extra_depth < j < grid_size - 1 - extra_depth
            ):
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def u_maze(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(grid_size - 2, 1): 1.0}
    blocks = []
    extra_depth = 2
    for i in range(grid_size):
        for j in range(grid_size):
            if extra_depth < j < grid_size - 1 - extra_depth and i > extra_depth:
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def two_rooms(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    mid = int(grid_size // 2)
    blocks_b = [[mid, i] for i in range(grid_size)]
    blocks = blocks_b
    blocks.remove([mid, mid])
    if grid_size == 17:
        blocks_a = [[mid - 1, i] for i in range(grid_size)]
        blocks_c = [[mid + 1, i] for i in range(grid_size)]
        blocks += blocks_a
        blocks += blocks_c
        blocks.remove([mid - 1, mid])
        blocks.remove([mid + 1, mid])
    return blocks, agent_start, reward_locs


def obstacle(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    mid = int(grid_size // 2)
    if grid_size == 11:
        blocks_b = [[mid, i] for i in range(2, grid_size - 2)]
    else:
        blocks_b = [[mid, i] for i in range(3, grid_size - 3)]
    blocks = blocks_b
    return blocks, agent_start, reward_locs


def s_maze(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    mid_a = int(grid_size // 3)
    mid_b = int(grid_size // 3 + grid_size // 3) + 1
    blocks_a = [[i, mid_a] for i in range(grid_size // 2 + 1 + grid_size // 4)]
    blocks_b = [[i, mid_b] for i in range(grid_size // 4 + 1, grid_size - 1)]
    blocks = blocks_a + blocks_b
    return blocks, agent_start, reward_locs


def hairpin(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    mid_a = int(grid_size // 5)
    mid_b = 2 * int(grid_size // 5)
    mid_c = 3 * int(grid_size // 5)
    mid_d = 4 * int(grid_size // 5)
    blocks_a = [[i, mid_a] for i in range(grid_size // 2 + 1 + grid_size // 4)]
    blocks_b = [[i, mid_b] for i in range(grid_size // 4 + 1, grid_size - 1)]
    blocks_c = [[i, mid_c] for i in range(grid_size // 2 + 1 + grid_size // 4)]
    blocks_d = [[i, mid_d] for i in range(grid_size // 4 + 1, grid_size - 1)]
    blocks = blocks_a + blocks_b + blocks_c + blocks_d
    return blocks, agent_start, reward_locs


def circle(grid_size):
    agent_start = [grid_size - 2, grid_size // 2]
    reward_locs = {(1, grid_size // 2): 1.0}
    blocks = []
    mask = utils.create_circular_mask(grid_size, grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            if mask[i, j] == 0:
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def ring(grid_size):
    agent_start = [grid_size - 2, grid_size // 2]
    reward_locs = {(1, grid_size // 2): 1.0}
    blocks = []
    big_mask = utils.create_circular_mask(grid_size, grid_size)
    small_mask = utils.create_circular_mask(grid_size, grid_size, radius=grid_size // 4)
    for i in range(grid_size):
        for j in range(grid_size):
            if big_mask[i, j] == 0 or small_mask[i, j] != 0:
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def t_maze(grid_size):
    agent_start = [grid_size - 2, grid_size // 2]
    reward_locs = {(1, 1): 1.0}
    width = 3
    half_width = width // 2
    middle_pos = grid_size // 2
    blocks = []
    for i in range(grid_size):
        for j in range(grid_size):
            if i >= width + 1 and (
                j < middle_pos - half_width or j > middle_pos + half_width
            ):
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def i_maze(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}
    width = 3
    half_width = width // 2
    middle_pos = grid_size // 2
    blocks = []
    for i in range(grid_size):
        for j in range(grid_size):
            if width + 1 <= i <= grid_size - width - 2 and (
                j < middle_pos - half_width or j > middle_pos + half_width
            ):
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def hallways(grid_size):
    agent_start = [grid_size - 2, grid_size - 2]
    reward_locs = {(1, 1): 1.0}

    blocks = []
    extra_depth = 1
    for i in range(grid_size):
        for j in range(grid_size):
            check_outer = (
                extra_depth < i < grid_size - extra_depth - 1
                and extra_depth < j < grid_size - extra_depth - 1
            )
            check_inner = i == grid_size // 2 or j == grid_size // 2
            if check_outer and not check_inner:
                blocks.append([i, j])
    return blocks, agent_start, reward_locs


def detour(grid_size):
    agent_start = [grid_size - 2, grid_size // 2]
    reward_locs = {(1, grid_size // 2): 1.0}
    blocks = []
    extra_depth = 1
    for i in range(grid_size):
        for j in range(grid_size):
            if (
                extra_depth < i < grid_size - 1 - extra_depth
                and extra_depth < j < grid_size - 1 - extra_depth
            ):
                blocks.append([i, j])
    for i in range(grid_size):
        if [i, grid_size // 2] in blocks:
            blocks.remove([i, grid_size // 2])
    return blocks, agent_start, reward_locs


def detour_block(grid_size):
    agent_start = [grid_size - 2, grid_size // 2]
    reward_locs = {(1, grid_size // 2): 1.0}
    blocks = []
    extra_depth = 1
    for i in range(grid_size):
        for j in range(grid_size):
            if (
                extra_depth < i < grid_size - 1 - extra_depth
                and extra_depth < j < grid_size - 1 - extra_depth
            ):
                blocks.append([i, j])
    for i in range(grid_size):
        if [i, grid_size // 2] in blocks and i != grid_size // 2:
            blocks.remove([i, grid_size // 2])
    return blocks, agent_start, reward_locs


class GridTopography(enum.Enum):
    empty = "empty"
    four_rooms = "four_rooms"
    outer_ring = "outer_ring"
    two_rooms = "two_rooms"
    u_maze = "u_maze"
    t_maze = "t_maze"
    hallways = "hallways"
    ring = "ring"
    s_maze = "s_maze"
    circle = "circle"
    i_maze = "i_maze"
    hairpin = "hairpin"
    detour = "detour"
    detour_block = "detour_block"
    four_rooms_split = "four_rooms_split"
    obstacle = "obstacle"


topography_map = {
    GridTopography.empty: empty,
    GridTopography.four_rooms: four_rooms,
    GridTopography.outer_ring: outer_ring,
    GridTopography.two_rooms: two_rooms,
    GridTopography.u_maze: u_maze,
    GridTopography.t_maze: t_maze,
    GridTopography.hallways: hallways,
    GridTopography.ring: ring,
    GridTopography.s_maze: s_maze,
    GridTopography.circle: circle,
    GridTopography.i_maze: i_maze,
    GridTopography.hairpin: hairpin,
    GridTopography.detour: detour,
    GridTopography.detour_block: detour_block,
    GridTopography.four_rooms_split: four_rooms_split,
    GridTopography.obstacle: obstacle,
}


class GridSize(enum.Enum):
    micro = 7
    small = 11
    large = 17


def generate_topography(topography=GridTopography.empty, grid_size=GridSize.small):
    grid_size = grid_size.value
    if type(topography) == str:
        topography = GridTopography(topography)
    blocks, agent_start, reward_locs = topography_map[topography](grid_size)
    blocks = add_outer_blocks(blocks, grid_size)
    return blocks, agent_start, reward_locs


def add_outer_blocks(blocks, grid_size):
    for i in range(grid_size):
        for j in range(grid_size):
            if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1:
                blocks.append([i, j])
    return blocks
