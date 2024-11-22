import numpy as np


class GridLangRenderer:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.reward_types = {
            True: "gem",  # positive reward
            False: "lava",  # negative reward
        }

    def _get_region(self, pos):
        """Helper method to determine region in the maze."""
        third = self.grid_size / 3
        x, y = pos[1], pos[0]

        regions = []
        if y < third:
            regions.append("north")
        elif y > 2 * third:
            regions.append("south")
        if x < third:
            regions.append("west")
        elif x > 2 * third:
            regions.append("east")

        return "-".join(regions) if regions else "middle"

    def _get_object_descriptions(self, positions, obj_type, agent_pos, reward_val=None):
        """Helper method to generate descriptions for a group of objects.
        If reward_val is provided, positions should be a dictionary of position: reward pairs.
        """
        descriptions = []

        # Convert positions to a list of (pos, obj_type) tuples
        pos_type_pairs = []
        if reward_val is not None:
            # Handle rewards dictionary {pos: reward}
            for pos, reward in positions.items():
                reward_val = reward[0] if isinstance(reward, list) else reward
                pos_type_pairs.append((pos, self.reward_types[reward_val > 0]))
        else:
            # Handle regular objects list [pos1, pos2, ...]
            pos_type_pairs = [(pos, obj_type) for pos in positions]

        # Generate descriptions
        for pos, item_type in pos_type_pairs:
            direction, distance = self._get_direction_and_distance(pos, agent_pos)
            if direction == "same position":
                descriptions.append(f"There is a {item_type} at your position.")
            else:
                descriptions.append(
                    f"There is a {item_type} {direction} of you, {distance} meters away."
                )

        return descriptions

    def _get_direction_and_distance(self, obj_pos, agent_pos):
        """Helper method to get cardinal direction and distance."""
        diff = np.array(obj_pos) - agent_pos
        y_dir = "north" if diff[0] < 0 else "south" if diff[0] > 0 else ""
        x_dir = "east" if diff[1] > 0 else "west" if diff[1] < 0 else ""

        direction = (
            f"{y_dir}-{x_dir}"
            if (y_dir and x_dir)
            else (y_dir or x_dir or "same position")
        )
        distance = abs(diff[0]) + abs(diff[1]) if direction != "same position" else 0

        return direction, round(distance, 2)

    def make_language_obs(self, agent_pos: list, objects: dict, keys: int):
        agent_pos = np.array(agent_pos)
        descriptions = [
            f"You are in the {self._get_region(agent_pos)} region of a {self.grid_size}x{self.grid_size} meter maze. "
            f"You have {keys} {'key' if keys == 1 else 'keys'}."
        ]
        object_descriptions = []

        # Describe walls
        wall_positions = [
            (i, j)
            for i in range(0, self.grid_size)
            for j in range(0, self.grid_size)
            if [i, j] in objects["walls"]
        ]
        wall_descs = self._get_object_descriptions(wall_positions, "wall", agent_pos)
        if wall_descs:
            object_descriptions.append("\n".join(wall_descs))

        # Describe rewards
        reward_descs = self._get_object_descriptions(
            objects["rewards"], None, agent_pos, reward_val=True
        )
        if reward_descs:
            object_descriptions.append("\n".join(reward_descs))

        # Describe other objects
        for obj_type in ["keys", "doors", "warps"]:
            obj_descs = self._get_object_descriptions(
                objects[obj_type], obj_type[:-1], agent_pos
            )
            if obj_descs:
                object_descriptions.append("\n".join(obj_descs))

        # Handle "other" objects with custom names
        if "other" in objects:
            other_pos_type_pairs = [
                (pos, name) for pos, name in objects["other"].items()
            ]
            other_descs = []
            for pos, item_type in other_pos_type_pairs:
                direction, distance = self._get_direction_and_distance(pos, agent_pos)
                if direction == "same position":
                    other_descs.append(f"There is a {item_type} at your position.")
                else:
                    other_descs.append(
                        f"There is a {item_type} {direction} of you, {distance} meters away."
                    )
            if other_descs:
                object_descriptions.append("\n".join(other_descs))

        if not object_descriptions:
            return f"{descriptions[0]}\nThere are no objects or walls near you."

        return "\n\n".join([descriptions[0]] + object_descriptions)
