import numpy as np


class GridLangRenderer:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.reward_types = {True: "gem", False: "lava"}
        self.obj_name_mapping = {
            "keys": "door key",
            "doors": "locked door",
            "warps": "warp pad",
        }

    def _get_region(self, pos):
        """Helper method to determine region in the maze."""
        third = self.grid_size / 3
        x, y = pos[1], pos[0]

        # Determine vertical region (north/center/south)
        if y < third:
            v_region = "north"
        elif y > 2 * third:
            v_region = "south"
        else:
            v_region = "center"

        # Determine horizontal region (west/center/east)
        if x < third:
            h_region = "west"
        elif x > 2 * third:
            h_region = "east"
        else:
            h_region = "center"

        # Combine regions based on position
        if v_region == "center" and h_region == "center":
            return "center"
        elif v_region == "center":
            return f"center {h_region}"
        elif h_region == "center":
            return f"center {v_region}"
        else:
            return f"{v_region}-{h_region}"

    def _get_object_descriptions(
        self,
        positions,
        obj_type,
        agent_pos,
        reward_val=None,
        pronouns=None,
        vision_range=None,
    ):
        """Unified helper method to generate descriptions for any type of object."""
        descriptions = []
        pos_type_pairs = []
        if isinstance(positions, dict):
            if reward_val is not None:
                # Handle rewards dictionary
                pos_type_pairs = [
                    (
                        pos,
                        self.reward_types[
                            reward[0] if isinstance(reward, list) else reward > 0
                        ],
                    )
                    for pos, reward in positions.items()
                ]
            elif obj_type is "other":
                pos_type_pairs = [(pos, name) for pos, name in positions.items()]
            elif obj_type is "locked door":
                pos_type_pairs = [(pos, "locked door") for pos in positions]
        else:
            pos_type_pairs = [(pos, obj_type) for pos in positions]

        for pos, item_type in pos_type_pairs:
            direction, distance = self._get_direction_and_distance(pos, agent_pos)
            # Skip objects beyond vision range if specified
            if vision_range is not None and distance > vision_range:
                continue
            descriptions.append(
                f"There is a {item_type} at {pronouns['possessive']} position."
                if direction == "same position"
                else f"There is a {item_type} {direction} of {pronouns['subject']}, {distance} meters away."
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

    def _get_boundary_descriptions(self, agent_pos, pronouns):
        """Helper method to describe adjacent outer walls."""
        descriptions = []

        # Check each boundary
        if agent_pos[0] == 0:  # North wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the north wall of the maze."
            )
        elif agent_pos[0] == self.grid_size - 1:  # South wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the south wall of the maze."
            )

        if agent_pos[1] == 0:  # West wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the west wall of the maze."
            )
        elif agent_pos[1] == self.grid_size - 1:  # East wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the east wall of the maze."
            )

        return descriptions

    def make_language_obs(
        self,
        agent_pos: list,
        objects: dict,
        keys: int,
        first_person: bool = True,
        vision_range: float = None,
    ):
        # Create pronouns dictionary locally based on first_person parameter
        pronouns = {
            "subject": "you" if first_person else "the agent",
            "possessive": "your" if first_person else "the agent's",
            "be": "are" if first_person else "is",
        }

        agent_pos = np.array(agent_pos)
        base_description = (
            f"{pronouns['subject'].capitalize()} {pronouns['be']} in the {self._get_region(agent_pos)} region of a {self.grid_size}x{self.grid_size} meter maze. "
            f"{pronouns['subject'].capitalize()} {pronouns['be']} carrying {keys} {'key' if keys == 1 else 'keys'}."
        )

        # Get boundary descriptions
        all_descriptions = self._get_boundary_descriptions(agent_pos, pronouns)

        # Process all object types
        all_descriptions.extend(
            self._get_object_descriptions(
                objects["walls"],
                "wall",
                agent_pos,
                pronouns=pronouns,
                vision_range=vision_range,
            )
        )

        # Handle rewards
        if objects["rewards"]:
            all_descriptions.extend(
                self._get_object_descriptions(
                    objects["rewards"],
                    None,
                    agent_pos,
                    reward_val=True,
                    pronouns=pronouns,
                    vision_range=vision_range,
                )
            )

        # Handle standard objects (keys, doors, warps)
        for obj_type, display_name in self.obj_name_mapping.items():
            if objects[obj_type]:
                all_descriptions.extend(
                    self._get_object_descriptions(
                        objects[obj_type],
                        display_name,
                        agent_pos,
                        pronouns=pronouns,
                        vision_range=vision_range,
                    )
                )

        # Handle other objects
        if "other" in objects and objects["other"]:
            all_descriptions.extend(
                self._get_object_descriptions(
                    objects["other"],
                    "other",
                    agent_pos,
                    pronouns=pronouns,
                    vision_range=vision_range,
                )
            )

        return (
            f"{base_description}\nThere are no objects or walls near you."
            if not all_descriptions
            else f"{base_description}\n\n" + "\n".join(all_descriptions)
        )
