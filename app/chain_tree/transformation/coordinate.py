from typing import List, Optional, Union, Dict, Tuple
from chain_tree.transformation.operation import Operations
import numpy as np
import uuid


class Coordinate(Operations):
    @classmethod
    def create(
        cls,
        id: str = None,
        depth_args: Union[int, List[int]] = 0,
        sibling_args: Union[int, List[int]] = 0,
        sibling_count_args: Union[int, List[int]] = 0,
        time_args: Union[int, List[int]] = 0,
        n_parts_args: Union[int, List[int]] = 0,
    ):
        if id is None:
            id = str(uuid.uuid4())
        return cls(
            id=id,
            x=first_element_or_value(depth_args),
            y=first_element_or_value(sibling_args),
            z=first_element_or_value(sibling_count_args),
            t=first_element_or_value(time_args),
            n_parts=first_element_or_value(n_parts_args),
        )

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def merge_coordinates(coords: List["Coordinate"]) -> "Coordinate":
        """Merge a list of coordinates, calculating average values for each dimension."""
        avg_x = sum([coord.x for coord in coords]) / len(coords)
        avg_y = sum([coord.y for coord in coords]) / len(coords)
        avg_z = sum([coord.z for coord in coords]) / len(coords)
        avg_t = sum([coord.t for coord in coords]) / len(coords)

        return Coordinate(x=avg_x, y=avg_y, z=avg_z, t=avg_t)

    def path_distance(self, other: Optional["Coordinate"] = None) -> int:
        """
        Calculate the tree distance between two coordinate nodes.

        Args:
            other (Optional[Coordinate]): Another coordinate object. Defaults to None.

        Returns:
            int: Distance in the tree structure.
        """
        if other is None:
            raise ValueError("The 'other' parameter cannot be None")

        if not isinstance(other, Coordinate):
            raise TypeError(
                f"Expected 'Coordinate' type for 'other', got {type(other)}"
            )

        if self == other:
            return 0

        if not hasattr(self, "parent") or not hasattr(other, "parent"):
            raise AttributeError("Both nodes must have a 'parent' attribute")

        if self.is_same_depth_as(other):
            return abs(self.y - other.y)

        distance_to_root = self.x
        other_distance_to_root = other.x

        current_self = self
        current_other = other

        # Align depths
        while current_self.x != current_other.x:
            if current_self.x > current_other.x:
                current_self = current_self.parent
                if current_self is None:
                    raise ValueError("Node has no parent, cannot align depths")
                distance_to_root -= 1
            else:
                current_other = current_other.parent
                if current_other is None:
                    raise ValueError("Node has no parent, cannot align depths")
                other_distance_to_root -= 1

        # Traverse to the common ancestor
        while current_self != current_other:
            current_self = current_self.parent
            current_other = current_other.parent
            if current_self is None or current_other is None:
                raise ValueError("Node has no parent, cannot find common ancestor")
            distance_to_root += 1
            other_distance_to_root += 1

        return distance_to_root + other_distance_to_root

    @staticmethod
    def normalize_coordinates(
        tetra_dict: Dict[str, "Coordinate"]
    ) -> Dict[str, "Coordinate"]:
        """
        Normalizes the coordinates of the to the nearest 100.

        Args:
            tetra_dict: A dictionary mapping message IDs to their coordinates.

        Returns:
            Dict[str, Any]: The normalized tetra_dict with updated coordinates.
        """
        attributes = [
            attr for attr in Coordinate.__annotations__ if attr not in ["n_parts"]
        ]
        min_values = Coordinate(
            **{
                axis: np.min(
                    np.array([getattr(coord, axis) for coord in tetra_dict.values()])
                )
                for axis in attributes
            }
        )
        max_values = Coordinate(
            **{
                axis: np.max(
                    np.array([getattr(coord, axis) for coord in tetra_dict.values()])
                )
                for axis in attributes
            }
        )

        decimal_places = 3

        def normalize_coord(coord):
            for axis in attributes:
                value = (getattr(coord, axis) - getattr(min_values, axis)) / (
                    getattr(max_values, axis) - getattr(min_values, axis)
                )
                setattr(coord, axis, np.round(value, decimal_places))
            return coord

        normalized_tetra_dict = {
            message_id: normalize_coord(coord)
            for message_id, coord in tetra_dict.items()
        }

        return normalized_tetra_dict

    @staticmethod
    def custom_scale_x_coordinates(
        tetra_dict: Dict[str, Operations],
        scaling_technique: str,
        scaled_t: Dict[str, float],
        desired_range: Tuple[float, float] = (0, 1),
    ) -> Dict[str, float]:
        """
        Apply custom scaling techniques to the x coordinates in the Coordinate Dataclasss.
        taking into account the scaled t coordinates.

        Args:
            scaling_technique (str): The scaling technique to apply. Choose from 'linear', 'logarithmic', 'sigmoid', 'discrete'.
            scaled_t (Dict[str, float]): Th
            e scaled t coordinates.
            desired_range (Tuple[float, float]): The desired range for the scaled x coordinates.

        Returns:
            Dict[str, float]: The scaled x coordinates.
        """

        # Extract the x coordinates from Coordinate Class
        x_coords = np.array([coord.x for coord in tetra_dict.values()])

        if scaling_technique == "linear":
            # Linear scaling
            x_min = np.min(x_coords)
            x_max = np.max(x_coords)
            scaled_x = (x_coords - x_min) / (x_max - x_min) * (
                desired_range[1] - desired_range[0]
            ) + desired_range[0]

        elif scaling_technique == "logarithmic":
            # Logarithmic scaling
            scaled_x = np.log(x_coords + 1)

        elif scaling_technique == "sigmoid":
            # Sigmoid scaling
            scaled_x = 1 / (1 + np.exp(-x_coords))

        elif scaling_technique == "discrete":
            # Discrete scaling
            n_intervals = len(np.unique(x_coords))
            scaled_x = np.floor(x_coords) % n_intervals

        else:
            raise ValueError(
                "Invalid scaling technique. Choose from 'linear', 'logarithmic', 'sigmoid', or 'discrete'."
            )

        # Convert scaled_x to a dictionary with regular Python integers
        scaled_x_dict = {
            message_id: int(scaled_x[i])
            for i, message_id in enumerate(tetra_dict.keys())
        }

        # Adjust scaled x coordinates based on the scaled t coordinates
        for message_id, t_coord in scaled_t.items():
            scaled_x_dict[message_id] = int(scaled_x_dict[message_id] + t_coord)

        return scaled_x_dict

    @staticmethod
    def custom_scale_t_coordinates(
        tetra_dict: Dict[str, Operations],
        scaling_technique: str,
        desired_range: Tuple[float, float] = (0, 1),
    ) -> Dict[str, float]:
        """
        Apply custom scaling techniques to the t coordinates in the Coordinate Dataclass.

        Args:
            scaling_technique (str): The scaling technique to apply. Choose from 'temporal_resolution',
                                        'time_weighting', 'time_clustering', 'time_based_relationships'.
            desired_range (Tuple[float, float]): The desired range for the scaled t coordinates.

        Returns:
            Dict[str, float]: The scaled t coordinates.
        """

        # Extract the t coordinates from Coordinate Class

        t_coords = np.array([coord.t for coord in tetra_dict.values()])

        if scaling_technique == "temporal_resolution":
            # Apply temporal resolution adjustment
            min_t = np.min(t_coords)
            scaled_t = t_coords - min_t

        elif scaling_technique == "time_weighting":
            # Apply time weighting scaling
            min_t = np.min(t_coords)
            max_t = np.max(t_coords)
            scaled_t = (t_coords - min_t) / (max_t - min_t) * (
                desired_range[1] - desired_range[0]
            ) + desired_range[0]

        elif scaling_technique == "time_based_relationships":
            # Apply time-based relationship scaling
            scaled_t = np.diff(t_coords)
            scaled_t = np.insert(scaled_t, 0, 0)  # Pad with zero for consistent length

        else:
            raise ValueError(
                "Invalid scaling technique. Choose from 'temporal_resolution', 'time_weighting', or 'time_based_relationships'."
            )

        # Create a dictionary mapping message IDs to scaled t coordinates
        scaled_t_dict = {
            message_id: float(scaled_t[i])
            for i, message_id in enumerate(tetra_dict.keys())
        }

        return scaled_t_dict

    @staticmethod
    def adaptive_temporal_scaling(
        tetra_dict: Dict[str, "Coordinate"],
        x_scaling_technique: str = "discrete",
        t_scaling_technique: str = "time_based_relationships",
        t_scaling_range: Tuple[float, float] = (0, 1),
    ) -> Dict[str, "Coordinate"]:
        """
        Applies adaptive temporal scaling to the coordinates in the Coordinate Dataclass.

        Args:
            tetra_dict (Dict[str, Coordinate]): The original Representation mapping message IDs to Coordinate objects.
            x_scaling_technique (str): The desired scaling technique for x coordinates. Options: 'linear', 'logarithmic', 'sigmoid', 'discrete'.
            t_scaling_technique (str): The desired scaling technique for t coordinates. Options: 'temporal_resolution', 'time_weighting', 'time_based_relationships'.
            t_scaling_range (Tuple[float, float]): The desired range for t scaling. Only applicable for certain t scaling techniques.
            topological_method (str): The topological method to use. Options: 'persistent_homology', 'graph_based'.
            temporal_modeling_method (str): The temporal modeling method to use. Options: 'autoregressive', 'state_space', 'recurrent_neural_network', etc.
            temporal_modeling_params (Dict[str, Any]): Additional parameters for the temporal modeling method.

        Returns:
            Dict[str, Coordinate]: The transformed Representation with scaled x and t coordinates.
        """
        # Normalize the coordinates

        tetra_dict = Coordinate.normalize_coordinates(tetra_dict)

        # Apply custom scaling for t coordinates
        scaled_t_coords = Coordinate.custom_scale_t_coordinates(
            tetra_dict, t_scaling_technique, t_scaling_range
        )

        # Apply custom scaling for x coordinates
        scaled_x_coords = Coordinate.custom_scale_x_coordinates(
            tetra_dict, x_scaling_technique, scaled_t_coords
        )

        # Update the coordinates in the Coordinate Dataclass
        transformed_representation = {
            message_id: Coordinate(
                x=scaled_x_coords[message_id],
                y=coord.y,
                z=coord.z,
                t=scaled_t_coords[message_id],
            )
            for message_id, coord in tetra_dict.items()
        }

        return transformed_representation


def first_element_or_value(value: Union[int, List[int]]) -> int:
    """Return the first element if the input is a list; otherwise, return the value itself."""
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return value


def is_list_of_ints(value: Union[int, List[int]]) -> bool:
    """Return True if the input is a list of ints; otherwise, return False."""
    if isinstance(value, list) and len(value) > 0:
        return all(isinstance(val, int) for val in value)
    return False
