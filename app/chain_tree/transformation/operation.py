from typing import List, Dict, Union
from scipy.spatial.distance import pdist, squareform
from chain_tree.base import BaseOperations
from chain_tree.type import DistanceMode
import pandas as pd
import numpy as np


class Operations(BaseOperations):
    create_time: float = None

    def to_human_readable(self) -> str:
        """Convert the coordinate to a human-readable string."""
        return f"Depth: {self.x}, Sibling Position: {self.y}, Child Position: {self.z}, Timestamp: {self.t}"

    def compare_depth(self, other: "Operations") -> int:
        """
        Compare the depth of this coordinate with another.

        Args:
            other (Operations): Another coordinate object.

        Returns:
            int: Positive if this coordinate is deeper, negative if shallower, and zero if same depth.
        """
        if other is None:
            raise ValueError("The 'other' parameter cannot be None")
        return self.x - other.x

    def compare_time(self, other: "Operations") -> float:
        """
        Compare the time of this coordinate with another.

        Args:
            other (Operations): Another coordinate object.

        Returns:
            float: Positive if this coordinate is after, negative if before, and zero if at the same time.
        """
        if other is None:
            raise ValueError("The 'other' parameter cannot be None")
        return float(self.create_time) - float(other.create_time)

    def is_deeper_than(self, other: "Operations") -> bool:
        """Check if this coordinate is deeper than another."""
        return self.compare_depth(other) > 0

    def is_shallower_than(self, other: "Operations") -> bool:
        """Check if this coordinate is shallower than another."""
        return self.compare_depth(other) < 0

    def is_same_depth_as(self, other: "Operations") -> bool:
        """Check if this coordinate is at the same depth as another."""
        return self.compare_depth(other) == 0

    def is_before(self, other: "Operations") -> bool:
        """Check if this coordinate was created before another."""
        return self.compare_time(other) < 0

    def is_after(self, other: "Operations") -> bool:
        """Check if this coordinate was created after another."""
        return self.compare_time(other) > 0

    def is_next_sibling_of(self, other: "Operations") -> bool:
        """Check if this coordinate is the immediate next sibling of another."""
        return self.is_same_depth_as(other) and (self.y == other.y + 1)

    def is_previous_sibling_of(self, other: "Operations") -> bool:
        """Check if this coordinate is the immediate previous sibling of another."""
        return self.is_same_depth_as(other) and (self.y == other.y - 1)

    def is_sibling_of(self, other: "Operations", position: str) -> bool:
        """Check if this coordinate is the immediate previous sibling of another."""
        if position == "next":
            return self.is_next_sibling_of(other)
        elif position == "previous":
            return self.is_previous_sibling_of(other)
        else:
            raise ValueError(f"Unknown position: {position}")

    def is_cousin_of(self, other: "Operations") -> bool:
        """Check if this coordinate is a cousin of another."""
        return self.is_sibling_of(other, position="next") and self.is_shallower_than(
            other
        )

    def is_parent_of(self, other: "Operations") -> bool:
        """Check if this coordinate is the parent of another."""
        return self.is_sibling_of(
            other, position="previous"
        ) and self.is_shallower_than(other)

    def is_child_of(self, other: "Operations") -> bool:
        """Check if this coordinate is the child of another."""
        return self.is_sibling_of(other, position="previous") and self.is_deeper_than(
            other
        )

    def is_same_structure_as(self, other: "Operations") -> bool:
        """
        Check if this coordinate has the same structure as another.

        Args:
            other (Operations): Another coordinate object to compare.

        Returns:
            bool: True if the two coordinates have the same structure; otherwise, False.
        """
        if not isinstance(other, Operations):
            raise TypeError(
                f"Expected 'Operations' type for 'other', but got {type(other)}"
            )

        # Check if the coordinates are at the same depth and have the same sibling and child positions.
        if self.is_same_depth_as(other) and self.y == other.y and self.z == other.z:
            return True
        else:
            return False

    def is_same_topology_as(self, other: "Operations") -> bool:
        """
        Check if this coordinate has the same topology as another.

        Args:
            other (Operations): Another coordinate object to compare.

        Returns:
            bool: True if the two coordinates have the same topology; otherwise, False.
        """
        if not isinstance(other, Operations):
            raise TypeError(
                f"Expected 'Operations' type for 'other', but got {type(other)}"
            )

        # Check if the coordinates have the same depth, sibling, child, and time positions.
        if (
            self.is_same_depth_as(other)
            and self.y == other.y
            and self.z == other.z
            and self.x == other.x
        ):
            return True
        else:
            return False

    @staticmethod
    def is_conversation_progressing(
        coord_old: "Operations", coord_new: "Operations"
    ) -> bool:
        """Determine if the conversation is progressing based on comparing two coordinates in time."""
        return coord_new.is_after(coord_old)

    @staticmethod
    def get_relative_depth(coord1: "Operations", coord2: "Operations") -> int:
        """Get the relative depth difference between two conversation nodes."""
        return coord1.compare_depth(coord2)

    @staticmethod
    def is_sibling(coord1: "Operations", coord2: "Operations") -> bool:
        """Determine if two coordinates are siblings."""
        return coord1.is_same_depth_as(coord2) and abs(coord1.y - coord2.y) == 1

    @staticmethod
    def is_parent_child_relation(parent: "Operations", child: "Operations") -> bool:
        """Determine if there is a parent-child relationship between two nodes."""
        return parent.is_previous_sibling_of(child) and parent.y == child.y

    @staticmethod
    def conversation_temporal_gap(coord1: "Operations", coord2: "Operations") -> float:
        """Calculate the temporal gap between two messages in a conversation."""
        return abs(coord1.compare_time(coord2))

    def _get_norm(self, exclude_t: bool = False) -> float:
        return np.linalg.norm(self.to_reduced_array(exclude_t))

    def _get_difference(
        self, other: "Operations", exclude_t: bool = False
    ) -> np.ndarray:
        if other is None:
            raise ValueError("The 'other' parameter cannot be None")
        return self.to_reduced_array(exclude_t) - other.to_reduced_array(exclude_t)

    def euclidean_distance(self, other: "Operations", exclude_t: bool = False) -> float:
        return np.linalg.norm(self._get_difference(other, exclude_t))

    def calculate_cosine_metric(
        self, other: "Operations", exclude_t: bool = False, return_distance: bool = True
    ) -> float:
        """
        Calculate the cosine similarity or distance between this coordinate and another.

        Args:
            other (Operations): Another coordinate object.
            exclude_t (bool): Whether to exclude the time dimension.
            return_distance (bool): If True, return cosine distance; otherwise, return cosine similarity.

        Returns:
            float: Cosine similarity or distance between the two coordinates.
        """
        a = self.to_reduced_array(exclude_t)
        b = other.to_reduced_array(exclude_t)
        similarity = np.dot(a, b) / (
            self._get_norm(exclude_t) * other._get_norm(exclude_t)
        )

        return 1 - similarity if return_distance else similarity

    @staticmethod
    def _calculate_pairwise_cosine_similarity(coords_array: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise cosine similarity for a set of coordinates.

        Args:
            coords_array (np.ndarray): Array of coordinates.

        Returns:
            np.ndarray: Pairwise cosine similarity matrix.
        """
        norms = np.linalg.norm(coords_array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Prevent division by zero
        normalized_coords = coords_array / norms
        return np.clip(normalized_coords @ normalized_coords.T, -1.0, 1.0)

    @staticmethod
    def _calculate_pairwise_manhattan_distances(coords_array: np.ndarray) -> np.ndarray:
        """
        Calculate the pairwise Manhattan distances between coordinates in the given array.

        Parameters:
            coords_array (np.ndarray): Array of coordinates.

        Returns:
            np.ndarray: Array of pairwise Manhattan distances.
        """
        return np.sum(
            np.abs(coords_array[:, np.newaxis, :] - coords_array[np.newaxis, :, :]),
            axis=-1,
        )

    def _calculate_pairwise_euclidean_distances(coords_array: np.ndarray) -> np.ndarray:
        """
        Calculate the pairwise Euclidean distances between coordinates in the given array.

        Parameters:
            coords_array (np.ndarray): Array of coordinates.

        Returns:
            np.ndarray: Pairwise Euclidean distances between coordinates.
        """
        pairwise_distances = squareform(pdist(coords_array, metric="euclidean"))
        return pairwise_distances

    @staticmethod
    def calculate_pairwise_distances(
        coords_array: np.ndarray, mode: DistanceMode = DistanceMode.EUCLIDEAN
    ) -> np.ndarray:
        if mode == DistanceMode.EUCLIDEAN:
            return Operations._calculate_pairwise_euclidean_distances(coords_array)

        elif mode == DistanceMode.MANHATTAN:
            return Operations._calculate_pairwise_manhattan_distances(coords_array)

        elif mode == DistanceMode.COSINE:
            return Operations._calculate_pairwise_cosine_similarity(coords_array)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def calculate_distance_matrix(
        coordinates: List[Union["Operations", np.ndarray]], mode: DistanceMode
    ) -> np.ndarray:
        """
        Calculate the distance matrix between a list of coordinates.

        Args:
            coordinates (List[Union["Operations", np.ndarray]]): The list of coordinates.
            mode (DistanceMode): The distance mode to use.

        Returns:
            np.ndarray: The distance matrix.
        """
        coords_array = np.array(
            [
                (
                    coord.to_reduced_array()
                    if hasattr(coord, "to_reduced_array")
                    else coord
                )
                for coord in coordinates
            ]
        )

        distances = Operations.calculate_pairwise_distances(coords_array, mode)
        np.fill_diagonal(distances, 1)
        return distances

    @staticmethod
    def calculate_and_assess(
        coordinates: List[Union["Operations", np.ndarray]],
        coordinate_ids: Dict,
        mode: DistanceMode = DistanceMode.EUCLIDEAN,
        csv_filename: str = "distance_matrix.csv",
        save_to_csv: bool = False,
    ) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
        """
        Calculates the distance matrix between the given coordinates and assesses the distances.

        Args:
            coordinates (List[Union["Operations", np.ndarray]]): The list of coordinates to calculate distances for.
            coordinate_ids (Dict): A dictionary mapping coordinate names to their corresponding IDs.
            mode (DistanceMode, optional): The distance mode to use. Defaults to DistanceMode.EUCLIDEAN.
            csv_filename (str, optional): The filename for saving the distance matrix as a CSV file. Defaults to "distance_matrix.csv".
            save_to_csv (bool, optional): Whether to save the distance matrix to a CSV file. Defaults to False.

        Returns:
            Dict[str, Union[float, np.ndarray, pd.DataFrame]]: A dictionary containing the calculated distances and the distance matrix DataFrame.
        """
        distances = Operations.calculate_distance_matrix(coordinates, mode)

        df = pd.DataFrame(distances)
        coordinate_names = list(coordinate_ids.keys())
        df.columns = coordinate_names
        df.index = coordinate_names

        if save_to_csv:
            prefixed_filename = f"{mode.value}_{csv_filename}"
            df.to_csv(prefixed_filename)

        return df
