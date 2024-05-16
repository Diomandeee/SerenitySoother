from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from chain_tree.transformation.coordinate import Coordinate
from chain_tree.utils import log_handler
from collections import defaultdict
from pydantic import Field
import pandas as pd
import numpy as np
import functools
import torch
import glob
import json
import os


def memoize(f):
    cache = {}

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            result = f(*args, **kwargs)
            cache[key] = result
        return cache[key]

    return wrapped


class CoordinateTree(Coordinate):
    parent: Optional[Union[str, "CoordinateTree"]] = Field(
        None, description="The parent of the node."
    )
    children: List["CoordinateTree"] = Field(
        default_factory=list, description="The children of the node."
    )

    text: Optional[str] = Field(None, description="The text of the node.")

    author: Optional[str] = Field(None, description="The author of the node.")

    embeddings: Optional[Any] = Field(None, description="The embeddings of the node.")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    def add_node(self, node: "CoordinateTree") -> None:
        """Add a node to the tree."""
        self.children.append(node)

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child

    def __len__(self):
        return sum(1 for _ in self)

    def get_nodes(self) -> List["CoordinateTree"]:
        """Get a list of all nodes in the tree."""
        return list(self)

    def get_node_ids(self) -> List[str]:
        """Get a list of all node IDs in the tree."""
        return [node.id for node in self.get_nodes()]

    def to_tuple(self) -> Tuple[float, float, float, float, int]:
        """Convert the object to a tuple representation."""
        return (self.x, self.y, self.z, self.t, self.n_parts)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the tree into a DataFrame representation."""
        return pd.DataFrame.from_dict(self.to_dict())

    def to_json(self) -> str:
        """Convert the object to a JSON string."""
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary representation."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
            "author": self.author,
            "parent": self.parent,
            "n_parts": self.n_parts,
            "text": self.text,
            "children": [child.id for child in self.children],
            "create_time": self.create_time,
            "embeddings": self.embeddings if self.embeddings is not None else None,
        }

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float, float, int]) -> "CoordinateTree":
        """Create an instance from a tuple representation."""
        return cls(x=t[0], y=t[1], z=t[2], t=0, n_parts=0)

    @classmethod
    def from_list(cls, l: List["CoordinateTree"]) -> Dict[str, "CoordinateTree"]:
        """Create an instance from a list representation."""
        return {node.id: node for node in l}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CoordinateTree":
        """Create an instance from a DataFrame representation."""
        return cls.from_dict(df.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoordinateTree":
        """Create an instance from a dictionary representation."""
        node = cls(
            id=d["id"],
            x=d["x"],
            y=d["y"],
            z=d["z"],
            t=d["t"],
            parent=d["parent"],
            n_parts=d["n_parts"],
            create_time=d["create_time"],
            text=d["text"],
            children=[],
        )
        node.children = [
            cls.from_dict(child_dict) for child_dict in d.get("children", [])
        ]
        return node

    @classmethod
    def from_json_file(cls, json_file: str) -> "CoordinateTree":
        """Create an instance from a JSON file."""
        with open(json_file, "r") as f:
            json_data = json.load(f)
        return cls.from_dict(json_data)

    @classmethod
    def from_jsons(cls, jsons: List[str]) -> List["CoordinateTree"]:
        trees = []
        for json_file in jsons:
            try:
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                trees.append(cls.from_dict(json_data))
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in file: {json_file}")
                print(f"Error details: {e}")
                continue
        return trees

    @classmethod
    def from_json(
        cls, json_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> "CoordinateTree":
        """Create an instance from a JSON file path, a dictionary, or a list of dictionaries."""
        if isinstance(json_data, str) and os.path.isfile(json_data):
            return cls.from_json_file(json_data)
        elif isinstance(json_data, dict):
            return cls.from_dict(json_data)
        elif isinstance(json_data, list):
            # Assuming the list represents children of a single node
            return [cls.from_dict(item) for item in json_data]
        else:
            raise ValueError(
                "Invalid input: Please provide either a valid JSON file path, a dictionary, or a list of dictionaries."
            )

    @classmethod
    def flatten_coordinates(tree: "CoordinateTree") -> dict[str, "CoordinateTree"]:
        data = tree.to_dict()
        flat_dict = {tree.id: tree}

        flat_dict = {tree.id: tree}
        for child_data in data["children"]:
            flat_dict[child_data["id"]] = child_data

        return flat_dict

    @staticmethod
    def create_tree(scenario_chunks, author_switch=True, x=1):
        if not scenario_chunks:
            return None

        head, *tail = scenario_chunks

        if author_switch:
            author = "user"
        else:
            author = "assistant"

        child_tree = CoordinateTree.create_tree(tail, not author_switch, x + 1)

        return CoordinateTree(
            x=x,
            y=0,
            z=0,
            t=0,
            text=head[0],
            author=author,
            children=[
                CoordinateTree(
                    x=x + 1,
                    y=0,
                    z=0,
                    t=0,
                    text="\n\n".join(head[1:]),
                    author="assistant" if author == "user" else "user",
                    children=[child_tree] if child_tree is not None else [],
                )
            ],
        )

    @staticmethod
    def flatten(coordinate: "CoordinateTree") -> np.ndarray:
        """Flatten the Coordinate instance into a numpy array."""
        return np.array(
            [
                coordinate.x,
                coordinate.y,
                coordinate.z,
                coordinate.t,
                coordinate.text,
                coordinate.author,
                coordinate.n_parts,
                coordinate.create_time,
                coordinate.parent,
                coordinate.id,
            ]
        )

    @staticmethod
    def tree_flatten(
        coordinates_dict: Dict[str, "CoordinateTree"],
        as_dictionary: bool = True,
    ) -> Dict[str, Tuple[float, float, float, float, int]]:
        """
        Flattens a dictionary of Coordinate objects.

        Args:
            coordinates_dict: The dictionary of Coordinate objects.

        Returns:
            A tuple containing a list of flattened Coordinate numpy arrays and a list of auxiliary data needed for unflattening.
        """
        # Flatten the Coordinate objects.
        flattened_coordinates_list = [
            CoordinateTree.flatten(coord) for coord in coordinates_dict.values()
        ]

        # Get the auxiliary data needed for unflattening (keys of the original dictionary).
        aux_data = list(coordinates_dict.keys())

        if as_dictionary:
            # Convert the list of flattened Coordinate numpy arrays to a dictionary.
            flattened_coordinates_dict = dict(zip(aux_data, flattened_coordinates_list))
            return flattened_coordinates_dict
        else:
            return flattened_coordinates_list, aux_data

    @staticmethod
    def flattened_coordinates_to_json(
        coordinates_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    ) -> Dict[str, str]:
        """
        Convert a dictionary of flattened Coordinate numpy arrays to a dictionary of JSON strings.

        Args:
            coordinates_dict (Dict[str, List[np.ndarray]]): A dictionary of flattened Coordinate numpy arrays.

        Returns:
            Dict[str, str]: A dictionary of JSON strings.
        """
        return {
            key: json.dumps(value.tolist(), default=lambda x: x.tolist())
            for key, value in coordinates_dict.items()
        }

    @staticmethod
    def tree_to_tetra_dict(
        tree: Union["CoordinateTree", List["CoordinateTree"]]
    ) -> Dict[str, Tuple[float, float, float, float, int]]:
        """
        Transforms a given CoordinateTree structure into a dictionary representation, mapping nodes to their coordinates.

        This function traverses the CoordinateTree starting from the root. For each node in the tree, it extracts its
        coordinates (x, y, z, t, n_parts) and maps them to the node's ID in the resulting dictionary.

        """

        tetra_dict = {}
        if isinstance(tree, list):
            stack = tree

        else:
            stack = [tree]

        while stack:
            node = stack.pop()

            if not node.id:
                log_handler("Node has no ID. Skipping.", level="WARNING")
                continue

            # Validate the coordinates
            if any(
                val is None for val in [node.x, node.y, node.z, node.t, node.n_parts]
            ):
                log_handler(
                    f"Node {node.id} has missing coordinates. Skipping.",
                    level="WARNING",
                )
                continue

            # Add to tetra_dict
            tetra_dict[node.id] = (node.x, node.y, node.z, node.t, node.n_parts)

            # Add children to stack
            stack.extend(node.children)

        return tetra_dict

    @staticmethod
    def tree_unflatten(
        flattened_coordinates_list: List[np.ndarray],
    ) -> Dict[str, "CoordinateTree"]:
        """
        Unflattens a list of flattened Coordinate numpy arrays.

        Args:
            flattened_coordinates_list: The list of flattened Coordinate numpy arrays.
            aux_data: The auxiliary data needed for unflattening (keys of the original dictionary).

        Returns:
            A dictionary of Coordinate objects.
        """
        # Unflatten the Coordinate numpy arrays.
        coordinates_list = [
            CoordinateTree.unflatten(coord) for coord in flattened_coordinates_list
        ]

        # Convert the list of Coordinate objects to a dictionary.
        coordinates_dict = CoordinateTree.from_list(coordinates_list)

        return coordinates_dict

    @staticmethod
    def coordinates_to_tensor(
        coordinates_dict: Union[
            Dict[str, "CoordinateTree"],
            List["CoordinateTree"],
            Dict[str, Tuple[float, float, float, float, int]],
        ],
        as_dictionary: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert a dictionary of CoordinateTree objects to a tensor.

        Args:
            coordinates_dict (Dict[str, CoordinateTree]): A dictionary of CoordinateTree objects.
            as_dictionary (bool, optional): A flag to determine if the CoordinateTree objects should be returned as a dictionary. Defaults to True.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: A tensor or a dictionary of tensors.
        """
        if as_dictionary:
            coordinates = CoordinateTree.tree_to_tetra_dict(coordinates_dict)

            coordinates_dict = {key: value for key, value in coordinates.items()}

            return torch.tensor(list(coordinates_dict.values()))

        return torch.tensor([coordinate.to_list() for coordinate in coordinates_dict])

    def distance_to(self, other: "CoordinateTree") -> int:
        """
        Calculate the distance between two nodes.

        Args:
            other (CoordinateTree): The node to calculate the distance to.

        Returns:
            int: The distance.
        """
        # Calculate the distance between the current node and the other node.
        distance = self.euclidean_distance(other)

        return distance

    def tree_distance(self, other: "CoordinateTree") -> int:
        """
        Calculate the tree distance between two trees.

        Args:
            other (CoordinateTree): The tree to calculate the distance to.

        Returns:
            int: The tree distance.
        """
        # Calculate the distance between the current node and the other node.
        distance = self.distance_to(other)

        # Calculate the distance between the current node and the other node's children.
        for child in other.children:
            distance += self.distance_to(child)

        return distance

    def distance_to_root(self) -> int:
        """
        Calculate the distance to the root node.

        Returns:
            int: The distance to the root node.
        """
        # Calculate the distance between the current node and the root node.
        distance = self.euclidean_distance(CoordinateTree(x=0, y=0, z=0, t=0))

        return distance

    def tree_distance_to_root(self) -> int:
        """
        Calculate the tree distance to the root node.

        Returns:
            int: The tree distance to the root node.
        """
        # Calculate the distance between the current node and the root node.
        distance = self.distance_to_root()

        # Calculate the distance between the current node and the root node's children.
        for child in self.children:
            distance += self.distance_to(child)

        return distance

    @classmethod
    def load_coordinate_trees_from_dir(
        cls, directory: str, file_pattern: str = "**/coordinate_tree.json"
    ) -> List["CoordinateTree"]:
        """
        Load CoordinateTree instances from JSON files with the name 'coordinate_tree.json' within a specified directory, including nested directories.

        Args:
            directory (str): The directory to search for 'coordinate_tree.json' files.
            file_pattern (str, optional): The pattern to match 'coordinate_tree.json' files within the directory and its subdirectories. Defaults to "**/coordinate_tree.json".

        Returns:
            List[CoordinateTree]: A list of loaded tree instances.
        """
        json_files = glob.glob(os.path.join(directory, file_pattern), recursive=True)

        if not json_files:
            raise ValueError(
                "No 'coordinate_tree.json' files found in the specified directory."
            )

        tree_instances = cls.from_jsons(json_files)

        if not tree_instances:
            raise ValueError(
                "No valid 'coordinate_tree.json' files found in the specified directory."
            )

        return tree_instances

    @classmethod
    def pairwise_tree_distance(
        cls, coordinates: List["CoordinateTree"], to_dataframe: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute pairwise tree distances between a list of coordinates.

        Args:
            coordinates (List[Operations]): A list of coordinates.

        Returns:
            Dict[str, Dict[str, int]]: A dictionary of dictionaries containing the pairwise distances between coordinates.
        """
        for coord in coordinates:
            if not isinstance(coord, CoordinateTree):
                raise TypeError(f"Expected 'CoordinateTree' type, got {type(coord)}")

        n = len(coordinates)

        # Use a list comprehension to generate a list of tuples (i, j, distance)
        distances = [
            (i, j, coordinates[i].tree_distance(coordinates[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]

        result = defaultdict(dict)
        for i, j, distance in distances:
            result[coordinates[i].id][coordinates[j].id] = distance
            result[coordinates[j].id][coordinates[i].id] = distance  # leverage symmetry

        if to_dataframe:
            df = pd.DataFrame.from_dict(result)
            df = df.reindex(index=df.index[::-1])
            return df

        return result

    def combine(
        self, other: "CoordinateTree", increment_x: bool = False
    ) -> Tuple["CoordinateTree", int]:
        """
        Combine two trees into one, creating a new tree with an incremented x-coordinate if specified.

        Args:
            other (CoordinateTree): The tree to combine with the current tree.
            increment_x (bool, optional): If True, increment the x-coordinate of the combined tree. Defaults to False.

        Returns:
            Tuple[CoordinateTree, int]: A tuple containing the combined tree and the tree distance.
        """
        combined_x = max(self.x, other.x) + 1 if increment_x else self.x
        combined_tree = CoordinateTree(
            x=combined_x,
            y=self.y,
            z=self.z,
            t=self.t,
            parent=self.id,
            n_parts=self.n_parts,
            create_time=self.create_time,
            text=self.text,
        )

        combined_tree.children = self.children + other.children
        for child in combined_tree.children:
            child.parent = combined_tree.id

        tree_distance = sum(
            combined_tree.tree_distance(child)
            for child in combined_tree.children + other.children
        )
        tree_distance += combined_tree.tree_distance(other)

        return combined_tree, tree_distance

    def compute_sibling_sequences(
        self, nodes: List["CoordinateTree"]
    ) -> List[List["CoordinateTree"]]:
        """
        Group nodes into sequences based on their y-coordinates and create_time.

        This method sorts the nodes first by their y-coordinates and then by their
        create_time. A new group or sequence is started when:
        - The difference in y-coordinates between adjacent nodes is greater than 1.
        - The y-coordinate is the same, but the next node has an earlier create_time.

        :param nodes: A list of CoordinateTree objects.
        :return: A list of lists where each inner list represents a sequence of sibling nodes.
        """

        if not nodes:
            return []

        # Convert list of nodes to a structured numpy array
        dtype: List[Tuple[str, Union[type, np.dtype]]] = [
            ("y", int),
            ("create_time", float),
            ("node", object),
        ]
        nodes_array = np.array(
            [(node.y, node.create_time, node) for node in nodes], dtype=dtype
        )

        # Sort nodes by y-coordinate and then by create_time
        sorted_nodes = np.sort(nodes_array, order=["y", "create_time"])

        # Compute the difference for y-coordinates and create_times between adjacent nodes
        y_diff = np.diff(sorted_nodes["y"])
        time_diff = np.diff(sorted_nodes["create_time"])

        # Find the breaking points where a new group should start
        breaks = np.where((y_diff > 1) | ((y_diff == 0) & (time_diff < 0)))[0] + 1

        # Split the array into groups based on the found breaks
        node_groups = np.split(sorted_nodes, breaks)

        return [list(group["node"]) for group in node_groups]

    @classmethod
    def combine_coordinate_trees(
        cls, tree_instances: List["CoordinateTree"]
    ) -> "CoordinateTree":
        """
        Combine a list of CoordinateTree instances into a single combined tree.

        Args:
            tree_instances (List[CoordinateTree]): A list of tree instances to combine.

        Returns:
            CoordinateTree: The combined tree.
        """
        if not tree_instances:
            raise ValueError("No tree instances provided for combining.")

        # Start with the first tree as the base for combining.
        combined_tree = tree_instances[0]

        # Combine the remaining trees with the base tree.
        for tree in tree_instances[1:]:
            combined_tree, _ = combined_tree.combine(tree)

        return combined_tree

    def flatten_tree(self) -> "CoordinateTree":
        """
        Flatten the tree by incrementing the x-coordinate of each node.

        Returns:
            CoordinateTree: The flattened tree.
        """
        # Create a copy of the tree.
        flattened_tree = self.copy()

        # Increment the x-coordinate of each node.
        for node in flattened_tree:
            node.x += 1

        return flattened_tree

    @classmethod
    def list_to_dict(
        cls, coordinates: List["CoordinateTree"], flatten_tree: bool = False
    ) -> Dict[str, "CoordinateTree"]:
        """
        Convert a list of CoordinateTree objects to a dictionary.

        Args:
            coordinates (List[CoordinateTree]): A list of CoordinateTree objects.
            flatten_tree (bool, optional): A flag to determine if the CoordinateTree objects should be flattened. Defaults to False.

        Returns:
            Dict[str, CoordinateTree]: A dictionary of CoordinateTree objects.
        """
        if flatten_tree:
            return {
                coordinate.id: coordinate.flatten_tree() for coordinate in coordinates
            }
        return {coordinate.id: coordinate for coordinate in coordinates}

    @staticmethod
    def depth_first_search(
        tree: "CoordinateTree", predicate: Callable[["CoordinateTree"], bool]
    ) -> Optional["CoordinateTree"]:
        if predicate(tree):
            return tree
        else:
            for child in tree.children:
                result = CoordinateTree.depth_first_search(child, predicate)
                if result is not None:
                    return result
            return None

    @staticmethod
    def breadth_first_search(
        tree: "CoordinateTree", predicate: Callable[["CoordinateTree"], bool]
    ) -> Optional["CoordinateTree"]:
        queue = [tree]
        while queue:
            node = queue.pop(0)
            if predicate(node):
                return node
            else:
                queue.extend(node.children)
        return None

    @staticmethod
    def depth_first_search_all(
        tree: "CoordinateTree", predicate: Callable[["CoordinateTree"], bool]
    ) -> List["CoordinateTree"]:
        results = []
        if predicate(tree):
            results.append(tree)
        for child in tree.children:
            results.extend(CoordinateTree.depth_first_search_all(child, predicate))
        return results

    @staticmethod
    def breadth_first_search_all(
        tree: "CoordinateTree", predicate: Callable[["CoordinateTree"], bool]
    ) -> List["CoordinateTree"]:
        results = []
        queue = [tree]
        while queue:
            node = queue.pop(0)
            if predicate(node):
                results.append(node)
            else:
                queue.extend(node.children)
        return results