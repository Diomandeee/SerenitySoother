from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from chain_tree.transformation.animate import animate_conversation_tree
from chain_tree.transformation.tree import CoordinateTree
from pydantic import BaseModel
import pandas as pd

CoordinateOperation = Callable[[CoordinateTree], CoordinateTree]
AnyOperation = Callable[[CoordinateTree], Any]
Predicate = Callable[[CoordinateTree], bool]
Reducer = Callable[[Any, CoordinateTree], Any]


def reduce(reducer: Reducer, coordinates: List[CoordinateTree], initial: Any) -> Any:
    """
    Reduce the coordinates by applying a reducer function that accumulates the results starting with an initial value.

    Args:
        reducer (Reducer): A reducer function that takes two arguments, the accumulator and a Coordinate, and returns the updated accumulator.
        initial (Any): The initial value for the accumulation.

    Returns:
        Any: The result of the reduction.
    """
    accumulator = initial
    for coordinate in coordinates:
        accumulator = reducer(accumulator, coordinate)
    return accumulator


class CoordinateTreeCollection(BaseModel):
    trees: Union[
        List[CoordinateTree], Dict[str, CoordinateTree], List[Dict[str, float]]
    ]

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        return self.trees[index]

    def __setitem__(self, index, value):
        self.trees[index] = value

    def __delitem__(self, index):
        del self.trees[index]

    def __iter__(self):
        return iter(self.trees)

    def __add__(self, other):
        if isinstance(other, CoordinateTreeCollection):
            return self.__class__(trees=self.trees + other.trees)
        elif isinstance(other, CoordinateTree):
            return self.__class__(trees=self.trees + [other])
        else:
            raise ValueError(
                f"Invalid input: Please provide either a {self.__class__.__name__} or a Coordinatechain_tree.tree."
            )

    def __radd__(self, other):
        if isinstance(other, CoordinateTreeCollection):
            return self.__class__(trees=other.trees + self.trees)
        elif isinstance(other, CoordinateTree):
            return self.__class__(trees=[other] + self.trees)
        else:
            raise ValueError(
                f"Invalid input: Please provide either a {self.__class__.__name__} or a Coordinatechain_tree.tree."
            )

    def __contains__(self, item):
        return item in self.trees

    def is_same_structure_as(
        self, other: "CoordinateTreeCollection"
    ) -> Union[bool, Dict[str, List[int]]]:
        """
        Check if two coordinate tree collections have the same structure.

        Args:
            other (CoordinateTreeCollection): Another coordinate tree collection.

        Returns:
            Union[bool, Dict[str, List[int]]]:
                - True if the two coordinate tree collections have the same structure.
                - False if the lengths are different.
                - Dictionary containing detailed differences if not the same.
        """
        if not isinstance(other, CoordinateTreeCollection):
            raise TypeError(
                f"Expected 'CoordinateTreeCollection' type for 'other', got {type(other)}"
            )

        if len(self.trees) != len(other.trees):
            return False

        differences = {}
        for i in range(len(self.trees)):
            subtree_differences = self.trees[i].is_same_structure_as(other.trees[i])
            if subtree_differences is not True:
                differences[i] = subtree_differences

        if differences:
            return differences
        else:
            return True

    def is_same_topology_as(
        self, other: "CoordinateTreeCollection"
    ) -> Union[bool, Dict[str, List[int]]]:
        """
        Check if two coordinate tree collections have the same topology.

        Args:
            other (CoordinateTreeCollection): Another coordinate tree collection.

        Returns:
            Union[bool, Dict[str, List[int]]]:
                - True if the two coordinate tree collections have the same topology.
                - False if the lengths are different.
                - Dictionary containing detailed differences if not the same.
        """
        if not isinstance(other, CoordinateTreeCollection):
            raise TypeError(
                f"Expected 'CoordinateTreeCollection' type for 'other', got {type(other)}"
            )

        if len(self.trees) != len(other.trees):
            return False

        differences = {}
        for i in range(len(self.trees)):
            subtree_differences = self.trees[i].is_same_topology_as(other.trees[i])
            if subtree_differences is not True:
                differences[i] = subtree_differences

        if differences:
            return differences
        else:
            return True

    def distance(self, other: "CoordinateTreeCollection") -> float:
        """
        Calculate the distance between two coordinate tree collections.

        Args:
            other (CoordinateTreeCollection): Another coordinate tree collection.

        Returns:
            float: Distance between the two coordinate tree collections.
        """
        if not isinstance(other, CoordinateTreeCollection):
            raise TypeError(
                f"Expected 'CoordinateTreeCollection' type for 'other', got {type(other)}"
            )

        def find_matching_subsets(other: "CoordinateTreeCollection") -> List[int]:
            matching_subsets = []

            for i in range(len(self.trees)):
                for j in range(len(other.trees)):
                    if self.trees[i].is_same_structure_as(other.trees[j]):
                        matching_subsets.append(i)
                        break

            return matching_subsets

        matching_subsets = find_matching_subsets(self, other)

        if not matching_subsets:
            raise ValueError(
                "No matching subsets found. The collections have different structures."
            )

        total_distance = sum(
            [
                self.trees[i].euclidean_distance(other.trees[j])
                for i, j in enumerate(matching_subsets)
            ]
        )

        return total_distance

    def is_relative_to(
        self,
        other: "CoordinateTreeCollection",
        relation: str,
    ) -> bool:
        """
        Check the relationship between this coordinate tree collection and another.

        Args:
            other (CoordinateTreeCollection): Another coordinate tree collection.
            relation (str): The relationship to check (e.g., 'parent', 'child', 'sibling', 'cousin', etc.).
            position (str): The position of the other coordinate tree collection relative to this one.

        Returns:
            bool: True if the specified relationship is satisfied; otherwise, False.
        """
        if not isinstance(other, CoordinateTreeCollection):
            raise TypeError(
                f"Expected 'CoordinateTreeCollection' type for 'other', got {type(other)}"
            )

        if relation == "parent":
            return self.trees[0].is_parent_of(other.trees[0])
        elif relation == "child":
            return self.trees[0].is_child_of(other.trees[0])
        elif relation == "cousin":
            return self.trees[0].is_cousin_of(other.trees[0])
        elif relation == "supertree":
            return any(
                [
                    self.trees[i].is_same_structure_as(other.trees[0])
                    and self.trees[i].is_same_topology_as(other.trees[0])
                    for i in range(len(self.trees))
                ]
            )
        elif relation == "subtree":
            return any(
                [
                    self.trees[0].is_same_structure_as(other.trees[i])
                    and self.trees[0].is_same_topology_as(other.trees[i])
                    for i in range(len(other.trees))
                ]
            )
        else:
            raise ValueError(f"Invalid relation: {relation}")

    def is_relative_to_any(
        self,
        others: List["CoordinateTreeCollection"],
        relation: str,
        position: Optional[str] = None,
    ) -> bool:
        """
        Check the relationship between this coordinate tree collection and another.

        Args:
            others (List[CoordinateTreeCollection]): A list of other coordinate tree collections.
            relation (str): The relationship to check (e.g., 'parent', 'child', 'sibling', 'cousin', etc.).
            position (str): The position of the other coordinate tree collection relative to this one.

        Returns:
            bool: True if the specified relationship is satisfied; otherwise, False.
        """
        if not isinstance(others, list):
            raise TypeError(f"Expected 'list' type for 'others', got {type(others)}")

        return any([self.is_relative_to(other, relation, position) for other in others])

    def is_relative_to_all(
        self,
        others: List["CoordinateTreeCollection"],
        relation: str,
        position: Optional[str] = None,
    ) -> bool:
        """
        Check the relationship between this coordinate tree collection and another.

        Args:
            others (List[CoordinateTreeCollection]): A list of other coordinate tree collections.
            relation (str): The relationship to check (e.g., 'parent', 'child', 'sibling', 'cousin', etc.).
            position (str): The position of the other coordinate tree collection relative to this one.

        Returns:
            bool: True if the specified relationship is satisfied; otherwise, False.
        """
        if not isinstance(others, list):
            raise TypeError(f"Expected 'list' type for 'others', got {type(others)}")

        return all([self.is_relative_to(other, relation, position) for other in others])


class CoordinateTreeList(CoordinateTreeCollection):
    def __init__(self, trees: List[CoordinateTree]):
        super().__init__(trees)


class CoordinateTreeDict(CoordinateTreeCollection):
    def __init__(self, trees: Dict[str, CoordinateTree]):
        super().__init__(list(trees.values()))
        self.trees_dict = trees


class CoordinateHandler:
    """
    Custom container for Coordinate objects.
    Attributes:
        coordinates (List[Coordinate]): A list of Coordinate objects.
    """

    def __init__(self, coordinates: Union[CoordinateTreeList, CoordinateTreeDict]):
        self.coordinates = coordinates

    def __repr__(self):
        return f"CoordinateHandler(coordinates={self.coordinates})"

    def tree_flatten(self):
        leaves = [coord.tree_flatten() for coord in self.coordinates]
        return leaves, self

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(aux_data)

    def animate(self, coordinates) -> "CoordinateHandler":
        return animate_conversation_tree(coordinates)

    @classmethod
    def from_list(cls, coordinates: List[CoordinateTree]) -> "CoordinateHandler":
        return cls(CoordinateTreeList(coordinates))

    @classmethod
    def from_dict(cls, coordinates: Dict[str, CoordinateTree]) -> "CoordinateHandler":
        return cls(CoordinateTreeDict(coordinates))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoordinateHandler":
        coordinates = [CoordinateTree.from_dict(coord) for coord in d["coordinates"]]
        return cls(coordinates)

    @classmethod
    def from_tuple(cls, coordinates: List[CoordinateTree]) -> "CoordinateHandler":
        return cls(coordinates)

    @classmethod
    def from_json(cls, json_file_path: str) -> "CoordinateHandler":
        return cls(CoordinateTree.from_json(json_file_path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CoordinateHandler":
        return cls(CoordinateTree.from_dataframe(df))

    def to_dict(self) -> Dict[str, Any]:
        return {"coordinates": [coord.to_dict() for coord in self.coordinates]}

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([coord.to_dict() for coord in self.coordinates])

    def distance(self, other: "CoordinateHandler") -> float:
        """
        Calculates the distance between two coordinate handlers.

        Args:
            other (CoordinateHandler): Another coordinate handler.

        Returns:
            float: The distance between the two coordinate handlers.
        """
        if not isinstance(other, CoordinateHandler):
            raise TypeError(
                f"Expected 'CoordinateHandler' type for 'other', got {type(other)}"
            )

        if len(self.coordinates) != len(other.coordinates):
            raise ValueError(
                "The two coordinate handlers must have the same number of coordinates."
            )

        total_distance = sum(
            [
                self.calculate_distance(self.coordinates[i], other.coordinates[i])
                for i in range(len(self.coordinates))
            ]
        )

        return total_distance

    def calculate_distance(
        self, segment_coordinate: CoordinateTree, state: CoordinateTree
    ) -> float:
        """
        Calculates the distance between two coordinates.

        Args:
            segment_coordinate (Coordinate): A segment coordinate.
            state (Coordinate): A state coordinate.

        Returns:
            float: The distance between the two coordinates.
        """
        # Calculate the distance between the segment coordinate and the state
        distance = segment_coordinate.euclidean_distance(state)

        return distance

    def get_coordinate_tree(
        self, segment_coordinates: List[CoordinateTree]
    ) -> CoordinateTree:
        """
        Creates a coordinate tree from a list of segment coordinates.

        Args:
            segment_coordinates (List[Coordinate]): A list of segment coordinates.

        Returns:
            CoordinateTree: A coordinate chain_tree.tree.
        """
        # Create a coordinate tree from the segment coordinates
        coordinate_tree = CoordinateTree.from_tuple(segment_coordinates)

        return coordinate_tree

    def reduce(self, reducer: Reducer, initial: Any) -> Any:
        return reduce(reducer, self.coordinates, initial)

    def aggregate(
        self, initial: Any, operation: Callable[[Any, CoordinateTree], Any]
    ) -> Any:
        """
        Aggregate the coordinates by applying an operation that accumulates
        the results starting with an initial value.

        :param initial: The initial value for the accumulation.
        :param operation: A Callable that takes two arguments, the accumulator and a Coordinate, and returns the updated accumulator.
        :return: The result of the aggregation.
        """
        return self.reduce(operation, initial)

    def map(self, func: Callable[[CoordinateTree], Any]) -> List[Any]:
        return [func(coord) for coord in self.coordinates]

    def filter(
        self, predicate: Callable[[CoordinateTree], bool]
    ) -> "CoordinateHandler":
        filtered_coordinates = [coord for coord in self.coordinates if predicate(coord)]
        return CoordinateHandler(filtered_coordinates)

    def transform(
        self, operation: Callable[[CoordinateTree], CoordinateTree]
    ) -> "CoordinateHandler":
        transformed_coordinates = [operation(coord) for coord in self.coordinates]
        return CoordinateHandler(transformed_coordinates)

    def sort(
        self,
        key: Optional[Callable[[CoordinateTree], Any]] = None,
        reverse: bool = False,
    ) -> "CoordinateHandler":
        sorted_coordinates = sorted(self.coordinates, key=key, reverse=reverse)
        return CoordinateHandler(sorted_coordinates)

    def chain_operations(
        self, operations: List[Tuple[str, Union[Callable, Tuple]]]
    ) -> "CoordinateHandler":
        result = self
        for operation, args in operations:
            if operation == "map":
                result = result.map(*args)
            elif operation == "filter":
                result = result.filter(*args)
            elif operation == "transform":
                result = result.transform(*args)
            elif operation == "sort":
                result = result.sort(*args)
            elif operation == "reduce":
                result = result.reduce(*args)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        return result
