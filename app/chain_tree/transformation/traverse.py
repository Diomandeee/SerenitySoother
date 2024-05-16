from typing import Callable, List, Optional, Tuple
from chain_tree.transformation.tree import CoordinateTree
from chain_tree.transformation.coordinate import Coordinate
from heapq import heappush, heappop
import pandas as pd
import numpy as np


class CoordinateTreeTraverser:
    def __init__(self, tree: CoordinateTree):
        self.tree = tree

    def traverse(
        self,
    ) -> List[Coordinate]:
        """Traverse the tree and return a list of nodes."""
        return self.tree.to_list()

    def to_list(self) -> List[Coordinate]:
        """Convert the tree into a list of Coordinates."""
        return self.tree.to_list()

    def to_dict(self) -> dict:
        """Convert the tree into a dictionary representation."""
        return self.tree.to_dict()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the tree into a DataFrame representation."""
        return self.tree.to_dataframe()

    def traverse_depth_first(
        self, predicate: Callable[[Coordinate], bool]
    ) -> Optional[Coordinate]:
        """
        Traverse the tree depth-first and returns the first node that matches the predicate.

        Args:
            predicate (Callable[[Coordinate], bool]): A function to test each node. Returns True for matches.

        Returns:
            Optional[Coordinate]: The first matching node or None if not found.
        """
        if not callable(predicate):
            raise ValueError("Provided predicate must be callable.")
        return self.tree.depth_first_search(predicate)

    def traverse_breadth_first(
        self, predicate: Callable[[Coordinate], bool]
    ) -> Optional[Coordinate]:
        """
        Traverse the tree breadth-first and returns the first node that matches the predicate.

        Args:
            predicate (Callable[[Coordinate], bool]): A function to test each node. Returns True for matches.

        Returns:
            Optional[Coordinate]: The first matching node or None if not found.
        """
        if not callable(predicate):
            raise ValueError("Provided predicate must be callable.")
        return self.tree.breadth_first_search(predicate)

    def traverse_depth_first_all(
        self, predicate: Callable[[Coordinate], bool]
    ) -> List[Coordinate]:
        """
        Traverse the tree depth-first and returns all nodes that match the predicate.

        Args:
            predicate (Callable[[Coordinate], bool]): A function to test each node. Returns True for matches.

        Returns:
            List[Coordinate]: A list of all matching nodes.
        """
        if not callable(predicate):
            raise ValueError("Provided predicate must be callable.")
        return self.tree.depth_first_search_all(tree=self.tree, predicate=predicate)

    def traverse_similarity(
        self,
        target_sentence: str,
        fast_predicate: Optional[Callable[[Coordinate], bool]] = None,
        top_k: int = 1,
        keyword: Optional[str] = None,
        encoding_function: Optional[Callable[[str], np.ndarray]] = None,
    ) -> List[Tuple[Coordinate, float]]:
        most_similar_nodes = []
        if not encoding_function:
            target_embedding = encoding_function([target_sentence])
        else:
            target_embedding = target_sentence

        candidate_nodes = (
            self.traverse_depth_first_all(predicate=fast_predicate)
            if fast_predicate
            else self.traverse_depth_first_all(predicate=lambda x: True)
        )
        # Convert target_embedding to a unit vector if it isn't already
        target_norm = np.linalg.norm(target_embedding)
        if target_norm > 0:
            target_unit_vector = target_embedding / target_norm
        else:
            target_unit_vector = target_embedding

        for node in candidate_nodes:
            # If keyword filtering is enabled, skip nodes that don't contain the keyword
            if keyword and keyword.lower() not in node.text.lower():
                continue

            node_embedding = node.embedding  # Assuming embeddings are stored here

            # Convert node_embedding to a unit vector if it isn't already
            node_norm = np.linalg.norm(node_embedding)
            if node_norm > 0:
                node_unit_vector = node_embedding / node_norm
            else:
                node_unit_vector = node_embedding

            # Compute cosine similarity
            cosine_sim = np.dot(target_unit_vector, node_unit_vector)

            heappush(most_similar_nodes, (cosine_sim, node))

            # If heap is too big, remove smallest
            if len(most_similar_nodes) > top_k:
                heappop(most_similar_nodes)

        # Extract nodes and actual similarity scores, and sort by similarity
        return [
            (node, similarity)
            for similarity, node in sorted(most_similar_nodes, reverse=True)
        ]
