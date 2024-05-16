from typing import Dict, Any, List, Union
from chain_tree.engine.engine import calculate_similarity
from chain_tree.models import ChainMap, Chain
import networkx as nx
import numpy as np
import collections


class ChainRelationships:
    """
    Abstract class for representing relationships between messages.
    """

    RELATIONSHIP_TYPES = {
        "siblings": "_get_message_siblings",
        "cousins": "_get_message_cousins",
        "uncles_aunts": "uncles_aunts",
        "nephews_nieces": "nephews_nieces",
        "grandparents": "grandparents",
        "ancestors": "_get_message_ancestors",
        "descendants": "_get_message_descendants",
    }

    def __init__(self, mapping: Dict[str, ChainMap]):
        self.mapping = mapping
        self.message_dict = self._message_representation()
        self.conversation_dict = self._conversation_representation()
        self.relationship_dict = self._create_relationship_dict()
        self._message_tree = None
        self._initialize_graph()

    def _initialize_graph(self):
        self._message_tree = self._get_message_tree()

    def _conversation_representation(self) -> Dict[str, List[Any]]:
        """
        Creates a dictionary representation of the conversation chain_tree.tree.

        The keys are the message IDs and the values are a list of the form [parent, children].
        The parent is the ID of the parent message, or None if the message is the root.
        The children is a list of the IDs of the children messages, or an empty list if the message has no children.

        Returns:
            A dictionary representation of the conversation chain_tree.tree.
        """
        conversation_dict = {}
        for mapping in self.mapping.values():
            conversation_dict[mapping.id] = [mapping.parent, mapping.children]
        return conversation_dict

    def _message_representation(self) -> Dict[str, Dict[str, Chain]]:
        """
        Creates a dictionary representation of the conversation chain_tree.tree.
        The keys are the message IDs and the values are the corresponding Message of the form:

        Returns:
            A dictionary representation of the conversation chain_tree.tree.
        """
        message_dict = {}
        for message in self.mapping.values():
            if message is not None and message.id is not None:
                message_dict[message.id] = message
            else:
                print(f"Warning: Invalid message or message id: {message}")
        return message_dict

    @property
    def depth(self) -> int:
        """
        Returns the maximum depth of the conversation chain_tree.tree.

        Returns:
            depth: The maximum depth of the conversation chain_tree.tree.
        """
        return self.get_message_depth(self.root_message_id)

    @property
    def root_message_id(self) -> Union[str, None]:
        """Returns the root message of the conversation, or None if it doesn't exist."""
        for message in self.mapping.values():
            if message.parent is None:
                return self.message_dict[message.id].id if self.message_dict else None
        return None

    def calculate_similarity_score(self, child_id: str, next_child_id: str) -> float:
        """
        Calculate the cosine similarity score between two child messages.

        Args:
            child_id (str): The ID of the first child message.
            next_child_id (str): The ID of the second child message.

        Returns:
            float: The cosine similarity score between the two messages.
        """
        try:
            # Get the embeddings from the message dictionary
            child_embedding = self.message_dict[child_id].message.embedding
            next_child_embedding = self.message_dict[next_child_id].message.embedding

            # Check if embeddings are empty or contain NaN values
            if (
                not child_embedding
                or np.isnan(child_embedding).any()
                or not next_child_embedding
                or np.isnan(next_child_embedding).any()
            ):
                return 0.0

            # Call the standalone calculate_similarity function
            similarity_score = calculate_similarity(
                child_embedding, next_child_embedding
            )

            return similarity_score
        except Exception as e:
            print(f"An error occurred while calculating the similarity score: {e}")
            return 0.0  # Return a default similarity score

    def _gather_similarity_scores(self, children_ids: List[str]) -> List[float]:
        """
        Obtain Similarity Metrics for Child Messages.

        For a provided list of child messages, this method retrieves their similarity scores by
        comparing each child message with its adjacent siblings (both previous and next). The goal
        is to determine how related or similar adjacent messages in a sequence are.

        Parameters:
            children_ids (List[str]): A collection of message IDs that represent the child messages
                                    for which similarity scores should be gathered.

        Returns:
            List[float]: An array of similarity scores representing the degree of similarity between
                        each pair of adjacent child messages.
        """

        # Pre-calculate previous and next sibling IDs for all children
        prev_sibling_ids = [
            self._get_previous_sibling_id(child_id) for child_id in children_ids
        ]
        next_sibling_ids = [
            self._get_next_sibling_id(child_id) for child_id in children_ids
        ]

        # Gather similarity scores where both prev and next siblings exist
        similarity_scores = [
            self.calculate_similarity_score(prev_id, next_id)
            for prev_id, next_id in zip(prev_sibling_ids[:-1], next_sibling_ids[1:])
            if prev_id and next_id
        ]

        return similarity_scores

    def get_child_message(self, message_id: str):
        message_object = self.message_dict.get(message_id, None)
        if (
            message_object is None
            or message_object.message is None
            or message_object.message.children is None
        ):
            return None

        for child in message_object.message.children:
            child.dict()

        return message_object.message.children

    def get_message_content(self, message_id: str) -> str:
        """
        Get the content of a message with a given id.

        Args:
            message_id: The id of the message.

        Returns:
            The content of the message.
        """
        message_object = self.message_dict.get(message_id, None)
        if (
            message_object is None
            or message_object.message is None
            or message_object.message.content is None
        ):
            return None
        return message_object.message.content.text

    def get_message_embeddings(self, message_id: str) -> List[float]:
        """
        Get the embeddings of a message with a given id.

        Args:
            message_id: The id of the message.

        Returns:
            The embeddings of the message.
        """
        message_object = self.message_dict.get(message_id, None)
        if (
            message_object is None
            or message_object.message is None
            or message_object.message.embedding is None
        ):
            return None
        return message_object.message.embedding

    def get_message_coordinate(self, message_id: str) -> List[float]:
        """
        Get the coordinates of a message with a given id.

        Args:
            message_id: The id of the message.

        Returns:
            The coordinates of the message.
        """
        message_object = self.message_dict.get(message_id, None)
        if (
            message_object is None
            or message_object.message is None
            or message_object.message.coordinate is None
        ):
            return None
        return message_object.message.coordinate

    def get_message_author_role(self, message_id: str) -> str:
        """
        Get the author role of a message with a given id.

        Args:
            message_id: The id of the message.

        Returns:
            The author role of the message.
        """
        message_object = self.message_dict.get(message_id, None)
        if (
            message_object is None
            or message_object.message is None
            or message_object.message.author is None
        ):
            return None
        return message_object.message.author.role

    def get_message_create_time(self, message_id: str) -> str:
        """
        Get the creation time of a message with a given id.

        Args:
            message_id: The id of the message.

        Returns:
            The creation time of the message.
        """
        message_object = self.message_dict.get(message_id, None)
        if message_object is None or message_object.message is None:
            return None
        return message_object.message.create_time

    def _get_message_tree(self) -> nx.DiGraph:
        """
        Creates a networkx DiGraph representation of the conversation chain_tree.tree.

        Returns:
            A networkx DiGraph representation of the conversation chain_tree.tree.
        """
        tree = nx.DiGraph()
        for message_id, (parent_id, children_ids) in self.conversation_dict.items():
            tree.add_node(message_id, weight=self.message_dict[message_id])
            if parent_id is not None:
                tree.add_edge(parent_id, message_id)
        return tree

    def _child_message_representation(self) -> Dict[str, Dict[str, Any]]:
        """
        Creates a dictionary representation of the child messages in the conversation chain_tree.tree.

        Returns:
            A dictionary representation of the child messages in the conversation chain_tree.tree.
        """

        message_dict = self._message_representation()
        child_message_dict = {}

        for message_id, message in message_dict.items():
            if message is not None:
                children = message.children
                for child_message in children:
                    if child_message is not None and child_message.id is not None:
                        child_message_dict[child_message.id] = child_message
                    else:
                        print(
                            f"Warning: Invalid child message or child message id: {child_message}"
                        )

        return child_message_dict

    def _create_relationship_dict(self) -> Dict[str, Dict[str, int]]:
        """Creates a dictionary of relationships between messages"""
        relationship_dict = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        for message_id, (parent_id, children_ids) in self.conversation_dict.items():
            # Add parent relationship
            if parent_id:
                relationship_dict[message_id][parent_id] += 1

            # Add child relationship
            for child_id in children_ids:
                relationship_dict[message_id][child_id] += 1

        return relationship_dict

    def get_relationship(
        self, message_id: str
    ) -> Dict[str, Union[List[Dict[str, Any]], int, None]]:
        """Returns all relationships for the message with the given ID in a dictionary."""
        if message_id not in self.message_dict:
            return None

        relationship_dict = {}

        for relationship_type, method in self.RELATIONSHIP_TYPES.items():
            try:
                relationship_dict[relationship_type] = getattr(self, method)(message_id)
            except ValueError as e:
                print(f"An error occurred: {str(e)}")

        return relationship_dict

    def _get_message_relationship_ids(
        self, message_id: str, relationship_type: str
    ) -> List[str]:
        relationships = self.get_message_reference(message_id, relationship_type)
        if relationships is None:
            return []
        return [rel.id for rel in relationships]

    def get_relationship_ids(self, message_id: str) -> Dict[str, List[str]]:
        """Returns a list of all message IDs that are related to the message with the given ID."""
        result = {}
        # exclude the descendants, ancestors, and siblings
        for relationship_type in [
            "cousins",
            "uncles_aunts",
            "nephews_nieces",
            "grandparents",
            "ancestors",
            "descendants",
            "siblings",
        ]:
            result[relationship_type] = self._get_message_relationship_ids(
                message_id, relationship_type
            )

        return result

    def _get_previous_sibling_id(self, message_id: str) -> Union[str, None]:
        """Returns the ID of the previous sibling of the message with the given ID, or None if the message is the first child or doesn't exist"""
        parent_id = self.get_parent_id(message_id)
        if parent_id is None:
            return None
        siblings_ids = self.get_children_ids(parent_id)
        try:
            index = siblings_ids.index(message_id)
        except ValueError:
            return None
        return siblings_ids[index - 1] if index > 0 else None

    def _get_next_sibling_id(self, message_id: str) -> Union[str, None]:
        """Returns the ID of the next sibling of the message with the given ID, or None if the message is the last child or doesn't exist"""
        parent_id = self.get_parent_id(message_id)
        if parent_id is None:
            return None
        siblings_ids = self.get_children_ids(parent_id)
        try:
            index = siblings_ids.index(message_id)
        except ValueError:
            return None
        return siblings_ids[index + 1] if index < len(siblings_ids) - 1 else None

    def get_message_reference(
        self, message_id: str, relationship_type: str
    ) -> Union[List[Dict[str, Any]], int, None]:
        """Returns the requested relationship for the message with the given ID."""
        if message_id not in self.message_dict:
            return None

        if relationship_type in self.RELATIONSHIP_TYPES:
            return getattr(self, self.RELATIONSHIP_TYPES[relationship_type])(message_id)
        else:
            raise ValueError(f"Unsupported relationship type: {relationship_type}")

    def get_parent_id(self, message_id: str) -> Union[str, None]:
        """Returns the ID of the parent of the message with the given ID, or None if the message doesn't exist or is the root"""
        if message_id not in self.message_dict:
            return None
        return self.conversation_dict[message_id][0]

    def get_children_ids(self, message_id: str) -> Union[List[str], None]:
        """Returns a list of the IDs of the children of the message with the given ID, or None if the message doesn't exist"""
        if message_id not in self.message_dict:
            return None
        return self.conversation_dict[message_id][1]

    def get_message_parent(self, message_id: str) -> Union[Dict[str, Any], None]:
        """
        Returns the parent of the message with the given ID or None if the message doesn't exist or is the root.

        Args:
            message_id (str): The ID of the message.

        Returns:
            Union[Dict[str, Any], None]: Dictionary representation of the parent message or None.
        """
        if message_id not in self.message_dict:
            return None
        parent_id = self.conversation_dict.get(message_id, [None])[0]
        if not parent_id:
            return None
        return self.message_dict.get(parent_id)

    def get_message_children(
        self, message_id: str
    ) -> Union[List[Dict[str, Any]], None]:
        """Returns a list of the children of the message with the given ID, or None if the message doesn't exist"""
        if message_id not in self.message_dict:
            return None
        children_ids = self.conversation_dict[message_id][1]
        children = [self.message_dict[child_id] for child_id in children_ids]
        return children

    def _get_message_ancestors(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the ancestors of the message with the given ID"""
        ancestors = []
        stack = [message_id]

        while stack:
            current_id = stack.pop()
            parent = self.get_message_parent(current_id)
            if parent is None:
                continue
            ancestors.append(parent)
            stack.append(parent.id)
        return ancestors

    def _get_message_descendants(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the descendants of the message with the given ID"""
        descendants = []
        stack = [message_id]

        while stack:
            current_id = stack.pop()
            children = self.get_message_children(current_id) or []

            descendants.extend(children)
            stack.extend([child.id for child in children])
        return descendants

    def _get_message_siblings(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the siblings of the message with the given ID"""
        parent = self.get_message_parent(message_id)
        if parent is None:
            return []
        siblings = self.get_message_children(parent.id)
        siblings = [sibling for sibling in siblings if sibling.id != message_id]
        return siblings

    def _get_message_cousins(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the cousins of the message with the given ID"""
        parent = self.get_message_parent(message_id)
        if parent is None:
            return []
        uncles_and_aunts = self._get_message_siblings(parent.id)
        cousins = []
        for uncle_or_aunt in uncles_and_aunts:
            cousins.extend(self.get_message_children(uncle_or_aunt.id))
        return cousins

    def uncles_aunts(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the uncles and aunts of the message with the given ID"""
        parent = self.get_message_parent(message_id)
        if parent is None:
            return []
        return self._get_message_siblings(parent.id)

    def nephews_nieces(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the nephews and nieces of the message with the given ID"""
        uncles_and_aunts = self.uncles_aunts(message_id)
        nephews_nieces = []
        for uncle_or_aunt in uncles_and_aunts:
            nephews_nieces.extend(self.get_message_children(uncle_or_aunt.id))
        return nephews_nieces

    def grandparents(self, message_id: str) -> List[Dict[str, Any]]:
        """Returns a list of the grandparents of the message with the given ID"""
        parent = self.get_message_parent(message_id)
        if parent is None:
            return []
        return self._get_message_ancestors(parent.id)

    def get_message_depth(self, message_id: str) -> int:
        """Returns the depth of the message with the given ID"""
        return self._get_message_depth(message_id, 0)

    def _get_message_depth(self, message_id: str, depth: int) -> int:
        """Returns the depth of the message with the given ID"""
        parent = self.get_message_parent(message_id)
        if parent is None:
            return depth
        return self._get_message_depth(parent.id, depth + 1)

    def get_parents_ids(self, message_id: str) -> List[str]:
        """Returns a list of the IDs of the parents of the message with the given ID"""
        return self._get_parents_ids(message_id, [])

    def _get_parents_ids(self, message_id: str, parents_ids: List[str]) -> List[str]:
        """Returns a list of the IDs of the parents of the message with the given ID"""
        parent = self.get_message_parent(message_id)
        if parent is None:
            return parents_ids
        parents_ids.append(parent.id)
        return self._get_parents_ids(parent.id, parents_ids)
