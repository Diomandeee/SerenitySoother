from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from chain_tree.models import ChainTree, Content, Chain
from chain_tree.engine.embedder import get_text_chunks
from chain_tree.transformation.tree import CoordinateTree
from chain_tree.engine.structure import ChainStruct
from chain_tree.type import NodeRelationship
import networkx as nx
import pandas as pd
import numpy as np


class Representation(ChainStruct):
    RELATIONSHIP_WEIGHTS = {
        "siblings": 1,
        "cousins": 2,
        "uncles_aunts": 3,
        "nephews_nieces": 3,
        "grandparents": 4,
        "ancestors": 5,
        "descendants": 5,
        NodeRelationship.PARENT: 1,
        NodeRelationship.CHILD: 1,
        NodeRelationship.PREVIOUS: 1,
        NodeRelationship.NEXT: 1,
        NodeRelationship.SOURCE: 1,
    }

    def __init__(
        self,
        conversation_tree: ChainTree,
        message_dict: Dict[str, Chain] = None,
        tetra_dict: Dict[str, Tuple[float, float, float, float, int]] = None,
        root_component_values: Dict[str, Any] = None,
    ):
        self.conversation = conversation_tree
        self.mapping = conversation_tree.mapping
        self.message_dict = message_dict
        self.tetra_dict = tetra_dict
        self.conversation_dict = self._conversation_representation()
        self.relationships = {}
        self.graph = None
        self.default_root_component_values = {
            "depth_args": [0],
            "sibling_args": [0],
            "sibling_count_args": [0],
            "time_args": [0],
        }
        # If root component values are provided, update the default ones
        if root_component_values:
            self.default_root_component_values.update(root_component_values)

        # Construct root coordinate with updated component values
        self.root_coordinate = CoordinateTree.create(
            **self.default_root_component_values
        )

    @classmethod
    def from_conversation_json(cls, conversation_json_path: str) -> "Representation":
        """
        Create a CoordinateRepresentation object from a conversation JSON file.

        Args:
            conversation_json_path (str): Path to the conversation JSON file.

        Returns:
            CoordinateRepresentation: A CoordinateRepresentation object.
        """
        conversation_tree = ChainTree.from_json(conversation_json_path)
        return cls(conversation_tree)

    def _sort_children_by_time(self, children_ids) -> List:
        return sorted(
            children_ids,
            key=lambda id: (
                self.message_dict[id].message.create_time
                if id in self.message_dict
                else 0
            ),
        )

    def calculate_ewp(self, siblings, message_dict):
        """
        Calculate the Engagement Weighted Position (EWP) for a message among its siblings.
        EWP is a weighted average position based on the engagement of each message, measured
        by the number of children.

        Args:
            siblings (List[str]): List of IDs of sibling messages.
            message_dict (Dict[str, Message]): Dictionary mapping message IDs to message objects.

        Returns:
            float: The engagement weighted position of the message.
        """
        if not siblings:
            return 0

        engagement_scores = [
            len(message_dict[sibling].children) for sibling in siblings
        ]
        total_engagement = sum(engagement_scores)

        if total_engagement == 0:
            return 0  # Consider whether another default value is more appropriate

        ewp_scores = [eng / total_engagement for eng in engagement_scores]
        weighted_positions = [score * (idx + 1) for idx, score in enumerate(ewp_scores)]
        return sum(weighted_positions)

    def calculate_mean_sibling_depth(self, parent_id: str) -> float:
        """
        Calculate the mean depth of all siblings of a given node.
        """
        siblings = [
            self.message_dict[sib]
            for sib in self.message_dict
            if self.message_dict[sib].parent == parent_id
        ]
        if not siblings:
            return 0
        return np.mean([sib.depth for sib in siblings])

    def calculate_max_siblings_at_depth(self, depth: int) -> int:
        """
        Determine the maximum number of siblings any node has at a given depth.
        """
        nodes_at_depth = [
            node for node in self.message_dict.values() if node.depth == depth
        ]
        return max([len(node.children) for node in nodes_at_depth], default=0)

    def calculate_other_components(
        self, depth: int, index: int, message_id: str, children: List[str]
    ) -> Tuple[int, float, int, int, float, int, int, float]:
        """
        Calculate remaining spatial components for a message node.
        """
        message = self.message_dict[message_id]

        # Depth Components
        x = depth + 1  # Depth level incremented as we go deeper in the tree
        s_x = self.calculate_mean_sibling_depth(message.parent) if message.parent else 0
        c_x = max((self.message_dict[child].depth for child in children), default=0)

        # Sibling Components
        y = (
            index + 1
        )  # Position among siblings, assuming 1-indexed for clarity in usage

        # Sibling Count Component
        z = (
            0 if len(children) == 1 else -0.5 * (len(children) - 1)
        )  # Position among siblings in the z-axis
        m_z = self.calculate_max_siblings_at_depth(
            depth
        )  # Max sibling count at this depth across the tree

        # Time Component
        t = message.create_time  # Assuming create_time is an integer or float timestamp
        p_y = self.calculate_ewp(
            self._sort_children_by_time(children), self.message_dict
        )  # Engagement Weighted Position

        return (x, s_x, c_x, y, z, m_z, t, p_y)

    def _update_coordinates(
        self, depth: int, index: int, message: any, children: List[str]
    ) -> Tuple[int, int, int, int, int]:
        """
        Calculate spatial coordinates (x, y, z, t, p) for a message node.

        Args:
            depth (int): The depth level of the message node in the chain_tree.tree.
            index (int): The index of the message node within its level.
            message (any): The message object for the node.
            children (List[str]): The list of message IDs of child nodes.

        Returns:
            Tuple[int, int, int, int, int]: A tuple containing:
                - x_coord (int): The x-coordinate (depth) of the node.
                - y_coord (int): The y-coordinate (index) of the node.
                - z_coord (int): The z-coordinate (reverse index) of the node.
                - t_coord (int): The t-coordinate (timestamp) of the node.
                - p_coord (int): The p-coordinate (paragraph count) of the node.
        """
        try:
            x_coord = depth
            y_coord = index
            z_coord = 0 if len(children) == 1 else -0.5 * (len(children) - 1)
            p_coord = len(message.message.content.text.split("\n\n"))
            t_coord = message.message.create_time
            return (x_coord, y_coord, z_coord, t_coord, p_coord)
        except Exception as e:
            # Log the exception and return a default value or re-raise the exception
            print(f"Error in update_coordinates: {e}")
            return (0, 0, 0, 0, 0)

    def _calculate_part_weight(self, n_parts: int) -> float:
        """
        Calculate the weight of each part in a multi-part message.

        Args:
            n_parts: The number of parts in the message.

        Returns:
            The weight of each part as a float.
        """
        return round(1.0 / n_parts, 2) if n_parts > 0 else 0

    def _get_mapping(self, child_id: str) -> Any:
        """
        Retrieve the mapping object for a message by its ID.

        Args:
            child_id: The ID of the child message.

        Returns:
            The mapping object associated with the child message.

        Raises:
            ValueError: If the message is not found in the message dictionary.
        """
        mapping = self.message_dict.get(child_id)
        if not mapping:
            raise ValueError(f"Message {child_id} not found in message_dict")
        return mapping

    def _calculate_unique_child_id(self, child_id: str, index: int) -> str:
        """
        Calculates a unique ID for a child message by appending an index to its parent ID.

        Parameters:
        - child_id (str): The ID of the child message.
        - index (int): The index to append to the child_id.

        Returns:
        - str: A unique ID for the child message.
        """
        return f"{child_id}_{index}"

    def add_relationship(
        self, from_id: str, to_id: str, relationship: NodeRelationship
    ):
        """
        Establishes a directed relationship between two messages identified by their IDs.

        Operations:
        1. Checks if the source message (represented by 'from_id') already has any relationships recorded in the system.
        2. If the source message doesn't have any existing relationships, it initializes an empty dictionary for it.
        3. Registers the directed relationship from the source message (from_id) to the target message (to_id) with the specified type.

        Args:
            from_id (str): Identifier of the source message.
            to_id (str): Identifier of the target message.
            relationship (NodeRelationship): The type of relationship being established.
        """

        if from_id not in self.relationships:
            self.relationships[from_id] = {}

        self.relationships[from_id][to_id] = relationship

    def get_message_attribute(self, message_id: str, *attributes: str):
        """
        Get a specific attribute of a message given its id.

        Args:
            message_id: The id of the message.
            attributes: The sequence of attributes to fetch (e.g., "content", "text").

        Returns:
            The desired attribute of the message.
        """
        try:
            value = self.message_dict[message_id].message
            for attribute in attributes:
                if hasattr(value, attribute):
                    value = getattr(value, attribute)
                else:
                    raise AttributeError(f"Attribute {attribute} not found in message.")
            return value
        except KeyError:
            raise ValueError(f"Message with id {message_id} not found.")

    def _get_message_attributes(self, child_id: str) -> Tuple[Any, Any, Any]:
        """
        Retrieve attributes of a message by its ID.

        Args:
            child_id: The ID of the child message.

        Returns:
            A tuple containing the create_time, author, and text of the message.
        """
        create_time = self.get_message_attribute(child_id, "create_time")
        author = self.get_message_attribute(child_id, "author")
        text = self.get_message_attribute(child_id, "content", "text")
        return create_time, author, text

    def _create_representation(self, include_system: bool = False) -> nx.DiGraph:
        """
        Creates a NetworkX directed graph representation of the conversation chain_tree.tree.
        Each node in the graph represents a message, and each edge indicates a
        relationship between messages. The relationships can be:
        - From the previous message to the current message.
        - From a parent message to the current message.
        - From the current message to all its references.

        Nodes are annotated with message content and authors. The edges don't have
        any additional annotations in the current implementation.

        Args:
            include_system (bool): If False, system messages are excluded from the graph.

        Returns:
            A NetworkX directed graph representation of the conversation chain_tree.tree.
        """
        graph = nx.DiGraph()
        prev_node = None

        for mapping_id, mapping in self.mapping.items():
            if mapping.message is None:
                continue

            # Skip system messages if include_system is False
            if not include_system and mapping.message.author.role == "system":
                continue

            # Add the node to the graph
            graph.add_node(mapping_id, **mapping.message.dict())

            # If this isn't the first node, create an edge from the previous node
            if prev_node is not None:
                graph.add_edge(prev_node, mapping_id)

            # If the mapping has a parent, create an edge from the parent
            if mapping.parent is not None:
                graph.add_edge(mapping.parent, mapping_id)

            # Add edges to all references
            for ref_id in mapping.references:
                if ref_id in self.mapping:
                    graph.add_edge(mapping_id, ref_id)

            # Update the previous node
            prev_node = mapping_id

        self.graph = graph
        return graph

    def create_representation(
        self,
        node_ids: Optional[List[str]] = None,
        attribute_filter: Optional[Dict[str, Any]] = None,
        include_system: bool = True,
        time_range: Optional[Dict[str, Union[str, int]]] = None,
    ) -> nx.DiGraph:
        """
        Transforms a conversation tree into a NetworkX directed graph. The graph represents the structure
        and flow of the conversation, with messages as nodes and their response relationships as edges.

        Args:
            node_ids (Optional[List[str]]): Specifies which node IDs should be considered for the graph.
                If given, this will be the primary filter applied to shape the resulting graph.
            attribute_filter (Optional[Dict[str, Any]]): Provides criteria to include nodes that match
                specified attributes. This filter is applied if `node_ids` isn't given.
            include_system (bool): Decides whether messages from the system should be a part of the graph.
            time_range (Optional[Dict[str, Union[str, int]]]): Defines a period in which the messages were
                created. Messages outside this range are excluded. This filter is considered if both
                `node_ids` and `attribute_filter` aren't provided.

        Returns:
            A directed graph capturing the essence of the conversation. Filters are applied in the order they
            appear in the arguments, so only the first available filter will be used.
        """
        # Get the full graph representation, consider the include_system flag
        graph = self._create_representation(include_system)

        # If node_ids are provided, use them to create the subgraph
        if node_ids is not None:
            subgraph = graph.subgraph(node_ids)

        # If attribute_filter is provided, select nodes based on attributes
        elif attribute_filter is not None:
            selected_nodes = [
                node
                for node, data in graph.nodes(data=True)
                if all(item in data.items() for item in attribute_filter.items())
            ]
            subgraph = graph.subgraph(selected_nodes)

        # Additional filtering based on time_range
        elif time_range is not None:
            start_time, end_time = time_range.get("start", None), time_range.get(
                "end", None
            )
            selected_nodes = [
                node
                for node, data in graph.nodes(data=True)
                if (
                    start_time is None
                    or data["message"].get("create_time") >= start_time
                )
                and (end_time is None or data["message"].get("create_time") <= end_time)
            ]
            subgraph = graph.subgraph(selected_nodes)

        # If no filters are provided, return the full graph
        else:
            subgraph = graph

        return subgraph

    def initialize_representation(
        self,
        node_ids: Optional[List[str]] = None,
        attribute_filter: Optional[Dict[str, Any]] = None,
        include_system: bool = False,
        time_range: Optional[Dict[str, Union[str, int]]] = None,
        RELATIONSHIP_TYPE=NodeRelationship,
    ) -> Tuple[Dict, Callable, Dict, str, Any]:
        """
        Prepares the initial setup for conversation visualization. Depending on the `use_graph` parameter,
        it either transforms the conversation into a directed graph or leverages the existing conversation data structure.
        The method then identifies the root node and prepares relationship mappings, and spatial coordinates for visualization.

        Args:
            node_ids (Optional[List[str]]): Specifies node IDs to be considered if using graph representation.
            attribute_filter (Optional[Dict[str, Any]]): Sets criteria for including nodes in graph representation.
            include_system (bool): Decides if system messages should be part of the representation.
            time_range (Optional[Dict[str, Union[str, int]]]): Filters messages based on creation time, if using graph.
            RELATIONSHIP_TYPE: Enum that defines node relationships. Default is `NodeRelationship`.

        Returns:
            A tuple containing:
            - relationships: A dictionary mapping node IDs to their relationships.
            - tetra_dict: A dictionary mapping node IDs to spatial coordinates for visualization.
            - root_id: The ID of the root node.
            - root_coordinate: The spatial coordinate of the root node.
        """
        relationships = {}

        G = self.create_representation(
            node_ids=node_ids,
            attribute_filter=attribute_filter,
            include_system=include_system,
            time_range=time_range,
        )

        if G.number_of_nodes() == 0:
            raise ValueError("No nodes found in the graph.")

        # Get the root node
        root_id = list(nx.topological_sort(G))[0]

        # Get the tetra dict
        relationships[root_id] = {RELATIONSHIP_TYPE.SOURCE: root_id}

        tetra_dict = {}
        tetra_dict[root_id] = self.root_coordinate.flatten(self.root_coordinate)

        return (
            relationships,
            tetra_dict,
            root_id,
            self.root_coordinate,
        )

    def _assign_relationships(
        self,
        message_id: str,
        child_id: str,
        children_ids: List[str],
        i: int,
        relationships: Dict[str, Dict[str, str]],
        RELATIONSHIP_TYPE=NodeRelationship,
    ) -> Dict[str, Dict[str, str]]:
        """
        Determines the hierarchical and sequential relationships of a child message in relation to its siblings and parent.

        Given a child message and its parent, this method establishes:
        - The parent-child relationship.
        - The previous sibling of the child, if it exists.
        - The next sibling of the child, if it exists.
        Additionally, it fetches any extended relationships for the child message and integrates them with the established relationships.

        Args:
            message_id (str): ID of the parent message.
            child_id (str): ID of the current child message being processed.
            children_ids (List[str]): List of IDs representing all children of the parent message.
            i (int): Index of the child message in the children_ids list.
            relationships (Dict[str, Dict[str, str]]): A dictionary containing relationships of all messages processed so far.
            RELATIONSHIP_TYPE (Optional[Enum]): Defines the types of node relationships, defaulting to `NodeRelationship`.

        Returns:
            Dict[str, Dict[str, str]]: Updated dictionary containing relationships for messages, inclusive of the current child.
        """

        # Define relationships for the child message
        child_relationships = {
            RELATIONSHIP_TYPE.PARENT: message_id,
            RELATIONSHIP_TYPE.CHILD: [],
            RELATIONSHIP_TYPE.PREVIOUS: children_ids[i - 1] if i > 0 else None,
            RELATIONSHIP_TYPE.NEXT: (
                children_ids[i + 1] if i < len(children_ids) - 1 else None
            ),
        }

        # Get extended relationships, if any
        extended_relationships = self.get_relationship_ids(child_id)

        # Merge the two dictionaries
        relationships[child_id] = {**child_relationships, **extended_relationships}

        return relationships

    def _create_child_messages(
        self,
        child_id: str,
        x_coord: float,
        y_coord: float,
        z_coord: float,
        t_coord: float,
        content_parts: List[str],
        create_time: str,
        author: str,
        part_weight: float,
    ) -> List[Chain]:
        """
        Constructs and interlinks child messages derived from content parts for a parent message.

        For each content part:
        1. Computes a unique child ID.
        2. Calculates its coordinates in the context.
        3. Creates a new message for the given content part.
        4. Associates the child message with the previous child (if any) to maintain the order.
        5. Links the child message with its parent.

        Returns:
            List[Chain]: A list of child messages created from the content parts.
        """
        child_messages = []
        prev_child = None  # To keep track of the previous child Chain object

        for index, part in enumerate(content_parts):
            new_child_id = f"{child_id}_{index}"  # Ensures a unique ID for each child
            children_coordinate = CoordinateTree(
                id=new_child_id,
                x=x_coord,
                y=y_coord,
                z=z_coord,
                t=t_coord,
                n_parts=index,
            )
            # Determine the next child in advance
            next_child_id = (
                f"{child_id}_{index+1}" if index + 1 < len(content_parts) else None
            )

            child_message = self.create_chain_message(
                parent_id=child_id,
                message_id=new_child_id,
                content_part=part,
                create_time=create_time,
                author=author,
                weight=part_weight,
                coordinate=children_coordinate,
                next=next_child_id,  # Set the next child ID (could be None if this is the last part)
                prev=prev_child.id if prev_child else None,  # Set the previous child ID
            )

            if prev_child:
                # Link the previous child's next to this new child id
                prev_child.next = child_message.id

            child_messages.append(child_message)
            prev_child = child_message  # Update the previous child to this child for the next iteration

        return child_messages

    def _assign_coordinates(
        self,
        child_id: str,
        i: int,
        children_ids: List[str],
        depth: int,
        message_id: str,
        create_child_message: bool = False,
        **kwargs: Dict[str, Union[str, float]],
    ) -> None:
        """
        Determines and assigns tetrahedral coordinates to a specific child message based on its position and role in the conversation chain_tree.tree.

        Given a child message and its parent, this method:
        1. Fetches the current mapping associated with the child ID.
        2. Calculates the tetrahedral coordinates for the child message based on its index within its sibling group, its depth in the tree, and other factors.
        3. Computes the weight or importance of each content segment.
        4. Retrieves the necessary attributes of the child message, such as the timestamp, author, and text content.
        5. Divides the text content into multiple parts or chunks.
        6. Constructs a coordinate tree representation for the child message using the calculated coordinates and the number of content parts.
        7. Produces and associates child messages from the text chunks, each having its unique coordinate in the tetrahedral space.
        8. Updates the internal mapping for the child message with its children and coordinates.

        Args:
            child_id (str): Identifier of the child message whose coordinates are being determined.
            i (int): Position of the child message among its siblings.
            children_ids (List[str]): Identifiers of all sibling messages.
            depth (int): Depth level of the message in the tree hierarchy.
            kwargs (Dict[str, Union[str, float]]): Additional optional parameters.

        Returns:
            flattened_coordinate: The calculated coordinate in a flattened format.
        """

        mapping = self._get_mapping(child_id)

        x_coord, y_coord, z_coord, t_coord, n_parts = self._update_coordinates(
            depth,
            i,
            mapping,
            children_ids,
        )

        create_time, author, text = self._get_message_attributes(child_id)
        if create_child_message:
            content_parts = get_text_chunks(text)

        child_coordinate = CoordinateTree(
            parent_id=message_id,
            id=child_id,
            x=x_coord,
            y=y_coord,
            z=z_coord,
            t=t_coord,
            text=text,
            n_parts=n_parts,
            author=author.role,
            parent=mapping.parent,
        )

        if create_child_message:
            child_messages = self._create_child_messages(
                child_id,
                x_coord,
                y_coord,
                z_coord,
                t_coord,
                content_parts,
                create_time,
                author,
                self._calculate_part_weight(len(content_parts)),
            )
            mapping.message.children = child_messages

        mapping.message.coordinate = child_coordinate
        flattened_coordinate = child_coordinate.flatten(child_coordinate)

        return flattened_coordinate

    def create_chain_message(
        self,
        parent_id: str,
        message_id: str,
        content_part: str,
        create_time: float,
        author: str,
        weight: float,
        coordinate: CoordinateTree,
        next: Optional[str] = None,
        prev: Optional[str] = None,
    ) -> Chain:
        """
        Creates a new message in the chain with associated attributes.

        Parameters:
        - message_id (str): The unique identifier for the new message.
        - content_part (str): The content or part of the message text.
        - create_time (float): The time at which the message was created.
        - author (str): The author or sender of the message.
        - weight (float): The weight to assign to this part of the message.
        - coordinate (Coordinate): The coordinate object representing the position of the message.

        Returns:
        - Chain: The newly created message object.
        """
        child_message = Chain(
            parent=parent_id,
            id=message_id,
            author=author,
            content=Content(parts=[content_part]),
            coordinate=coordinate,
            create_time=create_time,
            weight=weight,
            next=next,
            prev=prev,
        )
        return child_message


class HierarchyRepresentation(Representation):
    def create_representation(self) -> Dict[str, Any]:
        """
        Creates a dictionary representation of the conversation tree.
        The dictionary has the following structure:

        {
            <parent id>: {
                "message": <message content>,
                "children": [
                    <child id>,
                    ...
                ]
            },
            ...
        }
        """
        hierarchy = {}
        for id, node in self.mapping.items():
            children_ids = [child for child in node.children]
            message = self.message_dict[id]
            if message is not None:
                hierarchy[node.id] = {
                    "message": message.content,
                    "children": children_ids,
                }
            else:
                hierarchy[node.id] = {"message": "", "children": children_ids}
        return hierarchy


class ListRepresentation(Representation):
    def create_representation(self) -> Dict[str, Any]:
        """
        Creates a dictionary representation of the conversation tree.
        The dictionary has the following structure:

        [
            {
                "id": <message id>,
                "message": <message content>,
                "children": [<child id>, ...]
            },
            ...
        ]
        """
        rep_list = []
        for id, node in self.mapping.items():
            children_ids = [child for child in node]
            message = self.message_dict[id]
            content = message.content.parts if message is not None else ""
            rep_list.append(
                {"id": node.id, "message": content, "children": children_ids}
            )
        return rep_list


class SequentialMessagesRepresentation(Representation):
    def create_representation(self) -> List[Tuple[str, str]]:
        """
        Creates a sequential representation of the conversation tree.
        The  representation has the following structure:
        [
            {
                "message": <message content>,
                "author": <message author>
            },
            ...
        ]

        """

        message_list = []

        def _traverse_sequentially(node_id: str):
            node = self.mapping[node_id]
            message_list.append((node.message.content, node.message.author))
            for child in node.children:
                _traverse_sequentially(child.id)

        for node in self.mapping.values():
            if node.parent is None:
                _traverse_sequentially(node.id)

        return message_list


class AdjacencyMatrixRepresentation(Representation):
    def create_representation(self) -> pd.DataFrame:
        message_ids = [message_id for message_id in self.mapping.keys()]
        adjacency_matrix = pd.DataFrame(0, index=message_ids, columns=message_ids)

        for message_id, mapping in self.mapping.items():
            for child in mapping.children:
                adjacency_matrix.at[message_id, child] = 1

        return adjacency_matrix


class ChildParentRepresentation(Representation):
    def create_representation(self) -> Dict[str, str]:
        """
        Creates a dictionary representation where each child id is mapped to its parent id.
        This representation will be helpful to quickly lookup the parent of any message.

        Returns:
            children (Dict[str, str]): The dictionary mapping each child id to its parent id.
        """
        children = {}
        for message in self.mapping.values():
            for child in message.children:
                children[child] = message.id
        return children


class RootChildRepresentation(Representation):
    def create_representation(self) -> Dict[str, List[str]]:
        """
        Creates a dictionary representation where the root id maps to all the leaf nodes.
        This representation can be useful to get all leaf nodes from the root.

        Returns:
            root_child (Dict[str, List[str]]): The dictionary mapping the root id to all the leaf nodes.
        """
        root_child = {}
        for message in self.mapping.values():
            if message.parent is None:  # this is the root node
                root_child[message.id] = self._find_leaf_nodes(message)
        return root_child

    def _find_leaf_nodes(self, node) -> List[str]:
        leaf_nodes = []
        for child_id in node.children:
            child_node = self.mapping[child_id]
            if child_node.children:
                leaf_nodes.extend(self._find_leaf_nodes(child_node))
            else:
                leaf_nodes.append(child_id)
        return leaf_nodes


class MessagesWithMetadata(Representation):
    def create_representation(self) -> List[Dict[str, Any]]:
        messages_metadata = []

        for node in self.mapping.values():
            if node.parent is None:
                self._traverse_tree(node.id, messages_metadata)

        return messages_metadata

    def _traverse_tree(self, node_id: str, messages_metadata: List[Dict[str, Any]]):
        node = self.mapping[node_id]
        messages_metadata.append(
            {"message": node.message.content, "metadata": node.message.metadata}
        )

        for child in node.children:
            self._traverse_tree(child, messages_metadata)


class MessagesByAuthorRole(Representation):
    def create_representation(self) -> Dict[str, List[Dict[str, Any]]]:
        messages_by_role = {}

        for node in self.mapping.values():
            if node.parent is None:
                self._traverse_tree(node.id, messages_by_role)

        return messages_by_role

    def _traverse_tree(
        self, node_id: str, messages_by_role: Dict[str, List[Dict[str, Any]]]
    ):
        node = self.mapping[node_id]
        role = node.message.author.role

        if role not in messages_by_role:
            messages_by_role[role] = []

        messages_by_role[role].append(
            {"message": node.message.content, "metadata": node.message.metadata}
        )

        for child in node.children:
            self._traverse_tree(child, messages_by_role)


class ThreadRepresentation(Representation):
    def create_representation(self) -> List[List[Dict[str, Any]]]:
        """
        Creates a list representation of the conversation tree, where each entry is a list representing a thread.

        [
            [  # Thread 1
                {  # Document 1
                    "text": <text content>,
                    "extra_info": <extra_info>,
                    "relationships": {
                        "parent": <parent id>,
                        "children": [<child id>, ...],
                        "previous": <previous id>,
                        "next": <next id>,
                    },
                },
                ...  # More documents in thread 1
            ],
            ...  # More threads
        ]
        """

        threads = []

        # A helper function to construct a document dictionary
        def _create_document(node):
            text = node.message.content if node.message else ""
            parent_id = node.parent.id if node.parent else None
            children_ids = [child.id for child in node.children]
            previous_id = None  # TODO: Define the previous document logic
            next_id = None  # TODO: Define the next document logic

            document = {
                "text": text,
                "relationships": {
                    "parent": parent_id,
                    "children": children_ids,
                    "previous": previous_id,
                    "next": next_id,
                },
            }

            return document

        # A helper function to traverse the conversation tree and construct threads
        def _traverse_tree(node, current_thread):
            current_thread.append(_create_document(node))
            for child in node.children:
                _traverse_tree(child, current_thread)

        # Start the traversal
        for node in self.mapping.values():
            if node.parent is None:  # It's a root node, start a new thread
                current_thread = []
                _traverse_tree(node, current_thread)
                threads.append(current_thread)

        return threads


class ConversationAsDataFrame(Representation):
    def create_representation(self) -> pd.DataFrame:
        data = []
        for message in self.mapping.values():
            data.append(message.message.to_dict())

        return pd.DataFrame(data)


class FlatDictRepresentation(Representation):
    def create_representation(self) -> Dict[str, Any]:
        flat_dict = {}
        for mapping in self.mapping.values():
            flat_dict[mapping.id] = mapping.message
        return flat_dict


class NestedDictRepresentation(Representation):
    def create_representation(self) -> Dict[str, Any]:
        nested_dict = {}
        for node in self.mapping.values():
            if node.parent is None:
                nested_dict[node.id] = self._build_nested_dict(node.id)
        return nested_dict

    def _build_nested_dict(self, node_id: str) -> Dict[str, Any]:
        node = self.mapping[node_id]
        children = [self._build_nested_dict(child.id) for child in node.children]
        return {node.id: {"message": node.message, "children": children}}


class ChainRepresentation(Representation):

    def _conversation_representation(self) -> Dict[str, Chain]:
        """
        Create a dictionary representation of the conversation tree.

        Returns:
            Dict[str, Chain]: A dictionary mapping message IDs to Chain objects.
        """
        conversation_dict = {}
        for message_id, message in self.message_dict.items():
            children_ids = [child.id for child in message.children]
            conversation_dict[message_id] = Chain(
                id=message_id,
                content=message.content,
                children=children_ids,
                next=message.next,
                prev=message.prev,
            )
        return conversation_dict
