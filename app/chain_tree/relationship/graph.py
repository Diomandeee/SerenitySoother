from typing import Tuple, List, Union, Optional, Any, Dict, Deque
from chain_tree.engine.match import compute_stable_matching, train_model
from chain_tree.transformation.animate import animate_conversation_tree
from chain_tree.relationship.base import Representation
from chain_tree.services.preprocessing import build_dataframe
from chain_tree.transformation.tree import CoordinateTree
from chain_tree.relationship.estimator import Estimator
from chain_tree.services.metrics import relation_metrics
from sklearn.metrics.pairwise import cosine_similarity
from chain_tree.engine.embedder import OpenAIEmbedding
from chain_tree.services.context import process_data
from chain_tree.models import Chain, ChainTree
from chain_tree.type import NodeRelationship
from chain_tree.utils import log_handler
import pandas as pd
import numpy as np
import collections
import logging
import torch
import json
import os


def expand_coordinate_column(main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 'coordinate' column into separate columns in the main DataFrame.
    If 'coordinate' column doesn't exist, create it with a default value.

    Parameters:
    - main_df (pd.DataFrame): The main DataFrame.

    Returns:
    - pd.DataFrame: Updated DataFrame with expanded columns.
    """
    coord_names = CoordinateTree.get_coordinate_names()
    if "coordinate" not in main_df.columns:
        main_df[coord_names] = 0
        for index, name in enumerate(coord_names):
            if name == "depth_x":
                main_df["depth_x"] = main_df.index + 1
        return main_df
    else:
        main_df[coord_names] = main_df["coordinate"].apply(pd.Series)
        return main_df


TetraCoordinate = Tuple[float, float, float, float, int]
TetraDict = Dict[str, TetraCoordinate]
StackItem = Tuple[str, TetraCoordinate, int]
Stack = List[StackItem]
RelationshipDict = Dict[str, Dict[str, Union[str, List[str]]]]
RelationshipList = List[RelationshipDict]


class ChainDocument(Representation):
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
        spatial_similarity: Optional[OpenAIEmbedding] = None,
        message_dict: Optional[Dict[str, Any]] = None,
        conversation_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(conversation_tree)
        self.message_dict = (
            self._message_representation() if message_dict is None else message_dict
        )
        self.conversation_dict = (
            self._conversation_representation()
            if conversation_dict is None
            else conversation_dict
        )
        self.spatial_similarity = (
            spatial_similarity
            if spatial_similarity
            else OpenAIEmbedding(api_key=api_key)
        )

        self.estimator = Estimator(self.message_dict, self.conversation_dict)

    def _generate_embeddings(
        self,
    ) -> Tuple[List[List[float]], List[str], List[List[float]]]:
        """
        Generate embeddings for messages.

        Returns:
        - Tuple: A tuple containing lists of embeddings, message IDs, and message embeddings.
        """
        return self.spatial_similarity.generate_embeddings(self.message_dict)

    def _calculate_and_validate_similarity(
        self, parent_id: str, child_id: str
    ) -> Optional[float]:
        """
        Calculates and validates the similarity score between a parent and child node.

        Args:
            parent_id (str): The ID of the parent node.
            child_id (str): The ID of the child node.

        Returns:
            float: The similarity score if it's within a valid range (0 to 1).
            None: If the score is outside the valid range or there's an error in calculation.
        """
        try:
            similarity_score = self.calculate_similarity_score(parent_id, child_id)

            if 0 <= similarity_score <= 1:
                return similarity_score
            else:
                log_handler(
                    f"Unexpected similarity score {similarity_score} for child ID {child_id}. Skipping."
                )
                return None
        except Exception as e:
            log_handler(
                f"Error calculating similarity for child ID {child_id}: {str(e)}. Skipping."
            )
            return None

    def _sort_by_dissimilarity(
        self, current_node_id: str, child_ids: List[str]
    ) -> List[str]:
        """
        Sorts child node IDs based on their dissimilarity to the parent node ID.

        For each child node ID, this method calculates a dissimilarity score relative to the parent node ID.
        The dissimilarity is computed as 1 minus the similarity score. The child node IDs are then sorted in
        descending order based on their dissimilarity scores.

        Args:
            current_node_id (str): The ID of the current node (parent node).
            child_ids (List[str]): A list of child node IDs to be sorted based on their dissimilarity to the parent node.

        Returns:
            List[str]: A list of child node IDs sorted in descending order of dissimilarity to the parent node.
                    IDs with failed or invalid similarity scores are excluded from the returned list.
        """
        dissimilarity_scores = [
            (child_id, 1 - similarity)
            for child_id in child_ids
            if (
                similarity := self._calculate_and_validate_similarity(
                    current_node_id, child_id
                )
            )
            is not None
        ]

        return [
            x[0] for x in sorted(dissimilarity_scores, key=lambda x: x[1], reverse=True)
        ]

    def _sort_by_similarity(
        self, current_node_id: str, child_ids: List[str]
    ) -> List[str]:
        """
        Sorts child node IDs based on their similarity to the parent node ID.

        For each child node ID, this method calculates a similarity score relative to the parent node ID.
        The child node IDs are then sorted in descending order based on their similarity scores.

        Args:
            current_node_id (str): The ID of the current node (parent node).
            child_ids (List[str]): A list of child node IDs to be sorted based on their similarity to the parent node.

        Returns:
            List[str]: A list of child node IDs sorted in descending order of similarity to the parent node.
                    IDs with failed or invalid similarity scores are excluded from the returned list.
        """
        similarity_scores = [
            (child_id, similarity)
            for child_id in child_ids
            if (
                similarity := self._calculate_and_validate_similarity(
                    current_node_id, child_id
                )
            )
            is not None
        ]

        return [
            x[0] for x in sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        ]

    def _sort_by_importance(
        self, current_node_id: str, child_ids: List[str]
    ) -> List[str]:
        """
        Sorts child node IDs based on their importance to the parent node ID.

        For each child node ID, this method calculates a importance score relative to the parent node ID.
        The child node IDs are then sorted in descending order based on their importance scores.

        Args:
            current_node_id (str): The ID of the current node (parent node).
            child_ids (List[str]): A list of child node IDs to be sorted based on their importance to the parent node.

        Returns:
            List[str]: A list of child node IDs sorted in descending order of importance to the parent node.
                    IDs with failed or invalid importance scores are excluded from the returned list.
        """
        importance_scores = [
            (child_id, importance)
            for child_id in child_ids
            if (
                importance := self._calculate_and_validate_importance(
                    current_node_id, child_id
                )
            )
            is not None
        ]

        return [
            x[0] for x in sorted(importance_scores, key=lambda x: x[1], reverse=True)
        ]

    def _calculate_and_validate_importance(
        self, parent_id: str, child_id: str
    ) -> Optional[float]:
        """
        Calculates and validates the importance score between a parent and child node.

        Args:
            parent_id (str): The ID of the parent node.
            child_id (str): The ID of the child node.

        Returns:
            float: The importance score if it's within a valid range (0 to 1).
            None: If the score is outside the valid range or there's an error in calculation.
        """
        try:
            importance_score = self.estimator.calculate_importance_score(
                parent_id, child_id
            )

            if 0 <= importance_score <= 1:
                return importance_score
            else:
                log_handler(
                    f"Unexpected importance score {importance_score} for child ID {child_id}. Skipping."
                )
                return None
        except Exception as e:
            log_handler(
                f"Error calculating importance for child ID {child_id}: {str(e)}. Skipping."
            )
            return None

    def prepare_base_document(
        self,
        message_id: str,
        embedding_data: Any,
        coordinate: CoordinateTree,
    ):
        """
        Prepare the base attributes for the ChainDocument.

        Args:
            message_id: The id of the message.
            embedding_data: The embedding data of the message.
            coordinate: The coordinate of the message.
            relationship: The relationships of the message.

        Returns:
            A dictionary containing the base attributes for a ChainDocument.
        """

        return {
            "id": message_id,
            "text": self.get_message_content(message_id),
            "author": self.get_message_author_role(message_id),
            "coordinate": list(coordinate),
            "embedding": embedding_data,
            "create_time": self.get_message_create_time(message_id),
            "title": self.conversation.title,
            "conversation_id": self.conversation.id,
        }

    def _generate_local_message_embeddings(
        self,
        embeddings: List[List[float]],
        message_ids: List[str],
        message_embeddings: List[List[float]],
    ) -> List[List[float]]:
        """
        Generate local embeddings for messages.

        Parameters:
        - embeddings: List of embeddings.
        - message_ids: List of message IDs.
        - message_embeddings: List of message embeddings.

        Returns:
        - List[List[float]]: A list of local message embeddings.
        """
        return self.spatial_similarity.generate_message_embeddings(
            self.estimator,
            self.message_dict,
            embeddings,
            message_ids,
            message_embeddings,
        )

    def get_chain_document_parameters(
        self,
        message_id: str,
        embedding_data: Any,
        coordinate: CoordinateTree,
    ):
        """
        Extract the parameters for a ChainDocument from the given inputs.

        Args:
            message_id: The id of the message.
            embedding_data: The embedding data of the message.
            coordinate: The coordinate of the message.

        Returns:
            A dictionary containing the parameters for a ChainDocument.
        """
        base_document = self.prepare_base_document(
            message_id, embedding_data, coordinate
        )

        return {
            **base_document,
        }

    def create_chain_document(
        self,
        message_id: str,
        embedding_data: Any,
        tetra_dict: TetraDict,
    ) -> Chain:
        """
        Create a ChainDocument for a given message id.

        Args:
            message_id: The id of the message.
            embedding_data: The embedding data of the message.
            tetra_dict: The dictionary containing the coordinates.
            relationship: The relationships of the message.
            subgraph: The subgraph of the graph.

        Returns:
            A ChainDocument object for the given message id.
        """
        if message_id not in tetra_dict:
            # skip if message_id is not in tetra_dict
            return None
        chain_document_parameters = self.get_chain_document_parameters(
            message_id, embedding_data, tetra_dict[message_id]
        )

        return chain_document_parameters


class CoordinateRepresentation(ChainDocument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_coordinate_tree_node(
        self, node_id: str, tetra_dict: "TetraDict"
    ) -> "CoordinateTree":
        """
        Constructs a new CoordinateTree node based on coordinates and additional information.

        This function constructs and returns a new instance of the CoordinateTree class for a given node ID.
        The coordinates for the node are sourced from the `tetra_dict` argument. Additional attributes
        such as creation time, content text, and embeddings are retrieved via respective helper methods of the class.

        Args:
            node_id (str):
                The unique identifier of the node. It is used to index into various data sources to retrieve node-specific attributes.

            tetra_dict (TetraDict):
                A dictionary containing coordinates for nodes. Each key is a node ID, and the corresponding value is
                a tuple containing the (x, y, z, t, n_parts) coordinates for that node.

        Returns:
            CoordinateTree:
                The constructed CoordinateTree node with the provided attributes and the additional attributes fetched using helper methods.

        Raises:
            This function may raise exceptions if the provided node ID doesn't exist in `tetra_dict` or if there are issues in fetching
            additional attributes for the node via helper methods.

        Example:
            Given:
            tetra_dict = {
                "node_1": (0.5, 0.7, 0.2, 0.9, 3),
                ...
            }

            Calling `create_coordinate_tree_node("node_1", tetra_dict)` will return a CoordinateTree node with
            coordinates (0.5, 0.7, 0.2, 0.9), n_parts as 3, and additional attributes like creation time, content, and embeddings
            as fetched from the respective helper methods.
        """

        # Extract coordinates from the dictionary
        x, y, z, t, n_parts, *_ = tetra_dict[node_id]

        # Construct and return the CoordinateTree node
        return CoordinateTree(
            id=node_id,
            x=x,
            y=y,
            z=z,
            t=t,
            parent=None,
            n_parts=n_parts,
            create_time=self.get_message_create_time(node_id),
            text=self.get_message_content(node_id),
            author=self.get_message_author_role(node_id),
            embeddings=(
                self.get_message_embeddings(node_id)
                if self.get_message_embeddings(node_id) is not None
                else None
            ),
        )

    def construct_tree(
        self,
        root_id: str,
        tetra_dict: "TetraDict",
        combined_tensor: Optional[Any] = None,
    ) -> Dict[str, "CoordinateTree"]:
        """
        Build and return a dictionary mapping node IDs to CoordinateTree nodes
        from given tetrahedral coordinates and relationships.

        Args:
            root_id (str): The ID of the root node.
            tetra_dict (TetraDict): A dictionary containing tetrahedral coordinates for nodes.
            combined_tensor (Optional[Any]): Additional data, not used in this example.

        Returns:
            Dict[str, "CoordinateTree"]: A dictionary where each key is a node ID and each value is the corresponding CoordinateTree node.
        """
        # Initialize the root node
        root = self.create_coordinate_tree_node(root_id, tetra_dict)

        # Create a dictionary to store the CoordinateTree nodes
        tree_dict = {root_id: root}

        # Initialize a queue to keep track of nodes
        queue = collections.deque([root])

        while queue:
            current_node = queue.popleft()

            # Assuming get_children_ids is a method that retrieves child node IDs for a given node ID
            child_ids = self.get_children_ids(current_node.id)

            for child_id in child_ids:
                child_node = self.create_coordinate_tree_node(child_id, tetra_dict)
                child_node.parent = (
                    current_node.id if current_node is not None else None
                )
                current_node.children.append(child_node)

                # Add the child node to the dictionary
                tree_dict[child_id] = child_node

                # Add the child node to the queue
                queue.append(child_node)

        return tree_dict

    def _construct_representation(
        self,
        message_embeddings: Dict[str, Any],
        tetra_dict: "TetraDict",
    ) -> List["Chain"]:
        """
        Constructs a representation of the messages as a list of Chain objects.

        This method generates a representation of a conversation as a list of Chain
        objects by utilizing the provided embeddings, coordinates (from
        tetra_dict), and relationship data (from relationships). Each Chain object
        encapsulates the essence of a single message in the conversation, capturing
        both its content (through embeddings) and its context (through
        coordinates and relationship data).

        Args:
            message_embeddings (Dict[str, Any]):
                A dictionary mapping message IDs to their corresponding embeddings.
                Each embedding captures the semantic essence of a message, potentially
                in a dense vector format.

            tetra_dict (TetraDict):
                A dictionary capturing coordinates for each message.
                These coordinates provide a spatial representation of messages,
                allowing one to understand the structure and flow of the conversation.

            relationships (RelationshipDict):
                A dictionary mapping message IDs to their relationships with other
                messages. Relationships can be parent-child, peer, or any other
                custom-defined type.

        Returns:
            List[Chain]:
                A list of Chain objects representing the conversation. Each Chain
                object corresponds to a single message in the conversation, capturing
                its embedding, coordinates, and relationships.

        """

        chain_documents = []

        for message_id, embedding_data in message_embeddings.items():
            chain_document = self.create_chain_document(
                message_id, embedding_data, tetra_dict
            )
            chain_documents.append(chain_document)

        return chain_documents


class GraphRepresentation(CoordinateRepresentation):
    def __init__(
        self,
        train_model=False,
        stable_match=False,
        create=False,
        filter_words: Optional[List[str]] = [
            "I'm sorry",
            "as a language model",
            "Sorry",
            "I apologize",
            "Unfortunately",
        ],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.train_model = train_model
        self.stable_match = stable_match
        self.filter_words = filter_words
        self.create = create
        self.state_map = {}  # Maps message IDs to their states

    def extract_coordinates(
        self, coordinate: Union[CoordinateTree, List[Union[int, float]]]
    ) -> Optional[List[Union[int, float]]]:
        """
        Extracts spatial coordinates from a Coordinate object or a list of coordinates.

        Parameters:
        - coordinate (Union[Coordinate, List[Union[int, float]]]): A Coordinate object or a list of coordinates.

        Returns:
        - Optional[List[Union[int, float]]]: A list of spatial coordinates (x, y, z, t, n_parts) if the provided
        object is valid. If the object is None or if any of the attributes are missing, the function
        returns None.
        """
        if coordinate is None:
            return [0, 0, 0, 0, 0]

        if isinstance(coordinate, CoordinateTree):
            try:
                return [
                    coordinate.x,
                    coordinate.y,
                    coordinate.z,
                    coordinate.t,
                    coordinate.n_parts,
                ]
            except AttributeError as e:
                print(f"AttributeError encountered: {e}")
        elif isinstance(coordinate, list) and len(coordinate) == 5:
            return coordinate

        return [0, 0, 0, 0, 0]

    def collect_messages(
        self,
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[List[Union[int, float]]],
        List[List[Union[int, float]]],
        List[List[float]],
        List[List[float]],
    ]:
        """
        Gathers detailed data associated with valid messages within the system.

        Messaging systems often consist of a vast array of messages exchanged between participants. This function
        delves into this dataset, sifting through individual messages, and meticulously collecting relevant data
        linked with each valid message, along with their subsequent responses.

        Returns:
        - Tuple: A comprehensive tuple capturing a multitude of details associated with each message. The tuple

        Process:
        The function commences by initializing empty lists for each of the aforementioned categories. It then iterates
        over all messages in the system's message dictionary. For every message that is deemed valid through the
        'is_message_valid' check, it identifies the message's author and fetches its immediate children or responses.

        It further filters these children based on the validity of the message pair (parent message and child message)
        using the 'is_message_pair_valid' check. Once a valid child message is identified, the function employs
        'append_message_details' to populate all initialized lists with appropriate details.

        """
        prompts, responses, prompt_ids, response_ids = [], [], [], []
        created_times, prompt_coordinates, response_coordinates = [], [], []
        prompt_encodings, response_encodings = [], []

        for message_id, _ in self.message_dict.items():
            if self.is_message_valid(message_id):
                author = self.get_message_author_role(message_id)
                children = self.conversation_dict.get(message_id, [None, []])[1]

                for child_id in children:
                    if self.is_message_pair_valid(message_id, child_id, author):
                        self.append_message_details(
                            message_id,
                            child_id,
                            prompts,
                            responses,
                            prompt_ids,
                            response_ids,
                            created_times,
                            prompt_coordinates,
                            response_coordinates,
                            prompt_encodings,
                            response_encodings,
                        )

        return (
            prompts,
            responses,
            prompt_ids,
            response_ids,
            created_times,
            prompt_coordinates,
            response_coordinates,
            prompt_encodings,
            response_encodings,
        )

    def is_message_valid(self, message_id: str) -> bool:
        """
        Validates a message based on its content, creation time, spatial coordinates, and author role.


        """
        return all(
            [
                self.get_message_content(message_id) is not None,
                self.get_message_create_time(message_id) is not None,
                self.get_message_coordinate(message_id) is not None,
                self.get_message_author_role(message_id) is not None,
            ]
        )

    def is_message_pair_valid(
        self, parent_id: str, child_id: str, parent_author: str
    ) -> bool:
        """
        Validates the legitimacy of a parent-child message pair within a conversation.

        """
        child_author = self.get_message_author_role(child_id)
        return (
            child_author is not None
            and parent_author == "user"
            and child_author == "assistant"
        )

    def append_message_details(
        self,
        parent_id: str,
        child_id: str,
        prompts: List[str],
        responses: List[str],
        prompt_ids: List[str],
        response_ids: List[str],
        created_times: List[str],
        prompt_coordinates: List[Optional[List[Union[int, float]]]],
        response_coordinates: List[Optional[List[Union[int, float]]]],
        prompt_encodings: List[List[float]],
        response_encodings: List[List[float]],
    ) -> None:
        """
        Enriches provided lists with details extracted from a valid message pair.


        Returns:
        - None: This function operates in-place, enhancing the provided lists with new data. It doesn't have a return value.

        """

        # Append message contents, ids, and created_times
        prompts.append(self.get_message_content(parent_id))
        responses.append(self.get_message_content(child_id))
        prompt_ids.append(parent_id)
        response_ids.append(child_id)
        created_times.append(self.get_message_create_time(parent_id))

        # Append coordinates if they exist, otherwise skip
        prompt_coord = self.extract_coordinates(self.get_message_coordinate(parent_id))
        if prompt_coord is not None:
            prompt_coordinates.append(prompt_coord)

        response_coord = self.extract_coordinates(self.get_message_coordinate(child_id))
        if response_coord is not None:
            response_coordinates.append(response_coord)

        # Append encodings
        prompt_encodings.append(
            list(map(float, self.message_dict[parent_id].message.embedding))
        )
        response_encodings.append(
            list(map(float, self.message_dict[child_id].message.embedding))
        )

    def embed_messages(self, collected_messages, embed_text_batch):
        """
        Transforms raw text data of collected conversation messages into vector representations.

        Process:
        1. **Data Extraction**:
        Retrieves prompts and responses from the `collected_messages` tuple.

        2. **Embedding Generation**:
        Uses the `embed_text_batch` method to create embeddings for both the prompts and the responses. The method
        may leverage deep learning models or other mechanisms to transform the text into high-dimensional vectors.

        Returns:
        - tuple: A tuple that consists of the vector representations (embeddings) of prompts and responses.
        """

        # Extract prompts and responses from the collected message tuple
        prompts, responses = collected_messages[:2]

        # Generate embeddings for the extracted text data
        return embed_text_batch(prompts, responses)

    def get_similarity_matrix(
        self,
        exclude_keys: Optional[List[Any]] = None,
        csv_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Computes a pairwise cosine similarity matrix for the embedded representations of messages.

        In the realm of natural language processing and information retrieval, cosine similarity is a widely accepted metric
        to compare the similarity between two vectors. In this function, it serves as a quantifier for the likeness between
        different messages based on their embedded representations.

        Process:
        1. **Filtering**:
        From the entire collection of message IDs, remove any IDs listed in `exclude_keys`.

        2. **Embedding Extraction**:
        Retrieve the embedding for each message. Only consider messages with valid embeddings.

        3. **Cosine Similarity Computation**:
        Calculate pairwise cosine similarities between the embeddings of all valid messages, resulting in a matrix.

        4. **DataFrame Creation**:
        The similarity matrix is encapsulated within a pandas DataFrame with both rows and columns labeled by valid
        message IDs.

        5. **CSV Export** (if applicable):
        If a filename is provided in `csv_filename`, the similarity matrix is saved as a CSV file, facilitating
        further analysis or visualization in external tools.

        Returns:
        - pd.DataFrame: A structured table where each cell (i, j) represents the similarity between messages i and j.

        """

        # Derive a filtered list of message_ids by excluding any present in the exclude_keys list
        message_ids = [
            key for key in self.message_dict.keys() if key not in (exclude_keys or [])
        ]

        embeddings = []
        valid_message_ids = []  # Store message_ids that have valid embeddings

        for message_id in message_ids:
            embedding = self.get_message_embeddings(message_id)

            # Ensure that the retrieved embedding is valid (not None)
            if embedding is not None:
                valid_message_ids.append(message_id)
                embeddings.append(np.array(embedding))

        # Convert the list of embeddings into a NumPy array for efficient computation
        embeddings_array = np.array(embeddings)

        # Use the cosine_similarity function to generate a matrix of pairwise similarities
        similarity_matrix = cosine_similarity(embeddings_array)

        # Convert the similarity matrix into a pandas DataFrame for structured representation
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=valid_message_ids,
            columns=valid_message_ids,
        )

        # If a filename is specified, export the DataFrame to CSV format
        if csv_filename:
            similarity_df.to_csv(csv_filename)

        return similarity_df

    def _calculate_and_assess_distances(
        self,
        coordinates_list: List[CoordinateTree],
        tetra_dict: Dict[str, CoordinateTree],
    ) -> pd.DataFrame:
        """
        Calculate and assess distances based on coordinates.

        Parameters:
        - coordinates_list: List of Coordinate objects.
        - tetra_dict: Dictionary mapping IDs to Coordinates.

        Returns:
        - pd.DataFrame: A DataFrame containing calculated distances.
        """
        return CoordinateTree.calculate_and_assess(
            coordinates=coordinates_list, coordinate_ids=tetra_dict
        )

    def get_distances_and_similarity(
        self, coordinates_list: List[List[Union[int, float]]], tetra_dict: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves and returns distance and similarity matrices as DataFrames.

        Two types of matrices are fetched and processed:
        1. The distance matrix derived from the spatial (or coordinate) information of the messages.
        2. The similarity matrix computed from the embeddings of the messages.

        Process:
        1. **Matrix Retrieval**:
        Extracts the distance and similarity matrices for the messages.

        2. **Sorting**:
        To guarantee a consistent structure, both matrices are sorted by their indices and columns.

        Returns:
        - pd.DataFrame: The distance matrix as a DataFrame.
        - pd.DataFrame: The similarity matrix as a DataFrame.
        """

        # Retrieve the distance and similarity matrices (DataFrames)
        distances_df = self._calculate_and_assess_distances(
            coordinates_list, tetra_dict
        )

        return distances_df

    def create_combined_tensor(
        self, distances_df: pd.DataFrame, similarity_df: pd.DataFrame
    ) -> Union[torch.Tensor, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Combines distance and similarity matrices into a single tensor or returns DataFrames.

        Args:
        - distances_df (pd.DataFrame): The distance matrix as a DataFrame.
        - similarity_df (pd.DataFrame): The similarity matrix as a DataFrame.
        - return_dataframes (bool, optional): If True, return DataFrames. If False, return the combined tensor.

        Returns:
        - torch.Tensor: A combined tensor where the first half of the columns represent distances between messages and
        the second half represents their similarities (if `return_dataframes` is False).
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing distances_df and similarity_df (if `return_dataframes` is True).
        """

        # Convert the sorted DataFrames to PyTorch tensors
        distance_tensor = torch.tensor(distances_df.to_numpy(), dtype=torch.float32)
        similarity_tensor = torch.tensor(similarity_df.to_numpy(), dtype=torch.float32)

        combined_tensor = torch.cat((distance_tensor, similarity_tensor), dim=1)
        return combined_tensor

    def create_prompt_response_df(
        self,
        pre_computed_embeddings: bool = True,
        embed_text_batch: callable = None,
    ) -> pd.DataFrame:
        """
        Assembles a structured DataFrame encapsulating the interactions between conversation prompts and their respective responses.

        Conversations, especially in digital platforms, have a clear structure where one entity (a user or a system) provides
        a prompt, and another responds. This function curates these interactions into a DataFrame. Beyond mere textual data,
        the DataFrame includes vector representations (embeddings) of the messages and potentially other metrics that
        offer insights into the relationship dynamics of the conversation.

        Parameters:
        - pre_computed_embeddings (bool): Specifies whether the function should leverage already available embeddings
                                        for messages. If set to False, embeddings are generated on-the-fly. Default is True.

        Process:
        1. **Message Collection**:
        Leverages the `collect_messages` method to harvest conversation pairs, including their content,
        spatial attributes, and potentially pre-existing embeddings.

        2. **Embedding Generation (Conditional)**:
        If `pre_computed_embeddings` is set to False, the function computes embeddings for the collected messages.


        3. **DataFrame Construction**:
        Uses the `build_dataframe` method to systematically assemble a DataFrame from the collected/generate data.

        4. **Relationship Metrics Computation**:
        Enriches the DataFrame with additional metrics and features that provide deeper insights into the relationship
        between prompts and responses.

        Returns:
        - pd.DataFrame: A structured table that captures and represents the dynamics of conversation pairs, inclusive
                        of embeddings and relationship metrics.
        """
        try:
            # Collect message details and, if required, generate embeddings
            collected_messages = self.collect_messages()
        except Exception as e:
            logging.error(f"Error in collecting messages: {e}")
            raise

        try:
            if not pre_computed_embeddings:
                collected_messages = self.embed_messages(
                    collected_messages, embed_text_batch
                )
        except Exception as e:
            logging.error(f"Error in embedding messages: {e}")
            raise

        try:
            # Build and enrich the DataFrame with metrics
            relationship_df = build_dataframe(
                collected_messages,
                self.conversation.id,
                filter_words=self.filter_words,
            )
        except Exception as e:
            logging.error(f"Error in DataFrame construction: {e}")
            raise

        try:
            if self.stable_match:
                node_response_df = compute_stable_matching(relationship_df)
            else:
                node_response_df = relationship_df
        except Exception as e:
            logging.error(f"Error in computing stable matching: {e}")
            node_response_df = relationship_df  # Fallback to original DataFrame

        try:
            main_df = relation_metrics(node_response_df)
        except Exception as e:
            logging.error(f"Error in computing relationship metrics: {e}")
            raise

        if self.train_model:
            try:
                model = train_model(main_df)
            except Exception as e:
                logging.error(f"Error in training model: {e}")
                model = None  # Fallback if model training fails

            return main_df, model

        return main_df

    def update_children_structure(self, parent_id, sorted_children):
        """
        Update the structure of children based on the sorted children IDs.
        """
        self.message_dict[parent_id].children = sorted_children

    def rebalance_children(self, parent_id, children_ids):
        """
        Rebalance the children of a parent node based on the provided parent ID and children IDs.
        """
        sorted_children = self._sort_by_similarity(parent_id, children_ids)
        self.update_children_structure(parent_id, sorted_children)

    def dynamic_steepness_adjustment(self, num_children: int) -> float:
        """Adjust the steepness of the penalty curve based on the current state of the conversation tree."""
        average_children = np.mean(
            [len(node.children) for node in self.message_dict.values()]
        )
        if num_children > average_children:
            return 0.7  # More steepness to avoid too much branching if already high
        return 0.3  # Less steepness to encourage more branching if currently low

    def optimal_child_count_penalty(self, num_children, optimal=1):
        """Calculates a dynamic penalty peaking at `optimal` number of children."""
        # Dynamically adjust steepness based on the context or performance metrics
        steepness = self.dynamic_steepness_adjustment(num_children)
        return np.exp(-steepness * (num_children - optimal) ** 2)

    def calculate_structure_loss(self, children_counts):
        """Calculate structure-based loss for a batch of nodes using the optimal_child_count_penalty."""
        loss = 0
        for count in children_counts:
            loss += self.optimal_child_count_penalty(count)
        return loss / len(children_counts) if children_counts else 0

    def rebalance_tree(self, queue):
        """
        Rebalance the tree structure based on the provided queue of message IDs.
        """
        for message_id in queue:
            children_ids = self.get_children_ids(message_id)
            children_counts = [
                len(self.get_children_ids(child_id)) for child_id in children_ids
            ]
            structural_loss = self.calculate_structure_loss(children_counts)
            if structural_loss > 0.5:
                self.rebalance_children(message_id, children_ids)

    def update_tree_structure_based_on_loss(self, queue):
        """
        Update the tree structure by evaluating potential losses due to current configurations.
        """
        children_counts = [
            len(self.get_children_ids(message_id)) for message_id in queue
        ]
        structural_loss = self.calculate_structure_loss(children_counts)
        if structural_loss > 0.5:
            self.rebalance_tree(queue)

        print(f"Structural Loss: {structural_loss}")

    def get_all_level_queues(self):
        """
        Get all level queues in the conversation tree.
        """
        level_queues = []
        queue = collections.deque([self.root_message_id])
        while queue:
            level_queues.append(queue)
            children = []
            for message_id in queue:
                children.extend(self.get_children_ids(message_id))
            queue = collections.deque(children)

        return level_queues

    def evaluate_tree_structure(self):
        """
        Evaluate the overall structure of the tree after its construction.
        """
        level_queues = (
            self.get_all_level_queues()
        )  # Assuming a method to collect all queues
        for queue in level_queues:
            self.update_tree_structure_based_on_loss(queue)

    def adjust_tree_real_time(self, new_message_id):
        """
        Adjust the conversation tree in real time as new messages are added.
        """
        parent_id = self.get_parent_id(new_message_id)
        siblings = self._get_message_siblings(new_message_id)
        if parent_id:
            self.adjust_tree_real_time(parent_id)

            self.rebalance_children(parent_id, siblings)

    def process_queue_level(
        self,
        queue: collections.deque,
        depth: int,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Tuple[int, int, int, int, int]], collections.deque]:
        """
        Process a level of the message tree queue and calculate coordinates for child nodes.

        Args:
            queue (collections.deque): A collections.deque containing message IDs to be processed at the current level.
            depth (int): The depth level of the current processing level.

        Returns:
            Tuple[Dict[str, Tuple[int, int, int, int, int]], collections.deque]: A tuple containing:
                - children_coords (Dict[str, Tuple[int, int, int, int, int]]): A dictionary
                mapping child message IDs to their calculated coordinates.
                - level_queue (collections.deque): A collections.deque containing message IDs of child nodes for the
                next level of processing.
        """
        try:
            level_queue = collections.deque()
            children_coords = {}
            for message_id in list(queue):
                children_ids = list(self.get_children_ids(message_id))
                coords = {
                    child_id: self._assign_coordinates(
                        child_id=child_id,
                        depth=depth,
                        i=i,
                        children_ids=children_ids,
                        message_id=message_id,
                        **kwargs,
                    )
                    for i, child_id in enumerate(children_ids)
                    if self.message_dict[child_id]
                }
                children_coords.update(coords)
                level_queue.extend(coords.keys())
            return children_coords, level_queue
        except Exception as e:
            print(f"Error in _process_queue_level: {e}")
            return {}, collections.deque()

    def create_coordinates(
        self,
        **kwargs: Dict[str, Union[str, float]],
    ) -> Tuple[
        Dict[str, CoordinateTree],
        Dict[str, Dict[str, float]],
        str,
        List[List[Union[int, float]]],
    ]:
        """
        Creates a coordinate representation of the conversation chain_tree.tree.

        Args:
            alpha_final_z (float): The final z-coordinate value for the root node.
            alpha_scale (float): The scaling factor for the alpha value.
            method (MethodType, optional): The method to use for assigning coordinates.

        Returns:
            Tuple[Dict[str, CoordinateTree], Dict[str, Dict[str, float]], str]: A tuple containing:
            - tetra_dict (Dict[str, CoordinateTree]): A dictionary mapping message IDs to their
                corresponding Coordinate objects.
            - relationships (Dict[str, Dict[str, float]]): A dictionary mapping message IDs to a
                dictionary of child message IDs and their corresponding relationship strengths.
            - root_id (str): The ID of the root message.

        """
        try:
            if not self.conversation_dict:
                return {}, {}, {}, []

            (
                relationships,
                tetra_dict,
                root_id,
                root_coordinate,
            ) = self.initialize_representation()

            queue = collections.deque([root_id])
            depth = 1
            level_queues = []

            while queue:
                children_coords, level_queue = self.process_queue_level(
                    queue,
                    depth,
                    **kwargs,
                )
                tetra_dict.update(children_coords)
                queue = level_queue
                level_queues.insert(0, level_queue)
                depth += 1

            for (message_id_head,), (message_id_tail,) in zip(
                level_queues[0], reversed(level_queues[-1])
            ):
                for message_id, children_ids in [
                    (message_id_head, list(self.get_children_ids(message_id_head))),
                    (message_id_tail, list(self.get_children_ids(message_id_tail))),
                ]:
                    for i, child_id in enumerate(children_ids):
                        if self.message_dict[child_id]:
                            tetra_dict[child_id] = self._assign_coordinates(
                                child_id=child_id,
                                depth=depth,
                                i=i,
                                children_ids=children_ids,
                                message_id=message_id,
                                **kwargs,
                            )

            stack = [(root_id, root_coordinate, 1)]
            parent_coordinates = []

            while stack:
                message_id, parent_coords, depth = stack.pop()
                children_ids = self.get_children_ids(message_id)
                for i, child_id in enumerate(children_ids):
                    relationships = self._assign_relationships(
                        message_id, child_id, children_ids, i, relationships
                    )
                    child_coords = tetra_dict[child_id]
                    stack.append((child_id, child_coords, depth + 1))
                    parent_coordinates.append(parent_coords)

            return tetra_dict, relationships, root_id, parent_coordinates

        except Exception as e:
            print(f"Error in create_coordinates: {e}")
            raise Exception(f"Error in create_coordinates: {e}")

    def _construct_relationships(
        self, relationships: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Constructs a DataFrame representation of relationships between messages.

        Args:
            relationships (Dict[str, Dict[str, float]]): A dictionary mapping message IDs to a dictionary of child
                message IDs and their corresponding relationship strengths.

        Returns:
            pd.DataFrame: A DataFrame representation of relationships between messages.
        """
        relationship_df = pd.DataFrame.from_dict(relationships, orient="index")
        relationship_df.fillna(0, inplace=True)
        return relationship_df

    def _update_and_create_main_df(
        self, tree_docs: Optional[List[Dict]]
    ) -> pd.DataFrame:
        """
        A sequential procedure that first integrates additional document features into the coordinates' mapping.
        Subsequent to this, it constructs a primary DataFrame based on the provided document features.

        Args:
            tree_docs (Optional[List[Dict]]): A list of dictionaries, where each dictionary embodies the features of a specific document or message node in the conversation chain_tree.tree. The list can be None.

        Returns:
            pd.DataFrame: A DataFrame that displays the features for each document or message node, extracted from 'tree_docs'. If there's a problem or if 'tree_docs' is None, a None object is returned.

        """
        # Check if 'tree_docs' is valid. If it's not, log a warning and return None.
        if tree_docs is None:
            log_handler(
                "tree_docs is None, skipping update.",
                level="warning",
                verbose=True,
            )
            return None

        # Integrate the new features into the coordinates' mapping and subsequently create the main DataFrame.
        try:
            main_df = pd.DataFrame(tree_docs)
            return main_df
        except TypeError as e:
            log_handler(
                f"Caught error: {e}. Skipping update.",
                level="warning",
                verbose=True,
            )
            return None

    def save_coordinates(
        self,
        coordinates_list: List[List[Union[int, float]]],
        main_df: pd.DataFrame,
        relationship_df: pd.DataFrame,
        message_dict: Dict[str, Any],
        base_path: str,
        title: str,
    ):
        """
        Save the coordinates list to a JSON file.

        Args:
            coordinates_list (List[List[Union[int, float]]]): A list of lists containing the coordinates of each message.
            base_path (str): The base path where the JSON file will be saved.
            title (str): The title of the conversation.
        """

        try:
            # create the base path if it doesn't exist
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            # create the title path if it doesn't exist
            if not os.path.exists(f"{base_path}/{title}"):
                os.makedirs(f"{base_path}/{title}")

            # save the dataframe to csv
            main_df.to_csv(f"{base_path}/{title}/main_df.csv", index=False)
            relationship_df.to_csv(
                f"{base_path}/{title}/relationship_df.csv", index=False
            )

            # save the message_dict to json
            with open(f"{base_path}/{title}/message_dict.json", "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in message_dict.items()}, f, indent=4
                )

            # save a numpy array to a file
            np.save(
                f"{base_path}/{title}/coordinates_list.npy", np.array(coordinates_list)
            )

            base_path = f"{base_path}/{title}"
            process_data(relationship_df, base_path)

        except Exception as e:
            print(f"Error in save_coordinates: {e}")

    def process_coordinates_graph(
        self,
        animate: bool = True,
        pre_computed_embeddings: bool = True,
        local_embeddings: bool = True,
        base_path: Optional[str] = "Mega",
        **kwargs: Dict[str, Union[str, float]],
    ) -> Tuple[
        Any,
        Union[np.ndarray, None],
        CoordinateTree,
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        pd.DataFrame,
        Any,
        Any,
    ]:
        """
        Processes coordinates to generate the appropriate coordinate representation for the given data.

        Args:
            animate (bool): A flag indicating whether to animate the conversation tree.
            method (MethodType): The method to use for assigning coordinates.
            alpha_final_z (float): The final z-coordinate value for the root node.
            alpha_scale (float): The scaling factor for the alpha value.
            local_embeddings (bool): A flag indicating whether to use local embeddings for the representation.
            **kwargs: Additional keyword arguments to pass to the coordinate assignment functions.

        Returns:
            Tuple[Any, Union[np.ndarray, None], CoordinateTree, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, Any, Any]:
            A tuple containing:
            - tree_doc (Any): The document representation of the conversation tree.
            - message_embeddings (Union[np.ndarray, None]): The message embeddings.
            - relationships (pd.DataFrame): A DataFrame representation of relationships between messages.
            - parent_coords (pd.DataFrame): A DataFrame containing parent coordinates.
            - parent_coords (pd.DataFrame): A DataFrame containing parent coordinates.
            - combined_tensor (np.ndarray): A combined tensor of distances and similarities.
            - distances_df (pd.DataFrame): A DataFrame containing calculated distances.
            - similarity_df (pd.DataFrame): A DataFrame containing calculated similarities.
            - model (Any): A trained model for the conversation tree.
        """
        model = None
        main_df = pd.DataFrame()

        if local_embeddings:
            embeddings, message_ids, message_embeddings = self._generate_embeddings()
        else:
            embeddings = message_ids = message_embeddings = None

        try:
            (
                tetra_dict,
                relationships,
                root_id,
                parent_coords,
            ) = self.create_coordinates(
                **kwargs,
            )

            coordinates_list = []
            for tetra in tetra_dict.values():
                # drop the last 4 elements of the tetra list
                coordinates_list.append(tetra[:-4])

            title = self.conversation.title

            if animate:
                animate_conversation_tree(coordinates=coordinates_list)

            if local_embeddings:
                if self.train_model:
                    relationship_df, model = self.create_prompt_response_df(
                        pre_computed_embeddings=pre_computed_embeddings
                    )
                else:
                    if self.create:
                        relationship_df = self.create_prompt_response_df(
                            pre_computed_embeddings=pre_computed_embeddings
                        )

                    else:
                        relationship_df = self.create_prompt_response_df(
                            pre_computed_embeddings=pre_computed_embeddings
                        )

                main_df = self._update_and_create_main_df(
                    self._construct_representation(message_embeddings, tetra_dict)
                )

            else:
                relationship_df = self._construct_relationships(relationships)

            # self.save_coordinates(
            #     coordinates_list=coordinates_list,
            #     main_df=main_df,
            #     relationship_df=relationship_df,
            #     message_dict=self.message_dict,
            #     base_path=base_path,
            #     title=title,
            # )

            return (model, main_df, relationship_df, self.message_dict)

        except Exception as e:
            print(f"Error in _process_coordinates_graph: {e}")
            return (model, main_df, pd.DataFrame(), self.message_dict)
