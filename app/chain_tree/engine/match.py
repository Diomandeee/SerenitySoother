from typing import Any, Dict, List, Tuple, Callable, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from chain_tree.utils import log_handler
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    InputExample,
    losses,
)
import pandas as pd
import numpy as np
import json


class DataMatcher:
    """
    Class for performing stable matching between nodes and responses.
    Supports matching with indifference if specified.
    """

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        responses: List[Dict[str, Any]],
        similarity_func: Callable = cosine_similarity,
    ):
        """
        Initializes the StableMatching class with nodes and responses.

        Args:
        - nodes (List[Dict]): List of nodes with keys 'id', 'prompt', 'embedding'.
        - responses (List[Dict]): List of responses with keys 'id', 'response', 'embedding'.
        - similarity_func (Callable): Function to compute similarity, default is cosine similarity.
        """

        self.nodes = nodes
        self.responses = responses
        self.similarity_func = similarity_func

        # Compute similarity scores
        self.similarity_scores = self.compute_similarity_scores()

        # Initialize empty matches and proposals
        self.node_matches = {}
        self.response_matches = {}
        self.node_proposals = {
            node["id"]: [response["id"] for response in responses] for node in nodes
        }

    def convert_embedding(self, embedding):
        """
        Convert a stringified embedding to a numerical array.

        Args:
            embedding (str or list): The embedding to convert.

        Returns:
            np.array: The numerical array representation of the embedding.
        """
        if isinstance(embedding, str):
            try:
                # Assuming the embedding is JSON string or space-separated values
                return np.array(json.loads(embedding))
            except json.JSONDecodeError:
                # Fallback for space-separated string
                return np.fromstring(embedding, sep=" ")
        return np.array(embedding)

    def compute_similarity_scores(self) -> Dict[Tuple[str, str], float]:
        """
        Computes the similarity scores between nodes and responses.

        Notation:
            N: Set of nodes
            R: Set of responses
            s(n, r): Similarity function for node n and response r

        Algorithm:
            1. Initialize similarity_scores as an empty dictionary
            2. For each node n ∈ N and response r ∈ R:
                - Compute s(n, r) and add it to similarity_scores

        Returns:
            - Dict[Tuple[str, str], float]: Dictionary of similarity scores, where each key is a tuple of the node ID and response ID, and the value is the similarity score.
        """

        similarity_scores = {}

        for node in self.nodes:
            node_embedding = self.convert_embedding(node["embedding"])
            for response in self.responses:
                response_embedding = self.convert_embedding(response["embedding"])
                similarity_scores[(node["id"], response["id"])] = self.similarity_func(
                    [node_embedding], [response_embedding]
                )[0][0]

        return similarity_scores

    def get_matches(self) -> List[Tuple[str, str]]:
        """
        Executes the stable matching algorithm to find optimal node-response pairings.

        Args:
            None: Operates on instance variables

        Instance Variables:
            self.node_proposals (Dict[str, List[str]]): A dictionary mapping each node to a list of potential responses.
            self.similarity_scores (Dict[Tuple[str, str], float]): A dictionary storing the similarity scores between nodes and responses.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains a node ID and a corresponding response ID that forms the best match.

        Notation:
            - N: Set of nodes (from self.node_proposals)
            - R: Set of responses
            - s(n, r): Similarity function for node n and response r (from self.similarity_scores)
            - M: Resulting matching (returned value)

        Algorithm Steps:
            0. Initialize `rejected_proposals` to keep track of rejected proposals for each node.
            1. Loop until `self.node_proposals` is empty:
                - For each node n and its list of responses r:
                    - Check against past rejections.
                    - Calculate the similarity score.
                    - Update `best_response` based on similarity.
                - If `best_response` is a tuple, handle each through `handle_match`.
                - Otherwise, handle the single best response.
            2. Return the final matching, M.

        Side Effects:
            - Modifies `self.node_proposals` by removing processed nodes.
            - Modifies `self.node_matches` to store the final matching results.

        Example:
            Given initial state:
            self.node_proposals = {'node1': ['response1', 'response2']}
            self.node_matches = {}
            self.response_matches = {}
            Running get_matches() will modify:
                self.node_matches = {'node1': 'response1'} (or 'response2' depending on scores)
                self.response_matches = {'response1': 'node1'} (or 'response2': 'node1')
        """

        # Initialize the set of rejected proposals for each node
        rejected_proposals = {node_id: set() for node_id in self.node_proposals.keys()}

        # Loop until all node proposals are empty
        while len(self.node_proposals) > 0:
            # Pop a node and its associated responses; skip if no nodes left
            node_id, response_ids = self.node_proposals.popitem()

            best_response = None
            best_score = float("-inf")

            # Iterate over each response associated with the node
            for response_id in response_ids:
                # Skip this response if it was previously rejected by this node
                if response_id in rejected_proposals[node_id]:
                    continue

                # Retrieve the precomputed similarity score
                score = self.similarity_scores[(node_id, response_id)]

                if best_response is None:
                    best_response = (
                        (best_response, response_id)
                        if isinstance(best_response, tuple)
                        else (best_response,) + (response_id,)
                    )
                elif score > best_score:
                    best_response = response_id
                    best_score = score

            if best_response is not None:
                if isinstance(best_response, tuple):
                    for response_id in best_response:
                        success = self.handle_match(node_id, response_id)
                        if not success:
                            rejected_proposals[node_id].add(response_id)
                else:
                    success = self.handle_match(node_id, best_response)
                    if not success:
                        rejected_proposals[node_id].add(best_response)

        return [(node, response) for node, response in self.node_matches.items()]

    def handle_match(self, node_id: str, response_id: str) -> None:
        """
        Handles matching between a node (n) and a response (r) within the stable matching framework.

        Args:
            node_id (str): Unique identifier for the node (typically a user) proposing the match.
            response_id (str): Unique identifier for the response (typically a job position, course, etc.) being proposed to.

        Returns:
            None: This function modifies the internal state but doesn't return any value.

        Notation:
            - N: Set of nodes (e.g., users, participants)
            - R: Set of responses (e.g., job positions, courses)
            - s(n, r): A function that returns the similarity score between node n and response r
            - M: A dictionary storing the current state of matches
            - n: The node currently under consideration for matching
            - r: The response currently under consideration for matching

        Algorithm Steps:
            1. Check if the response (r) is currently unmatched:
                - If so, create a new match (n, r) and add it to the internal matching dictionaries `node_matches` and `response_matches`.
            2. If the response (r) is already matched with another node (m):
                - Calculate the current matching score s(m, r) and the proposed score s(n, r).
                - If s(n, r) > s(m, r):
                    - Remove the existing match (m, r).
                    - Create the new match (n, r).
                    - Re-add the old node (m) to `node_proposals` for future matching.
                - Otherwise:
                    - Re-add the proposing node (n) to `node_proposals` for future matching.

        Side Effects:
            - Modifies `node_matches` to update or insert new node-response matches.
            - Modifies `response_matches` to update or insert new response-node matches.
            - Modifies `node_proposals` to allow for re-proposal from either the old or new node as needed.

        Theoretical Guarantees:
            In accordance with the Indifference Stability Theorem, this function ensures that each match is as optimal as possible under the current conditions. It also allows for re-proposals from nodes to ensure that all nodes have the opportunity to find their most compatible matches.

        Example:
            Given node_proposals = {'node1': ['response1', 'response2']}
            And node_matches = {}
            And response_matches = {}
            Running handle_match('node1', 'response1') will result in:
                node_matches = {'node1': 'response1'}
                response_matches = {'response1': 'node1'}
        """

        if response_id not in self.response_matches:
            self.node_matches[node_id] = response_id
            self.response_matches[response_id] = node_id
        else:
            current_match_score = self.similarity_scores[
                (self.response_matches[response_id], response_id)
            ]
            proposed_match_score = self.similarity_scores[(node_id, response_id)]

            if proposed_match_score > current_match_score:
                old_node = self.response_matches[response_id]
                self.node_matches.pop(old_node)
                self.node_matches[node_id] = response_id
                self.response_matches[response_id] = node_id
                if old_node not in self.node_proposals:
                    self.node_proposals[old_node] = [response_id]
                else:
                    self.node_proposals[old_node].append(
                        response_id
                    )  # Allow old node to propose again
            else:
                if node_id not in self.node_proposals:
                    self.node_proposals[node_id] = [response_id]
                else:
                    self.node_proposals[node_id].append(
                        response_id
                    )  # Allow this node to propose again


def get_node_response_pairs_stable_matching(
    nodes: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    similarity_func: Callable = cosine_similarity,
    compute_similarity: bool = False,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Gets the node-response pairs for the given nodes and responses using stable matching.

    Notation:
        N: Set of nodes
        R: Set of responses
        s: Similarity function
        P: Stable matching problem defined as (N, R, I, V, s)
        M: Matching as a result of solving P

    Algorithm:
        1. Validate inputs
        2. Initialize stable matching problem P = (N, R, I, V, s)
        3. Execute stable matching algorithm on P to get M
        4. Return M as a list of node-response pairs

    Args:
        - nodes (List[Dict[str, Any]]): List of nodes, each represented by a dictionary with keys:
            - 'id': Unique identifier for the node
            - 'prompt': Prompt associated with the node
            - 'embedding': Embedding vector for the prompt
        - responses (List[Dict[str, Any]]): List of responses, each represented by a dictionary with keys:
            - 'id': Unique identifier for the response
            - 'response': Text of the response
            - 'embedding': Embedding vector for the response
         - similarity_func (Callable): Optional. Function to compute similarity, default is cosine similarity.

    Returns:
        - List[Tuple[Dict[str, Any], Dict[str, Any]]]: List of node-response pairs, where each pair is a tuple of the node and response dictionaries.

    Raises:
        - ValueError: If nodes or responses are empty or if there are more nodes than responses.

    Constraints:
        - Node IDs and response IDs must be unique.
        - Embedding vectors should be in a format compatible with the provided similarity function.

    Dependencies:
        - Requires `DataMatcher` class for performing stable matching. Make sure to import it before using this function.
        - Default similarity function is `similarity`, make sure it is defined or imported.

    Side Effects:
        - None. This function is pure and does not modify its inputs.

    Example:
        nodes = [{'id': 'n1', 'prompt': 'prompt1', 'embedding': [0.1, 0.2]},
                 {'id': 'n2', 'prompt': 'prompt2', 'embedding': [0.3, 0.4]}]
        responses = [{'id': 'r1', 'response': 'response1', 'embedding': [0.1, 0.2]},
                     {'id': 'r2', 'response': 'response2', 'embedding': [0.3, 0.4]}]
        pairs = get_node_response_pairs_stable_matching(nodes, responses)
        # Output could be: [(node_dict1, response_dict1), (node_dict2, response_dict2)]

    """

    # Validate input
    if not nodes or not responses:
        raise ValueError("Nodes and responses must be non-empty lists.")
    if len(nodes) > len(responses):
        raise ValueError("Number of nodes must not exceed the number of responses.")

    # Perform stable matching between nodes and responses using the DataMatcher class
    stable_matching = DataMatcher(nodes, responses, similarity_func=similarity_func)
    matches = stable_matching.get_matches()

    node_response_records = []
    for node_id, response_id in matches:
        node = next((n for n in nodes if n["id"] == node_id), None)
        response = next((r for r in responses if r["id"] == response_id), None)

        if not node or not response:
            continue  # or raise an error, depending on your use case

        record = {
            "prompt_id": node["id"],
            "response_id": response["id"],
            "created_time": node["created_time"],
            "prompt": node["prompt"],
            "response": response["response"],
            "prompt_coordinate": node["prompt_coordinate"],
            "response_coordinate": response["response_coordinate"],
            "prompt_embedding": node["embedding"],
            "response_embedding": response["embedding"],
        }

        if compute_similarity:
            similarity_score = similarity_func(
                [node["embedding"]], [response["embedding"]]
            )[0][0]
            record["similarity"] = similarity_score

        stacked_coords = np.vstack(
            (node["prompt_coordinate"], response["response_coordinate"])
        )
        record["stacked_coordinate"] = stacked_coords

        node_response_records.append(record)

    return pd.DataFrame(node_response_records)


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that the DataFrame has all the required columns."""
    return all(column in df.columns for column in required_columns)


def prepare_training_examples(df: pd.DataFrame) -> List[InputExample]:
    """Prepare training examples from DataFrame."""
    examples = []
    for i, row in df.iterrows():
        input_example = InputExample(
            texts=[row["prompt"], row["response"]], label=row["similarity"]
        )
        examples.append(input_example)
    return examples


def extract_nodes_and_responses(
    df: pd.DataFrame,
) -> Any:
    """
    Extracts and transforms data from the DataFrame into 'nodes' and 'responses' format.

    Parameters:
    - df (pd.DataFrame): DataFrame generated from the 'build_dataframe' function containing conversational data.

    Returns:
    - tuple: A tuple containing two lists. The first list consists of 'nodes' and the second list consists of 'responses'.
    """

    nodes = [
        {
            "id": prompt_id,
            "prompt": prompt,
            "embedding": embedding,
            "prompt_coordinate": prompt_coordinate,
            "created_time": created_time,
        }
        for prompt_id, prompt, embedding, prompt_coordinate, created_time in zip(
            df["prompt_id"],
            df["prompt"],
            df["prompt_embedding"],
            df["prompt_coordinate"],
            df["created_time"],
        )
    ]

    responses = [
        {
            "id": response_id,
            "response": response,
            "embedding": embedding,
            "response_coordinate": response_coordinate,
        }
        for response_id, response, embedding, response_coordinate, created_time in zip(
            df["response_id"],
            df["response"],
            df["response_embedding"],
            df["response_coordinate"],
            df["created_time"],
        )
    ]

    return nodes, responses


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares nodes and responses from the relationship data."""
    try:
        nodes, responses = extract_nodes_and_responses(df)
    except Exception as e:
        raise ValueError("Failed to prepare data.")
    return nodes, responses


def compute_stable_matching(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes stable matching based on nodes and responses."""
    try:
        nodes, responses = prepare_data(df)

        stable_df = get_node_response_pairs_stable_matching(
            nodes=nodes, responses=responses
        )
    except Exception as e:
        raise ValueError("Failed to compute stable matching.")
    return stable_df


def initialize_and_train_model(
    result_df: pd.DataFrame,
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    num_epochs: int = 1,
    verbose: bool = True,
    warmup_steps: int = 100,
    evaluation_steps: int = 100,
) -> Optional[SentenceTransformer]:
    """
    Initialize and train a Sentence Transformer model using the provided DataFrame.
    """

    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    # Check DataFrame validity
    required_columns = ["prompt", "response", "similarity"]
    if not validate_dataframe_columns(result_df, required_columns):
        raise ValueError("The DataFrame does not contain all the required columns.")

    try:
        # Initialize the model
        model = SentenceTransformer(model_name)

        # Prepare training data
        training_examples = prepare_training_examples(result_df)

        # Create DataLoader and Loss objects
        train_dataset = SentencesDataset(training_examples, model)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )
        train_loss = losses.CosineSimilarityLoss(model)

        # define the sentence evaluator
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=result_df["prompt"].tolist(),
            sentences2=result_df["response"].tolist(),
            scores=result_df["similarity"].tolist(),
            batch_size=batch_size,
            name="similarity_evaluation",
        )

        # Train the model with the evaluator
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
        )

        return model
    except Exception as e:
        if verbose:
            log_handler(f"An error occurred during training: {e}", level="error")
        raise


def train_model(
    df: pd.DataFrame, test_csv_path: str = "test.csv", save: bool = False
) -> Optional[SentenceTransformer]:
    """Trains a model if specified and saves test data."""

    def save_to_csv(df: pd.DataFrame, path: str):
        """Saves a DataFrame to a CSV file."""
        try:
            df.to_csv(path, index=True)
        except Exception as e:
            raise ValueError("Failed to save DataFrame to CSV.")

    model = None
    try:
        train_df, test_df = train_test_split(df, test_size=0.2)
        if save:
            save_to_csv(test_df, test_csv_path)
        model = initialize_and_train_model(train_df)
    except Exception as e:
        raise ValueError("Failed to train model.")
    return model
