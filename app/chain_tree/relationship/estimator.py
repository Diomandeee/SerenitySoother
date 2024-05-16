from typing import Dict, Any, Optional, Deque, List
from chain_tree.engine.relation import ChainRelationships
from collections import deque
import networkx as nx
import numpy as np
import time

ESTIMATE_HISTORY_MAXLEN = 100
default_params = {
    "weight": 0.11,
    "frequency": 0,
    "importance": 1.0,
    "last_observed": 0,
}

variable_params = {
    "siblings": {"importance": 1.0},
    "cousins": {"importance": 0.8},
    "uncles_aunts": {"importance": 0.7},
    "nephews_nieces": {"importance": 0.6},
    "grandparents": {"weight": 0.16, "importance": 0.9},
    "ancestors": {"weight": 0.2, "importance": 0.5},
    "descendants": {"weight": 0.2, "importance": 1.2},
}


class Estimator(ChainRelationships):
    """
    The Estimator class embodies the ChainLink Interaction Framework, orchestrating a nuanced exploration
    and interpretation of relationships amongst messages within a conversational milieu.

    Framework: ChainLink Interaction Framework

    Representation:

    Let:
    - M denote the set of all messages in the system.
    - R be a function such that R(m1, m2) ∈ [0,1], representing the relational strength between two messages m1 and m2.

    The framework explores four primary estimations:

    1. Baseline Estimation (B):
       B(m) = sqrt(|M|), ∀ m ∈ M.
       Volume of the dataset.

    2. Relationship Size Estimation (S):
       S(m) = max_{m'} (|R(m, m')|), ∀ m' ∈ M.
       Depth of each message's relationships.

    3. Type-based Estimation (T):
       T(m) = |{ m' : R(m, m') > 0 }|.
       Evaluating the array of message links based on types.

    4. Weighted Relationship Estimation (W):
       W(m) = Σ_{m' ∈ M} W_m' × R(m, m').
       Incorporating relationship type weights into the equation.

    The unified, weighted estimate is articulated as:
    E(m) = N( ω_B × B(m) + ω_S × S(m) + ω_T × T(m) + ω_W × W(m) )
    Where ω values symbolize weight multipliers for each estimation component and N is a normalization function.

    Attributes:
    - RELATIONSHIPS (dict): A dictionary mapping relationship types to their parameters.
    - message_dict (dict): Dictionary encapsulating all messages.
    - conversation_dict (dict, optional): An optional dictionary containing conversations.
    - _message_references (dict): A dictionary to store message references.
    - estimate_history (dict): A history log of past estimations.
    - estimate_weights (dict): Weights associated with different types of estimates.
    - n_neighbors_weighted (int): The computed weighted number of neighbors.

    Methods:
    - Inherits methods from ChainRelationships: The class inherits several crucial methods from ChainRelationships
      class which include determining relationships, updating relationship frequencies and weights, and calculating
      the number of neighbors for a given message.
    - __init__(): Initializes the Estimator with a given message dictionary and optionally a conversation dictionary.
    - update_estimate_history_and_weights(): Updates the estimate history and recalculates weights based on new estimates.
    - compute_new_estimates(): Computes new estimates for the number of neighbors based on the given relationship dictionary.
    - determine_n_neighbors(): Determines the number of neighbors for a given message based on its relationships.
    - _update_history(): Updates the estimate history with new estimates.
    - _recalculate_weights(): Recalculates weights based on the updated history.
    - _normalize_weights(): Normalizes weights ensuring they sum up to 1.
    - get_root_node(): Determines the root node of the conversation.tree.

    """

    RELATIONSHIPS = {
        key: {**default_params, **params} for key, params in variable_params.items()
    }

    def __init__(
        self,
        message_dict: Dict[str, Any],
        conversation_dict: Optional[Dict[str, Any]] = None,
        relationship_threshold: float = 0.1,
    ):
        """
        Initialize the Estimator with message and optional conversation data.

        Args:
            message_dict (Dict[str, Any]): A dictionary of messages.
            conversation_dict (Optional[Dict[str, Any]]): Optionally, a dictionary of conversation data.
        """

        self.message_dict = message_dict
        self.conversation_dict = conversation_dict

        # Prepare references for each message ID
        self._message_references = {msg_id: {} for msg_id in self.message_dict.keys()}

        # Setup history of estimates for each type with a max length
        self.estimate_history: Dict[str, Deque[int]] = {
            "baseline": deque(maxlen=ESTIMATE_HISTORY_MAXLEN),
            "relationships": deque(maxlen=ESTIMATE_HISTORY_MAXLEN),
            "types": deque(maxlen=ESTIMATE_HISTORY_MAXLEN),
            "weighted": deque(maxlen=ESTIMATE_HISTORY_MAXLEN),
        }

        # Initialize default weights for each estimate type
        self.estimate_weights: Dict[str, float] = {
            "baseline": 1.0,
            "relationships": 1.0,
            "types": 1.0,
            "weighted": 1.0,
        }

        # Initial count of weighted neighbors set to 0
        self.n_neighbors_weighted = 0
        self.tree = nx.DiGraph()
        self.relationship_threshold = relationship_threshold

    def get_relationship_strength(self, msg1: str, msg2: str) -> float:
        """
        Retrieves the strength of the relationship between two specified messages.

        Within the ChainLink Interaction Framework, relationships not only connect messages but also carry a
        quantifiable strength, reflecting the degree of connection. This method facilitates the extraction of this
        relational strength for a pair of messages.

        Args:
            msg1 (str): The ID of the first message.
            msg2 (str): The ID of the second message.

        Returns:
            float: The strength of the relationship between `msg1` and `msg2`. It returns 0 if there is no established
                relationship between the two.

        Process:
        1. Invoke the `get_relationship` method, derived from the `ChainRelationships` class, to procure the relationships
        of `msg1`.
        2. Use the `.get()` method on the returned relationship dictionary to obtain the strength associated with `msg2`.
        If `msg2` is not present in the dictionary, a default value of 0 is returned, indicating no relationship.

        This method simplifies the process of discerning the intensity of connection between two messages, thereby aiding
        in more granular analysis and interpretations of the conversational landscape.
        """
        relationships = self.get_relationship(msg1)

        return relationships.get(msg2, 0)

    def update_relationship_frequency(self, relationship_dict: Dict[str, Any]):
        """
        Iterates through the provided relationship dictionary and updates the frequency
        and last observed time of each relationship type in the self.RELATIONSHIPS dictionary.

        Process:
        1. Retrieves the current time using the time() function.
        2. Iterates through each relationship type present in the provided relationship_dict.
        3. For each relationship type found in both the input dictionary and the self.RELATIONSHIPS dictionary,
           it increments the frequency count and updates the last observed time with the current time.

        Args:
            relationship_dict (Dict[str, Any]):
                A dictionary mapping relationship types to some values, the nature of which
                aren't clear from the snippet. Only the keys (relationship types) are used.

        Notes:
            This method assumes that the self.RELATIONSHIPS dictionary is structured with
            each relationship type as a key, and each value being another dictionary with
            keys "frequency" and "last_observed" to keep track of the count and last
            observation time of each relationship type.
        """
        current_time = time.time()  # Retrieves the current time
        for (
            rel_type
        ) in (
            relationship_dict.keys()
        ):  # Iterates through each relationship type in the input dictionary
            if (
                rel_type in self.RELATIONSHIPS
            ):  # Checks if the relationship type is recognized in self.RELATIONSHIPS
                self.RELATIONSHIPS[rel_type][
                    "frequency"
                ] += 1  # Increments the frequency count for this relationship type
                self.RELATIONSHIPS[rel_type][
                    "last_observed"
                ] = current_time  # Updates the last observed time for this relationship type

    def get_dynamic_decay_factor(self) -> float:
        """
        Computes a dynamic decay factor influenced by the average time elapsed since the
        last observed relationship for all relationship types.

        Process:
        1. For each relationship type stored in self.RELATIONSHIPS, the method retrieves the
           'last_observed' timestamp, which is assumed to be the time since epoch in seconds.
           If no 'last_observed' key is present for a relationship, it defaults to 0.

        2. Calculates the sum of the time since last observation for all relationships,
           which gives the total time since last observed across all relationship types.

        3. Computes the average time since the last observation by dividing the total time
           by the number of relationship types.

        4. The dynamic decay factor is determined by multiplying a base decay factor (0.9)
           by a ratio. This ratio is (1 + average time since last observation / 100), which
           ensures that as the average time since the last observed relationship increases,
           the decay factor gradually grows above the base decay value.

        Returns:
            float: The computed dynamic decay factor.

        Notes:
            - The base decay factor is currently set to 0.9. A higher average time since the
              last observation would push the dynamic decay factor above this base value.
            - The function assumes that self.RELATIONSHIPS is structured such that each
              relationship type is mapped to a dictionary. Within this dictionary, the
              'last_observed' key holds a timestamp (time since epoch in seconds).
        """
        total_time_since_last = sum(
            rel.get("last_observed", 0) for rel in self.RELATIONSHIPS.values()
        )
        avg_time_since_last = total_time_since_last / len(self.RELATIONSHIPS)

        base_decay = 0.9
        return base_decay * (1 + avg_time_since_last / 100)

    def update_relationship_weights(self, coefficient: float = [0.4, 0.3, 0.3]) -> None:
        """
        Modifies the weights of relationships within the system according to a combination
        of metrics, including the frequency of the relationship's occurrence,
        its recency (or the time since its last occurrence), and its pre-defined importance.

        Process:
        1. Compute the total frequency of all relationships in the system.

        2. If no relationship has ever been observed (total frequency is 0), exit the function.

        3. Determine the most recent timestamp across all relationships, termed as 'max_recency'.

        4. Extract a dynamic decay factor which affects the calculation of the recency weight.

        5. Traverse through each relationship and compute the following weights:
           - Frequency Weight: Uses logarithmic scaling to give diminishing returns as the
             frequency of a relationship increases.
           - Recency Weight: Utilizes an exponential decay formula, making older relationships
             decay in significance more rapidly.
           - Importance Weight: Uses quadratic scaling to emphasize or de-emphasize relationships
             based on their intrinsic importance.

        6. These weights are combined linearly to get a new weight for each relationship.

        7. Normalize the weights so that they sum up to 1.

        Returns:
            None: This method updates the internal state of the 'self.RELATIONSHIPS' dictionary
            with new weights but does not return any value.

        Notes:
            - The base weight computation coefficients (e.g., 0.4, 0.3, 0.3) can be adjusted to
              emphasize or de-emphasize certain metrics.
            - The 'self.RELATIONSHIPS' dictionary is expected to hold, for each relationship type,
              keys including 'frequency', 'last_observed', and 'importance' with their respective values.
        """
        total_frequency = sum(rel["frequency"] for rel in self.RELATIONSHIPS.values())
        if total_frequency == 0:
            return

        max_recency = max(
            rel.get("last_observed", 0) for rel in self.RELATIONSHIPS.values()
        )

        decay_factor = self.get_dynamic_decay_factor()

        new_weights = {}

        for rel_type, rel_data in self.RELATIONSHIPS.items():
            frequency_weight = np.log(rel_data["frequency"] + 1)

            time_since_last_observed = max_recency - rel_data.get("last_observed", 0)
            recency_weight = np.exp(-decay_factor * time_since_last_observed)

            importance_weight = (rel_data["importance"]) ** 2

            new_weight = (
                coefficient[0] * frequency_weight
                + coefficient[1] * recency_weight
                + coefficient[2] * importance_weight
            )

            new_weights[rel_type] = new_weight

        total_new_weight = sum(new_weights.values())
        for rel_type in self.RELATIONSHIPS:
            self.RELATIONSHIPS[rel_type]["weight"] = (
                new_weights[rel_type] / total_new_weight
            )

    def _update_history(self, new_estimates: Dict[str, int]) -> None:
        """
        Supplements the historical data with fresh estimates.

        Useful for persisting a trace of estimates over time, enabling future analysis
        or decision-making based on historical trends. It updates the 'self.estimate_history'
        dictionary with new data provided in the 'new_estimates' dictionary.

        Process:
        1. Traverse through each estimate type and its corresponding new value in
           the 'new_estimates' dictionary.
        2. If the estimate type already exists in 'self.estimate_history', append
           the new estimate to its list of historical values.
        3. If the estimate type does not exist in 'self.estimate_history', raise
           a ValueError indicating that an unrecognized estimate type has been encountered.

        Parameters:
        - new_estimates (Dict[str, int]): A dictionary wherein keys represent the type of
          estimate (e.g., "cost_estimate", "time_estimate") and values represent the new
          estimate for that type. These new values will be added to the respective estimate
          type's history in 'self.estimate_history'.

        Raises:
        - ValueError: This error is raised if a provided estimate type in 'new_estimates'
          is not recognized in the 'self.estimate_history' dictionary. It indicates that
          the input might have some inconsistencies or erroneous entries.

        Returns:
            None: This method only modifies the internal state of the 'self.estimate_history'
            dictionary and does not return any value.
        """
        for estimate_type, new_estimate in new_estimates.items():
            if estimate_type in self.estimate_history:
                self.estimate_history[estimate_type].append(new_estimate)
            else:
                raise ValueError(f"Invalid estimate type: {estimate_type}")

    def _recalculate_weights(self) -> None:
        """
        Dynamically determines the weights for each estimate type using its historical data.

        This method takes into consideration the history of estimates and, based on this historical
        data, recalculates the weight for each type of estimate. The methodology is straightforward:
        the weight for a specific estimate type is determined as the inverse of its average from
        the historical data. This means that if the average estimate is higher, its weight will be
        lower, and vice versa. If the historical average is zero, the weight is explicitly set to zero,
        indicating no importance or relevance.

        The main advantage of this approach is its adaptability. As new estimates are added to the
        history and older ones become less relevant, the weights adjust to reflect the most recent
        trends or changes in the data.

        Process:
        1. For each estimate type present in the 'self.estimate_history':
            a. If there's no history for an estimate type, it's skipped.
            b. Calculate the mean of the historical values for that estimate type.
            c. If the mean is greater than zero, set the weight as its inverse.
            d. If the mean is zero, explicitly set the weight to zero.

        Parameters:
            None

        Modifies:
            self.estimate_weights: The internal dictionary that stores the calculated weight for
            each estimate type. This dictionary is updated with new weights as a result of this method.

        Returns:
            None: This method solely alters the internal state and does not return any value.
        """
        for estimate_type, history in self.estimate_history.items():
            if not history:  # Check if history is empty and skip if so
                continue
            mean_estimate = np.mean(history)
            if mean_estimate > 0:
                self.estimate_weights[estimate_type] = 1 / mean_estimate
            else:
                self.estimate_weights[estimate_type] = 0.0

    def _normalize_weights(self) -> None:
        """
        Adjusts the weights within 'self.estimate_weights' to ensure they collectively amount to 1.

        Process:
        1. Compute the total sum of all weights.
        2. If this total is greater than zero:
            a. Adjust each weight proportionally so that their combined sum becomes 1.
        3. If the total sum is zero (an edge case which might arise in specific scenarios):
            a. Assign equal weights to all entries, ensuring they still sum up to 1.

        It's noteworthy that in scenarios where all weights are initially zero, the method will
        distribute equal weights to all entries, effectively treating them with equal importance.

        Parameters:
            None

        Modifies:
            self.estimate_weights: The internal dictionary that stores the weights for each estimate type.
            This dictionary gets updated to reflect the normalized weights.

        Returns:
            None: This method only modifies the internal state and does not return any value.
        """
        total_weight = sum(self.estimate_weights.values())
        if total_weight > 0:
            self.estimate_weights = {
                k: v / total_weight for k, v in self.estimate_weights.items()
            }
        else:
            # When the sum is zero, distribute equal weights to all entries
            self.estimate_weights = {
                k: 1.0 / len(self.estimate_weights) for k in self.estimate_weights
            }

    def update_estimate_history_and_weights(
        self, new_estimates: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Incorporate new data into the historical estimates and recompute corresponding weights.

        Process:
        1. History Update:
            For each estimate type provided in 'new_estimates', this method appends the new value
            to its corresponding history in 'self.estimate_history'. If a provided estimate type
            doesn't exist in the history, an error is raised. This ensures that only valid estimate
            types are added to the history.

        2. Weight Recalculation:
            The weights for each estimate type are recalculated based on their historical data. The
            weight for a specific estimate type is determined as the inverse of its historical mean.
            If the mean is zero, its weight is set to zero. This approach gives higher importance
            (or weight) to estimate types with lower average values.

        3. Normalization:
            After weights are recalculated, they are adjusted such that their sum amounts to 1. This
            step ensures the weights represent valid proportions of a whole, which is commonly required
            in mathematical or statistical models. If the combined sum of weights is zero, equal weights
            are assigned to all estimate types, treating them with equal significance.

        Args:
            new_estimates (Dict[str, int]): A dictionary that contains the new estimate values, where
            each key represents an estimate type and its corresponding value represents the estimate
            amount.

        Returns:
            Dict[str, float]: A dictionary mapping each estimate type to its recalculated and normalized
            weight. This provides an overview of the relative importance of each estimate type based on
            historical data.

        Modifies:
            self.estimate_history: The internal history of past estimates is updated with the new values.
            self.estimate_weights: The internal weights are recalculated and normalized based on the
            updated history.
        """
        self._update_history(new_estimates)
        self._recalculate_weights()
        self._normalize_weights()

        return self.estimate_weights

    def compute_new_estimates(
        self, relationship_dict: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Compute estimates for a message's neighbor count based on its relationships.

        This method computes four estimates for the number of neighbors a message has based on its
        relationships. The four estimates are:
        1. Baseline Estimation (B): The square root of the total number of messages in the system.
        2. Relationship Size Estimation (S): The maximum number of relationships a message has with
              any other message in the system.
        3. Type-based Estimation (T): The number of messages linked to the given message based on
                relationship types.
        4. Weighted Relationship Estimation (W): The sum of the weights of all relationships a message
                has with other messages.

        The four estimates are combined linearly to get a unified estimate, which is returned as a dictionary
        with the four estimates as keys and their corresponding values as the estimates.

        Process:
        1. Compute the total number of messages in the system.
        2. Extract the values from the provided relationship dictionary.
        3. Compute the four estimates using the extracted values.
        4. Return the four estimates in a dictionary.

        Args:
            relationship_dict (Dict[str, Any]): Dictionary mapping relationship types to associated
            messages/entities. The key is the relationship type, and the value is a list/set of related entities.

        Returns:
            Dict[str, int]: Dictionary with estimates. Keys represent estimation methods ("baseline",
            "relationships", "types", "weighted"), and values are the computed estimates.

        Accesses:
            self.message_dict: Dictionary of all messages for baseline estimation.
            self.RELATIONSHIPS: Dictionary with relationship types and their weights.
        """
        n_total_messages = len(self.message_dict)
        relationship_values = list(relationship_dict.values())
        relationship_items = list(relationship_dict.items())

        n_neighbors_baseline = int(np.sqrt(n_total_messages))
        n_neighbors_relationships = max(len(value) for value in relationship_values)
        n_neighbors_types = len(
            [rel_type for rel_type, rel_list in relationship_items if rel_list]
        )
        n_neighbors_weighted = sum(
            self.RELATIONSHIPS[rel_type]["weight"] * len(rel)
            for rel_type, rel in relationship_items
        )

        return {
            "baseline": n_neighbors_baseline,
            "relationships": n_neighbors_relationships,
            "types": n_neighbors_types,
            "weighted": n_neighbors_weighted,
        }

    def determine_n_neighbors(self, message_id: str) -> int:
        """
        Determine a message's neighbor count based on its relationships and historical data.

        It first computes the four estimates for the number of neighbors, as described
        in the `compute_new_estimates` method. It then combines these estimates linearly to get a unified
        estimate, which is returned as an integer.

        Process:
        1. Validates the existence of the message.
        2. Fetches and processes the message's relationships.
        3. Computes various neighbor count estimates.
        4. Updates internal history and weights based on these estimates.
        5. Returns an average count derived from the new estimates.

        Args:
            message_id (str): Unique identifier for a message in the system.

        Returns:
            int: Average estimated number of neighbors the message has.

        Raises:
            ValueError: If the message ID is not found in the system.

        Accesses:
            Multiple internal methods/attributes, including self.message_dict, self.get_relationship,
            self.update_relationship_frequency, self.update_relationship_weights, and others.
        """

        if message_id not in self.message_dict:
            raise ValueError(
                f"Message ID {message_id} not found in message dictionary."
            )

        relationship_dict = self.get_relationship(message_id)
        self.update_relationship_frequency(relationship_dict)
        self.update_relationship_weights()
        new_estimates = self.compute_new_estimates(relationship_dict)
        self.update_estimate_history_and_weights(new_estimates)

        return int(np.mean(list(new_estimates.values())))

    def get_root_node(self) -> str:
        """
        Determines and returns the message ID that is estimated to have the most neighbors (i.e., relationships).

        Each message' significance, in part, is determined by its number of neighbors – the more relationships a message has, the more
        pivotal it is in the conversational context. The root node is chosen as the message that stands out with the highest
        number of estimated neighbors, making it the most central or significant message in the conversation.

        Proccess:
        1. For each message in the message dictionary:
        - Calculate the number of neighbors using the `determine_n_neighbors` method.
        - Store the result in the `message_estimates` dictionary with the message ID as the key.
        2. Return the message ID with the highest number of estimated neighbors from the `message_estimates` dictionary.

        Returns:
            str: The message ID of the message that has the highest estimated number of neighbors.

        This method offers an effective means to kickstart the tree-building process, ensuring the tree is rooted
        at the most central message, thereby providing a comprehensive perspective of the conversation's dynamics.
        """
        message_estimates = {
            msg_id: self.determine_n_neighbors(msg_id) for msg_id in self.message_dict
        }
        return max(message_estimates, key=message_estimates.get)

    def analyze_tree(self) -> Dict[str, Any]:
        """
        Provides a comprehensive analysis of the conversation tree's structure and characteristics.

        Conversations, when represented as trees, can be intricate, with messages branching out in various directions
        based on their relationships. This method dissects the tree to offer vital insights into its structure,
        depth, branching patterns, and overall size, enabling a deeper understanding of the conversation's flow and
        complexity.

        Returns:
            dict: A dictionary containing:
                - "depth" (int): The longest path from the root to the furthest leaf in the.tree.
                - "average_branching_factor" (float): The average number of child nodes branching out from each node in the.tree.
                - "total_nodes" (int): The total number of messages (nodes) present in the.tree.
                - "total_edges" (int): The total number of relationships (edges) present in the.tree.

        Proccess:
        1. "depth" is calculated using the `dag_longest_path_length` function from the NetworkX library, which gives
        the length of the longest path in the.tree.
        2. "average_branching_factor" is derived by taking the sum of successors (child nodes) each node has and
        dividing it by the total number of nodes in the.tree.
        3. "total_nodes" and "total_edges" are direct measures obtained from the NetworkX tree object using its
        `number_of_nodes` and `number_of_edges` functions, respectively.

        """
        return {
            "depth": nx.dag_longest_path_length(self.tree),
            "average_branching_factor": sum(
                len(list(self.tree.successors(node))) for node in self.tree.nodes()
            )
            / self.tree.number_of_nodes(),
            "total_nodes": self.tree.number_of_nodes(),
            "total_edges": self.tree.number_of_edges(),
        }

    def visualize_tree(self):
        """
        Visualizes the constructed conversation tree using NetworkX and Matplotlib.

        The visualization helps in graphically representing the conversational dynamics captured in the.tree. Each node
        in the tree corresponds to a message, and the edges represent relationships between messages. The nodes' colors
        are based on their estimation values, providing a gradient representation of their relative importance. The edge
        widths indicate the strength of relationships between messages.

        Prerequisites:
            - matplotlib
            - NetworkX

        Returns:
            None

        Proccess:
        1. Extracts node colors based on the "estimate" attribute of each node. This helps in visually distinguishing
        messages based on their importance or significance.
        2. Extracts edge widths from the "weight" attribute of each edge, showcasing the strength of the relationship
        between two connected messages.
        3. Uses NetworkX's `spring_layout` to determine the positioning of the nodes in the visualization.
        4. Draws the graph using NetworkX's `draw` method, where node colors, sizes, labels, and edge widths are all
        configured.
        5. Finally, Matplotlib's `show` method is used to display the visualization.

        """
        import matplotlib.pyplot as plt

        node_colors = [self.tree.nodes[node]["estimate"] for node in self.tree.nodes()]
        edge_widths = [self.tree.edges[edge]["weight"] for edge in self.tree.edges()]

        pos = nx.spring_layout(self.tree)
        nx.draw(
            self.tree,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=8,
            width=edge_widths,
            edge_color="gray",
        )
        plt.show()

    def calculate_importance_score(self, parent_id: str, child_id: str) -> float:
        """
        Calculates the importance score of a child node based on its parent node.

        The importance score is calculated using the following formula:
        importance_score = parent_importance_score * relationship_strength

        Args:
            parent_id (str): The ID of the parent node.
            child_id (str): The ID of the child node.

        Returns:
            float: The importance score of the child node.

        Proccess:
        1. Fetch the importance score of the parent node.
        2. Fetch the relationship strength between the parent and child nodes.
        3. Calculate the importance score of the child node using the formula above.

        This method helps in determining the relative importance of a child node based on its parent node.
        """
        parent_importance_score = self.tree.nodes[parent_id]["importance_score"]
        relationship_strength = self.tree.edges[(parent_id, child_id)]["weight"]

        return parent_importance_score * relationship_strength

    def get_related_messages(self, message_id: str) -> List[str]:
        """
        Retrieves the messages related to the specified message based on their relationships.

        The relationships of a message are indicative of its significance and influence in the conversation. This method
        fetches the messages related to the specified message based on their relationships, providing a list of messages
        that are most relevant to the specified message.

        Args:
            message_id (str): The ID of the message whose related messages are to be fetched.

        Returns:
            List[str]: A list of message IDs representing the messages related to the specified message.

        Proccess:
        1. Fetch the relationships of the specified message.
        2. For each relationship, extract the messages associated with it.
        3. Return a list of all the messages associated with the specified message's relationships.

        This method offers a quick way to determine the most relevant messages in the conversation based on their
        relationships with a specific message.
        """
        relationships = self.get_relationship(message_id)
        related_messages = []
        for rel in relationships.values():
            related_messages.extend(rel)
        return related_messages

    def build_tree(self, root: Optional[str] = None, depth: int = 5, k: int = 5):
        """
        Constructs the conversational tree (or directed graph) for the conversation by recursively adding related messages.

        Given a starting message (or root), this method fetches the most related messages based on their estimated
        significance and relationship strength with the root. These messages are then added to the tree as children of the
        root. This process is recursive, where each child becomes the root for its own subtree until the specified depth
        is reached or no more related messages are found.

        The tree provides a visual representation of the conversation hierarchy, showing how messages are interconnected
        and the relative significance of each message based on its position in the.tree.

        Args:
            root (str, optional): The message ID of the starting message to begin tree construction.
                                If not provided, the method determines the root using the get_root_node method.
            depth (int): The maximum depth to which the tree should be built. It limits the recursion to prevent
                        overly expansive trees. Default is 5.

        Proccess:
        1. If no root is provided, determine the root using the `get_root_node` method.
        2. If the depth is zero, terminate the recursion.
        3. Add the root node to the tree with its estimated significance.
        4. Fetch related messages for the root.
        5. Sort the related messages based on their estimated significance and relationship strength with the root.
        6. For the top k related messages:
        - If the message is not already in the tree and its relationship strength with the root is above the threshold:
            - Add the message to the tree as a child of the root.
            - Recursively build the subtree for this child.

        This approach ensures that the tree captures the most relevant messages in the conversation and presents them
        in a structured hierarchy, providing insights into the conversational flow and focal points.
        """
        if not root:
            root = self.get_root_node()

        if depth == 0:
            return

        self.tree.add_node(root, estimate=self.determine_n_neighbors(root))

        related_messages = self.get_related_messages(root)

        sorted_related_messages = sorted(
            related_messages,
            key=lambda x: (
                self.determine_n_neighbors(x),
                self.get_relationship_strength(root, x),
            ),
            reverse=True,
        )

        for msg in sorted_related_messages[:k]:
            relationship_strength = self.get_relationship_strength(root, msg)

            if (
                msg not in self.tree
                and relationship_strength > self.relationship_threshold
            ):
                self.tree.add_edge(root, msg, weight=relationship_strength)
                self.build_tree(msg, depth=depth - 1)


class EnhancedEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def semantic_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_relationship_strength(self, msg1, msg2):
        vec1 = self.get_message_embedding(msg1)
        vec2 = self.get_message_embedding(msg2)
        semantic_strength = self.semantic_similarity(vec1, vec2)
        existing_strength = super().get_relationship_strength(msg1, msg2)
        return 0.5 * (existing_strength + semantic_strength)

    def update_relationships_online(self, new_data):
        # Example method to update relationship weights dynamically
        for msg_id, interactions in new_data.items():
            for interaction, strength in interactions.items():
                if interaction in self.message_dict:
                    current_strength = self.get_relationship_strength(
                        msg_id, interaction
                    )
                    updated_strength = 0.9 * current_strength + 0.1 * strength
                    self.RELATIONSHIPS[msg_id][interaction] = updated_strength
        self._recalculate_weights()


def prepare_data(
    message_dict, relationships, variable_params, default_params, Data, torch
):
    node_features = torch.tensor(
        [msg["features"] for msg in message_dict.values()], dtype=torch.float
    )
    edge_index = (
        torch.tensor([[rel[0], rel[1]] for rel in relationships], dtype=torch.long)
        .t()
        .contiguous()
    )

    # Edge attributes based on relationship types and their parameters
    edge_attr = torch.tensor(
        [
            [
                variable_params.get(rel[2], default_params).get("weight", 0.11),
                variable_params.get(rel[2], default_params).get("importance", 1.0),
            ]
            for rel in relationships
        ],
        dtype=torch.float,
    )

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def create_torch_geometric_data(message_dict, relationships, Data, torch):
    x = torch.tensor(
        [message_dict[f"msg{i}"]["features"] for i in range(len(message_dict))],
        dtype=torch.float,
    )
    edge_index = torch.tensor(list(zip(*relationships)), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    data.train_mask = torch.tensor([0, 1], dtype=torch.bool)
    data.y = torch.tensor([0, 1], dtype=torch.long)
    return data
