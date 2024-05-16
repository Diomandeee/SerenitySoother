from typing import Dict, Tuple, Any, List, Union, Optional
from chain_tree.engine.relation import ChainRelationships
import itertools
import collections
import networkx as nx


class ChainStruct(ChainRelationships):
    def __init__(
        self,
        *args,
        kwargs,
    ):
        super().__init__(*args, kwargs)

    def _fallback_method(self, message_id: str) -> List[str]:
        """
        Returns a list of neighboring messages for the provided message ID.

        Args:
        - message_id (str): The ID of the message for which neighboring messages are sought.

        Returns:
        - List[str]: A list of neighboring message IDs.

        Workflow:
        1. Fetch Neighbors:
        Fetch the children of the given message from the `conversation_dict`.

        2. Return Result:
        Return the list of neighboring messages (children).

        Note:
        The function assumes a specific structure for `conversation_dict` where each entry is
        a tuple and the second element of the tuple ([1]) contains a list of children messages.
        """

        # Retrieve and return the children messages from the conversation_dict for the given message_id
        return self.conversation_dict[message_id][1]

    def _get_referencing_messages(self, message_id: str) -> List[str]:
        """
        Returns a list of all messages that reference the given message.

        Args:
        - message_id (str): The ID of the message for which referencing messages are sought.

        Returns:
        - List[str]: A list of message IDs that reference the provided message.

        Workflow:
        1. Validation:
        Check if the given message ID is valid, i.e., it exists in the conversation. If it doesn't, return None.

        2. Fetch Predecessors:
        For the given message ID, retrieve all its predecessors (or parents) in the conversation chain_tree.tree. In the context of a conversation tree, predecessors are those messages that reference (or point to) the current message.

        3. Return Result:
        Return the list of predecessors.
        """

        # Validate the provided message_id
        if message_id not in self.message_dict:
            return None

        # Use networkx's predecessors method to fetch referencing messages
        return [node for node in self._get_message_tree().predecessors(message_id)]

    def _get_n_hop_neighbors(
        self, message_id: str, n: int
    ) -> Union[List[Dict[str, Any]], None]:
        """
        Extracts all the neighbors of a specific message that are 'n' hops (or steps) away.

        Args:
        - message_id (str): The ID of the target message for which we need to find the n-hop neighbors.
        - n (int): Specifies how many hops or steps away the neighbors should be.

        Returns:
        - Union[List[Dict[str, Any]], None]: A list of message information dictionaries for all messages
        that are 'n' hops away from the target message. If the provided message ID does not exist
        within the conversation, the function returns None.

        Workflow:
        1. Validation:
        Before any operations, the function confirms if the provided `message_id` exists within
        the conversation. If not, it returns None, indicating the absence of the message.

        2. N-hop Neighbor Extraction:
        For an existing message, the function uses NetworkX's `single_source_shortest_path_length`
        method. This method returns a dictionary with nodes as keys and the shortest path length
        to reach them from the source node as values. By setting a `cutoff=n`, the function ensures
        it gets nodes that are exactly 'n' hops away. Thus, effectively finding the n-hop neighbors.

        """

        # Validate if the given message ID exists in the conversation
        if message_id not in self.message_dict:
            return None

        # Utilize NetworkX to identify n-hop neighbors of the provided message ID
        return nx.single_source_shortest_path_length(
            self._get_message_tree(), message_id, cutoff=n
        )

    def _get_paths_within_hop_range(
        self,
        message_id_1: str,
        message_id_2: str,
        hop_range: Union[int, Tuple[int, int]],
        return_message_info: bool = False,
    ) -> Union[List[List[Union[str, Dict[str, Any]]]], None]:
        """
        Identifies all conversation paths between two given messages that fall within a specified range of hops (or steps).

        This function probes for all possible conversation paths between two distinct messages constrained within a certain
        number of steps. Using NetworkX to traverse the conversation's graphstructure and pinpoint such paths.

        Args:
        - message_id_1 (str): The ID of the starting message.
        - message_id_2 (str): The ID of the ending message.
        - hop_range (Union[int, Tuple[int, int]]): Defines the range of hops or steps within which paths should be found.
                                                If it's an integer, it represents a fixed number of hops. If it's a tuple
                                                of two integers, it outlines the minimum and maximum number of hops.
        - return_message_info (bool, optional): If set to True, the function will return detailed information about the
                                            messages on the path, rather than just their IDs.

        Returns:
        - Union[List[List[Union[str, Dict[str, Any]]]], None]: A list of all identified paths within the specified hop range.
                                                            Each path is represented as a list, either of message IDs or
                                                            dictionaries containing message details, depending on the
                                                            `return_message_info` parameter. If no paths exist within the
                                                            specified range, the function returns None.

        Workflow:
        1. Input Validation:
        The function ensures that the provided hop_range is either an integer or a tuple of two integers. It also
        confirms the presence of the two message IDs within the conversation. If any of these conditions fail, it raises
        a ValueError.

        2. Hop Range Transformation:
        If the hop_range is provided as an integer, it's transformed into a tuple with the same integer repeated, ensuring
        consistent processing in subsequent steps.

        3. Cache Neighbors:
        To improve efficiency, the function caches the neighbors of both message IDs for every hop in the specified range.
        The neighbors for a specific number of hops are computed only once and stored for reuse.

        4. Path Discovery:
        The function then identifies common neighbors between the two messages for each hop in the range. For each common
        neighbor, it probes all possible paths from `message_id_1` to the neighbor and from the neighbor to `message_id_2`.
        These partial paths are then combined to create complete paths from `message_id_1` to `message_id_2`.

        5. Return Message Information:
        Depending on the `return_message_info` parameter, the function either appends the raw message IDs or their
        corresponding message information to the final paths.

        """

        # Input validation for hop_range and message IDs
        if not isinstance(hop_range, (int, tuple)) or (
            isinstance(hop_range, tuple)
            and not len(hop_range) == 2
            and isinstance(hop_range[0], int)
            and isinstance(hop_range[1], int)
        ):
            raise ValueError("hop_range must be an int or a tuple of two ints.")

        if (
            message_id_1 not in self.message_dict
            or message_id_2 not in self.message_dict
        ):
            raise ValueError(
                "Both message_id_1 and message_id_2 must be valid message IDs."
            )

        # Transforming hop_range if it's an integer for consistent processing
        if isinstance(hop_range, int):
            hop_range = (hop_range, hop_range)

        # Caching neighbors to avoid redundant computations
        neighbor_cache_1 = {}
        neighbor_cache_2 = {}

        paths = []
        for hops in range(hop_range[0], hop_range[1] + 1):
            # Compute and cache n-hop neighbors if not done previously
            if hops not in neighbor_cache_1:
                neighbor_cache_1[hops] = self._get_n_hop_neighbors(message_id_1, hops)
            if hops not in neighbor_cache_2:
                neighbor_cache_2[hops] = self._get_n_hop_neighbors(message_id_2, hops)

            # Identify common neighbors for the current number of hops
            common_neighbors = list(
                set(neighbor_cache_1[hops]) & set(neighbor_cache_2[hops])
            )

            # Compute paths using common neighbors and append them to the final list
            for neighbor in common_neighbors:
                paths_1 = nx.all_simple_paths(
                    self._get_message_tree(), message_id_1, neighbor, cutoff=hops
                )
                paths_2 = nx.all_simple_paths(
                    self._get_message_tree(), neighbor, message_id_2, cutoff=hops
                )
                for path_1 in paths_1:
                    for path_2 in paths_2:
                        path = path_1 + path_2[1:]  # Combine the partial paths
                        if return_message_info:
                            path = [self.message_dict[msg_id] for msg_id in path]
                        paths.append(path)

        return paths if paths else None

    def _find_connecting_chain(self, message_id_1: str, message_id_2: str) -> List[str]:
        """
        Identifies a chain of messages that serve as a connection between two specific messages within a conversation structure.

        This function searches for a sequence of messages that can link two given messages. Such chains can help in
        understanding the flow of a conversation and identifying how distinct segments of a conversation are related.

        Args:
        - message_id_1 (str): The ID of the starting message for which a connecting chain is sought.
        - message_id_2 (str): The ID of the ending message for which a connecting chain is sought.

        Returns:
        - List[str]: A list of message IDs that together form the connecting chain between the two input messages.
                    If no such chain exists, an empty list is returned.

        Workflow:
        1. Input Validation:
        Initially, the function checks if both input message IDs are present in the conversation. If not, a
        `ValueError` is raised.

        2. Find Reachable Messages:
        The function determines all messages that are reachable from `message_id_1` and all messages that can reach
        `message_id_2`. This is achieved using NetworkX's `descendants` and `ancestors` functions, respectively.

        3. Identify Common Messages:
        By finding the intersection of the two sets (reachable messages from `message_id_1` and messages that can reach
        `message_id_2`), the function identifies messages that potentially lie in the connecting chain.

        4. Select Optimal Connecting Message:
        Among the common messages, the function chooses a message that minimizes the overall length of the connecting
        chain. This is done by considering the shortest paths from `message_id_1` to the common message and from the
        common message to `message_id_2`.

        5. Construct the Chain:
        The function then constructs the connecting chain by concatenating the paths from `message_id_1` to the optimal
        connecting message and from this message to `message_id_2`. The connecting message is only included once in the
        final chain to avoid redundancy.

        """

        # Ensure both message IDs are part of the conversation structure
        if (
            message_id_1 not in self.message_dict
            or message_id_2 not in self.message_dict
        ):
            raise ValueError("Both message IDs must exist in the conversation")

        # Determine the set of messages that can be reached starting from message_id_1 and the set that can reach message_id_2
        reachable_from_1 = set(nx.descendants(self._get_message_tree(), message_id_1))
        can_reach_2 = set(nx.ancestors(self._get_message_tree(), message_id_2))

        # Identify any common messages between the two sets
        common_messages = reachable_from_1 & can_reach_2

        # If there are no common messages, there isn't a connecting chain
        if not common_messages:
            return []

        # Choose the optimal connecting message that ensures the shortest overall chain
        connecting_message = min(
            common_messages,
            key=lambda node: nx.shortest_path_length(
                self._get_message_tree(), message_id_1, node
            )
            + nx.shortest_path_length(self._get_message_tree(), node, message_id_2),
        )

        # Determine the paths to and from the connecting message
        path_to_connecting = nx.shortest_path(
            self._get_message_tree(), message_id_1, connecting_message
        )
        path_from_connecting = nx.shortest_path(
            self._get_message_tree(), connecting_message, message_id_2
        )

        # Return the concatenated paths, ensuring the connecting message isn't duplicated
        return path_to_connecting + path_from_connecting[1:]

    def _find_path_or_chain(
        self, message_id_1: str, message_id_2: str
    ) -> Optional[List[str]]:
        """
        Identifies a connection path or chain between two given messages within a defined hop range.

        This function tries to identify a connection between two specific messages. Initially, it looks for a direct path
        (defined as being within a specified number of "hops" or steps) between the messages. If such a direct path
        exists, it is returned. If not, the function then looks for a connecting chain, which may involve more
        intermediate messages than the direct path.

        The primary objective of the function is to determine how two messages are related or connected in the
        conversation structure, providing insights into the flow and context of the dialogue.

        Parameters:
        - message_id_1 (str): ID of the first message.
        - message_id_2 (str): ID of the second message.

        Returns:
        - Optional[List[str]]: A list of message IDs that make up the path or chain between the two input messages.
                            If no such path or chain exists, returns None.

        Steps:
        1. Direct Path Check:
        Using `_get_paths_within_hop_range()`, the function checks for any paths between the two messages that are
        within a predefined hop range (in this case, 2 hops).

        2. Chain Search:
        If no direct path is found, the function utilizes `_find_connecting_chain()` to search for any potential
        connecting chains that link the two messages.

        3. Return Path or Chain:
        The function then returns the identified path or chain if one exists. If neither a path nor a chain can be
        found, it returns None.

        """

        # Check for paths within the specified hop range (2 hops in this case)
        paths_within_hop_range = self._get_paths_within_hop_range(
            message_id_1, message_id_2, 2
        )

        # If a path within the hop range is found, return the first one
        if paths_within_hop_range:
            return paths_within_hop_range[0]

        # If no path is found within the hop range, look for a connecting chain
        else:
            return self._find_connecting_chain(message_id_1, message_id_2)

    def get_bifurcation_points(
        self, message_id_1: str, message_id_2: str
    ) -> Optional[List[str]]:
        """
        Determines the bifurcation points in a path or chain between two specified messages.

        Args:
        - message_id_1 (str): The ID of the starting message.
        - message_id_2 (str): The ID of the target message.

        Returns:
        - Optional[List[str]]: A list of message IDs representing bifurcation points in the path
        or chain between the two provided messages. If no such chain or path exists, the function
        returns None.

        Workflow:
        1. Finding Path or Chain:
        The function calls the helper method `_find_path_or_chain` to obtain a path or chain
        between the two specified messages. This method provides either a direct path between
        the two messages or, if such a direct path doesn't exist, a connecting chain that links
        the two messages.

        2. Identifying Bifurcation Points:
        If a path or chain is identified, the function iterates over each node in this sequence.
        For each node, it checks the number of child messages (responses). If the node has more
        than one child message, it is considered a bifurcation point and added to the list of
        bifurcation points to be returned.

        """

        # Get the path or chain between the two provided messages
        path_or_chain = self._find_path_or_chain(message_id_1, message_id_2)

        # If no path or chain is found, return None
        if path_or_chain is None:
            return None

        # Identify and return the bifurcation points in the obtained path or chain
        return [
            node for node in path_or_chain if len(self.conversation_dict[node][1]) > 1
        ]

    def _get_n_hop_neighbors_ids(
        self, message_id: str, n: int
    ) -> Union[List[str], None]:
        """
        Extracts the IDs of all neighbors of a specific message that are 'n' hops (or steps) away.

        Args:
        - message_id (str): The ID of the target message for which we need to find the n-hop neighbors.
        - n (int): Specifies how many hops or steps away the neighbors should be.

        Returns:
        - Union[List[str], None]: A list of message IDs for all messages that are 'n' hops away
        from the target message. If the provided message ID does not exist within the conversation,
        the function returns None.

        Workflow:
        1. Fetching Neighbors:
        The function initially calls `_get_n_hop_neighbors` to fetch neighbors of the given message
        that are 'n' hops away.

        2. Returning IDs:
        If the neighbors exist, it iterates through the dictionary keys (which are the message IDs)
        and compiles a list of those IDs.

        """

        # Fetch n-hop neighbors of the provided message ID
        neighbors = self._get_n_hop_neighbors(message_id, n)

        # If neighbors don't exist, return None
        if neighbors is None:
            return None

        # Extract and return only the message IDs from the neighbors dictionary
        return [neighbor for neighbor in neighbors]

    def _get_paths_between_messages(
        self, message_id_1: str, message_id_2: str, all_paths: bool = False
    ) -> Union[List[str], List[List[str]], None]:
        """
        Extracts the shortest path or all shortest paths between two specified messages within a network or graph.

        Args:
        - message_id_1 (str): The ID of the starting message.
        - message_id_2 (str): The ID of the target message.
        - all_paths (bool, optional): Determines the type of output. If set to True, the function
        returns all possible shortest paths. If False, it returns a single shortest path. Defaults to False.

        Returns:
        - Union[List[str], List[List[str]], None]: Depending on the 'all_paths' argument, the function
        either returns a single shortest path (List of message IDs), all shortest paths (List of Lists where
        each inner List represents a distinct path), or None if no path exists.

        Workflow:
        1. Determining Path Type:
        - If 'all_paths' is True: The function uses the `nx.all_shortest_paths` method from the
            NetworkX library to fetch all shortest paths between the two message IDs. It returns
            a list of lists.
        - If 'all_paths' is False: The function uses the `nx.shortest_path` method to get a
            single shortest path between the two messages.

        2. Handling No Path Scenarios:
        If the two messages are not connected (i.e., there's no path between them in the graph),
        the function captures the `nx.NetworkXNoPath` exception and returns an empty list.

        """

        # Attempt to retrieve the path or paths
        try:
            if all_paths:
                return list(
                    nx.all_shortest_paths(
                        self._get_message_tree(),
                        message_id_1,
                        message_id_2,
                    )
                )
            else:
                return nx.shortest_path(
                    self._get_message_tree(),
                    message_id_1,
                    message_id_2,
                )
        # Handle the scenario where no path exists between the two messages
        except nx.NetworkXNoPath:
            return []

    def _are_commonly_co_referenced(self, message_id_1: str, message_id_2: str) -> bool:
        """
        Checks if two given messages are commonly co-referenced.

        Args:
        - message_id_1 (str): ID of the first message.
        - message_id_2 (str): ID of the second message.

        Returns:
        - bool: True if the messages are commonly co-referenced, otherwise False.

        Workflow:
        1. Validation:
        Validates that the provided message IDs exist in the conversation.

        2. Get Referencing Messages:
        For each message ID, it fetches the list of messages that reference it.

        3. Identify Common Referencing Messages:
        Determines messages that reference both message_id_1 and message_id_2 by taking an intersection of the sets of referencing messages for each ID.

        4. Check for Commonality:
        If there's no common referencing message, the two messages aren't co-referenced and the function returns False.

        5. Select Optimal Common Reference:
        From the common references, the function identifies the message which results in the shortest combined path to both messages.

        6. Check Disjoint Paths:
        Finds paths from message_id_1 to the selected common reference and from the common reference to message_id_2. If these paths don't intersect, it implies a genuine co-reference, and the function returns True.

        7. Return Result:
        If the paths intersect, it means the common reference is a part of the direct path between the two messages, and they are not necessarily co-referenced.
        """

        # Validate the provided message_id's
        if (
            message_id_1 not in self.message_dict
            or message_id_2 not in self.message_dict
        ):
            raise ValueError("Invalid message_id(s)")

        # Fetch the referencing messages for each ID
        references_1 = set(self._get_referencing_messages(message_id_1))
        references_2 = set(self._get_referencing_messages(message_id_2))

        # Identify common referencing messages
        common_references = references_1 & references_2
        if not common_references:
            return False

        # Select the optimal common reference
        common_reference = min(
            common_references,
            key=lambda node: nx.shortest_path_length(
                self._get_message_tree(), node, message_id_1
            )
            + nx.shortest_path_length(self._get_message_tree(), node, message_id_2),
        )

        # Find paths using the common reference
        path_to_common_reference = nx.shortest_path(
            self._get_message_tree(), message_id_1, common_reference
        )
        path_from_common_reference = nx.shortest_path(
            self._get_message_tree(), common_reference, message_id_2
        )

        # Check if the paths are disjoint
        return not set(path_to_common_reference) & set(path_from_common_reference)

    def get_commonly_co_referenced_messages(
        self, message_id: str, n: int
    ) -> Dict[Tuple[str, str], int]:
        """
        Finds pairs of messages within n hops of the provided message_id that are commonly co-referenced.

        Args:
        - message_id (str): The ID of the anchor message.
        - n (int): The number of hops around the message_id to consider.

        Returns:
        - Dict[Tuple[str, str], int]: A dictionary where the keys are pairs of message IDs and the values
        indicate how frequently these pairs of messages are co-referenced.

        Workflow:
        1. Validation:
        The function starts by checking if the provided message_id exists in the conversation.
        If not, it raises a ValueError.

        2. Finding Neighbors:
        The function then identifies all the neighbors within n hops of the provided message_id.

        3. Generate Pairs:
        Using the `itertools.combinations` function, it creates pairs of message IDs from these neighbors.
        This will generate all possible unique combinations of 2 message IDs from the list of neighbors.

        4. Identify Commonly Co-referenced Pairs:
        The function then sifts through these pairs to identify which ones are commonly co-referenced
        (using an internal method `_are_commonly_co_referenced`). The output is counted using the
        `collections.Counter` function to track how many times each pair is co-referenced.

        5. Return Result:
        The function returns a dictionary of these commonly co-referenced pairs and their count.

        """

        # Validate the provided message_id
        if message_id not in self.message_dict:
            raise ValueError("Invalid message_id")

        # Get all neighbors within n hops
        neighbors = self._get_n_hop_neighbors_ids(message_id, n)
        if neighbors is None:
            return None

        # Generate all unique pairs from the list of neighbors
        pairs = list(itertools.combinations(neighbors, 2))

        # Count pairs that are commonly co-referenced
        commonly_co_referenced = collections.Counter(
            [
                tuple(sorted(pair))
                for pair in pairs
                if self._are_commonly_co_referenced(pair[0], pair[1])
            ]
        )

        return commonly_co_referenced

    def get_co_reference_chain_between_messages(
        self, message_id_1: str, message_id_2: str, n: int
    ) -> Optional[List[str]]:
        """
        Determines the co-reference chain of messages between two specified messages.

        Args:
        - message_id_1 (str): The ID of the starting message.
        - message_id_2 (str): The ID of the target message.
        - n (int): The number of co-references to be considered.

        Returns:
        - Optional[List[str]]: A list of message IDs representing the co-referenced messages in the
        path or chain between the two provided messages. If no such chain or path exists or if there
        are no co-referenced messages, the function returns None.

        Workflow:
        1. Finding Path or Chain:
        The function calls the helper method `_find_path_or_chain` to obtain a path or chain between
        the two specified messages. This method provides either a direct path between the two messages
        or, if such a direct path doesn't exist, a connecting chain that links the two messages.

        2. Getting Commonly Co-referenced Messages:
        The function uses another method `get_commonly_co_referenced_messages` (which is not provided)
        that presumably returns a list of message IDs that are commonly referred to by other messages.
        The exact mechanism on how this is determined is not specified in the provided code.

        3. Identifying Co-referenced Messages:
        If a path or chain is identified, the function iterates over each node in this sequence and checks
        if the node is in the list of commonly co-referenced messages. If it is, then it is considered as
        a co-referenced message and added to the list to be returned.
        """

        # Get the path or chain between the two provided messages
        path_or_chain = self._find_path_or_chain(message_id_1, message_id_2)

        # If no path or chain is found, return None
        if path_or_chain is None:
            return None

        # Get the commonly co-referenced messages
        commonly_co_referenced = self.get_commonly_co_referenced_messages(
            message_id_1, n
        )

        # Identify and return the co-referenced messages in the obtained path or chain
        return [node for node in path_or_chain if node in commonly_co_referenced]

    def get_merge_points(self, message_id: str, n: int) -> List[str]:
        """
        Identifies message IDs within 'n' hops from the given 'message_id' that have more than one parent.

        Args:
        - message_id (str): The ID of the base message from which we want to start our search.
        - n (int): The number of hops or steps away from the base message that we want to search.

        Returns:
        - List[str]: A list of message IDs which have more than one parent, within 'n' hops from the base message.

        Workflow:
        1. Finding Neighbors:
        The function initiates by identifying all neighbors within 'n' hops of the given 'message_id'.
        This is achieved using the `_get_n_hop_neighbors` method.

        2. Building Child-Parent Dictionary:
        A dictionary is constructed (`child_parent_dict`) where keys are child message IDs and values
        are lists of their parent message IDs. This helps to map which children have more than one parent.

        3. Identifying Merge Points:
        Iterate through the previously identified neighbors and check if they have more than one parent
        in the `child_parent_dict`. If so, they are considered as merge points.

        """

        # Get neighbors within 'n' hops from 'message_id'
        neighbors = self._get_n_hop_neighbors(message_id, n)

        # Construct a child-parent dictionary from the conversation dictionary
        child_parent_dict = {}
        for parent, (_, children) in self.conversation_dict.items():
            for child in children:
                if child in child_parent_dict:
                    child_parent_dict[child].append(parent)
                else:
                    child_parent_dict[child] = [parent]

        # Return all neighbors that have more than one parent
        return [node for node in neighbors if len(child_parent_dict.get(node, [])) > 1]

    def get_all_relationships_between_messages(
        self, message_id_1: str, message_id_2: str, return_message_info: bool = True
    ) -> List[List[Union[str, Dict[str, Any]]]]:
        """
        Finds and returns all potential paths between two messages in the conversation.

        Args:
        - message_id_1 (str): The ID of the starting message.
        - message_id_2 (str): The ID of the ending message.
        - return_message_info (bool): Flag to decide if the function should return detailed message information
        (from the message dictionary) or just the message IDs.

        Returns:
        - List[List[Union[str, Dict[str, Any]]]]: A list of paths, where each path is a list of message IDs or
        message information (based on 'return_message_info' flag).

        Workflow:
        1. Validation:
        The function starts by verifying if both provided message IDs exist in the conversation.
        If not, it raises an error.

        2. Retrieving Paths:
        The function then uses the `_get_paths_between_messages` method to fetch all potential paths
        between the two given messages.

        3. Processing Output:
        Depending on the 'return_message_info' flag, the function either:
        a) Returns the paths with detailed message information.
        b) Returns the paths with just the message IDs.

        """

        # Validate message IDs
        if (
            message_id_1 not in self.message_dict
            or message_id_2 not in self.message_dict
        ):
            raise ValueError("Invalid message_id(s)")

        # Get all paths between the two messages
        paths = self._get_paths_between_messages(
            message_id_1, message_id_2, all_paths=True
        )
        if paths is None:
            return None

        # Process the paths based on 'return_message_info' flag
        if return_message_info:
            return [[self.message_dict[msg_id] for msg_id in path] for path in paths]
        else:
            return paths

    def create_chain_representation(
        self,
        message_id_1: str,
        operation: str = None,
        message_id_2: str = None,
        n: int = None,
        return_message_info: bool = False,
        return_df: bool = False,
        return_dict: bool = False,
    ) -> Any:
        """
        Creates a chain representation of messages based on the specified operation and parameters.

        Args:
        - message_id_1 (str): The ID of the starting message for the operation.
        - operation (str, optional): The operation to be performed. It can be one of
            - "bifurcation_points"
            - "merge_points"
            - "cross_references"
            - "commonly_co_referenced"
            - "relationships_between"
        Default is None.
        - message_id_2 (str, optional): The ID of the ending message for certain operations like "relationships_between".
        - n (int, optional): The number of hops away from `message_id_1` to consider for certain operations.
        - return_message_info (bool, optional): If True, returns detailed message information. If False, returns message IDs. Default is False.
        - return_df (bool, optional): If True, returns the result as a pandas DataFrame. Default is False.
        - return_dict (bool, optional): If True, returns the result as a dictionary. Default is False.

        Returns:
        - Any: A list, DataFrame, or dictionary representation of the chain of messages based on the specified operation and parameters.

        Workflow:
        1. Setup Operation Dictionary:
        Map each operation string to the respective function that should be called.

        2. Validate Inputs:
        Ensure that the provided operation is supported, the message IDs exist, the number of hops is an integer, and the return type flags are booleans.

        3. Call the Relevant Function:
        Use the operation dictionary to call the respective function based on the provided operation string and other input arguments.

        4. Format the Result:
        Based on the flags `return_message_info`, `return_df`, and `return_dict`, format the result as a list, DataFrame, or dictionary.

        """

        operation_dict = {
            "bifurcation_points": self.get_bifurcation_points,
            "merge_points": self.get_merge_points,
            "cross_references": self.get_commonly_co_referenced_messages,
            "commonly_co_referenced": self.get_co_reference_chain_between_messages,
            "relationships_between": self.get_all_relationships_between_messages,
        }

        # Validate the operation
        if operation not in operation_dict:
            raise ValueError(
                f"Invalid operation. Valid operations are: {list(operation_dict.keys())}"
            )

        # Validate the message IDs
        if message_id_1 not in self.message_dict:
            raise ValueError("Invalid message_id_1")

        if message_id_2 is not None and message_id_2 not in self.message_dict:
            raise ValueError("Invalid message_id_2")

        # Validate the number of hops
        if n is not None and not isinstance(n, int):
            raise ValueError("n must be an int")

        # Validate the return type
        if not isinstance(return_message_info, bool):
            raise ValueError("return_message_info must be a boolean")

        if not isinstance(return_df, bool):
            raise ValueError("return_df must be a boolean")

        if not isinstance(return_dict, bool):
            raise ValueError("return_dict must be a boolean")

        # Perform the operation
        if operation == "relationships_between":
            result = operation_dict[operation](
                message_id_1, message_id_2, return_message_info=return_message_info
            )

        elif operation == "commonly_co_referenced":
            result = operation_dict[operation](
                message_id_1, message_id_2, n, return_message_info=return_message_info
            )

        else:
            result = operation_dict[operation](
                message_id_1, n, return_message_info=return_message_info
            )

        # Format the result
        if return_message_info:
            result = [self.message_dict[message_id] for message_id in result]

        if return_dict:
            result = {
                message_id: self.message_dict[message_id] for message_id in result
            }

        return result
