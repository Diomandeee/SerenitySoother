from typing import Dict, List, Optional, Tuple, Callable, Set, Any
from chain_tree.engine.handler import PhaseHandler, UnwantedHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from chain_tree.relationship.graph import GraphRepresentation
from chain_tree.infrence.generator import PromptGenerator
from chain_tree.models import ChainTree, ChainMap, Chain
from chain_tree.engine.aggregator import ChainAggregator
from chain_tree.engine.builder import ChainTreeBuilder
from sklearn.model_selection import train_test_split
from chain_tree.engine.filters import ChainFilter
from chain_tree.utils import log_handler
from tqdm import tqdm
import pandas as pd
import copy
from chain_tree.models import (
    DataParams,
    ModelParams,
    ChainParams,
    MiscParams,
    ProcessTreeOutput,
    ChainRunnerInput,
    ChainTree,
    ChainMap,
    Chain,
)
import json


class ChainMerger(ChainTreeBuilder):
    """
    TreeMerger is responsible for merging multiple conversation trees into batches,
    retrieving mappings from these trees, and updating parent-child relationships.
    """

    def __init__(
        self,
        path,
        base_persist_dir,
        api_key,
        strategy=None,
        target_num: int = 10,
        use_semantic_similarity=False,
        verbose=False,
        phrases: Optional[List[str]] = None,
        message_contains: Optional[List[str]] = None,
        exact_match=False,
        tree_range=None,
        top_k=10,
    ):

        super().__init__(
            path=path, base_persist_dir=base_persist_dir, target_num=target_num
        )

        self.chain_aggregator = ChainAggregator(
            strategy=strategy,
            use_semantic_similarity=use_semantic_similarity,
            verbose=verbose,
            phrases=phrases,
            message_contains=message_contains,
        )
        self.graphs = []
        self.verbose = verbose
        self.use_semantic_similarity = use_semantic_similarity
        self.exact_match = exact_match
        self.verbose = verbose
        self.tree_range = tree_range
        self.top_k = top_k
        self.api_key = api_key

    def get_message_indexes(
        self, keywords: List[str], start: int, end: int
    ) -> Set[int]:
        """
        Retrieves the indexes of messages that contain any of the specified keywords within their content.
        The method uses traditional string matching techniques to identify relevant messages, rather than employing methods like semantic similarity.

        Args:
            keywords (List[str]): A list containing keywords to search for within the messages.
            start (int): An integer specifying the starting index of the range of messages to consider.
            end (int): An integer marking the ending index (exclusive) of the range of messages to search through.

        Returns:
            Set[int]: A set of unique indexes corresponding to messages that contain any of the provided keywords.

        """
        return set(self.get_indexes_by_message_contains(keywords, start, end))

    def get_semantic_indexes(
        self,
        phrases: List[str],
        top_k: int,
        start: int,
        end: int,
    ) -> Set[int]:
        """
        Retrieves the indexes of titles that semantically match any of the specified phrases.
        The method uses semantic similarity techniques to identify relevant titles, rather than employing traditional string matching.

        Args:
            phrases (List[str]): A list containing phrases to search for within the titles.
            top_k (int): An integer specifying the number of top semantically similar results to return for each phrase.
            start (int): An integer specifying the starting index of the range of titles to consider.
            end (int): An integer marking the ending index (exclusive) of the range of titles to search through.

        Returns:
            Set[int]: A set of unique indexes corresponding to titles that match any of the provided phrases based on semantic similarity.

        """

        # Retrieving the list of titles within the specified range
        titles = self.get_titles()[start:end]

        # Using 'get_indexes_by_phrases_similarity' to find semantically similar title indexes and ensuring their uniqueness
        return set(
            self.chain_aggregator.get_indexes_by_phrases_similarity(
                titles,
                phrases,
                top_k,
                start,
            )
        )

    def get_indexes(
        self,
        phrases: List[str],
        start: int = 0,
        end: int = None,
        index_type: str = "message",
        top_k: Optional[int] = None,
        exact_match: Optional[bool] = None,
    ) -> Set[int]:
        """
        General method to get indexes based on different search strategies.

        Args:
            phrases (List[str]): Phrases or keywords to search for.
            start (int): Start index for search.
            end (int): End index for search.
            index_type (str): Type of indexes to retrieve ('message', 'semantic', 'traditional').
            top_k (Optional[int]): Number of top semantic matches to consider (only for 'semantic').
            exact_match (Optional[bool]): Whether to match phrases exactly (only for 'traditional').

        Returns:
            Set[int]: Set of indexes that match the search criteria.
        """

        if end is None:
            end = len(self.conversations)

        if index_type == "message":
            return self.get_message_indexes(keywords=phrases, start=start, end=end)
        elif index_type == "semantic":
            if top_k is None:
                raise ValueError("top_k must be provided for semantic indexes.")
            return self.get_semantic_indexes(
                phrases=phrases, top_k=top_k, start=start, end=end
            )
        elif index_type == "traditional":
            if exact_match is None:
                raise ValueError(
                    "exact_match must be provided for traditional indexes."
                )
            return self.get_traditional_indexes(
                phrases=phrases, start=start, end=end, exact_match=exact_match
            )
        else:
            raise ValueError(f"Invalid index_type provided: {index_type}")

    def _filter_conversation_trees(
        self, start=None, end=None, skip_indexes=None, phrase_indexes=None
    ):
        """
        Refines the set of conversation trees based on a combination of conditions
        like specific indices to include (those that match certain phrases) or exclude, as well as a given range.

        Args:
            start (int, optional): An integer representing the beginning index of the range of conversation trees to consider. If None, the beginning of the list is assumed. Defaults to None.
            end (int, optional): An integer representing the terminal index (exclusive) of the range of conversation trees to consider. If None, the end of the list is assumed. Defaults to None.
            skip_indexes (List[int], optional): A list of integer indexes pointing to conversation trees that must be omitted from the outp
            phrase_indexes (List[int], optional): A list of integer indexes pointing to conversation trees that should be specifically included based on some phrase matching criteria.

        Returns:
            List: A list containing the filtered conversation trees that meet the given conditions.

        """

        # Input validation for 'start' and 'end'
        if start is not None and not isinstance(start, int):
            raise TypeError("Start index must be an integer or None")
        if end is not None and not isinstance(end, int):
            raise TypeError("End index must be an integer or None")

        # Logging the initial filtering range
        log_handler(
            f"Filtering conversation trees from {start} to {end}...",
            verbose=self.verbose,
        )

        # If phrase_indexes are given, they take precedence over the range
        if phrase_indexes is not None:
            start = None

        # Determining which conversation trees are valid based on provided conditions
        if phrase_indexes is not None:
            valid_indexes = set(phrase_indexes)
        elif skip_indexes is not None:
            valid_indexes = set(range(start, end)) - set(skip_indexes)
        else:
            valid_indexes = set(range(start, end))

        # Filtering the conversation trees based on the 'valid_indexes'
        filtered_trees = [
            ct
            for i, ct in enumerate(self.conversations[start:end])
            if i in valid_indexes
        ]

        # Logging the number of found valid conversation trees
        log_handler(
            f"Found {len(filtered_trees)} conversation trees that match the given criteria.",
            verbose=self.verbose,
        )

        # Exception handling for the scenario where phrase-based filtering yields no results
        if phrase_indexes is not None and len(filtered_trees) == 0:
            raise ValueError(
                "No conversations found that contain the given phrases within the specified range"
            )

        return filtered_trees

    def process_and_filter_trees(
        self,
        top_k: int,
        tree_range: Tuple[int, int],
        skip_indexes: List[int],
        phrases: Optional[List[str]] = None,
        message_contains: Optional[List[str]] = None,
    ):
        """
        Process and filter conversation trees based on a range of criteria. This function allows for
        refined tree selection based on semantic similarity, traditional phrase matching, and specific
        message content.

        Args:
            top_k (int): The top K trees to consider based on the given matching criteria.
            tree_range (Tuple[int, int]): Specifies the range of tree indexes to be considered.
            skip_indexes (List[int]): A list of tree indexes to be explicitly excluded from consideration.
            phrases (Optional[List[str]]): A list of phrases. Trees containing these phrases are considered
                                        for inclusion.
            message_contains (Optional[List[str]]): A list of keywords. Trees containing messages with these
                                                keywords are selected.

        Returns:
            Tuple: A tuple containing a list of conversation trees that match the criteria and the starting
                index for the processed range.

        Notes:
            - When both 'phrases' and 'message_contains' are specified, the function will prioritize phrase-based matching.
            - Semantic similarity allows the function to identify trees that might not contain the exact specified
            phrases but are contextually relevant.
            - The use of exact or semantic matching can be determined by the instance variables 'use_semantic_similarity'
            and 'strategy'.
            - If neither phrases nor message content keywords are specified, the function returns trees based on the
            range and skip_indexes.
        """

        # Step 1: Validate and Process the provided parameters
        start, end = self._process_tree_range(tree_range)
        log_handler(
            f"Processing trees from {start} to {end}...",
            verbose=self.verbose,
        )

        # Initializing containers for identified tree indexes
        indexes = None
        traditional_indexes = set()
        semantic_indexes = set()

        # Step 2: Identify matching tree indexes based on the provided criteria
        if phrases is not None:
            # Check for semantic similarity
            if self.use_semantic_similarity:
                semantic_indexes = self.get_semantic_indexes(
                    phrases,
                    top_k,
                    start,
                    end,
                )

            # Check for traditional matching or if a strategy has been set
            if (
                not self.use_semantic_similarity
                or self.chain_aggregator.strategy is not None
            ):
                traditional_indexes = self.get_traditional_indexes(
                    phrases, start, end, self.exact_match
                )

            indexes = self.chain_aggregator.apply_strategy(
                traditional_indexes, semantic_indexes
            )

        # Check for keywords in message content
        elif message_contains is not None:
            message_indexes = self.get_message_indexes(message_contains, start, end)
            log_handler(
                f"Found {len(message_indexes)} indexes that match the keywords {message_contains} in message content.",
                verbose=self.verbose,
            )

            indexes = message_indexes

            if message_indexes is not None:
                start = None

        # Filter trees based on provided skip_indexes
        elif skip_indexes is not None:
            indexes = set(range(start, end)) - set(skip_indexes)

        else:
            indexes = set(range(start, end))

        # Step 3: Filter trees based on the identified tree indexes
        result = self._filter_conversation_trees(start, end, skip_indexes, indexes)

        return result, start

    def retrieve_mappings(
        self,
        conversation_trees: List[ChainTree] = None,
        verbose=True,
        start=None,
        end=None,
        batch_size=10,
        skip_indexes=None,
        phrases=None,
        message_contains=None,
        tree_range=None,
        top_k=10,
        **kwargs,
    ) -> Tuple[List[ChainMap], pd.DataFrame, pd.DataFrame]:

        
        filtered_trees = None

        if start is None:
            start = 0

        if conversation_trees is None:
            conversation_trees = self.conversations

        else:
        # Step 2: Determine the tree range and filter
            tree_range = (
                (start, end) if start is not None and end is not None else self.tree_range
            )

            # filter using process_and_filter_trees
            filtered_trees, start = self.process_and_filter_trees(
                top_k,
                tree_range,
                skip_indexes=skip_indexes,
                phrases=phrases,
                message_contains=message_contains,
            )

            conversation_trees = filtered_trees

        if not conversation_trees:
            log_handler(
                "No conversation trees found to process.",
                verbose=verbose,
            )
            return [], pd.DataFrame(), pd.DataFrame()

        total_trees = len(conversation_trees)
        log_handler(
            f"Processing {total_trees} conversation trees in batches...",
            verbose=verbose,
        )

        mappings = []
        main_dfs = []
        relationship_dfs = []

        def process_batch(trees):
            batch_results = []
            local_main_dfs = []
            local_relationship_dfs = []

            for conversation_tree in trees:
                graph = GraphRepresentation(
                    conversation_tree=conversation_tree, api_key=self.api_key
                )
                chain_filter = ChainFilter(
                    message_range=kwargs.get("message_range", None),
                    depth_range=kwargs.get("depth_range", None),
                    date_range=kwargs.get("date_range", None),
                    keyword_filter=kwargs.get("keyword_filter", None),
                    range_filter=kwargs.get("range_filter", None),
                    custom_filter=kwargs.get("custom_filter", None),
                )

                if not chain_filter.is_valid(
                    start, total_trees, conversation_tree, graph.depth
                ):
                    continue

                # add try except block
                try:
                    model, main_df, relationship_df, message_dict = (
                        graph.process_coordinates_graph(
                            animate=kwargs.get("animate", False),
                            local_embeddings=kwargs.get("local_embeddings", False),
                            base_path=kwargs.get("base_path", "Mega"),
                        )
                    )
                except Exception as e:
                    print(f"Error processing tree {conversation_tree.index}: {e}")
                    continue

                self.graphs.append(graph)
                if isinstance(main_df, pd.DataFrame):
                    local_main_dfs.append(main_df)
                if isinstance(relationship_df, pd.DataFrame):
                    local_relationship_dfs.append(relationship_df)
                batch_results.extend(list(message_dict.values()))
            return batch_results, local_main_dfs, local_relationship_dfs

        # Creating batches
        batches = [
            conversation_trees[i : i + batch_size]
            for i in range(0, total_trees, batch_size)
        ]

        # Use ThreadPoolExecutor with a specific number of threads
        num_threads = min(len(batches), 10) if len(batches) > 1 else 1
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                
                try:
                    batch_results, local_main_dfs, local_relationship_dfs = future.result()
                    mappings.extend(batch_results)
                    main_dfs.extend(local_main_dfs)
                    relationship_dfs.extend(local_relationship_dfs)
                except Exception as e:
                    print(e)
                    continue
        # Concatenate the main_df and relationship_df
        if main_dfs:
            combined_df = pd.concat(main_dfs, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        if relationship_dfs:
            combined_relationship_df = pd.concat(relationship_dfs, ignore_index=True)
        else:
            combined_relationship_df = pd.DataFrame()

        log_handler(
            f"Processed {len(mappings)} mappings from {total_trees} conversation trees.",
            verbose=verbose,
        )

        log_handler(
            f"Processed {len(combined_df)} messages and {len(combined_relationship_df)} relationships.",
            verbose=verbose,
        )

        log_handler(
            f"Processed {len(self.graphs)} graphs.",
            verbose=verbose,
        )

        # save the combined_df
        combined_df.to_csv("combined_df.csv")

        # save the relationship trees
        combined_relationship_df.to_csv("combined_relationship_df.csv")

        return mappings, combined_df, combined_relationship_df, conversation_trees

    def update_parent_child(self, mappings: List[ChainMap]) -> Dict[str, str]:
        """
        Update the parent-child relationships in the mappings and return a dictionary
        of new mapping IDs.

        This method traverses a list of ChainMap objects and updates the parent-child
        relationships based on the 'parent' and 'children' attributes in each ChainMap.
        It generates a new set of mapping IDs and a dictionary that associates each
        parent message ID with its children message IDs.

        Parameters:
            mappings (List[ChainMap]): A list of ChainMap objects containing conversation
            mappings.

        Returns:
            Dict[str, str]: A dictionary where each key-value pair represents a message
            ID and its new mapping ID.
        """
        print("Creating new IDs for mappings...")

        # If mappings is None or empty, return an empty dictionary
        if not mappings:
            return {}

        new_mapping_ids = {}
        parent_child_map = {}

        for mapping in tqdm(mappings):
            if mapping.message is not None:
                # Still retain the message ID mapping, as you did before
                new_mapping_ids[mapping.message.id] = mapping.message.id

                # Check for parent and establish a parent-child relationship
                parent_id = mapping.parent
                if parent_id:
                    # Store children IDs in a list against their parent
                    if parent_id not in parent_child_map:
                        parent_child_map[parent_id] = []
                    parent_child_map[parent_id].append(mapping.message.id)

        # Now, update the children information for each mapping based on the parent_child_map
        for mapping in mappings:
            if mapping.message and mapping.message.id in parent_child_map:
                mapping.children = parent_child_map[mapping.message.id]

        return new_mapping_ids

    def extract_and_sort_messages(
        self, mappings: List[ChainMap], new_mapping_ids: Dict[str, str]
    ) -> List[Chain]:
        """
        Extract and sort the messages based on their creation time.

        This method traverses a list of ChainMap objects and extracts each message
        object. It then updates the ID of each message with its new mapping ID and sorts
        the messages by their creation time.

        Parameters:
            mappings (List[ChainMap]): A list of ChainMap objects containing conversation
            mappings.

            new_mapping_ids (Dict[str, str]): A dictionary of new mapping IDs, where each
            key-value pair represents a message ID and its new mapping ID.

        Returns:
            List[Chain]: A list of Chain objects, representing messages, sorted by their
            creation time.
        """
        print("Extracting and sorting messages...")
        sorted_messages = []

        for mapping in tqdm(mappings):
            if mapping.message is not None:
                mapping.message.id = new_mapping_ids[mapping.message.id]
                sorted_messages.append(mapping.message)

        # Sort the messages based on their creation time
        sorted_messages.sort(key=lambda m: (m.create_time is None, m.create_time))

        return sorted_messages

    def create_linked_list(self, sorted_messages: List[Chain]) -> List[Chain]:
        """
        Create a doubly-linked list of sorted messages.

        This method iterates through a list of sorted messages (Chain objects) and
        updates the 'prev' and 'next' attributes for each message to establish a
        doubly-linked list.

        Parameters:
            sorted_messages (List[Chain]): A list of Chain objects, representing messages,
            sorted by their creation time.

        Returns:
            List[Chain]: A list of Chain objects, now organized as a doubly-linked list.
        """
        print("Creating linked list...")
        id_mapping = {}
        for i, message in tqdm(enumerate(sorted_messages)):
            # For each message, determine its previous and next based on its position in the sorted list
            message.prev = sorted_messages[i - 1].id if i > 0 else None
            message.next = (
                sorted_messages[i + 1].id if i < len(sorted_messages) - 1 else None
            )
            id_mapping[message.id] = message.id
        return sorted_messages

    def update_mappings(
        self, sorted_messages: List[Chain], conversation_trees: List[ChainTree]
    ) -> List[ChainMap]:
        """
        Update existing mappings or create new ones for sorted messages.

        This method iterates through a list of sorted messages and a list of conversation trees.
        It updates the existing mappings with new message information or creates new mappings
        if they do not already exist. If a message is by the system and is a prompt, it also
        updates the message content and creation time.

        Parameters:
            sorted_messages (List[Chain]): A list of Chain objects, representing sorted messages.

            conversation_trees (List[ChainTree]): A list of ChainTree objects representing
            the structure of conversation trees.

        Returns:
            List[ChainMap]: A list of updated ChainMap objects containing the new mappings.
        """
        print("Updating mappings...")

        combined_mappings = []

        # Create a message_id to ChainMap dictionary for quick look-up
        existing_mappings = {
            mapping.message.id: mapping
            for tree in conversation_trees
            for mapping in tree.mapping.values()
            if mapping.message is not None
        }

        # Initialize previous message variable
        prev_message = None

        for message in tqdm(sorted_messages):
            if message.id in existing_mappings:
                mapping = existing_mappings[message.id]
                mapping.message = message
            else:
                mapping = ChainMap(id=message.id, message=message)

            # Check if message is by system
            if message.author.role == "system":
                related_conversation = None
                for index, conv in enumerate(conversation_trees):
                    if conv.mapping.get(message.id):
                        related_conversation = conv
                        break

                if related_conversation:
                    # If message is a prompt, update the message content
                    message.content.text = (
                        f"Conversation {index + 1}: {related_conversation.title}"
                    )
                    message.content.parts = [message.content.text]
                    message.create_time = related_conversation.create_time

                if prev_message:
                    mapping.parent = prev_message.id
                    prev_mapping = existing_mappings.get(
                        prev_message.id,
                        ChainMap(id=prev_message.id, message=prev_message),
                    )
                    if prev_mapping.children:
                        prev_mapping.children.append(message.id)
                    else:
                        prev_mapping.children = [message.id]

                else:
                    mapping.parent = None

            else:
                # # connect the message to the previous message as a child
                mapping.parent = prev_message.id if prev_message else None

                # connect the previous message to the current message
                if prev_message:
                    prev_mapping = existing_mappings.get(
                        prev_message.id,
                        ChainMap(id=prev_message.id, message=prev_message),
                    )
                    if prev_mapping.children:
                        prev_mapping.children.append(message.id)
                    else:
                        prev_mapping.children = [message.id]

            combined_mappings.append(mapping)
            prev_message = message

        return combined_mappings

    def merge_conversations(
        self,
        filtered_trees: List[ChainTree] = None,
        title: str = "Combined Conversation",
        skip_indexes: Optional[List[int]] = None,
        batch_size=10,
        **kwargs,
    ) -> ChainTree:
        """
        Combine multiple conversation trees into a single conversation tree.

        This method retrieves the mappings from each of the filtered conversation trees
        and performs various operations like updating parent-child relationships, extracting and sorting
        messages, creating linked lists, and finally combining all these into a new single conversation tree.

        Steps involved:
        1. Retrieve mappings from the filtered conversation trees.
        2. Update the parent-child relationships in the mappings.
        3. Extract and sort the messages based on their creation time.
        4. Create a linked list from the sorted messages.
        5. Update the mappings with the new message information.
        6. Create a new combined conversation tree.

        Parameters:
            filtered_trees (List[ChainTree]): A list of ChainTree objects, each representing
                a filtered conversation tree.

            title (str, optional): The title to be assigned to the combined conversation. Defaults to 'Combined Conversation'.

        Returns:
            ChainTree: A ChainTree object representing the combined conversation tree.

        Raises:
            Exception: Any exception that might occur during the process will be caught and printed.

        """

        try:

            mappings, main_df, relationship_df, conversation_trees = (
                self.retrieve_mappings(
                    filtered_trees,
                    batch_size=batch_size,
                    skip_indexes=skip_indexes,
                    **kwargs,
                )
            )
            new_mapping_ids = self.update_parent_child(mappings)
            sorted_messages = self.extract_and_sort_messages(mappings, new_mapping_ids)
            sorted_messages = self.create_linked_list(sorted_messages)
            child_messages = Chain.flatten_all_chain_trees(sorted_messages)
            combined_mappings = self.update_mappings(
                sorted_messages, conversation_trees
            )

            print("Creating combined conversation...")
            # convert the combined mappings to a dictionary
            combined_mappings = {mapping.id: mapping for mapping in combined_mappings}
            # sort the combined mappings by create_time
            combined_mappings = dict(
                sorted(
                    combined_mappings.items(),
                    key=lambda item: item[1].message.create_time,
                )
            )

            combined_conversation = ChainTree(
                title=title,
                create_time=sorted_messages[0].create_time,
                update_time=sorted_messages[-1].create_time,
                mapping=combined_mappings,
                moderation_results=[],
                current_node=sorted_messages[-1].id,
            )
            # convert the combined tree to a dictionary

            # save the combined conversation tree
            combined_conversation.save("combined_conversation.json")

            return (
                combined_conversation,
                combined_mappings,
                child_messages,
                main_df,
                relationship_df,
            )

        except Exception as e:
            print(e)
            return None


class ChainMatrix(ChainMerger):
    def __init__(
        self,
        base_persist_dir: str,
        path: str,
        target_num: int,
        initial_unwanted_phrases: List[str] = None,
        input_data: ProcessTreeOutput = None,
        generate_data: ChainRunnerInput = None,
    ):
        """
        Initialize a Matrix obje

        Args:
            base_persist_dir (str): The base directory for persisting data.
            path (str): The path to the data file.
            input_data (Optional[ProcessTreeOutput], optional): The input data for the matrix. Defaults to None.
            graphs (Optional[List[GraphRepresentation]], optional): The list of graph representations. Defaults to None.
            conversations_df (Optional[pd.DataFrame], optional): The dataframe containing conversations data. Defaults to None.
            initial_unwanted_phrases (List[str], optional): The list of initial unwanted phrases. Defaults to None.
            build_graphs (bool, optional): Whether to build graphs. Defaults to False.
            build (bool, optional): Whether to build the matrix. Defaults to False.
            skip_trees (bool, optional): Whether to skip processing trees. Defaults to True.
            tree_range (Tuple[int, int], optional): The range of trees to process. Defaults to (0, None).
        """

        super().__init__(
            path=path, base_persist_dir=base_persist_dir, target_num=target_num
        )
        self.conversations_data = None
        self.data_processed = False

        self.message_map = self.create_message_map(format="df", skip_format=True)

        self.original_conversations_df = copy.deepcopy(self.message_map)
        self.unwanted_phrases = (
            initial_unwanted_phrases if initial_unwanted_phrases else []
        )
        # Initialize our handlers
        self.continue_handler = PhaseHandler()
        self.unwanted_response_handler = UnwantedHandler(self.unwanted_phrases)

        self.prompt_generator = PromptGenerator(
            **self.generate_data.generator_params,
            **self.generate_data.data_params,
            cloud=True,
        )

        # Initialize the counters
        self.unwanted_phrase_count = 0
        self.continue_count = 0

    def reset_data(self):
        """
        Reset the conversations_df to its original state and reset the processing flag.

        This method sets `conversations_df` back to its original state by deep copying `original_conversations_df`
        and resets the `data_processed` flag to False.
        """
        self.conversations_df = copy.deepcopy(self.original_conversations_df)
        self.data_processed = False

    def add_unwanted_phrase(self, phrase: str):
        """
        Add a new unwanted phrase to the list.

        Parameters:
            phrase (str): The phrase to add to the unwanted_phrases list.
        """
        if phrase not in self.unwanted_phrases:
            self.unwanted_phrases.append(phrase)

    def remove_unwanted_phrase(self, phrase: str):
        """
        Remove an unwanted phrase from the list.

        Parameters:
            phrase (str): The phrase to remove from the unwanted_phrases list.
        """
        if phrase in self.unwanted_phrases:
            self.unwanted_phrases.remove(phrase)

    def update_unwanted_phrases(self, new_phrases: List[str]):
        """
        Update the list of unwanted phrases.

        Parameters:
            new_phrases (List[str]): The new list of phrases to set or merge with the existing list.

        Note:
            This method replaces the entire existing list with the new list.
        """
        self.unwanted_phrases = new_phrases

    def handle_continue_responses(self, continue_pairs: List[Dict[str, int]]) -> None:
        """
        Handle 'continue' scenarios using the handler.

        Parameters:
            continue_pairs (List[Dict[str, int]]): A list of dictionaries with pairs of message IDs that fall under 'continue' scenarios.
        """
        self.continue_handler.handle(self.conversations_df, continue_pairs)

    def replace_unwanted_responses(self, message_pairs: List[Dict[str, int]]) -> None:
        """
        Replace unwanted responses using the handler.

        Parameters:
            message_pairs (List[Dict[str, int]]): A list of dictionaries with pairs of message IDs where unwanted phrases were used.
        """
        self.unwanted_response_handler.handle(self.conversations_df, message_pairs)

    def identify_continue_scenarios(self, user_phase: str) -> List[Dict[str, int]]:
        """
        Identify and count 'continue' scenarios based on the user's phase or query.

        Parameters:
            user_phase (str): The phase or query from the user that needs to be checked for 'continue' scenarios.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with pairs of message IDs that fall under 'continue' scenarios.
        """
        continue_pairs = self.continue_handler.identify(
            self.conversations_df, user_phase
        )
        self.continue_count += len(continue_pairs)  # Update the counter
        return continue_pairs

    def identify_unwanted_responses(
        self, unwanted_phrases: List[str]
    ) -> List[Dict[str, int]]:
        """
        Identify unwanted responses based on a list of unwanted phrases.

        This method updates the list of unwanted phrases, then identifies any message pairs
        where the assistant's response contains an unwanted phrase.

        Parameters:
            unwanted_phrases (List[str]): A list of phrases that are considered unwanted in the assistant's responses.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with pairs of message IDs where unwanted phrases were used.
            The keys are "assistant_message_id" and "user_message_id".
        """
        self.update_unwanted_phrases(unwanted_phrases)
        message_pairs = self.unwanted_response_handler.identify(self.conversations_df)
        self.unwanted_phrase_count += len(message_pairs)  # Update the counter
        return message_pairs

    def process_conversation(self, user_phase: str) -> None:
        """
        Process the conversation data to handle 'continue' scenarios and replace unwanted responses.

        This method goes through a pipeline of:
            1. Identifying "continue" scenarios based on the user's phase.
            2. Handling the identified "continue" scenarios.
            3. Identifying unwanted responses.
            4. Replacing unwanted responses.

        The method sets a flag to mark the data as processed upon completion.

        Parameters:
            user_phase (str): The phase or query from the user that needs to be checked for 'continue' scenarios.
        """
        # Check if data has been processed already
        if self.data_processed:
            return

        # 1. Identify "continue" scenarios
        continue_pairs = self.identify_continue_scenarios(user_phase)

        # 2. Handle the identified "continue" scenarios
        self.handle_continue_responses(continue_pairs)

        # 3. Identify unwanted responses after handling "continue" scenarios
        message_pairs = self.identify_unwanted_responses(self.unwanted_phrases)

        # 4. Replace unwanted responses
        self.replace_unwanted_responses(message_pairs)

        # Mark the data as processed
        self.data_processed = True

    def _add_instructions(
        self,
        conversations_df: pd.DataFrame,
        instruction_params: Dict[str, str],
        potential_phrases: List[str],
        phase: str = "continue",
    ):
        """
        Add instructions to the formatted text in the conversation DataFrame based on certain conditions.

        This method modifies the conversation DataFrame in-place by adding an instruction message
        either before or after the original text of messages. It adds a 'USER_CONTINUE_INSTRUCTION'
        if the 'continue' phrase appears in a user's text and an 'ASSISTANT_UNWANTED_INSTRUCTION'
        if any unwanted phrase appears in the assistant's text.

        Parameters:
            conversations_df (pd.DataFrame): The DataFrame containing the conversation data.
                This DataFrame should have at least the columns "formatted_text", "text", and "author".

            instruction_params (Dict[str, str]): A dictionary containing the instruction messages to be added.
                Should contain keys "USER_CONTINUE_INSTRUCTION" and "ASSISTANT_UNWANTED_INSTRUCTION"
                along with a "REVERSE_INSTRUCTION_PLACEMENT" flag indicating whether the instruction
                should be placed before or after the original text.

            potential_phrases (List[str]): A list of phrases that are considered unwanted in the assistant's responses.
        """

        def modify_text(row, instruction_key, condition_key=None):
            instruction = instruction_params[instruction_key]
            reverse_placement = instruction_params["REVERSE_INSTRUCTION_PLACEMENT"]
            if reverse_placement:
                return instruction + row["formatted_text"]
            else:
                return row["formatted_text"] + instruction

        conversations_df["formatted_text"] = conversations_df.apply(
            lambda row: (
                modify_text(row, "USER_CONTINUE_INSTRUCTION")
                if phase in row["text"].lower() and row["author"] == "user"
                else row["formatted_text"]
            ),
            axis=1,
        )
        conversations_df["formatted_text"] = conversations_df.apply(
            lambda row: (
                modify_text(row, "ASSISTANT_UNWANTED_INSTRUCTION")
                if any(
                    phrase.lower() in row["text"].lower()
                    for phrase in potential_phrases
                )
                and row["author"] == "assistant"
                else row["formatted_text"]
            ),
            axis=1,
        )

    def prepare_conversation_data(
        self,
        user_phase: str = "",
        test_size: float = 0.1,
        regenerate: bool = False,
        instruction_params: Dict[str, str] = None,
        use_instruction: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare the conversation data for training or validation.

        This method processes the conversation DataFrame, optionally adds instructions, regenerates prompts,
        and then splits the data into training and validation sets.

        Parameters:
            user_phase (str, optional): The specific phase or context of user interactions that need to be processed.
                Default is an empty string, which means no specific phase is considered.

            test_size (float, optional): The proportion of the dataset to be used as the validation set.
                Should be between 0 and 1. Default is 0.1.

            regenerate (bool, optional): Flag to indicate whether to regenerate the assistant's responses
                that include unwanted phrases. Default is False.

            instruction_params (Dict[str, str], optional): A dictionary containing instruction messages
                that can be added to the text in the DataFrame.
                Default is a set of placeholder instructions.

            use_instruction (bool, optional): Flag to indicate whether to add instructions to the DataFrame
                based on the conditions. Default is False.

        Returns:
            Tuple[List[str], List[str]]: Returns a tuple containing the list of training texts and the list
                of validation texts.

        Raises:
            ValueError: If test_size is not between 0 and 1.
        """

        if not (0 <= test_size <= 1):
            raise ValueError("test_size should be between 0 and 1.")

        if instruction_params is None:
            instruction_params = {
                "USER_CONTINUE_INSTRUCTION": "Your_Instruction_Here",
                "ASSISTANT_UNWANTED_INSTRUCTION": "Your_Other_Instruction_Here",
                "REVERSE_INSTRUCTION_PLACEMENT": True,
            }

        if not regenerate:
            self.process_conversation(user_phase)

        # Step 1: Extract Conversations
        conversations_df = self.conversations_df

        # Step 2: Preprocess Conversations
        conversations_df["formatted_text"] = (
            conversations_df["author"] + ": " + conversations_df["text"]
        )
        conversations_df["formatted_text"] = conversations_df[
            "formatted_text"
        ].str.replace(": ", ": \n\n")

        # Optional: Add instructions
        if use_instruction:
            self._add_instructions(
                conversations_df, instruction_params, self.unwanted_phrases
            )

        # Step 3: Combine Conversations
        grouped_conversations = (
            conversations_df.groupby("title")["formatted_text"]
            .apply(lambda x: "\n\n".join(x))
            .reset_index()
        )

        total_prompts = 0
        potential_phrases = self.unwanted_phrases

        for _, row in grouped_conversations.iterrows():
            convo_texts = row["formatted_text"].split("\n")
            total_prompts += sum(
                1
                for text in convo_texts
                if text.startswith("assistant:")
                and any(phrase in text for phrase in potential_phrases)
            )

        print(f"Total number of prompts to be regenerated: {total_prompts}")

        # Now proceed with regeneration.
        if (
            regenerate and total_prompts > 0
        ):  # Only proceed if there are prompts to regenerate
            with tqdm.tqdm(total=total_prompts, desc="Regenerating Prompts") as pbar:
                for _, row in grouped_conversations.iterrows():
                    convo_texts = row["formatted_text"].split("\n")
                    for i, text in enumerate(convo_texts):
                        if text.startswith("assistant:"):
                            if any(phrase in text for phrase in potential_phrases):
                                pbar.update(1)  # Update tqdm progress bar
                                prompt = convo_texts[i - 1] if i - 1 >= 0 else ""
                                new_response = (
                                    self.proccesor.prompt_generator.generate_prompt(
                                        prompt=prompt,
                                        response=text.split("assistant:")[-1],
                                    )
                                )
                                convo_texts[i] = "assistant: " + new_response[0]

                    row["formatted_text"] = "\n".join(convo_texts)

        # Step 5 (or Step 4 if not regenerating): Split Data
        train_texts, val_texts = train_test_split(
            grouped_conversations["formatted_text"].tolist(), test_size=test_size
        )

        # Save the conversation data to CSV
        self.conversations_data = grouped_conversations

        self.conversations_data.to_csv("conversations_data.csv", index=False)

        return train_texts, val_texts
