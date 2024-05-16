from typing import Optional, Dict, Any, List, Union, Set, Tuple
from chain_tree.models import ChainMap, ChainTree
from chain_tree.infrence.manager import Element
from chain_tree.utils import load_json
from collections import Counter
import pandas as pd
import numpy as np
import json
import os
import re


class ChainTreeBuilder:
    FILE_PATTERN_VARIABLE_MAP = {
        "**/conversation_chain_tree.json": "data",
        "**/coordinate_chain_tree.json": "coordinate_tree",
        "**/main_df.csv": "main_df",
        "**/relationship.csv": "relationship",
        "**/distance_df.csv": "distance_df",
        "**/similarity_df.csv": "similarity_df",
    }

    """
    Builds a chain tree from conversation data, possibly merging multiple trees.

    Attributes:
        base_persist_dir (str): The base directory where data will be saved.
        path (Optional[str]): The path to the JSON file containing the data.
        key (Optional[str]): The key to be used for building a dictionary of ChainTrees. Default is "title".
        data (Union[Dict, List[Dict], None]): The data for building the Chainchain_tree.tree.
        target_num (Optional[int]): The target number of messages to include in the chain_tree.tree. Default is 6.
        combine (bool): Whether to combine conversations or not.
        less_than_target (list): List of ChainTrees that have fewer than `target_num` messages.
        conversations (list): The built ChainTrees.

    Methods:
        create_conversation_trees: Creates a list of ChainTrees from the data.
        combine_conversations: Optionally combines the conversations.
    """

    def __init__(
        self,
        base_persist_dir: str,
        path: Optional[str] = None,
        key: Optional[str] = "title",
        data: Union[Dict, List[Dict], None] = None,
        target_num: Optional[int] = 0,
        pattern_variable_map: Optional[Dict] = None,
        combine: bool = False,
        database_uri: Optional[str] = None,
        range_index: Optional[Tuple[int, Optional[int]]] = None,
    ):
        self.database_uri = database_uri
        self.base_persist_dir = base_persist_dir
        self.key = key
        self.target_num = target_num
        self.combine = combine
        self.range_index = range_index
        self.less_than_target = []
        self.manager = Element(self.base_persist_dir)
        if pattern_variable_map:
            self.FILE_PATTERN_VARIABLE_MAP.update(pattern_variable_map)

        try:
            if path or data:
                self.path = (
                    None
                    if data
                    else (
                        path
                        if os.path.isabs(path)
                        else os.path.join(self.base_persist_dir, path)
                    )
                )
                if data:
                    self.data = data

                elif self.path:
                    self.data = load_json(self.path)
            else:
                # Dynamically assign instance variables based on file patterns
                for (
                    file_pattern,
                    variable_name,
                ) in self.FILE_PATTERN_VARIABLE_MAP.items():
                    data = self.load_files(file_pattern)
                    if data is not None:
                        setattr(self, variable_name, data)
                self.path = None

        except FileNotFoundError:
            self.data = None
            self.path = None

        if self.data is None:
            self.conversations = self.create_conversation_trees()
        else:
            self.conversations = ChainTreeBuilder.parse_chain_tree(self.data)

    def extract_files_by_pattern(self):
        """
        Extracts files matching patterns from the base directory and assigns
        them to attributes based on the FILE_PATTERN_VARIABLE_MAP.
        """
        for file_pattern, variable_name in self.FILE_PATTERN_VARIABLE_MAP.items():
            file_data = self.load_files(file_pattern)
            setattr(self, variable_name, file_data)

    def load_files(self, file_pattern, sort_function=None, load_function=None):
        loaded_objects = self.manager.loader(
            patterns=file_pattern,
            sort_function=sort_function,
            load_prompts=load_function,
            range_index=self.range_index,
        )
        renamed_objects = []

        for data in loaded_objects:
            if isinstance(data, pd.DataFrame):
                if data.columns[0] == "Unnamed: 0":
                    data = data.rename(columns={"Unnamed: 0": "id"})
            renamed_objects.append(data)

        return renamed_objects

    @staticmethod
    def parse_chain_tree(
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[ChainTree, List[ChainTree]]:
        """
        Parses a dictionary or list of dictionaries to produce ChainTree objects.

        Parameters:
            data (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to be parsed, either as a single
                dictionary or a list of dictionaries.

        Returns:
            Union[ChainTree, List[ChainTree]]: A ChainTree object if a dictionary is provided,
                or a list of ChainTree objects if a list of dictionaries is provided.

        Raises:
            ValueError: If the input data type is neither a dictionary nor a list of dictionaries.
        """
        if isinstance(data, dict):
            return ChainTree(**data)
        elif isinstance(data, list):
            return [ChainTree(**chain_tree) for chain_tree in data]
        else:
            raise ValueError("Invalid data type, should be dict or list of dicts.")

    def as_list(self) -> List[ChainTree]:
        """
        Converts the internal conversation trees to a list.

        Returns:
            List[ChainTree]: A list of ChainTree objects, each representing a conversation chain_tree.tree.
        """
        return self.conversations

    def as_dict(self) -> Dict[str, ChainTree]:
        """
        Converts the internal conversation trees to a dictionary.

        Note:
            A key must be provided when calling this function.

        Returns:
            Dict[str, ChainTree]: A dictionary where the keys are obtained from the attribute specified by `self.key`
                and the values are ChainTree objects.

        Raises:
            ValueError: If the `self.key` is not provided.
        """
        if not self.key:
            raise ValueError("Key must be provided when building a dictionary.")
        conversation_trees = self.conversations
        return {
            getattr(conversation, self.key): tree
            for conversation, tree in zip(self.conversations, conversation_trees)
        }

    def __iter__(self):
        """
        Allows for iteration over the conversation trees.

        Returns:
            iterator: An iterator over the ChainTree objects.
        """
        return iter(self.conversations)

    def __len__(self) -> int:
        """
        Returns the number of conversation trees.

        Returns:
            int: The number of ChainTree objects.
        """
        return len(self.conversations)

    def get(self, index: int) -> ChainTree:
        """
        Retrieves a specific conversation tree by its index.

        Parameters:
            index (int): The index of the conversation tree to retrieve.

        Returns:
            ChainTree: The ChainTree object representing the conversation tree at the given index.
        """
        return self.conversations[index]

    def get_index_by_title(self, title: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Retrieves the index(es) of conversations by their title(s).

        Parameters:
            title (Union[str, List[str]]): The title(s) of the conversation(s).

        Returns:
            Union[int, List[int]]: The index(es) of the conversation(s) if found, otherwise -1 or empty list.
        """
        # Always work with a list
        titles = [title] if isinstance(title, str) else title
        indexes = [
            i for i, tree in enumerate(self.conversations) if tree.title in titles
        ]

        # Return single index or -1 if a single title was provided, else return the list of indexes
        return indexes[0] if len(titles) == 1 else indexes

    def get_titles_by_index(self, indexes: List[int]) -> List[str]:
        """
        Retrieves the titles of conversations by their indexes.

        Parameters:
            indexes (List[int]): The indexes of the conversations.

        Returns:
            List[str]: The titles of the conversations.
        """
        return [self.get(index).title for index in indexes]

    def get_tree_by_title(self, title: str) -> ChainTree:
        """
        Retrieves a specific conversation tree by its title.

        Parameters:
            title (str): The title of the conversation.

        Returns:
            ChainTree: The ChainTree object representing the conversation with the given title.

        Raises:
            ValueError: If the conversation with the given title is not found.
        """
        index = self.get_index_by_title(title)
        if index == -1:
            raise ValueError(f"Conversation with title {title} not found.")
        return self.get(index)
    

    def get_trees_by_title(self, titles: List[str]) -> List[ChainTree]:
        """
        Retrieves a list of conversation trees by their titles.

        Parameters:
            titles (List[str]): The titles of the conversations.

        Returns:
            List[ChainTree]: A list of ChainTree objects representing the conversations with the given titles.
        """
        return [self.get_tree_by_title(title) for title in titles]
    
    def get_trees_not_in_titles(self, titles: List[str]) -> List[ChainTree]:
        """
        Retrieves a list of conversation trees that are not in the given titles.

        Parameters:
            titles (List[str]): The titles of the conversations to exclude.

        Returns:
            List[ChainTree]: A list of ChainTree objects representing the conversations that are not in the given titles.
        """
        return [tree for tree in self.conversations if tree.title not in titles]
    def get_titles(self) -> List[str]:
        """
        Retrieves the titles of all conversations.

        Returns:
            List[str]: The titles of all conversations.
        """
        return [tree.title for tree in self.conversations]

    def _process_tree_range(
        self, tree_range: Optional[Tuple[int, Optional[int]]]
    ) -> Tuple[int, int]:
        """
        Processes the tree range to determine the start and end indexes.

        Args:
            tree_range (Optional[Tuple[int, Optional[int]]]): The range of indexes to consider. Can be None.

        Returns:
            Tuple[int, int]: The start and end indexes.
        """

        if tree_range is None:
            start = 0
            end = len(self.conversations)
        else:
            start, end = tree_range
            if end is None:
                end = len(self.conversations)

        return start, end

    def get_titles_by_range(
        self, tree_range: Optional[Tuple[int, Optional[int]]] = None
    ) -> List[str]:
        """
        Retrieves the titles of conversations within the given range.

        Args:
            tree_range (Optional[Tuple[int, Optional[int]]]): The range of indexes to consider. Can be None.

        Returns:
            List[str]: The titles of the conversations within the given range.
        """
        start, end = self._process_tree_range(tree_range)
        return self.get_titles()[start:end]

    def get_traditional_indexes(
        self, phrases: List[str], start: int, end: int, exact_match: bool
    ) -> Set[int]:
        """
        Gets the indexes that match the given phrases using traditional matching.

        Args:
            phrases (List[str]): List of phrases to look for.
            start (int): Start index for the range to consider.
            end (int): End index for the range to consider.

        Returns:
            Set[int]: A set of matching indexes based on traditional matching.
        """
        return set(self.get_indexes_by_phrases(phrases, start, end, exact_match))

    def compute_and_combine(
        self,
        main_dfs: Optional[List[pd.DataFrame]] = None,
        relationship_dfs: Optional[List[pd.DataFrame]] = None,
    ) -> Tuple[float, float, pd.DataFrame]:
        combined_df = pd.concat(main_dfs, ignore_index=True)
        combined_relationship_df = pd.concat(relationship_dfs, ignore_index=True)

        total_sum = 0
        total_count = 0
        for df in main_dfs:
            n_neighbors_sum = df["n_neighbors"].sum()

            total_sum += float(n_neighbors_sum)
            total_count += len(df)

        mean_neighbors = total_sum / total_count if total_count != 0 else 0.0

        return mean_neighbors, combined_df, combined_relationship_df

    def create_conversation_trees(self) -> List[ChainTree]:
        """
        Creates a list of ChainTree objects from the input data, filtering them based on
        the target number of mappings.

        Attributes:
            self.data: The input data for creating ChainTree objects.
            self.target_num: The target number of mappings for a ChainTree to be considered "greater than target."
            self.less_than_target (List[ChainTree]): A list that stores ChainTree objects that
                have fewer than target_num mappings. This list is updated within the method.

        Returns:
            List[ChainTree]: A list of ChainTree objects that have greater than or equal to
                target_num mappings.
        """

        greater_than_target = []
        for i, conversation in enumerate(ChainTreeBuilder.parse_chain_tree(self.data)):
            if conversation is not None:
                if len(conversation.mapping) >= self.target_num:
                    greater_than_target.append(ChainTree(conversation))
                else:
                    conversation.title = str(i)
                    self.less_than_target.append(ChainTree(conversation))
        return greater_than_target

    def get_indexes_by_phrases(
        self,
        phrases: List[str] = None,
        start: int = 0,
        end: int = None,
        exact_match: bool = True,
    ) -> List[int]:
        """
        Get conversation indexes whose title contains any of the specified phrases within a specified range.

        Args:
            phrases (List[str]): List of phrases to look for. Defaults to None.
            start (int): Start index for the range to consider. Defaults to 0.
            end (int): End index for the range to consider. Defaults to None.
            exact_match (bool): Whether to look for exact matches. Defaults to False.

        Returns:
            List[int]: List of conversation indexes.
        """

        # If phrases is None, return an empty list
        if phrases is None:
            return []

        # Validate start and end indexes to be integers or None
        if start is not None and not isinstance(start, int):
            return []
        if end is not None and not isinstance(end, int):
            return []

        # Generate conversation trees
        conversation_trees = self.conversations[start:end]

        # Initialize empty list to hold matching indexes
        indexes = []

        for i, tree in enumerate(conversation_trees):
            if exact_match:
                if any(phrase.lower() == tree.title.lower() for phrase in phrases):
                    indexes.append(i + start)
            else:
                if any(phrase.lower() in tree.title.lower() for phrase in phrases):
                    indexes.append(i + start)

        return indexes

    def get_indexes_by_message_contains(
        self, keywords: List[str] = None, start: int = 0, end: int = None
    ) -> List[int]:
        """
        Get conversation indexes whose messages contain any of the specified keywords within a specified range.

        Args:
            keywords (List[str]): List of keywords to look for. Defaults to None.
            start (int): Start index for the range to consider. Defaults to 0.
            end (int): End index for the range to consider. Defaults to None.

        Returns:
            List[int]: List of conversation indexes.
        """

        # If keywords is None, return an empty list
        if keywords is None:
            return []

        # Validate start and end indexes to be integers or None
        if start is not None and not isinstance(start, int):
            return []
        if end is not None and not isinstance(end, int):
            return []

        # Generate conversation trees
        conversation_trees = self.conversations[start:end]

        # Initialize empty list to hold matching indexes
        indexes = []

        for i, tree in enumerate(conversation_trees):
            for message_map in tree.mapping.values():
                if message_map.message and message_map.message.content:
                    text = message_map.message.content.text.lower()

                    for keyword in keywords:
                        keyword_pattern = re.compile(
                            r"\b" + re.escape(keyword.lower()) + r"\b"
                        )  # Create a regex pattern
                        if keyword_pattern.search(
                            text
                        ):  # Search for the keyword in the text
                            indexes.append(i + start)
                            break
                        # Exit inner loop as we found a match in this conversation
                    if i + start in indexes:
                        break
                    # Exit the outer loop as well; we don't need to keep searching this conversation

        return indexes

    def get_the_std_of_messages(
        self,
        message_range_start: int = 0,
        message_range_end: int = None,
        start: int = 0,
        end: int = None,
        most_common: int = 10,
        plot: bool = True,
    ) -> List[int]:
        """
        Get the distribution of messages within a specified range.

        Args:
            start (int): Start index for the range to consider. Defaults to 0.
            end (int): End index for the range to consider. Defaults to None.

        Returns:
            List[int]: List of message counts.
        """

        # Validate start and end indexes to be integers or None
        if start is not None and not isinstance(start, int):
            return []
        if end is not None and not isinstance(end, int):
            return []

        # Generate conversation trees
        conversation_trees = self.conversations[start:end]

        # filter out the trees given the message range start and end

        conversation_trees = [
            tree
            for tree in conversation_trees
            if (message_range_start is None or len(tree.mapping) >= message_range_start)
            and (message_range_end is None or len(tree.mapping) <= message_range_end)
        ]

        # Initialize empty list to hold matching indexes
        message_counts = []

        for i, tree in enumerate(conversation_trees):
            message_counts.append(len(tree.mapping))

        # Generate a Gaussian distribution with the same mean and standard deviation as your message_counts
        mean = np.mean(message_counts)
        median = np.median(message_counts)
        std_dev = np.std(message_counts)
        size = len(
            message_counts
        )  # Use the same number of data points as message_counts
        variance = np.var(message_counts)

        # Generate a Gaussian distribution with the same mean and standard deviation as your message_counts
        gaussian = np.random.normal(mean, std_dev, size)

        # Plot the histogram of message_counts and the Gaussian distribution

        def plot_histogram(message_counts, gaussian):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(message_counts, bins=20, alpha=0.5, label="Message Counts")
            ax.hist(gaussian, bins=20, alpha=0.5, label="Gaussian")
            ax.set_xlabel("Message Count")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Message Counts")
            ax.legend()
            plt.show()

        if plot:
            plot_histogram(message_counts, gaussian)

        # Count the occurrences of each length in the message_counts list
        length_counts = Counter(message_counts)

        # Find the top 10 most common lengths and their counts
        top_lengths = length_counts.most_common(most_common)
        # Plot the Gaussian distribution
        return {
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "size": size,
            "length_counts": length_counts,
            "top_lengths": top_lengths,
            "message_counts": message_counts,
            "variance": variance,
            "gaussian": gaussian,
        }

    def create_message_map(
        self,
        neighbors: Optional[int] = 20,
        trees: Optional[List[ChainTree]] = None,
        format: Optional[str] = "df",
        visualize: bool = True,
        skip_format: bool = False,
        exclude_key: Optional[List[str]] = None,
        path: Optional[str] = None,
        embedding_model: Optional[object] = None,
        use_embeddings: bool = False,
        return_message: bool = False,
    ) -> Union[Dict, pd.DataFrame, None]:
        if trees is None:
            trees = self.conversations

        message_coord_map = self.extract_messages_from_trees(trees, exclude_key)
        if return_message:
            return message_coord_map

        if skip_format:
            df = pd.DataFrame.from_dict(message_coord_map, orient="index")
            return (
                self.process_with_embedding_model(
                    df, embedding_model, neighbors, use_embeddings, visualize
                )
                if embedding_model
                else df
            )

        return self.format_and_save_data(
            message_coord_map, format, path, neighbors, embedding_model
        )

    def _format_representation(
        self,
        data: Union[Dict, pd.DataFrame],
        path: Optional[str] = None,
        visualize: bool = True,
        combined_data: Optional[pd.DataFrame] = None,
        neighbors: int = 20,
        embedding_model: Optional[object] = None,
        use_embeddings: bool = False,
    ) -> pd.DataFrame:
        df_result, model = (
            self.process_with_embedding_model(
                data, embedding_model, neighbors, use_embeddings, visualize
            )
            if embedding_model
            else df_result
        )
        if path:
            self._save_data(df_result, path)
        if combined_data is not None:
            self._save_data(combined_data, path, suffix="_combined")

        if model:
            model.save(os.path.join(self.base_persist_dir, "model"))

        return df_result

    def extract_messages_from_trees(
        self,
        trees: List[ChainTree],
        exclude_key: List[str],
        extract_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Extracts messages from a list of conversation trees.

        Parameters:
            trees (List[ChainTree]): The conversation trees from which to extract messages.
            exclude_key (List[str]): The keys to be excluded from the message data.
            extract_all (bool): Flag to determine if all data should be extracted.

        Returns:
            Dict: A dictionary mapping message IDs to their corresponding data.
        """
        message_coord_map = {}
        for tree in trees:
            for message_id, mapping in tree.mapping.items():
                if self.should_include_message(tree, mapping):
                    message_data = self.extract_data(mapping, tree, extract_all)
                    self.exclude_specified_keys(message_data, exclude_key)
                    message_coord_map[message_id] = message_data
        return message_coord_map

    def extract_data(
        self, mapping: ChainMap, tree: ChainTree, extract_all: bool
    ) -> Dict:
        """
        Extracts message data from a given mapping and conversation tree based on extract_all flag.

        Parameters:
            mapping (ChainMap): The mapping containing the message data.
            tree (ChainTree): The conversation tree containing the message.
            extract_all (bool): Flag to determine if all data should be extracted.

        Returns:
            Dict: A dictionary containing the extracted message data.
        """
        message = mapping.message
        data = {
            "message_id": message.id,
            "text": getattr(message.content, "text", ""),
            "author": message.author.role,
            "create_time": message.create_time,
            "title": tree.title,
            "embeddings": message.embedding,
            "metadata": getattr(message, "metadata", {}),
            "parent_id": mapping.parent,
            "children": mapping.children,
        }

        if extract_all:
            coordinates = message.coordinate
            umap_embeddings = message.umap_embeddings
            data.update(
                {
                    "conversation_id": tree.id,
                    "x": umap_embeddings[0],
                    "y": umap_embeddings[1],
                    "z": umap_embeddings[2],
                    "depth_x": coordinates["x"],
                    "sibling_y": coordinates["y"],
                    "sibling_count_z": coordinates["z"],
                    "time_t": coordinates["t"],
                    "n_parts": coordinates["n_parts"],
                    "children_count": len(mapping.children),
                }
            )

        return data

    def process_with_embedding_model(
        self,
        df: pd.DataFrame,
        embedding_model: object,  # or specify the exact type
        neighbors: int,
        use_embeddings: bool,
        visualize: bool,
    ) -> pd.DataFrame:
        """
        Processes a DataFrame with a given embedding model.

        Args:
            df (pd.DataFrame): The DataFrame containing the messages.
            embedding_model (object): The model used for computing embeddings.
            neighbors (int): The number of neighbors to consider for each message.
            use_embeddings (bool): Flag indicating whether to use embeddings.
            visualize (bool): Flag indicating whether to visualize the embeddings.

        Returns:
            pd.DataFrame: The DataFrame with computed message embeddings.
        """
        return self._compute_embeddings(
            df,
            embedding_model,
            neighbors,
            use_embeddings=use_embeddings,
            visualize=visualize,
        )

    def format_and_save_data(
        self,
        message_coord_map: Dict,
        format: str,
        path: Optional[str],
        neighbors: int,
        embedding_model: Optional[object],  # or specify the exact type
    ) -> Union[Dict, str, None]:
        """
        Formats and saves the data based on the given format and path.

        Args:
            message_coord_map (Dict): The mapping of message coordinates.
            format (str): The format to save the data in ('json', 'df', or None).
            path (Optional[str]): The path where to save the data.
            neighbors (int): The number of neighbors to consider for each message.
            embedding_model (Optional[object]): The model used for computing embeddings.

        Returns:
            Union[Dict, str, None]: The data in the specified format.

        Raises:
            ValueError: If an invalid format is provided.
        """
        if format == "json":
            return self.format_as_json(message_coord_map, path)
        elif format == "df":
            return self._format_representation(
                message_coord_map, path, neighbors, embedding_model
            )
        elif format is None:
            return message_coord_map
        else:
            raise ValueError(
                "Invalid format. Accepted formats are 'json', 'df', or None."
            )

    def should_include_message(self, tree: ChainTree, mapping: ChainMap) -> bool:
        """
        Determines whether a message should be included based on given criteria.

        Parameters:
            tree (ChainTree): The conversation tree containing the message.
            mapping (ChainMap): The mapping containing the message data.

        Returns:
            bool: True if the message should be included, False otherwise.
        """
        return mapping.message is not None and mapping.message.author.role != "system"

    def exclude_specified_keys(
        self, message_data: Dict, exclude_key: List[str]
    ) -> None:
        """
        Removes specified keys from a message data dictionary.

        Parameters:
            message_data (Dict): The dictionary containing message data.
            exclude_key (List[str]): List of keys to be removed.

        Returns:
            None: The method modifies the message_data dictionary in-place.
        """

        if exclude_key:
            for key in exclude_key:
                message_data.pop(key, None)

    def _compute_embeddings(
        self,
        df: pd.DataFrame,
        embedding_model: object,  # or specify the exact type
        neighbors: int,
        use_embeddings: bool,
        visualize: bool,
        excude_coordinates: bool = False,
        add_embeddings: bool = False,
    ) -> pd.DataFrame:
        """
        Computes message embeddings for the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the messages.
            embedding_model (SpatialSimilarity): The spatial similarity model to use for computing embeddings.
            neighbors (int): The number of neighbors to consider for each message.
            use_embeddings (bool): Flag indicating whether to use embeddings.
            visualize (bool): Flag indicating whether to visualize the embeddings.

        Returns:
            pd.DataFrame: The DataFrame with computed message embeddings.
        """
        return embedding_model.compute_message_embeddings(
            neighbors=neighbors,
            main_df=df,
            use_embeddings=use_embeddings,
            visualize=visualize,
            excude_coordinates=excude_coordinates,
            add_embeddings=add_embeddings,
        )

    def format_dataframe(
        self, df: pd.DataFrame, exclude_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Formats the DataFrame by optionally excluding specified columns and resetting the index.

        Parameters:
            df (pd.DataFrame): The DataFrame to format.
            exclude_columns (Optional[List[str]]): List of columns to exclude.

        Returns:
            pd.DataFrame: The formatted DataFrame.
        """

        if exclude_columns is not None:
            df = df.drop(columns=exclude_columns)

        df = df.reset_index(drop=True)

        return df

    def format_as_json(self, data: Dict, path: Optional[str]) -> Optional[str]:
        """
        Formats data as JSON and optionally saves it to a file.

        Parameters:
            data (Dict): The data to format.
            path (Optional[str]): File path to save the JSON data. If None, the data is not saved.

        Returns:
            Optional[str]: The JSON string if path is None; otherwise None.

        """

        json_result = json.dumps(data)
        if path:
            with open(path, "w") as json_file:
                json_file.write(json_result)
            return None
        return json_result

    def _save_data(self, df: pd.DataFrame, path: str, suffix: str = "") -> None:
        """
        Private method to save a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            path (str): The relative path where the CSV file will be saved.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to an empty string.

        Returns:
            None
        """
        full_path = self.base_persist_dir + path + suffix + ".csv"
        df.to_csv(full_path, index=False)


def get_message_map(
    path: str,
    base_persist_dir: str,
    skip_format: bool = True,
    exclude_key: Optional[List[str]] = None,
    neighbors: Optional[int] = 20,
    embedding_model: Optional[object] = None,
    use_embeddings: bool = False,
    return_message: bool = False,
) -> Dict[str, Any]:
    """
    Creates a message map using ChainTreeBuilder and returns it.

    Args:
        path (str): The path to the data source.
        base_persist_dir (str): The directory where any persistent files will be saved.
        skip_format (bool, optional): Whether to skip formatting. Defaults to False.
        exclude_key (Optional[List[str]], optional): List of keys to exclude from the map. Defaults to None.
        neighbors (Optional[int], optional): The number of neighbors to consider for clustering. Defaults to 20.
        embedding_model (Optional[object], optional): The embedding model to use for message embedding. Defaults to None.
        use_embeddings (bool, optional): Whether to use embeddings or not. Defaults to False.

    Returns:
        Dict[str, Any]: The message map.
    """
    conversation_trees = ChainTreeBuilder(path=path, base_persist_dir=base_persist_dir)
    return conversation_trees.create_message_map(
        skip_format=skip_format,
        exclude_key=exclude_key,
        neighbors=neighbors,
        embedding_model=embedding_model,
        use_embeddings=use_embeddings,
        return_message=return_message,
    )


def initialize(
    SpatialSimilarity: object,
    text_column: str = None,
    label_column: str = None,
    api_key: str = None,
    neighbors: int = 10,
    min_cluster_size: int = 10,
    use_processed_text: bool = False,
    dataframe: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
    num_rows: Optional[int] = None,
    excude_coordinates: bool = False,
    update_coordinates: bool = False,
    unique: bool = False,
    add_embeddings: bool = False,
    visualize: bool = True,
):
    """
    Initialize a ElementProcessor with a specified directory and a semantic model.

    Parameters:
        prompt_col: str
            The column name for the prompt in the DataFrame.
        response_col: str
            The column name for the response in the DataFrame.
        text_column: str
            The column name containing text for semantic analysis.
        dataframe: Optional[pd.DataFrame]
            The DataFrame containing the data.
        compute_embeddings: bool  # Add this parameter description
            Whether to compute embeddings during initialization.
        verbose: bool
            Whether to print additional debugging information.
    """
    if api_key:
        semantic_model = SpatialSimilarity(api_key=api_key)
    else:
        semantic_model = SpatialSimilarity()

    if dataframe is None and csv_path:
        dataframe = pd.read_csv(csv_path)
        dataframe = dataframe.dropna(subset=[text_column])
        dataframe = dataframe.reset_index(drop=True)
        # If num_rows is specified, restrict the data to the specified number of rows
        if num_rows:
            dataframe = dataframe.head(num_rows)
    elif dataframe is not None and num_rows:
        dataframe = dataframe.head(num_rows)
    elif dataframe is None:
        raise ValueError("Either a dataframe or csv_path should be provided!")

    dataframe = semantic_model.compute_message_embeddings(
        main_df=dataframe,
        use_processed_text=use_processed_text,
        neighbors=neighbors,
        text_column=text_column,
        excude_coordinates=excude_coordinates,
        update_coordinates=update_coordinates,
        unique=unique,
        add_embeddings=add_embeddings,
        min_cluster_size=min_cluster_size,
        label_column=label_column,
        visualize=visualize,
    )

    return dataframe
