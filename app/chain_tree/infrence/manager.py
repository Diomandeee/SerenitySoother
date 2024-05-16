from typing import Dict, List, Union, Optional, Any, Callable, Tuple
from chain_tree.models import Content, Chain, ChainCoordinate, Author
from google.api_core.exceptions import BadRequest, GoogleAPIError
from concurrent.futures import ThreadPoolExecutor, as_completed
from chain_tree.infrence.validator import MessageIDValidator
from chain_tree.utils import load_json, log_handler
from chain_tree.infrence.state import StateMachine
from chain_tree.type import PromptStatus, RoleType
from chain_tree.type import ElementType
from pydantic import Field, BaseModel
from google.cloud import storage
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import networkx
import requests
import logging
import base64
import json
import glob
import uuid
import tqdm
import os
import re
import io


def generate_id() -> str:
    return str(uuid.uuid4())


class Element:
    def __init__(self, directory: Optional[str] = None):
        self.directory = directory

    def prepare_initial_data(
        self,
        dataframe: pd.DataFrame,
        element_type: ElementType = ElementType.PAGE,
        include_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Create a DataFrame from the existing DataFrame.
        Compute embeddings for each element and group similar terms.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the elements.
            element_type (ElementType, optional): The type of the elements (e.g., ElementType.STEP, ElementType.CHAPTER, ElementType.PAGE).
                Defaults to ElementType.STEP.
            include_missing (bool, optional): Whether to include missing elements. Defaults to True.

        Returns:
            pd.DataFrame: The DataFrame containing the elements and their embeddings.
        """
        try:
            elements = [col for col in dataframe.columns if element_type.value in col]

            if not elements and include_missing:
                # Generate column names based on element_type
                elements = [
                    f"{element_type.value} {i}" for i in range(dataframe.shape[1])
                ]

            filtered_dataframes = []
            for element in elements:
                if element in dataframe.columns:
                    filtered_dataframes.append(
                        dataframe[
                            dataframe[element].str.startswith(element + ":")
                        ].copy()
                    )
                elif include_missing:
                    filtered_dataframes.append(dataframe.copy())

            # Concatenate filtered DataFrames if any
            if filtered_dataframes:
                dataframe = pd.concat(filtered_dataframes, ignore_index=True)

        except Exception as e:
            print(f"Error processing elements: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of errors

        return dataframe

    def traverse_keys(
        self,
        data: dict,
        keys: List[str],
        return_all_values: bool = False,
        include_key_with_value: bool = False,
        callback: Optional[Callable] = None,
    ) -> Union[dict, List[dict], List[Tuple[str, dict]], None]:
        """
        Traverse through the keys in the given data.

        Args:
            data (dict): Data to traverse.
            keys (List[str]): List of keys to follow.
            return_all_values (bool, optional): If True, returns all values from the keys. Defaults to False.
            include_key_with_value (bool, optional): If True, returns a tuple of key and value. Defaults to False.
            callback (Optional[Callable], optional): A function to apply to each value as it is retrieved. Defaults to None.

        Returns:
            Union[dict, List[dict], List[Tuple[str, dict]], None]: Resulting value(s) or None if keys are not found.
        """

        # Initialize a list to store all values if return_all_values is True
        all_values = []

        try:
            # Iterate through the provided keys to traverse the data
            for key in keys:
                # Check if the key exists in the current level of the data
                if isinstance(data, dict) and key in data:
                    value = data[key]

                    # Apply the callback function to the value if provided
                    if callback:
                        value = callback(value)

                    # If return_all_values is True, store the value (and key if include_key_with_value is True)
                    if return_all_values:
                        result = (key, value) if include_key_with_value else value
                        all_values.append(result)

                    # Move to the next level of the data using the current key
                    data = value
                else:
                    # If the key is not found, return None
                    return None

            # Return either all the values or the final value, depending on return_all_values
            return all_values if return_all_values else data

        except Exception as e:
            # Log an error if an exception occurs during traversal
            log_handler(f"Error traversing keys: {str(e)}")
            return None

    def process_prompt_file(
        self,
        prompt_file: str,
        keys: List[str],
        return_all_values: bool,
        include_key_with_value: bool,
        callback: Optional[Callable],
    ) -> dict:
        try:
            file_data = load_json(prompt_file)
            return self.traverse_keys(
                file_data,
                keys,
                return_all_values=return_all_values,
                include_key_with_value=include_key_with_value,
                callback=callback,
            )
        except Exception as e:
            log_handler(f"Error processing prompt file: {str(e)}")
            return None

    def get_prompt_files(
        self,
        directory: Optional[str] = None,
        file_pattern: str = "**/*.json",
        sort_function: Optional[Callable] = None,
        convert_to_df: bool = False,
        concat_dfs: bool = False,  # New optional parameter to concatenate DataFrames
    ) -> List[str]:
        """
        Retrieve the list of prompt files from a specified directory.

        Args:
            directory (str): The directory to search for prompt files.
            file_pattern (str, optional): The pattern to match files. Defaults to "**/*.json" (matches all JSON files in the directory).
            sort_function (Callable, optional): A custom sorting function. Defaults to None.
            convert_to_df (bool, optional): Whether to convert loaded files to a DataFrame. Defaults to False.
            concat_dfs (bool, optional): Whether to concatenate the loaded DataFrames into a single DataFrame. Defaults to False.

        Returns:
            List[str] or List[pd.DataFrame] or pd.DataFrame: A list of paths to prompt files, DataFrames, or a single concatenated DataFrame.
        """

        dir_to_use = directory if directory else self.directory

        # Use glob to match all files in the directory with the specified pattern (e.g., all JSON files).
        prompt_files = glob.glob(os.path.join(dir_to_use, file_pattern))

        # Sort the files using the custom sort function if provided.
        if sort_function:
            prompt_files.sort(key=sort_function)
        else:
            # If no custom sort function is provided, sort the files based on the numeric part in their filenames.
            prompt_files.sort(
                key=lambda f: (
                    int(re.search(r"\d+", os.path.basename(f)).group())
                    if re.search(r"\d+", os.path.basename(f))
                    else 0
                )
            )

        if convert_to_df:
            # Initialize an empty list to store individual DataFrames
            dfs = []

            # Loop through the file paths and load each CSV file into a DataFrame
            for file_path in prompt_files:
                try:
                    df = pd.read_csv(file_path)

                    # Check if the DataFrame has both columns and data before appending
                    if not df.empty and not df.columns.empty:
                        dfs.append(df)
                    else:
                        print(f"Warning: Empty or no columns in file {file_path}")

                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty file {file_path}")

            if concat_dfs:
                # Concatenate all DataFrames into a single DataFrame
                concatenated_df = pd.concat(dfs, ignore_index=True)
                return concatenated_df
            else:
                # Return the list of DataFrames
                return dfs
        else:
            # Return the list of prompt files.
            return prompt_files

    def get_glob_files(
        self,
        pattern: str = "**/*.csv",
        file: Optional[str] = None,
        recursive: bool = True,
    ) -> List[str]:
        if file is not None:
            files = glob.glob(
                os.path.join(self.directory, pattern + file), recursive=recursive
            )
        else:
            files = glob.glob(
                os.path.join(self.directory, pattern), recursive=recursive
            )
        return files

    @staticmethod
    def topological_sort_files(
        files: List[str], dependency_resolver: Callable[[str], List[str]]
    ) -> List[str]:
        """
        Perform a topological sort on the files based on their dependencies.

        :param files: A list of file paths.
        :param dependency_resolver: A function that takes a file path and returns a list of file paths that the given file depends on.
        :return: A list of file paths sorted in topological order.
        """
        G = networkx.DiGraph()

        # Add nodes for each file
        for file in files:
            G.add_node(file)

        # Add edges based on dependencies
        for file in files:
            dependencies = dependency_resolver(file)
            for dependency in dependencies:
                G.add_edge(dependency, file)

        sorted_files = list(networkx.topological_sort(G))
        return sorted_files

    def sort_files(
        files: List[str], sort_function: Optional[Callable[[str], int]] = None
    ) -> List[str]:
        """Sort files either by custom function or by the numeric part in their filenames."""
        if sort_function:
            files.sort(key=sort_function)
        else:
            files.sort(
                key=lambda f: (
                    int(re.search(r"\d+", os.path.basename(f)).group())
                    if re.search(r"\d+", os.path.basename(f))
                    else 0
                )
            )
        return files

    @staticmethod
    def load_file(
        file: str, load_prompts: Optional[Union[bool, Callable]] = None
    ) -> any:
        """Load a single file based on its type."""
        extension = os.path.splitext(file)[1]
        if extension == ".json":
            with open(file, "r") as f:
                data = load_json(file)
                if callable(load_prompts):
                    return load_prompts(data, file)

                if load_prompts is True:
                    return (
                        data.get("response")
                        or data.get("prompt")
                        or data.get("text")
                        or data.get("revised_prompt")
                    )
        elif extension in [".txt", ".md"]:
            with open(file, "r") as f:
                return f.read()
        # elif extension == ".csv":
        #     return pd.read_csv(file)
        elif extension == ".npy":
            return np.load(file)
        elif extension == ".jsonl":
            return [json.loads(line) for line in open(file, "r")]
        elif extension == ".json":
            return load_json(file)
        elif extension == ".db":
            return pd.read_sql(file)
        elif extension in [".png", ".jpg", ".jpeg", ".gif", ".svg"]:
            return file
        else:
            return file

    def _sort_and_slice(
        self,
        files: List[str],
        sort_function: Optional[Callable],
        range_index: Optional[Tuple[int, Optional[int]]],
    ) -> List[str]:
        if sort_function:
            files = Element.sort_files(files, sort_function)
        if range_index:
            start, end = range_index
            files = files[start:end] if end else files[start:]
        return files

    def get_files_by_pattern(
        self,
        pattern: str,
        sort_function: Optional[Callable] = None,
        range_index: Optional[Tuple[int, Optional[int]]] = None,
    ) -> List[str]:
        # check if the file is a folder if so, list the files in the folder
        if os.path.isdir(pattern):
            files = os.listdir(pattern)
            # add the full path to the files
            files = [os.path.join(pattern, file) for file in files]
            # zip the files with the file names
            files = list(zip(files, [os.path.basename(file) for file in files]))
            return files

        files = self.get_glob_files(pattern)
        files = self._sort_and_slice(files, sort_function, range_index)

        data_files = []
        for file in files:
            # load only json files
            if file.endswith(".json"):
                data_files.append(load_json(file))

            elif file.endswith(".npy"):
                data_files.append(np.load(file))

            elif file.endswith(".jsonl"):
                data_files.append([json.loads(line) for line in open(file, "r")])

            elif file.endswith(".txt") or file.endswith(".md"):
                with open(file, "r") as f:
                    data_files.append(f.read())

            elif file.endswith(".csv"):
                try:
                    data_files.append(pd.read_csv(file))
                except pd.errors.EmptyDataError:
                    print(f"WARNING: Skipping empty CSV file: {file}")

            elif file.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                data_files.append(file)

            else:
                data_files.append(file)

        return data_files

    def _prepare_return(
        self,
        files: List[str],
        files_by_type: Optional[dict],
        return_files: bool,
        return_directories: bool,
        load_prompts: Optional[Union[bool, Callable]],
        group_by_data_type: bool,
    ) -> List:
        directories = list(set(os.path.dirname(f) for f in files))
        if return_files and return_directories:
            return (
                (files, directories)
                if not group_by_data_type
                else (files_by_type, directories)
            )
        elif return_files:
            return files_by_type if group_by_data_type else files
        elif return_directories:
            return directories
        else:
            loaded_files = (
                {
                    dt: [Element.load_file(f, load_prompts) for f in files]
                    for dt in files_by_type
                }
                if group_by_data_type
                else [Element.load_file(f, load_prompts) for f in files]
            )
            return loaded_files.values() if group_by_data_type else loaded_files

    def loader(
        self,
        patterns: List[str] = ["**/*"],
        sort_function: Optional[Callable] = None,
        load_prompts: Optional[Union[bool, Callable]] = None,
        return_files: bool = False,
        return_directories: bool = False,
        range_index: Optional[Tuple[int, Optional[int]]] = None,
        data_types: Optional[List[str]] = None,
        group_by_data_type: bool = False,
    ) -> List:
        """
        Retrieve and load files from a specified directory based on given patterns.
        Sort them if required, and optionally group by data type.

        Args:
            patterns (List[str]): Glob patterns to match files.
            sort_function (Callable): A function to sort the files.
            load_prompts (Union[bool, Callable]): Determines how to load files.
            return_files (bool): Whether to return files.
            return_directories (bool): Whether to return directories.
            range_index (Tuple[int, Optional[int]]): Range of indices for files to return.
            data_types (List[str]): Specific types of data folders to include.
            group_by_data_type (bool): Group files by data type.
        """

        data_types = data_types or [""]
        files_by_type = {dt: [] for dt in data_types} if group_by_data_type else None
        files = []

        for dt in data_types:
            search_directory = (
                os.path.join(self.directory, dt) if dt else self.directory
            )
            for pattern in patterns:
                matched_files = glob.glob(
                    os.path.join(search_directory, pattern), recursive=True
                )
                if group_by_data_type:
                    files_by_type[dt].extend(matched_files)
                else:
                    files.extend(matched_files)

        if group_by_data_type:
            files_by_type = {
                dt: self._sort_and_slice(files_by_type[dt], sort_function, range_index)
                for dt in data_types
            }
        else:
            files = self._sort_and_slice(files, sort_function, range_index)

        return self._prepare_return(
            files,
            files_by_type,
            return_files,
            return_directories,
            load_prompts,
            group_by_data_type,
        )


class ChainManager(BaseModel):
    conversations: Dict[str, StateMachine] = Field(
        {}, description="A dictionary mapping conversation IDs to conversations."
    )

    message_id_validator: MessageIDValidator = Field(
        MessageIDValidator(),
        description="A validator for checking if a message ID exists in a conversation.",
    )

    engine: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def conversation_exists(self, conversation_id: str) -> bool:
        return conversation_id in self.conversations

    def _validate_message_id_in_mapping(
        self, conversation: StateMachine, message_id: str
    ) -> None:
        if not self.message_id_validator.validate_message_id_in_mapping(
            conversation, message_id
        ):
            raise ValueError(
                f"Message with ID '{message_id}' does not exist in conversation with ID '{conversation.id}'."
            )

    def start_conversation(self, initial_message: str) -> str:
        conversation_id = self.create_conversation()
        self.handle_system_message(conversation_id, initial_message)
        return conversation_id

    def create_conversation(self) -> str:
        conversation_id = generate_id()
        conversation = StateMachine(conversation_id=conversation_id)
        self.conversations[conversation_id] = conversation
        return conversation_id

    def add_conversation(self, conversation: StateMachine) -> None:
        if not isinstance(conversation, StateMachine):
            raise TypeError(
                f"Expected 'Conversation' object, got '{type(conversation).__name__}'."
            )

        if conversation.conversation_id in self.conversations:
            raise ValueError(
                f"Conversation with ID '{conversation.conversation_id}' already exists."
            )

        self.conversations[conversation.conversation_id] = conversation

    def get_conversation(self, conversation_id: str) -> StateMachine:
        if conversation_id not in self.conversations:
            raise ValueError(
                f"Conversation with ID '{conversation_id}' does not exist."
            )

        return self.conversations[conversation_id]

    def rewind_conversation(self, conversation_id: str, steps: int = 1) -> None:
        conversation = self.get_conversation(conversation_id)
        conversation.rewind_conversation(steps)

    def print_conversation(self, conversation_id: str) -> None:
        conversation = self.get_conversation(conversation_id)
        conversation.print_conversation()

    def end_conversation(self, conversation_id: str) -> None:
        conversation = self.get_conversation(conversation_id)
        conversation.end_conversation()

    def restart_conversation(self, conversation_id: str) -> None:
        conversation = self.get_conversation(conversation_id)
        conversation.restart_conversation()

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            raise ValueError(f"No conversation with ID '{conversation_id}' found.")
        return conversation.get_history()

    def add_message(
        self,
        conversation_id: str,
        message_id: str,
        content: Content,
        author: Author,
        coordinate: Optional[ChainCoordinate] = None,
        embedding: List[float] = None,
        parent: str = None,
        save: bool = False,
    ) -> None:
        conversation = self.get_conversation(conversation_id)
        conversation.add_message(
            message_id=message_id,
            content=content,
            author=author,
            embedding=embedding,
            parent=parent,
            coordinate=coordinate,
        )
        if save:
            conversation.save_conversation(title=conversation_id)

    def update_message(
        self, conversation_id: str, message_id: str, new_message: Chain
    ) -> None:
        conversation = self.get_conversation(conversation_id)
        self._validate_message_id_in_mapping(conversation, message_id)

        conversation.update_message(message_id, new_message)

    def delete_message(self, conversation_id: str, message_id: str) -> bool:
        conversation = self.get_conversation(conversation_id)
        self._validate_message_id_in_mapping(conversation, message_id)

        return conversation.delete_message(message_id)

    def get_message(self, conversation_id: str, message_id: str) -> Chain:
        conversation = self.get_conversation(conversation_id)
        self._validate_message_id_in_mapping(conversation, message_id)

        return conversation.get_message(message_id)

    def move_message(
        self, conversation_id: str, message_id: str, new_parent_id: str
    ) -> None:
        conversation = self.get_conversation(conversation_id)
        self._validate_message_id_in_mapping(conversation, message_id)

        conversation.move_message(message_id, new_parent_id)

    def merge_conversations(
        self, conversation_id_1: str, conversation_id_2: str
    ) -> None:
        conversation_1 = self.get_conversation(conversation_id_1)
        conversation_2 = self.get_conversation(conversation_id_2)

        conversation_1.merge(conversation_2)
        self.delete_conversation(conversation_id_2)

    def get_conversations(self) -> List[StateMachine]:
        return list(self.conversations.values())

    def get_conversation_ids(self) -> List[str]:
        return list(self.conversations.keys())

    def get_conversation_titles(self) -> List[str]:
        return [conv.title for conv in self.conversations.values()]

    def get_conversation_titles_and_ids(self) -> List[Dict[str, str]]:
        return [
            {"title": conv.title, "id": conv.id} for conv in self.conversations.values()
        ]

    def delete_conversation(self, conversation_id: str) -> None:
        if conversation_id not in self.conversations:
            raise ValueError(
                f"Conversation with ID '{conversation_id}' does not exist."
            )

        del self.conversations[conversation_id]

    def delete_all_conversations(self) -> None:
        self.conversations = {}

    def cleanup_inactive_conversations(
        self, inactivity_threshold_in_hours: int = 1
    ) -> None:
        current_time = datetime.now().timestamp()
        inactive_conversations = []

        # identify inactive conversations
        for conversation_id, conversation in self.conversations.items():
            time_since_last_interaction = (
                current_time - conversation.last_interaction_time
            )
            if time_since_last_interaction > inactivity_threshold_in_hours * 60 * 60:
                inactive_conversations.append(conversation_id)

        # remove inactive conversations
        for conversation_id in inactive_conversations:
            del self.conversations[conversation_id]

    def merge_conversations(
        self, conversation_id_1: str, conversation_id_2: str
    ) -> None:
        conversation_1 = self.get_conversation(conversation_id_1)
        conversation_2 = self.get_conversation(conversation_id_2)

        conversation_1.merge(conversation_2)
        self.delete_conversation(conversation_id_2)

    def export_conversations_to_json(self) -> str:
        conversations_data = [conv.to_dict() for conv in self.conversations.values()]
        return json.dumps(conversations_data, indent=2)

    def import_conversations_from_json(self, json_data: str) -> None:
        try:
            conversations_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")

        if not isinstance(conversations_data, list):
            raise ValueError("JSON data should be a list of conversation dictionaries.")

        for conv_data in conversations_data:
            try:
                conversation = StateMachine.from_dict(conv_data)
            except ValueError as e:
                raise ValueError(f"Invalid conversation data: {e}")

            self.add_conversation(conversation)

    def _add_message(
        self,
        conversation_ids: Union[str, List[str]],
        messages: Union[str, List[str]],
        author_type: RoleType,
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        if isinstance(conversation_ids, str):
            conversation_ids = [conversation_ids]

        if isinstance(messages, str):
            messages = [messages]

        if isinstance(parent_ids, str):
            parent_ids = [parent_ids]
        elif parent_ids is None:
            parent_ids = [None] * len(messages)

        for conversation_id in conversation_ids:
            for message, parent_id in zip(messages, parent_ids):
                message_id = generate_id()

                content = Content(text=message)

                author = Author(role=author_type)

                self.add_message(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    content=content,
                    author=author,
                    parent=parent_id,
                )

    def handle_user_input(
        self,
        conversation_ids: Union[str, List[str]],
        user_input: Union[str, List[str]],
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        self._add_message(conversation_ids, user_input, RoleType.USER, parent_ids)

    def handle_agent_response(
        self,
        conversation_ids: Union[str, List[str]],
        agent_response: Union[str, List[str]],
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        self._add_message(
            conversation_ids, agent_response, RoleType.ASSISTANT, parent_ids
        )

    def handle_system_message(
        self,
        conversation_ids: Union[str, List[str]],
        system_message: Union[str, List[str]],
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        self._add_message(conversation_ids, system_message, RoleType.SYSTEM, parent_ids)

    def get_messages(self, conversation_id: str) -> List[Chain]:
        conversation = self.get_conversation(conversation_id)
        return conversation.get_messages()


class PromptManager(Element):
    def __init__(
        self,
        directory: str = "chain_database/prompts",
        filename_pattern: str = "synergy_chat_{}.json",
        credentials: dict = None,
    ):
        """
        Initialize the PromptManager.

        Args:
        - directory (str): The directory where prompts will be stored.
        - filename_pattern (str): The pattern for filenames. '{}' is replaced by the prompt number.
        """
        self.directory = directory
        self.filename_pattern = filename_pattern
        self.credentials = credentials

        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def create_directory(self, name: str) -> None:
        if not os.path.exists(name):
            os.makedirs(name)

    @staticmethod
    def create_dataset_from_images(
        path: str,
        file_patterns: List[str] = ["*.png", "*.jpg", "*.jpeg"],
        width: Optional[int] = None,
        height: Optional[int] = None,
        return_as_dataset: bool = True,
        data_types: Optional[List[str]] = None,
        format_type: str = "PNG",
        return_list: bool = True,
    ):
        import datasets

        def get_files(directory: str, patterns: List[str]) -> List[str]:
            files = []
            for pattern in patterns:
                full_pattern = os.path.join(directory, pattern)
                files.extend(glob.glob(full_pattern, recursive=True))
            return files

        files = get_files(directory=path, patterns=file_patterns)
        image_files = sorted(files, key=lambda x: int(re.findall(r"\d+", x)[0]))
        # If data_types is None or a single value, use default data types
        if data_types is None:
            data_types = ["image", "label", "bytes"]

        data_items = []
        for file_path in image_files:
            label = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, "rb") as image_file:
                image = Image.open(image_file)
                image.load()

                # Convert RGBA to RGB if necessary
                if image.mode == "RGBA":
                    image = image.convert("RGB")

                if width is not None and height is not None:
                    image = image.resize((width, height))

                # Save the image to a BytesIO object and get the byte value
                byte_stream = io.BytesIO()
                image.save(byte_stream, format=format_type)
                byte_array = byte_stream.getvalue()

                # Create a dictionary of data items
                if return_list:
                    data_item = [image, label, byte_array]
                else:
                    data_item = dict(zip(data_types, [image, label, byte_array]))

                data_items.append(data_item)

        # Check whether to return as dataset or as raw data items
        if return_as_dataset:
            # Construct the dataset dictionary based on provided data_types
            dataset_dict = {
                dtype: [item[dtype] for item in data_items] for dtype in data_types
            }
            # Convert the dictionary into a Hugging Face dataset
            dataset = datasets.Dataset.from_dict(dataset_dict)
            return dataset
        else:
            # Return the raw data items
            return data_items

    def _get_prompt_file_path(self, prompt_num: int) -> str:
        """Get the file path for a given prompt number."""
        filename = self.filename_pattern.format(prompt_num)
        return os.path.join(self.directory, filename)

    def load_prompt(self, prompt_num: Optional[int] = None) -> dict:
        """Load a prompt object."""
        if not prompt_num:
            prompt_num = self.get_last_prompt_num()

        prompt_path = self._get_prompt_file_path(prompt_num)
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                return json.load(f)
        else:
            self.logger.warning(f"Prompt {prompt_num} file does not exist.")
            return {"status": "NOT_FOUND"}

    def delete_all_prompts(self):
        """Remove all prompts."""
        for f in glob.glob(
            os.path.join(self.directory, self.filename_pattern.format("*"))
        ):
            os.remove(f)
        self.logger.info("All prompts deleted.")

    def get_last_prompt_num(self) -> int:
        """Retrieve the number of the latest prompt."""
        prompt_files = glob.glob(
            os.path.join(self.directory, self.filename_pattern.format("*"))
        )
        if prompt_files:
            prompt_files.sort(
                key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group())
            )
            return int(re.search(r"\d+", os.path.basename(prompt_files[-1])).group())
        return 0

    def format_markdown(self, data: Any, data_type: str) -> str:
        """
        Format the data into Markdown.

        Args:
            data: The data to be formatted.
            data_type: The type of data being formatted.

        Returns:
            A string formatted in Markdown.
        """
        if data_type in ["prompt", "response", "revised_prompt"]:
            return f"## {data_type.capitalize()}\n\n{data}"
        elif data_type == "embedding":
            return f"```json\n{json.dumps(data, indent=4)}\n```"
        else:
            return str(data)

    def _write_data_to_file(
        self, file_path: str, data, data_type: str, file_extension: str
    ) -> None:
        with open(file_path, "wb" if data_type == "image" else "w") as file:
            if data_type == "image":
                data.save(file, "PNG")
            else:
                json_data = {data_type: data} if file_extension == "json" else data
                if file_extension == "json":
                    file.write(json.dumps(json_data, indent=4))
                elif file_extension == "md":
                    file.write(self.format_markdown(json_data, data_type))
                else:
                    file.write(str(json_data))

    def _get_last_saved_prompt_for_data_type(self, specific_directory: str):
        last_saved_prompt = 0
        for file_name in os.listdir(specific_directory):
            if file_name.endswith(".json"):
                prompt_num = int(re.search(r"\d+", file_name).group())
                last_saved_prompt = max(last_saved_prompt, prompt_num)
        return last_saved_prompt

    def _get_last_saved_prompt(self, data_types: list):
        last_saved_prompt = 0
        for data_type in data_types:
            specific_directory = os.path.join(self.directory, data_type)
            if os.path.exists(specific_directory):
                last_saved_prompt = max(
                    last_saved_prompt,
                    self._get_last_saved_prompt_for_data_type(specific_directory),
                )
        return last_saved_prompt

    def _handle_save_operation(
        self,
        prompt_num,
        data_types,
        base_directory,
        extension,
        transformation_func,
        include_metadata,
    ):
        prompt_object = self.load_prompt(prompt_num)

        for data_type in data_types:
            try:
                data_to_save = (
                    prompt_object.get(data_type)
                    if data_type != "image"
                    else self.get_image(prompt_num)
                )
                if data_to_save:
                    if transformation_func:
                        data_to_save = transformation_func(data_to_save)

                    if include_metadata:
                        metadata = {
                            "prompt_number": prompt_num,
                            "creation_date": prompt_object.get(
                                "creation_date", "Unknown"
                            ),
                        }
                        data_to_save = {"metadata": metadata, data_type: data_to_save}

                    specific_directory = os.path.join(base_directory, data_type)
                    self.create_directory(specific_directory)
                    file_extension = "png" if data_type == "image" else extension
                    file_path = os.path.join(
                        specific_directory,
                        f"prompt_{prompt_num}_{data_type}.{file_extension}",
                    )
                    self._write_data_to_file(
                        file_path, data_to_save, data_type, file_extension
                    )
                    self.logger.info(
                        f"{data_type.capitalize()} for prompt {prompt_num} saved to {file_path}."
                    )
            except Exception as e:
                self.logger.error(
                    f"Error processing {data_type} for prompt {prompt_num}: {e}"
                )

    def save_data(
        self,
        data_types: list,
        base_directory: str,
        extension: str = "json",
        transformation_func=None,
        include_metadata: bool = False,
        max_workers: int = 5,
        incremental: bool = False,
    ) -> None:
        """
        Saves specified data types using parallel processing, with enhanced error handling and support for incremental saving.

        Args:
            data_types (list): The list of data types to save.
            base_directory (str): The base directory for saving files.
            extension (str): File extension for saved files. Default is 'json'.
            transformation_func (function): Function to transform data before saving.
            include_metadata (bool): Include metadata in saved files.
            max_workers (int): Number of threads for parallel processing.
            incremental (bool): Enable incremental saving.
        """
        last_saved_prompt = (
            self._get_last_saved_prompt(data_types) if incremental else 0
        )
        last_prompt_num = self.get_last_prompt_num()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for prompt_num in range(last_saved_prompt + 1, last_prompt_num + 1):
                futures.append(
                    executor.submit(
                        self._handle_save_operation,
                        prompt_num,
                        data_types,
                        base_directory,
                        extension,
                        transformation_func,
                        include_metadata,
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in parallel save operation: {e}")

    @classmethod
    def create_prompts_from_directory(
        cls,
        directory_path: str,
        use_simple_reader: bool = False,
        generate_id: bool = False,
        split: bool = False,
        as_dataframe: bool = False,
        SimpleDirectoryReader=None,
    ) -> list:
        """
        Create prompts from a directory recursively.

        Args:
        - directory_path (str): The path to the directory to use for prompt creation.
        - use_simple_reader (bool): If True, use the SimpleDirectoryReader, otherwise use recursive_read.
        - generate_id (bool): If True, generate an ID for the prompts.
        - split (bool): If True, split the text into prompt and content based on ":".
        - as_dataframe (bool): If True, return the results as a Pandas DataFrame.

        Returns:
        - list[dict or None]: A list of created prompt objects or None in case of failure for each file.
        """
        manager = cls()

        def recursive_read(dir_path: str) -> list:
            prompts = []

            for item_path in glob.glob(os.path.join(dir_path, "*")):
                if os.path.isdir(item_path):
                    prompts.extend(recursive_read(item_path))
                else:
                    with open(item_path, "r") as f:
                        text = f.read().strip()
                        if text:
                            prompts.append(text)

            return prompts

        results = []
        if use_simple_reader:
            reader = SimpleDirectoryReader(input_dir=directory_path, recursive=True)
            data = reader.load_data()
            for text in data:
                try:
                    result = manager.create_prompt_object([text])
                    results.append(result)
                except Exception as e:
                    print(f"Error processing text from SimpleDirectoryReader: {e}")
                    results.append({"status": "FAILURE"})

        else:
            for text in recursive_read(directory_path):
                try:
                    text_to_use = text

                    if split and ":" in text_to_use:
                        prompt_str, prompt_content = text_to_use.split(":", 1)
                        if len(prompt_str.split()) < 5:
                            text_to_use = prompt_content.strip()

                    if generate_id:
                        result = manager.create_prompt(
                            [text_to_use], id=str(uuid.uuid4()) if generate_id else None
                        )
                    else:
                        result = manager.create_prompt([text_to_use])

                    results.append(result)

                except Exception as e:
                    print(f"Error processing text: {e}")
                    results.append({"status": "FAILURE"})

        data = manager.loader(patterns=["*.json"], load_prompts=True)
        if as_dataframe:
            return pd.DataFrame(data)
        else:
            return data

    def get_image(
        self, prompt_num: Optional[int] = None, show: bool = False
    ) -> Image.Image:
        """Retrieve and optionally display the image from the latest prompt."""

        if not prompt_num:
            prompt_num = self.get_last_prompt_num()

        prompt_object = self.load_prompt(prompt_num)

        if "image" in prompt_object and prompt_object["image"]:
            try:
                # Decode the base64-encoded string to bytes and then open the image
                image_bytes = base64.b64decode(prompt_object["image"])
                image = Image.open(io.BytesIO(image_bytes))
                image.load()  # Load the image to catch any errors in the image file

                # Display the image if 'show' is True
                if show:
                    image.show()

                return image
            except (base64.binascii.Error, IOError) as e:
                # Handle errors in base64 decoding or PIL image opening
                self.logger.error(f"Error loading image for prompt {prompt_num}: {e}")
                return None
        else:
            self.logger.info(f"No image found in prompt number {prompt_num}.")
            return None

    def create_prompt_object(
        self,
        prompt_parts: list,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        id: Optional[str] = None,
        embedding: Optional[object] = None,
        image_format: str = "PNG",
    ) -> dict:
        """Construct a new prompt object and fetch image from URL to store in-memory."""

        # Validate input parameters
        if not prompt_parts:
            self.logger.error("No content provided for the prompt.")
            return {"status": "FAILURE"}

        prompt_num = self.get_last_prompt_num() + 1

        # Initialize the prompt object
        prompt_object = {
            "id": id,
            "create_time": str(datetime.now()),
            "prompt_num": prompt_num,
            "prompt": prompt,
            "response": prompt_parts,
            "revised_prompt": revised_prompt,
            "embedding": embedding,
            "image": None,
        }

        if prompt:
            prompt_object["prompt"] = prompt

        if revised_prompt:
            prompt_object["revised_prompt"] = revised_prompt

        # If an image URL is provided, fetch the image and store it in memory
        if image_url:
            try:
                response = requests.get(image_url)
                response.raise_for_status()

                # Load image into PIL
                image = Image.open(io.BytesIO(response.content))

                # Save the image in the specified format in-memory
                image_bytes_io = io.BytesIO()
                image.save(image_bytes_io, format=image_format)
                image_bytes_io.seek(0)

                # Convert image data to base64-encoded string
                base64_encoded_image = base64.b64encode(
                    image_bytes_io.getvalue()
                ).decode("utf-8")

                # Store the base64-encoded image data
                prompt_object["image"] = base64_encoded_image

            except requests.RequestException as e:
                self.logger.error(f"Request failed for image URL {image_url}: {e}")
                prompt_object["image"] = "Request failed"

            except IOError as e:
                self.logger.error(
                    f"IOError while processing image from URL {image_url}: {e}"
                )
                prompt_object["image"] = "IOError"

        return prompt_object

    def save_prompt_object(self, prompt_object: dict) -> dict:
        """Save the prompt object to a file."""
        try:
            prompt_num = prompt_object["prompt_num"]
            prompt_path = self._get_prompt_file_path(prompt_num)
            with open(prompt_path, "w") as f:
                json.dump(prompt_object, f, indent=4)

            # self.logger.info(f"Prompt {prompt_object['prompt_num']} saved.")
        except Exception as e:
            self.logger.error(f"Failed to save prompt: {e}")
            return {"status": PromptStatus.FAILURE.value}
        return {"status": PromptStatus.SUCCESS.value}

    def delete_prompt(self, prompt_num: int):
        """Remove a specific prompt."""
        try:
            prompt_path = self._get_prompt_file_path(prompt_num)
            if os.path.exists(prompt_path):
                os.remove(prompt_path)
                self.logger.info(f"Prompt {prompt_num} deleted.")
            else:
                self.logger.warning(f"Prompt {prompt_num} does not exist.")
        except Exception as e:
            self.logger.error(f"Failed to delete prompt {prompt_num}: {e}")

    def list_prompts(self) -> list:
        """List all available prompt filenames."""
        return [
            os.path.basename(f)
            for f in glob.glob(
                os.path.join(self.directory, self.filename_pattern.format("*"))
            )
        ]

    def create_prompt(
        self,
        prompt_parts: list,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        id: Optional[str] = None,
        embedding: Optional[object] = None,
        upload: Optional[bool] = False,
    ):
        """Save a prompt."""

        # Create the prompt object using the extracted or default information
        prompt_object = self.create_prompt_object(
            prompt_parts=prompt_parts,
            prompt=prompt,
            image_url=image_url,
            revised_prompt=revised_prompt,
            id=id,
            embedding=embedding,
        )
        if upload:
            return self.save_prompt_object(prompt_object)

        else:
            return self.save_prompt_object(prompt_object)


class CloudManager(PromptManager):
    def __init__(
        self,
        credentials: dict,
        directory: str = "prompts",
        filename_pattern: str = "synergy_chat_{}.json",
    ):
        """
        Initialize the CloudManager.

        Args:
        - directory (str): The directory where prompts will be stored.
        - filename_pattern (str): The pattern for filenames. '{}' is replaced by the prompt number.
        - bucket_name (str): The name of the bucket to use for cloud storage.
        - project: The project ID. If not passed, falls back to the default inferred from the environment.
        - credentials: The credentials to use when creating the client. If None, falls back to the default inferred from the environment.
        - http: The object to be used for HTTP requests. If None, an http client will be created.
        - client_info: The client info used to send a user-agent string along with API requests.
        - client_options: The options used to create a client.
        - use_auth_w_custom_endpoint: Boolean indicating whether to use the auth credentials when a custom API endpoint is set.
        """
        super().__init__(
            directory=directory,
            filename_pattern=filename_pattern,
        )
        self.client = storage.Client.from_service_account_json(
            json_credentials_path=credentials["service_account"]
        )

        bucket_name = credentials["bucket_name"]
        self.bucket = self.client.bucket(bucket_name)

    def save_prompt_object_bucket(self, prompt_object):
        """Save a prompt object to the 'synergy_chat' bucket."""
        try:
            # Specifying the bucket name for this operation
            synergy_chat_bucket = self.client.bucket("chain_trees")

            # Generate a unique UUID for each prompt object
            prompt_id = str(uuid.uuid4())
            filename = f"chain_trees_{prompt_id}.json"

            # Add the generated ID to the prompt object
            prompt_object["id"] = prompt_id

            # Creating the blob object
            blob = synergy_chat_bucket.blob(filename)
            blob.upload_from_string(json.dumps(prompt_object))
            self.logger.info(f"Prompt {prompt_id} saved to bucket.")

        except Exception as e:
            self.logger.error(f"Error saving prompt object: {str(e)}")
            raise

    def save_convo_object_bucket(self, prompt_object, convo_id):
        """Save a prompt object to the 'synergy_chat' bucket."""
        try:
            # Specifying the bucket name for this operation
            synergy_chat_bucket = self.client.bucket("conversation_tree")

            # Generate a unique UUID for each prompt object
            filename = f"conversation_tree_{convo_id}.json"

            # Add the generated ID to the prompt object
            prompt_object["id"] = convo_id

            # Creating the blob object
            blob = synergy_chat_bucket.blob(filename)
            blob.upload_from_string(json.dumps(prompt_object))

            self.logger.info(f"Convo {convo_id} saved to bucket.")

        except Exception as e:
            self.logger.error(f"Error saving prompt object: {str(e)}")
            raise

    def update_convo_object_bucket(self, prompt_object, convo_id):
        """Save a prompt object to the 'synergy_chat' bucket."""
        try:
            # Specifying the bucket name for this operation
            synergy_chat_bucket = self.client.bucket("conversation_tree")

            # Generate a unique UUID for each prompt object
            filename = f"conversation_tree_{convo_id}.json"

            # get the blob object
            blob = synergy_chat_bucket.blob(filename)
            blob.upload_from_string(json.dumps(prompt_object))

            self.logger.info(f"COnvo {convo_id} updated in bucket.")

        except Exception as e:
            self.logger.error(f"Error saving prompt object: {str(e)}")
            raise

    def check_if_convo_exists(self, convo_id):
        """Check if a conversation exists in the 'conversation_tree' bucket."""
        try:
            # Specifying the bucket name for this operation
            synergy_chat_bucket = self.client.bucket("conversation_tree")

            # Generate a unique UUID for each prompt object
            filename = f"conversation_tree_{convo_id}.json"

            # get the blob object
            blob = synergy_chat_bucket.blob(filename)
            return blob.exists()

        except Exception as e:
            self.logger.error(f"Error saving prompt object: {str(e)}")
            raise

    def _get_prompt_blob(self, prompt_num: int):
        blob_name = self.filename_pattern.format(prompt_num)
        return self.bucket.blob(blob_name)

    def get_last_prompt_num(self) -> int:
        """Retrieve the number of the latest prompt."""
        blobs = self.bucket.list_blobs()
        if blobs:
            prompt_nums = [
                int(re.search(r"\d+", os.path.basename(blob.name)).group())
                for blob in blobs
                if blob.name.endswith(".json")
            ]
            if prompt_nums:
                return max(prompt_nums)
        return 0

    def create_prompt(
        self,
        prompt_parts: list,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        id: Optional[str] = None,
        embedding: Optional[object] = None,
        upload: Optional[bool] = None,
    ):
        """Save a prompt."""
        # Create the prompt object using the extracted or default information
        prompt_object = self.create_prompt_object(
            prompt_parts=prompt_parts,
            prompt=prompt,
            image_url=image_url,
            revised_prompt=revised_prompt,
            id=id,
            embedding=embedding,
        )
        if upload:
            return self.upload_media_in_parallel(id)

        else:
            return self.save_prompt_object_bucket(prompt_object)

    def upload_file(
        self,
        file_path: str,
        bucket_subdir: str,
        verbose: bool = True,
        max_retries: int = 3,
    ) -> None:
        """
        Uploads a file to a specified bucket directory with retry logic.

        Args:
            file_path (str): The local path of the file to be uploaded.
            bucket_subdir (str): The subdirectory in the bucket where the file will be uploaded.
            verbose (bool): If True, enables verbose output.
            max_retries (int): Maximum number of retries for the upload.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                blob = self.bucket.blob(
                    os.path.join(bucket_subdir, os.path.basename(file_path))
                )

                blob.upload_from_filename(file_path)
                if verbose:
                    print(f"Uploaded {file_path} to {bucket_subdir}")

                    # return gsutil UR
                return f"gs://{self.bucket.name}/{blob.name}"

            except (TimeoutError, ConnectionError) as e:
                attempt += 1
                if verbose:
                    self.logger.info(
                        f"Retry {attempt}/{max_retries} for {file_path} due to error: {e}"
                    )
                if attempt == max_retries:
                    self.logger.info(
                        f"Failed to upload {file_path} after {max_retries} attempts"
                    )
            except Exception as e:
                if verbose:
                    self.logger.error(f"Error uploading {file_path}: {e}")
                break  # Break on other types of exceptions

    def upload_batch(
        self,
        mode: str,
        batch_files: list,
        conversation_id: str,
        path: str,
        verbose: bool,
    ) -> None:
        """
        Processes and uploads a batch of files.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            batch_files (list): List of file names to be uploaded.
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            verbose (bool): If True, enables verbose output.
            rate_limit_delay (float): The delay between each file upload to respect rate limits.
        """
        bucket_subdir = os.path.join(mode, conversation_id)
        for file_name in batch_files:
            file_path = os.path.join(path, mode, conversation_id, file_name)
            self.upload_file(file_path, bucket_subdir, verbose)

    def initialize_upload(
        self, mode: str, conversation_id: str, path: str, batch_size: int, verbose: bool
    ) -> list:
        """
        Initializes the upload process by creating batches of files.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            batch_size (int): The number of files in each batch.
            verbose (bool): If True, enables verbose output.

        Returns:
            list: A list of tuples, each containing the arguments for uploading a batch of files.
        """
        directory = os.path.join(path, mode, conversation_id)
        if not os.path.exists(directory):
            if verbose:
                self.logger.info(f"Directory not found: {directory}")
            return []

        file_list = os.listdir(directory)
        return [
            (mode, file_list[i : i + batch_size], conversation_id, path, verbose)
            for i in range(0, len(file_list), batch_size)
        ]

    def upload_media_in_parallel(self, conversation_id: str, directory=None, **kwargs):
        """
        Manages the parallel upload of media files across different modalities.

        Args:
            conversation_id (str): The conversation ID associated with the files.
            **kwargs: Additional keyword arguments including:
                      - path (str): The base path where the files are located.
                      - batch_size (int): The number of files in each batch.
                      - verbose (bool): If True, enables verbose output.
                      - rate_limit_delay (float): The delay between each file upload to respect rate limits.
        """
        modality = [
            "audio",
            "text",
            "image",
            "caption",
            "prompt",
            "zip",
            "csv",
            "content",
            "script",
        ]
        if directory is None:
            directory = self.directory
        else:
            directory = directory

        path = kwargs.get("path", directory)
        batch_size = kwargs.get("batch_size", 10)
        verbose = kwargs.get("verbose", True)

        with ThreadPoolExecutor(max_workers=len(modality)) as executor:
            futures = []
            for mode in modality:
                tasks = self.initialize_upload(
                    mode, conversation_id, path, batch_size, verbose
                )
                for task in tasks:
                    futures.append(executor.submit(self.upload_batch, *task))

            for future in tqdm.tqdm(futures, total=len(futures)):
                future.result()

    def upload_all_media_in_parallel(self, base_path):
        """
        Manages the parallel upload of media files across different modalities for all conversations.

        Args:
            **kwargs: Additional keyword arguments including:
                      - path (str): The base path where the files are located.
                      - batch_size (int): The number of files in each batch.
                      - verbose (bool): If True, enables verbose output.
                      - rate_limit_delay (float): The delay between each file upload to respect rate limits.
        """
        title_list = os.listdir(base_path)
        title_list = [title for title in title_list if title != ".DS_Store"]
        for title in title_list:
            directory = os.path.join(base_path, title)
            for folder_id in os.listdir(os.path.join(directory, "image")):
                self.upload_media_in_parallel(folder_id, path=directory)

    def upload_from_directory(
        self, directory_path: str, bucket_subdirectory: str = "", verbose: bool = True
    ):
        """
        Uploads all files from a specified local directory to a Google Cloud Storage bucket.

        Args:
            directory_path (str): The path to the local directory.
            bucket_subdirectory (str): The subdirectory in the bucket where files will be uploaded.
        """
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                # Construct the destination blob name
                destination_blob_name = (
                    os.path.join(bucket_subdirectory, filename)
                    if bucket_subdirectory
                    else filename
                )
                # Upload the file
                self.upload_file(file_path, destination_blob_name, verbose=verbose)
        # return url
        except FileNotFoundError as e:
            self.logger.error(f"Error uploading files from directory: {e}")

        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")

        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")

        except Exception as e:
            self.logger.error(f"Error uploading files from directory: {e}")

    def download_file(
        self, source_blob_name, destination_file_name, verbose: bool = True
    ):
        download_id = uuid.uuid4()
        try:
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            if verbose:
                self.logger.info(
                    f"Download ID {download_id}: File {source_blob_name} downloaded to {destination_file_name}."
                )
        except BadRequest as e:
            self.logger.error(f"Download ID {download_id}: Bad request error: {e}")

        except GoogleAPIError as e:
            self.logger.error(f"Download ID {download_id}: Service error: {e}")

        except Exception as e:
            self.logger.error(f"Download ID {download_id}: Unexpected error: {e}")

    def process_blob(self, blob, destination_directory, verbose):
        if not blob.name.endswith("/"):
            destination_file_name = os.path.join(
                destination_directory, os.path.basename(blob.name)
            )
            self.download_file(blob.name, destination_file_name, verbose)

    def download_batch(self, mode, blobs, destination_directory, verbose):
        for blob in blobs:
            self.process_blob(blob, destination_directory, verbose)

    def initialize_download(self, mode, destination_directory, verbose):
        blobs = self.bucket.list_blobs(prefix=mode)
        return [(mode, blobs, destination_directory, verbose)]

    def download_media_in_parallel(self, **kwargs):
        modality = ["audio", "text", "image", "video"]
        destination_directory = kwargs.get("destination_directory", "chain_database")
        verbose = kwargs.get("verbose", True)

        with ThreadPoolExecutor(max_workers=len(modality)) as executor:
            futures = []
            for mode in modality:
                tasks = self.initialize_download(mode, destination_directory, verbose)
                for task in tasks:
                    future = executor.submit(self.download_batch, *task)
                    futures.append(future)

            for future in tqdm.tqdm(futures, total=len(futures)):
                future.result()

    def download_prompt(self, prompt_num: int, destination_directory: str):
        """Download a prompt object."""
        blob_name = self.filename_pattern.format(prompt_num)
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            blob.download_to_filename(os.path.join(destination_directory, blob_name))
        else:
            self.logger.warning(f"Prompt {prompt_num} file does not exist.")

    def upload_blob(self, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)
            self.logger.info(
                f"File {source_file_name} uploaded to {destination_blob_name}."
            )
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def get_url(self, blob_name: str):
        blob = self.bucket.blob(blob_name)
        return blob.public_url

    def delete_bucket(self, bucket_name: str):
        try:
            bucket = self.client.bucket(bucket_name)
            if bucket.exists():
                bucket.delete()
                self.logger.info(f"Bucket {bucket_name} successfully deleted.")
            else:
                self.logger.warning(f"Bucket {bucket_name} does not exist.")
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def list_buckets(self):
        try:
            buckets = self.client.list_buckets()
            for bucket in buckets:
                self.logger.info(bucket.name)
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def list_files(self):
        try:
            blobs = self.bucket.list_blobs()
            for blob in blobs:
                self.logger.info(blob.name)
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def load_prompt(self, prompt_num: Optional[int] = None) -> dict:
        """Load a prompt object."""
        if not prompt_num:
            prompt_num = self.get_last_prompt_num()

        blob = self._get_prompt_blob(prompt_num)
        if blob.exists():
            return json.loads(blob.download_as_string())
        else:
            self.logger.warning(f"Prompt {prompt_num} file does not exist.")
            return {"status": "NOT_FOUND"}

    def get_file(self, blob_name: str):
        try:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob
            else:
                self.logger.warning(f"File {blob_name} does not exist.")
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
