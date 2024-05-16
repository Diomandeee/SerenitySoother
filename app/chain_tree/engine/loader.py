from typing import Union, Optional, Tuple, List, Iterator, Dict
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path
import pandas as pd
import itertools
import logging
import random
import json
import os


class DatasetLoader:
    def __init__(
        self,
        prompt_col: str,
        response_col: str,
        data_split: str = "train",
        prompt_directory: str = "chain_database/prompt",
        dataframe: Optional[pd.DataFrame] = None,
        local_dataset_path: Optional[str] = None,
        huggingface_dataset_name: Optional[str] = None,
    ):
        self.data = None
        self.output_directory = None

        # Ensure only one data source is provided
        sources = [dataframe, local_dataset_path, huggingface_dataset_name]
        if sum(source is not None for source in sources) > 1:
            logging.error("Please provide only one data source.")
            raise ValueError("Multiple data sources provided.")

        # Load the data
        if dataframe is not None:
            self.data = dataframe
            # rename columns
            self.output_directory = Path.cwd()
            logging.info("Loaded data from DataFrame.")
        elif local_dataset_path:
            self._load_data_from_local_path(local_dataset_path, prompt_col, response_col)
            self.output_directory = Path(os.path.dirname(local_dataset_path))
        elif huggingface_dataset_name:
            self.data = self._load_data_from_huggingface(
                huggingface_dataset_name, data_split, prompt_col, response_col
            )
            self.output_directory = Path.cwd()
        else:
            logging.warning("No data source provided.")
            self.output_directory = Path.cwd()  # Defaulting to the current directory

        if self.output_directory:
            self.prompt_directory = self.output_directory / prompt_directory
            self.prompt_directory.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Data loaded and prompt directory set to: {self.prompt_directory}"
        )
        self.prompt_col = prompt_col
        self.response_col = response_col
        self.data = self.data.rename(columns={prompt_col: "prompt", response_col: "response"})
        self.data = self.clean_data(self.data, "prompt", "response")


    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the loaded dataset."""
        return self.data.shape

    def __len__(self) -> int:
        """Return the number of rows in the loaded dataset."""
        return len(self.data)

    def preview(self, n: int = 5) -> None:
        """Preview the first n rows of the loaded dataset."""
        if n <= 0:
            print("Please provide a positive integer for preview.")
            return
        print(self.data.head(n))

    def filter_generated_parts(
        self, prompt_parts: List[str], min_length: int = 10
    ) -> List[str]:
        """
        Filter out undesired generated answers based on minimum length.
        """
        filtered_parts = [part for part in prompt_parts if len(part) >= min_length]
        return filtered_parts

    def get_dataset(self) -> pd.DataFrame:
        """Return the loaded dataset."""
        return self.data

    def get_data_columns(self):
        """Return the names of the prompt and response columns."""
        return self.prompt_col, self.response_col

    def to_csv(self, path: str):
        """Saves the dataset to a csv file"""
        self.data.to_csv(path, index=False)

    def save_data(self, file_path: str) -> None:
        self.data.to_csv(Path(file_path), index=False)

    def count_keyword(
        self, keyword: str, pair_type: str = "both"
    ) -> Union[int, Dict[str, int]]:
        data = self.data  # We get the data directly from the DataHelper instance

        if pair_type == "both":
            return {
                "prompt": data[self.prompt_col]
                .apply(lambda x: x.lower().count(keyword.lower()))
                .sum(),
                "response": data[self.response_col]
                .apply(lambda x: x.lower().count(keyword.lower()))
                .sum(),
            }
        elif pair_type == self.prompt_col:
            return (
                data[self.prompt_col]
                .apply(lambda x: x.lower().count(keyword.lower()))
                .sum()
            )
        elif pair_type == self.response_col:
            return (
                data[self.response_col]
                .apply(lambda x: x.lower().count(keyword.lower()))
                .sum()
            )
        else:
            raise ValueError(
                f"Invalid pair_type. Choose from '{self.prompt_col}', '{self.response_col}', 'both'"
            )

    def count_occurrences(self, word: str, pair_type: str = "prompt") -> int:
        """Counts the number of occurrences of a word in the data"""
        if pair_type not in [self.prompt_col, self.response_col, "both"]:
            raise ValueError(
                f"Invalid pair_type. Choose from '{self.prompt_col}', '{self.response_col}', 'both'"
            )

        text = ""
        if pair_type in [self.prompt_col, "both"]:
            text += " ".join(self.data[self.prompt_col].tolist())

        if pair_type in [self.response_col, "both"]:
            text += " ".join(self.data[self.response_col].tolist())

        return Counter(text.split())[word]

    def get_prompts(self) -> List[str]:
        return self.data[self.prompt_col].tolist()

    def get_responses(self) -> List[str]:
        return self.data[self.response_col].tolist()

    def get_example_pair(self, index: int) -> Tuple[str, str]:
        if index < 0 or index >= len(self):
            logging.error(
                f"Invalid index: {index}. Index should be between 0 and {len(self) - 1}."
            )
            raise IndexError(
                f"Invalid index: {index}. Index should be between 0 and {len(self) - 1}."
            )

        row = self.data.loc[index]
        return row["prompt"], row["response"]

    def get_examples(
        self, num_examples: int, random_order: bool = False
    ) -> List[Tuple[str, str]]:
        if num_examples <= 0 or num_examples > len(self):
            logging.error(f"Invalid number of examples requested: {num_examples}.")
            raise ValueError(f"Invalid number of examples requested: {num_examples}.")

        if random_order:
            return [
                self.get_example_pair(i) for i in self.data.sample(num_examples).index
            ]
        else:
            return [self.get_example_pair(i) for i in range(num_examples)]

    def get_examples_random(self, num_examples: int) -> List[Tuple[str, str]]:
        return self.get_examples(num_examples, random=True)

    def get_random_example_pair(self) -> Tuple[str, str]:
        return self.get_example_pair(random.randint(0, self.data.shape[0] - 1))

    def get_example_pairs(self) -> List[Tuple[str, str]]:
        return list(zip(self.get_prompts(), self.get_responses()))

    def get_next_example_pair(
        self, get_last_only: bool = False, id_col: str = "Number"
    ) -> Iterator[Tuple[str, str]]:
        """
        Get the next example pair. If get_last_only is True, return the last pair.
        Otherwise, return an iterator over all pairs.

        Args:
            get_last_only (bool): Flag to control whether to return just the last pair.

        Returns:
            Iterator[Tuple[str, str]]: An iterator over example pairs or the last pair.
        """
        if get_last_only:
            if self.data.shape[0] > 0:
                last_row = self.data.iloc[-1]
                return (
                    last_row[self.prompt_col],
                    last_row[self.response_col],
                    last_row[id_col],
                )
            else:
                raise ValueError("No data available to return the last example pair.")

        else:
            # Assuming self.data is iterable with each row containing a pair
            for index, row in self.data.iterrows():
                yield row[self.prompt_col], row[self.response_col], index

    def combination_factory(
        self, prompts: List[str], responses: List[str], file_name: str
    ) -> None:
        combinations = list(itertools.product(prompts, responses))
        df = pd.DataFrame(combinations, columns=["prompt", "response"])
        self.save_data(self.output_directory / file_name)
        df.to_csv(self.output_directory / file_name, index=False)

    def _load_data_from_local_path(self, dataset_path: str, prompt_col, response_col):
        """Load data from a given local path."""

        if not Path(dataset_path).exists():
            logging.error(f"Provided dataset path does not exist: {dataset_path}")
            raise FileNotFoundError("Dataset path not found.")

        _, file_extension = os.path.splitext(dataset_path)
        if file_extension == ".csv":
            self.data = pd.read_csv(dataset_path)

            # rename columns
            self.data = self.data.rename(columns={prompt_col: "prompt", response_col: "response"})



            logging.info(f"Loaded data from CSV at {dataset_path}")
        elif file_extension == ".json":
            with open(dataset_path, "r") as file:
                json_data = json.load(file)
                self.data = pd.DataFrame(json_data)
            logging.info(f"Loaded data from JSON at {dataset_path}")
        else:
            logging.error(
                f"Unsupported file format {file_extension}. Only .csv and .json are supported."
            )
            raise ValueError("Unsupported file format.")

    def _load_data_from_huggingface(
        self, dataset_name: str, split: str, prompt_col, response_col
    ):
        """Load data from the HuggingFace datasets library."""

        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)

        # Rename columns
        dataset = dataset.rename_column("input_text", prompt_col)
        dataset = dataset.rename_column("output_text", response_col)

        # Convert to pandas DataFrame
        self.data = dataset.to_pandas()

        logging.info(f"Loaded data from HuggingFace dataset: {dataset_name} ({split})")

        return self.data

    def generate_training_examples(
        self,
        data: Union[str, pd.DataFrame],
        filename: Optional[str] = "training_examples",
        system_message: str = "",
        return_data: bool = False,
        test_size: float = 0.1,
        random_state: Optional[int] = None,
    ) -> Optional[Tuple[str, str]]:
        # Check if data is a string (path to CSV) or a DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(
                "The 'data' argument must be either a path to a CSV file or a pandas DataFrame."
            )

        training_examples = []

        for index, row in df.iterrows():
            # Check for the existence of 'user_messages' and 'assistant_messages' columns
            if "user_messages" in df.columns and "assistant_messages" in df.columns:
                training_example = {
                    "messages": [{"role": "system", "content": system_message.strip()}]
                }
                user_messages = row["user_messages"]
                assistant_messages = row["assistant_messages"]

                if len(user_messages) != len(assistant_messages):
                    raise ValueError(
                        f"Mismatched number of user and assistant messages in row {index}"
                    )

                for user_msg, assistant_msg in zip(user_messages, assistant_messages):
                    training_example["messages"].append(
                        {"role": "user", "content": user_msg}
                    )
                    training_example["messages"].append(
                        {"role": "assistant", "content": assistant_msg}
                    )

            # Handle case where there's a single prompt-response pair in 'prompt' and 'response' columns
            else:
                training_example = {
                    "messages": [
                        {"role": "system", "content": system_message.strip()},
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                }

            training_examples.append(training_example)

        # Split the data into training and test sets
        train_data, test_data = train_test_split(
            training_examples,
            test_size=test_size,
            random_state=random_state,
        )

        if return_data:
            # convert to jsonl format
            train_data = [json.dumps(example) for example in train_data][0]
            test_data = [json.dumps(example) for example in test_data][0]

            return train_data, test_data

        else:
            # Save training examples to a .jsonl file
            with open(f"{filename}_train.jsonl", "w") as f:
                for example in train_data:
                    f.write(json.dumps(example) + "\n")

            # Save test examples to a separate .jsonl file
            with open(f"{filename}_test.jsonl", "w") as f:
                for example in test_data:
                    f.write(json.dumps(example) + "\n")

            print(f"Training examples saved to {filename}_train.jsonl")
            print(f"Test examples saved to {filename}_test.jsonl")

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        output_path: Union[str, Path],
        file_format: str = "csv",  # Default is csv, but can be set to json
        prompt_dir: str = "prompts",
        prompt_col: str = "prompt",  # New parameter
        response_col: str = "response",  # New parameter
    ) -> "DatasetLoader":
        """
        Create a DatasetLoader instance from a pandas DataFrame.

        Args:
        - dataframe (pd.DataFrame): The input dataframe.
        - output_path (Union[str, Path]): Path to save the processed dataset.
        - file_format (str): Desired output format, either "csv" or "json".
        - prompt_dir (str): Directory to save prompts.
        - prompt_col (str): Name of the prompt column.
        - response_col (str): Name of the response column.

        Returns:
        DatasetLoader: An instance of the DatasetLoader class.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        # Ensure that the dataframe contains the necessary columns
        required_columns = [prompt_col, response_col]
        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f"DataFrame is missing columns: {', '.join(missing_columns)}"
            )

        # Save to the appropriate format
        if file_format == "csv":
            dataframe.to_csv(output_path, index=False)
        elif file_format == "json":
            dataframe.to_json(output_path, orient="records", indent=4)
        else:
            raise ValueError(
                f"Unsupported file format: {file_format}. Supported formats are 'csv' and 'json'."
            )

        print(f"Data successfully saved to {output_path}")

        return cls(
            str(output_path),
            prompt_dir,
            prompt_col=prompt_col,
            response_col=response_col,
        )

    def filter_responses(
        self,
        use_specific_patterns: bool = False,
        min_elements: Optional[int] = 6,
        element_type: str = "STEP",
    ) -> pd.DataFrame:
        """
        Filter responses based on the data source and certain conditions.

        Args:
        - use_specific_patterns (bool): Whether to use specific patterns for filtering.
        - min_elements (int, optional): Minimum number of elements for filtering.
        - element_type (str, optional): Type of element for filtering.

        Returns:
        - pd.DataFrame: The filtered or original data.
        """

        SPF = [
            "Imagine That:",
            "Brainstorming:",
            "Thought Provoking Questions:",
            "Create Prompts:",
            "Synergetic Prompt:",
            "Category:",
        ]
        if not use_specific_patterns:
            logging.info("Using data directly without filtering.")
            return self.data

        # If specific patterns are to be used for filtering
        cleaned_data = self.data.copy()

        for pattern in SPF:
            cleaned_data = cleaned_data[
                cleaned_data[self.response_col].str.contains(pattern)
            ]

        cleaned_data = cleaned_data[
            cleaned_data[self.response_col].apply(
                lambda x: len([part for part in x.split(element_type) if ":" in part])
                >= min_elements
            )
        ]

        logging.info(
            f"Filtered data using specific patterns and found {len(cleaned_data)} relevant responses."
        )
        return cleaned_data

    def clean_data(
        dataframe: pd.DataFrame, prompt_col: str, response_col: str
    ) -> pd.DataFrame:
        """
        Clean the loaded dataset.

        Args:
        - dataframe (pd.DataFrame): The input DataFrame.
        - prompt_col (str): Name of the prompt column.
        - response_col (str): Name of the response column.

        Returns:
        - pd.DataFrame: The cleaned dataset.
        """

        if dataframe is None or not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Invalid dataframe object passed.")

        for col in [prompt_col, response_col]:
            if col not in dataframe.columns:
                raise ValueError(f"The column {col} is missing in the dataframe.")

        # Convert any non-string values to strings
        dataframe[prompt_col] = dataframe[prompt_col].astype(str)
        dataframe[response_col] = dataframe[response_col].astype(str)

        # Replace None or NaN with empty strings
        dataframe.fillna("", inplace=True)

        # Clean up leading and trailing whitespaces from both columns
        for col in [prompt_col, response_col]:
            dataframe[col] = dataframe[col].apply(
                lambda x: "\n".join([line.strip() for line in str(x).split("\n")])
            )

        # Remove duplicate rows
        dataframe.drop_duplicates(subset=[prompt_col, response_col], inplace=True)

        # Remove rows where the prompt and response are the same
        dataframe = dataframe[
            dataframe[prompt_col] != dataframe[response_col]
        ].reset_index(drop=True)

        # Remove rows where the prompt is empty
        dataframe = dataframe[dataframe[prompt_col] != ""].reset_index(drop=True)

        # Remove rows where the response is empty
        dataframe = dataframe[dataframe[response_col] != ""].reset_index(drop=True)

        return dataframe

    def save_prompts(self, prompts: List[str], filename: str) -> None:
        """
        Save prompts to a file.

        Args:
        - prompts (List[str]): A list of prompts to save.
        - filename (str): The name of the file to save the prompts to.
        """
        with open(self.prompt_directory / filename, "w") as file:
            for prompt in prompts:
                file.write(prompt + "\n")

        logging.info(f"Prompts saved to {self.prompt_directory / filename}")

    def load_prompts(self, filename: str) -> List[str]:
        """
        Load prompts from a file.

        Args:
        - filename (str): The name of the file to load the prompts from.

        Returns:
        - List[str]: A list of prompts loaded from the file.

        """

        with open(self.prompt_directory / filename, "r") as file:
            prompts = file.readlines()

        logging.info(f"Prompts loaded from {self.prompt_directory / filename}")
        return [prompt.strip() for prompt in prompts]

    def save_responses(self, responses: List[str], filename: str) -> None:
        """
        Save responses to a file.

        Args:
        - responses (List[str]): A list of responses to save.
        - filename (str): The name of the file to save the responses to.
        """
        with open(self.prompt_directory / filename, "w") as file:
            for response in responses:
                file.write(response + "\n")

        logging.info(f"Responses saved to {self.prompt_directory / filename}")
