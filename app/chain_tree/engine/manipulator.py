from typing import List, Callable, Union
from chain_tree.engine.loader import DatasetLoader
import pandas as pd
import logging


class DataManipulator:
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        use_specific_patterns: bool = False,
        min_elements: int = 6,
    ):
        self.use_specific_patterns = use_specific_patterns
        self.min_elements = min_elements
        self.dataset_loader = dataset_loader
        self.original_data = dataset_loader.filter_responses(
            use_specific_patterns=self.use_specific_patterns,
            min_elements=self.min_elements,
        )
        self.prompt_col, self.response_col = dataset_loader.get_data_columns()
        self.operations = []
        self.processed_data = self.original_data.copy()

    def start(self):
        """Initialize the chain."""
        logging.info(f"Starting with {len(self.processed_data)} records.")
        return self

    def get_operations(self) -> List[str]:
        return self.operations

    def __repr__(self):
        return f"DataManipulator with {len(self.processed_data)} entries. Operations: {', '.join(self.operations)}"

    def filter_prompts_by_complexity(
        self, min_words: int, max_words: int
    ) -> "DataManipulator":
        self.processed_data = self.processed_data[
            self.processed_data[self.prompt_col].apply(
                lambda x: min_words <= len(x.split()) <= max_words
            )
        ]
        self.operations.append(
            f"Filtered prompts by complexity: {min_words}-{max_words} words. Remaining records: {len(self.processed_data)}"
        )
        return self

    def _filter_by_keyword(self, keyword: str) -> "DataManipulator":
        self.processed_data = self.processed_data[
            self.processed_data[self.response_col].str.contains(keyword, case=False)
        ]
        self.operations.append(
            f"Filtered by keyword: {keyword}. Remaining records: {len(self.processed_data)}"
        )
        return self

    def filter_by_keywords(self, keywords: List[str]) -> "DataManipulator":
        for keyword in keywords:
            self._filter_by_keyword(keyword)
        return self

    def reset_filters(self) -> "DataManipulator":
        self.processed_data = self.original_data.copy()
        self.operations = ["Reset filters"]
        return self

    def filter_by_condition(
        self, column: str, condition_fn: Callable
    ) -> "DataManipulator":
        """Dynamic Filtering based on user-specified column and condition."""
        self.processed_data = self.processed_data[
            self.processed_data[column].apply(condition_fn)
        ]
        logging.info(
            f"Filtered by {column}. Remaining records: {len(self.processed_data)}"
        )
        self.operations.append(
            f"Filtered by {column}. Remaining records: {len(self.processed_data)}"
        )
        return self

    def randomize_order(self) -> "DataManipulator":
        """Shuffle the order of rows."""
        self.processed_data = self.processed_data.sample(frac=1).reset_index(drop=True)
        logging.info("Order randomized.")
        self.operations.append("Randomized order")
        return self

    def apply_transformation(
        self, column: str, transformation_fn: Callable
    ) -> "DataManipulator":
        """Apply transformations on a specific column."""
        self.processed_data[column] = self.processed_data[column].apply(
            transformation_fn
        )
        logging.info(f"Applied transformation to column: {column}")
        self.operations.append(f"Transformed column: {column}")
        return self

    def peek(self, rows: int = 5) -> "DataManipulator":
        """Allow users to peek into the current state of the data."""
        print(self.processed_data.head(rows))
        return self

    def finalize(self) -> pd.DataFrame:
        """Conclude the chain."""
        logging.info("Finalized.")
        return self.processed_data

    def handle_error(self, error_message: str) -> "DataManipulator":
        """Gracefully handle errors and continue the chain if possible."""
        logging.error(error_message)
        return self

    def chain(self, operations: List[Callable]) -> "DataManipulator":
        """
        Apply a series of operations sequentially on the data.

        Args:
            operations (List[Callable]): A list of functions (operations) to be applied in order.
                                        Each item in the list should be a tuple of (function, args, kwargs),
                                        where function is a reference to the function to be called,
                                        args is a tuple of arguments, and kwargs is a dictionary of keyword arguments.

        Returns:
            DataManipulator: The updated DataManipulator instance after applying operations.
        """

        for operation in operations:
            function, args, kwargs = operation

            if not callable(function):
                logging.error(f"{function} is not a callable operation.")
                continue

            try:
                function(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error executing {function}: {e}")
                self.handle_error(str(e))

        return self

    def remove_duplicates(self) -> "DataManipulator":
        """Remove duplicate rows."""
        self.processed_data.drop_duplicates(inplace=True)
        logging.info(
            f"Removed duplicates. Remaining records: {len(self.processed_data)}"
        )
        self.operations.append(
            f"Removed duplicates. Remaining records: {len(self.processed_data)}"
        )
        return self

    def remove_empty_rows(self) -> "DataManipulator":
        """Remove rows with empty prompts or responses."""
        self.processed_data = self.processed_data[
            self.processed_data[self.prompt_col].str.strip().astype(bool)
        ]
        self.processed_data = self.processed_data[
            self.processed_data[self.response_col].str.strip().astype(bool)
        ]
        logging.info(
            f"Removed empty rows. Remaining records: {len(self.processed_data)}"
        )
        self.operations.append(
            f"Removed empty rows. Remaining records: {len(self.processed_data)}"
        )
        return self

    def remove_rows(
        self,
        column_types: Union[str, List[str]],
        token_limit: int = 1000,
        remove_long: bool = True,
    ) -> "DataManipulator":
        """
        Remove rows with text that are too long or too short.

        Args:
            column_types (Union[str, List[str]]): Specifies the column(s) to check ('prompt', 'response', or both).
            token_limit (int): The token limit (max for long, min for short). Defaults to 1000 for long, 5 for short.
            remove_long (bool): If True, remove long texts; if False, remove short texts.

        Returns:
            DataManipulator: The instance with modified data.
        """
        if isinstance(column_types, str):
            column_types = [column_types]

        valid_columns = [self.prompt_col, self.response_col]
        for column_type in column_types:
            if column_type not in valid_columns:
                raise ValueError(
                    f"Invalid column_type '{column_type}'. Valid options are '{self.prompt_col}', '{self.response_col}'"
                )

            if remove_long:
                self.processed_data = self.processed_data[
                    self.processed_data[column_type].apply(
                        lambda x: len(x.split()) <= token_limit
                    )
                ]
                action = "long"
            else:
                self.processed_data = self.processed_data[
                    self.processed_data[column_type].apply(
                        lambda x: len(x.split()) >= token_limit
                    )
                ]
                action = "short"

            logging.info(
                f"Removed rows with {action} {column_type}. Remaining records: {len(self.processed_data)}"
            )
            self.operations.append(
                f"Removed rows with {action} {column_type}. Remaining records: {len(self.processed_data)}"
            )

        return self

    def remove_rows_by_length(
        self, column: str, min_length: int, max_length: int
    ) -> "DataManipulator":
        """Remove rows based on the length of the text in a specific column."""
        self.processed_data = self.processed_data[
            self.processed_data[column].apply(
                lambda x: min_length <= len(x.split()) <= max_length
            )
        ]
        logging.info(
            f"Removed rows with {column} length outside range {min_length}-{max_length}. Remaining records: {len(self.processed_data)}"
        )
        self.operations.append(
            f"Removed rows with {column} length outside range {min_length}-{max_length}. Remaining records: {len(self.processed_data)}"
        )
        return self
