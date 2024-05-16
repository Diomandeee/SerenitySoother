from typing import List, Optional, Callable, Any, Tuple, Union, Dict
from chain_tree.engine.manipulator import DataManipulator
from chain_tree.engine.retriever import DataRetriever
from chain_tree.engine.embedder import OpenAIEmbedding
from chain_tree.engine.loader import DatasetLoader
from chain_tree.engine.tuner import DataTuner
from chain_tree.models import BaseParams
from chain_tree.utils import log_handler
from chain_tree.type import ElementType
from numpy.linalg import norm
from torch import Tensor
import pandas as pd
import numpy as np
import itertools
import torch
import uuid
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calculate_similarity(embeddings1, embeddings2):
    """
    Calculate semantic similarity between two sets of embeddings using cosine similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Function to convert embeddings to numpy arrays
    def convert_to_numpy(embeddings):
        if isinstance(embeddings, Tensor):
            return embeddings.detach().cpu().numpy()
        elif isinstance(embeddings, list) or isinstance(embeddings, np.ndarray):
            return np.array(embeddings)
        elif isinstance(embeddings, list[0]):
            return np.array(embeddings)

        else:
            raise TypeError(
                "Unsupported embedding type. Must be a list, numpy array, or PyTorch tensor."
            )

    # Convert embeddings to numpy arrays
    embeddings1_array = convert_to_numpy(embeddings1).reshape(1, -1)
    embeddings2_array = convert_to_numpy(embeddings2).reshape(1, -1)

    # Check for NaN values
    if np.isnan(embeddings1_array).any() or np.isnan(embeddings2_array).any():
        print(
            "Warning: Embeddings contain NaN values. Returning similarity score as 0."
        )
        return 0.0

    # Normalize the embeddings
    embeddings1_array = embeddings1_array / norm(embeddings1_array)
    embeddings2_array = embeddings2_array / norm(embeddings2_array)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings1_array, embeddings2_array)
    similarity_score = np.clip(similarity_matrix[0][0], 0, 1)

    return similarity_score


def calculate_cross_entropy_loss(embedding1, embedding2) -> float:
    """
    Calculate the cross-entropy loss between two embeddings.

    Parameters:
        embedding1 (torch.Tensor or list): First embedding vector.
        embedding2 (torch.Tensor or list): Second embedding vector.

    Returns:
        float: The cross-entropy loss.
    """
    if not isinstance(embedding1, torch.Tensor):
        embedding1 = torch.tensor(embedding1)
    if not isinstance(embedding2, torch.Tensor):
        embedding2 = torch.tensor(embedding2)

    loss = torch.nn.functional.kl_div(
        torch.log_softmax(embedding1, dim=0), embedding2, reduction="sum"
    )
    return loss.item()


class ChainEngine:
    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        local_dataset_path: Optional[str] = None,
        huggingface_dataset_name: Optional[str] = None,
        prompt_subdir: Optional[str] = None,
        prompt_col: str = "prompt",
        conversation_dir: str = "conversation/",
        response_col: str = "response",
        root_directory: Optional[str] = None,
        verbose: bool = False,
        api_key: Optional[str] = None,
    ):
        # Default to the root directory of the current script if not provided
        if root_directory is None:
            root_directory = os.getcwd()

        # Here, concatenate root_directory and prompt_subdir to form the full path
        if prompt_subdir:
            full_prompt_directory = (
                os.path.join(root_directory, prompt_subdir)
                if root_directory
                else prompt_subdir
            )
        else:
            full_prompt_directory = root_directory

        self.dataset_loader = DatasetLoader(
            dataframe=dataframe,  # Pass the DataFrame here
            local_dataset_path=local_dataset_path,
            huggingface_dataset_name=huggingface_dataset_name,
            prompt_directory=full_prompt_directory,
            prompt_col=prompt_col,
            response_col=response_col,
        )

        self.data_embedder = OpenAIEmbedding(api_key=api_key)
        self.data_helper = DataManipulator(self.dataset_loader)
        self.data_tuner = DataTuner(
            self.dataset_loader,
            verbose=verbose,
            client=self.data_embedder.client,
        )
        self.data_retriever = DataRetriever(
            manipulator=self.data_helper, embedder=self.data_embedder
        )
        self.data = self.dataset_loader.data
        self.prompt_col = prompt_col
        self.response_col = response_col
        self.root_directory = root_directory
        self.conversation_dir = conversation_dir

    @staticmethod
    def concat_steps(
        df: pd.DataFrame,
        column_names: List[str],
        new_column_name="Steps",
        split_by="\n\n",
    ):
        """
        This function takes a dataframe and a list of column names and concatenates the columns into a new column.

        :param df: Pandas DataFrame containing the columns to concatenate.
        :param column_names: List of strings of the column names to concatenate.
        :param new_column_name: String, the name of the new column after concatenation.
        :param split_by: String, the separator used to split the concatenated strings.
        :return: Pandas DataFrame with the new concatenated column.
        """
        # Check if all the specified columns are in the dataframe
        missing_cols = [col for col in column_names if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following columns are missing from the dataframe: {missing_cols}"
            )

        # Concatenate the columns with the specified separator
        df[new_column_name] = df[column_names].apply(
            lambda row: split_by.join(row.values.astype(str)), axis=1
        )
        return df

    @classmethod
    def create_engines(
        cls, step_list, dataframe, root_directory, api_key, method="cartesian"
    ):
        """
        Class method to create a list of ChainEngine instances using either cartesian product or combinations.

        :param step_list: List of steps to create engines for.
        :param dataframe: Pandas DataFrame containing the data.
        :param root_directory: The root directory for the DataEngines.
        :param method: String, either 'cartesian' for Cartesian product or 'combination' for combinations.
        :return: List of DataEngine instances.
        """
        engines = []
        # Generate pairs using the specified method
        if method == "cartesian":
            pairs = itertools.product(step_list, repeat=2)
        elif method == "combination":
            pairs = itertools.combinations(step_list, 2)
        else:
            raise ValueError("Invalid method. Use 'cartesian' or 'combination'.")

        # Create DataEngine instances for each pair
        for prompt_col, response_col in pairs:
            # Skip creating an engine with the same prompt and response for cartesian
            if method == "cartesian" and prompt_col == response_col:
                continue
            engine = cls(
                dataframe=dataframe,
                prompt_col=prompt_col,
                response_col=response_col,
                root_directory=root_directory,
                api_key=api_key,
            )
            engines.append(engine)
        return engines

    def _prepare_conversation_data(
        self,
        num_prompts: Optional[int] = None,
        get_last_only: bool = False,
        id_col: str = None,
    ) -> List[Tuple[str, str]]:
        if num_prompts is None:
            num_prompts = len(self.data)

        example_pair_generator = self.dataset_loader.get_next_example_pair(
            get_last_only, id_col
        )
        example_pairs = []
        for _ in range(num_prompts):
            try:
                example_pairs.append(next(example_pair_generator))
            except StopIteration:
                break

        return example_pairs

    def execute_chain(self, operations: List[Callable]) -> "ChainEngine":
        """
        Apply a series of operations sequentially on the data using a chainable pattern.

        Args:
            operations (List[Callable]): A list of functions (operations) to be applied in order.
            Each item in the list should be a tuple of (function, args, kwargs),
            where function is a reference to the function to be called,
            args is a tuple of arguments, and kwargs is a dictionary of keyword arguments.

        Returns:
            DataEngine: The updated DataEngine instance after applying operations.
        """
        # Ensure the DataHelper's chain method is available
        if hasattr(self.data_helper, "chain"):
            self.data_helper.chain(operations)
        else:
            raise AttributeError(
                "DataHelper does not have a 'chain' method. Make sure it's defined."
            )

        return self

    @classmethod
    def fine_tune_and_return(
        cls,
        engine: "ChainEngine",
        dataframe_to_use: Optional[pd.DataFrame] = None,
        system_message_text: str = "",
        retrieve_model_name: bool = False,
        fine_tune: bool = True,
        base_persist_dir: str = "",
        output_name: str = "fine_tuned",
        return_data: bool = False,
        target_model: str = "gpt-3.5-turbo-0125",
    ) -> Any:
        """
        Conducts model fine-tuning based on provided data and configurations, then returns either the fine-tuned model's name or the original dataframe.

        This function orchestrates the model fine-tuning workflow leveraging the capabilities of a given `DataEngine` instance. It determines the fine-tuning process based on the `fine_tune` flag. If set to True, the method triggers the fine-tuning process, otherwise, it simply returns the input dataframe.

        Args:
            engine (ChainEngine): An instance of ChainEngine which encapsulates functionality for data processing and potentially model fine-tuning.
            dataframe_to_use (pd.DataFrame): DataFrame that will be returned if no fine-tuning is performed. This can be the data upon which decisions were based or any relevant data frame.
            system_message_text (str, optional): A system-specific message text to be used during the fine-tuning process. Defaults to an empty string.
            retrieve_model_name (bool, optional): Flag to decide whether the model's name should be retrieved after fine-tuning. Defaults to False.
            fine_tune (bool, optional): Determines if the model should undergo the fine-tuning process. Defaults to True.
            base_persist_dir (str, optional): The base directory where the model or any other persistence-related data will be stored. Defaults to an empty string.
            output_name (str, optional): The name assigned to the model post fine-tuning. Defaults to "fine_tuned".
            api_key (str, optional): The Groq API key required for the
            fine-tuning process. Defaults to an empty string.
            verbose (bool, optional): If set to True, the function will display detailed logs about its operations. Defaults to False.

        Returns:
            Any: Depending on the fine-tuning configuration, either the name of the fine-tuned model (str) or the original dataframe (pd.DataFrame) will be returned.

        Examples:
            >>> engine_instance = ChainEngine(...)
            >>> df_to_use = pd.DataFrame(...)
            >>> result = fine_tune_and_return(engine=engine_instance, dataframe_to_use=df_to_use, fine_tune=True)
            Fine-tuned model: fine_tuned_12345

        Notes:
            - It's important to ensure that the ChainEngine instance (`engine`) passed to this function has capabilities for fine-tuning.
            - If fine-tuning is not required, the input dataframe (`dataframe_to_use`) is directly returned without any modifications.
        """

        if not fine_tune and return_data:
            train_data, test_data = engine.dataset_loader.generate_training_examples(
                dataframe_to_use, output_name, system_message_text, return_data
            )
            return train_data, test_data

        if fine_tune:
            model_name = engine.data_tuner.process_and_fine_tune(
                system_message_text=system_message_text,
                output_filename=base_persist_dir + output_name,
                model_suffix=output_name,
                retrieve_model_name=retrieve_model_name,
                target_model=target_model,
            )
            log_handler(f"Fine-tuned model: {model_name}", level="info", verbose=True)
            return model_name

        return dataframe_to_use

    @staticmethod
    def preprocess_loaded_responses(
        loaded_responses: List[List[str]],
    ) -> List[List[str]]:
        """
        Preprocess the loaded responses data.
        """
        preprocessed_responses = []

        for response in loaded_responses:
            if response:
                response = [text for text in response if text.strip()]

                preprocessed_responses.append(response)

        return preprocessed_responses

    @staticmethod
    def process_simple_dataframe(
        processed_steps: List[List[str]],
        output_path: str = "processed_steps.csv",
        split_indicator: str = ":",
        message_split: Union[str, Callable[[str], List[str]], None] = "\n\n",
        element_type: ElementType = ElementType.STEP,
        start_index=0,
    ) -> pd.DataFrame:
        """
        Create a dataframe from the processed steps.
        Concatenate Prompt with Step 1 to Step n and add columns 'prompt', 'response', and 'both'.
        Compute embeddings for each step and group similar terms.
        Save the resulting dataframe to the specified output path.
        Return the dataframe.
        """

        # Find the maximum number of columns needed
        max_len = max(map(len, processed_steps))

        # Construct the data dictionary
        data = {}
        for i in range(max_len):
            col_name = "Prompt" if i == 0 else element_type.value + " " + str(i)

            # Splitting each row based on the split_indicator
            split_rows = [
                (
                    row[i].split(split_indicator, 1)
                    if len(row) > i and split_indicator in row[i]
                    else ["", ""]
                )
                for row in processed_steps
            ]

            # First part goes to 'prompt', second to 'response'
            data[col_name] = [split_row[0] for split_row in split_rows]

            if message_split:  # Only split response if message_split is provided
                if callable(message_split):  # If message_split is a function
                    responses = [
                        message_split(split_row[1]) for split_row in split_rows
                    ]
                else:  # Else, assume it's a string and split using it
                    responses = [
                        split_row[1].split(message_split) for split_row in split_rows
                    ]

                for idx, response_parts in enumerate(zip(*responses)):
                    data[element_type.value + " " + str(idx + start_index)] = list(
                        response_parts
                    )
            else:
                data["Response"] = [split_row[1] for split_row in split_rows]

            data["Both"] = [row[i] if len(row) > i else "" for row in processed_steps]

        df = pd.DataFrame(data)

        df.to_csv(output_path, index=False)

        return df

    @staticmethod
    def build_incremental_row(row, element_col, element_embed_col, element_id, i):
        return {
            "id": str(uuid.uuid4()),
            "element_id": element_id,
            "Element Type": element_col.split(" ")[0],
            "Element Index": i,
            "Element Text": row[element_col],
            "Embedding": row[element_embed_col],
        }

    @staticmethod
    def build_linear_row(row, element_col, element_embed_col, i):
        return {
            "id": str(uuid.uuid4()),
            "Prompt": row["Prompt"],  # Using the original "Prompt" column as desired
            "Element": i,
            "Element Text": row[element_col],
            "Embedding": row[element_embed_col],
        }

    @staticmethod
    def convert_to_long_format(
        dataframe: pd.DataFrame,
        element_type: ElementType = ElementType.STEP,
        format_type: str = "incremental",
    ) -> pd.DataFrame:
        long_format_data = []
        num_elements = dataframe.columns.str.startswith(element_type.value).sum()
        element_id_dict = {}

        for idx, row in dataframe.iterrows():
            prefix_text = row["Prompt"]
            element_id = element_id_dict.get(prefix_text)
            if element_id is None:
                element_id = str(uuid.uuid4())
                element_id_dict[prefix_text] = element_id

            for i in range(1, num_elements + 1):  # Adjusted to start from 1
                element_col = f"{element_type.value} {i}"
                element_embed_col = f"{element_type.value} {i} embedding"
                if (
                    element_col in dataframe.columns
                    and element_embed_col in dataframe.columns
                ):
                    if format_type == "incremental":
                        long_format_data.append(
                            ChainEngine.build_incremental_row(
                                row, element_col, element_embed_col, element_id, i
                            )
                        )
                    elif format_type == "linear":
                        long_format_data.append(
                            ChainEngine.build_linear_row(
                                row, element_col, element_embed_col, i
                            )
                        )

        long_df = pd.DataFrame(long_format_data)
        return long_df


def find_node_with_similarity(engine: ChainEngine):
    """
    Search for nodes in the coordinate tree that match user-specified conditions.

    Args:
        json_file_path (str): Path to the JSON file containing the coordinate tree data.
    """

    # Initialize history for prompt_toolkit
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit import prompt

    history = InMemoryHistory()

    while True:
        # Capture user input for the search condition
        user_condition = prompt(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
        )
        top_condition = user_condition.split("-")[0]
        top_k = int(user_condition.split("-")[1])
        title = user_condition.split("-")[2]
        try:
            retrived_data = engine.data_retriever.get_similar_examples(
                top_condition, return_pair_type=True, return_df=False, n=top_k
            )
            # save to txt file
            with open(f"/Users/mohameddiomande/Desktop/ret/{title}.txt", "w") as file:
                for i in range(len(retrived_data)):
                    file.write(retrived_data[i][1])
                    file.write("\n\n")
            print("Data saved to .retrived_data folder")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


class ChainGenerator(BaseParams):
    engine: Optional[Any] = None

    class Config:
        schema_extra = {"exclude": {"engine"}}

    class Config:
        arbitrary_types_allowed = True

    def validate_and_default_kwargs(self, kwargs: Dict) -> Dict:
        """
        Validates and sets default values for additional keyword arguments provided to the ChainGenerator.

        Args:
            kwargs (Dict): Additional keyword arguments that may be necessary for internal methods.

        Returns:
            Dict: A dictionary containing the validated and updated keyword arguments.

        """

        # Set default values for additional keyword arguments
        kwargs.setdefault("generate_data", False)
        kwargs.setdefault("num_prompts", 100)
        kwargs.setdefault("max_workers", 1)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("include_prompt", False)

        return kwargs

    def generate_data_or_finetune(self, dataframe_to_use: Any):
        """
        Utilizes the given DataFrame to either generate data or fine-tune a model based on user configuration.

        Args:
            dataframe_to_use (pd.DataFrame): The DataFrame containing the data which is to be
                either used for data generation or fine-tuning.
            **kwargs: Additional keyword arguments including:
                - generate_data (bool): A flag indicating whether data generation is required.
                - num_prompts (int): The number of prompts to be generated if `generate_data` is True.
                - max_workers (int): The max number of parallel workers for data generation.
                - batch_size (int): The number of items per batch during data generation.
                - include_prompt (bool): Whether to include the original prompt in generated data.

        Returns:
            pd.DataFrame: Returns a DataFrame that is either populated with the generated data or
                is the result of the fine-tuning process.
        """

        try:
            # Initialize ChainEngine for data handling
            self.engine = ChainEngine(dataframe=dataframe_to_use, verbose=self.verbose)

            return ChainEngine.fine_tune_and_return(
                self.engine,
                dataframe_to_use,
                self.system_message_text,
                self.retrieve_model_name,
                self.fine_tune,
                self.base_persist_dir,
                self.output_name,
                self.api_keys[1],
            )
        except Exception as e:
            print(f"An error occurred in generate_data_or_finetune: {e}")
            raise
