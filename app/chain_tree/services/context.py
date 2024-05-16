from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
import pandas as pd
import random
import os


def _write_dataset_mapped(dataset: Dataset, path: str):
    dataset.save_to_disk(path)


def _load_jsonl_file(original_file_path: str) -> list:
    with open(original_file_path, "r") as f:
        lines = f.readlines()
    return lines


def _write_jsonl_file(lines: list, file_path: str):
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)


def _save_data_to_jsonl(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    prompt_directory: str,
    file_name_prefix: str,
):
    train_file_path = f"{prompt_directory}{file_name_prefix}_train.jsonl"
    test_file_path = f"{prompt_directory}{file_name_prefix}_test.jsonl"
    train_data.to_json(train_file_path, orient="records", lines=True)
    test_data.to_json(test_file_path, orient="records", lines=True)
    return train_file_path, test_file_path


def _split_dataframe(
    data: pd.DataFrame, test_size: float = 0.1, random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    return train_data, test_data


def split_data(
    data: Optional[pd.DataFrame],
    prompt_directory: Optional[str] = None,
    test_size: float = 0.1,
    random_state: Optional[int] = None,
    file_name_prefix: Optional[str] = "split_data",
    original_file_path: Optional[str] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    if data is None and not original_file_path:
        raise ValueError(
            "Data is not loaded and no file path provided. Cannot perform split."
        )

    if data is not None:
        train_data, test_data = _split_dataframe(data, test_size, random_state)
        train_file_path, test_file_path = _save_data_to_jsonl(
            train_data, test_data, prompt_directory, file_name_prefix
        )
        return train_data, test_data
    else:
        lines = _load_jsonl_file(original_file_path)
        random.shuffle(lines)
        num_train = int((1 - test_size) * len(lines))
        train_lines, test_lines = lines[:num_train], lines[num_train:]
        _write_jsonl_file(
            train_lines, f"{prompt_directory}{file_name_prefix}_train.jsonl"
        )
        _write_jsonl_file(
            test_lines, f"{prompt_directory}{file_name_prefix}_test.jsonl"
        )
        print(
            f"Data split complete. {num_train} lines written to {train_file_path} and {len(lines) - num_train} lines written to {test_file_path}."
        )
        return None


def _load_and_preprocess_datasets(
    train_path: str, test_path: str, system_message: str, output_directory: str
) -> Tuple[Optional[object], Optional[object]]:
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    valid_dataset = load_dataset("json", data_files=test_path, split="train")
    train_dataset_mapped = train_dataset.map(
        lambda examples: {
            "text": [
                f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n"
                + prompt
                + " [/INST] "
                + response
                for prompt, response in zip(examples["prompt"], examples["response"])
            ]
        },
        batched=True,
    )
    valid_dataset_mapped = valid_dataset.map(
        lambda examples: {
            "text": [
                f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n"
                + prompt
                + " [/INST] "
                + response
                for prompt, response in zip(examples["prompt"], examples["response"])
            ]
        },
        batched=True,
    )
    _write_dataset_mapped(
        train_dataset_mapped, f"{output_directory}train_dataset_mapped"
    )
    _write_dataset_mapped(
        valid_dataset_mapped, f"{output_directory}valid_dataset_mapped"
    )

    return train_dataset_mapped, valid_dataset_mapped


def _split_data(
    original_data_path: str,
    output_directory: Optional[str],
    data: Optional[pd.DataFrame],
    test_size: float,
    random_state: Optional[int],
    file_name_prefix: str,
) -> Tuple[Optional[str], Optional[str]]:
    split_data(
        prompt_directory=output_directory,
        data=data,
        test_size=test_size,
        random_state=random_state,
        file_name_prefix=file_name_prefix,
        original_file_path=original_data_path,
    )
    train_path = f"{output_directory}{file_name_prefix}_train.jsonl"
    test_path = f"{output_directory}{file_name_prefix}_test.jsonl"
    return train_path, test_path


def build_datasets_task(
    original_data_path: str,
    output_directory: str,
    system_message: str = "",
    test_size: float = 0.2,
    data: Optional[pd.DataFrame] = None,
    random_state: Optional[int] = None,
    file_name_prefix: Optional[str] = "split_data",
) -> Tuple[Optional[object], Optional[object]]:
    try:
        train_path, test_path = _split_data(
            original_data_path,
            output_directory,
            data,
            test_size,
            random_state,
            file_name_prefix,
        )
        return _load_and_preprocess_datasets(
            train_path, test_path, system_message, output_directory
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def save_to_markdown(df: pd.DataFrame, file_path: str):
    # Extract the folder path from the file path
    folder_path = os.path.dirname(file_path)

    # Define the markdown file name and path
    markdown_path = os.path.join(folder_path, "main_df.md")

    # Extract the required columns
    required_columns = ["id", "text", "author", "coordinate"]
    data = df[required_columns].values.tolist()

    # Convert the list of lists to a Markdown formatted string
    markdown_content = "| " + " | ".join(required_columns) + " |\n"
    markdown_content += "| " + " | ".join(["---"] * len(required_columns)) + " |\n"
    for row in data:
        markdown_content += "| " + " | ".join(str(item) for item in row) + " |\n"

    # Save the markdown content to a file
    with open(markdown_path, "w") as md_file:
        md_file.write(markdown_content)


def process_data(relationship_df: pd.DataFrame, persist_dir: str) -> pd.DataFrame:
    """
    Processes the given DataFrame and performs data splitting and saving to markdown.

    Parameters:
    - relationship_df (pd.DataFrame): DataFrame containing conversation data including a "prompt" column.
    - persist_dir (str): Directory path to persist processed data.

    Returns:
    - pd.DataFrame: The processed DataFrame (main_df).
    """
    # Split the data into train and test sets
    build_datasets_task(
        original_data_path=persist_dir + "/train_data.jsonl",
        output_directory=persist_dir + "/",
        system_message="",
        test_size=0.1,
        data=relationship_df,
        random_state=None,
        file_name_prefix="split_data",
    )
