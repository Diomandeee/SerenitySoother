from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Iterable,
    Union,
    TypeVar,
)
from loguru import logger
import datetime
import random
import arrow
import json
import os
import re


T = TypeVar("T")


def split_text(text):
    # Use regular expressions to split text based on numbering patterns
    lines = re.split(r"(\d+\.\d+\.\d+\s)", text)

    # Filter out empty and None lines
    lines = [line.strip() for line in lines if line is not None and line.strip()]

    return lines


def split_by_markdown_delimiters(s: str) -> List[str]:
    # Split by headers from h1 to h6, italic/bold delimiters, block quotes, list items, and colons
    patterns = [
        r"\#{1,6} ",  # Headers (e.g., # Header1, ## Header2, ...)
        r"\*{1,2}",  # Italic and bold
        r"\>",  # Block quotes
        r"\- ",  # List items
        r"\* ",  # List items
        r"\+ ",  # List items
        r"\d+\. ",  # Numbered list items
        r"\:",  # Colons
    ]

    combined_pattern = "|".join(patterns)

    # Split by the combined pattern and remove empty strings
    return [
        segment.strip() for segment in re.split(combined_pattern, s) if segment.strip()
    ]


def split_by_multiple_delimiters(s: str, delimiters: List[str] = None) -> List[str]:
    if delimiters is None:
        delimiters = [";", ",", "|"]
    delimiter_pattern = "|".join(map(re.escape, delimiters))
    return re.split(delimiter_pattern, s)


def split_by_consecutive_spaces(s: str) -> List[str]:
    return re.split(r"\s{2,}", s)


def split_by_capital_letters(s: str) -> List[str]:
    return re.findall(r"[A-Z][a-z]*", s)


def split_string_to_parts(raw: str, delimiter: str = "\n") -> List[str]:
    return raw.split(delimiter)


def get_data(data, file):
    text = data.get("text")
    revised_prompt = data.get("revised_prompt")
    return [text, revised_prompt, file]


class InvalidChainTypeException(Exception):
    pass


class InvalidIdException(Exception):
    pass


class InvalidContentException(Exception):
    pass


class InvalidCoordinateException(Exception):
    pass


class APIFailureException(Exception):
    pass


class InvalidTreeException(Exception):
    pass


class SaveAndExitException(Exception):
    pass


def log_handler(
    message: str, level: str = "INFO", step=None, verbose: bool = False
) -> None:
    """
    Handle logging with different log levels.

    Args:
        message (str): The log message.
        level (str): The log level ('INFO', 'WARNING', 'ERROR').
        step (Optional[int]): The step number.
        verbose (bool): If True, log the message; otherwise, skip logging.
    """
    if not verbose:
        return

    if step is not None:
        message = f"Step {step}: {message}"

    if level.upper() == "INFO":
        logger.info(message)
    elif level.upper() == "WARNING":
        logger.warning(message)
    elif level.upper() == "ERROR":
        logger.error(message)
    else:
        raise ValueError(f"Invalid log level: {level}")


def setup_logging(path: str = None) -> None:
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Remove default handlers
    logger.remove()

    # Add handlers (one for the file, one for the console)
    logger.add(
        path,
        rotation="10 MB",
        level="INFO",
        format=log_format,
    )  # for file logging
    logger.add(
        lambda msg: print(msg, end=""), colorize=True, format=log_format, level="INFO"
    )


def _flatten_dict(
    nested_dict: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Iterable[Tuple[str, Any]]:
    """
    Generator that yields flattened items from a nested dictionary for a flat dict.

    Parameters:
        nested_dict (dict): The nested dictionary to flatten.
        parent_key (str): The prefix to prepend to the keys of the flattened dict.
        sep (str): The separator to use between the parent key and the key of the
            flattened dictionary.

    Yields:
        (str, any): A key-value pair from the flattened dictionary.
    """
    for key, value in nested_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            yield from _flatten_dict(value, new_key, sep)
        else:
            yield new_key, value


def flatten_dict(
    nested_dict: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    """Flattens a nested dictionary into a flat dictionary.

    Parameters:
        nested_dict (dict): The nested dictionary to flatten.
        parent_key (str): The prefix to prepend to the keys of the flattened dict.
        sep (str): The separator to use between the parent key and the key of the
            flattened dictionary.

    Returns:
        (dict): A flat dictionary.

    """
    flat_dict = {k: v for k, v in _flatten_dict(nested_dict, parent_key, sep)}
    return flat_dict


def filter_none_values(d):
    """
    Recursively filter out keys from dictionary d where value is None.
    """
    if not isinstance(d, dict):
        return d
    return {k: filter_none_values(v) for k, v in d.items() if v is not None}


def get_current_timestamp() -> float:
    """Return the current timestamp."""
    return datetime.datetime.now().timestamp()


def load_and_preprocess_data(
    path: str, key_field: str, target_num: int = None, verbose: bool = True
) -> Tuple[List[Dict[str, Any]], set]:
    def log(message: str):
        if verbose:
            print(message)

    # Load data
    if not os.path.exists(path):
        log(f"Error: File {path} does not exist.")
        return [], set()

    data = load_json(path)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        log(f"Error: File {path} doesn't contain a list of dictionaries.")
        return [], set()

    # Filter data
    if target_num is not None:
        data = [item for item in data if len(item.get("mapping", [])) >= target_num]

    # Extract keys
    keys = {item.get(key_field) for item in data if key_field in item}

    return data, keys


def convert_webm_to_flac(input_file, output_file):
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"The input file '{input_file}' does not exist.")
        return

    # Check if the input file is a WEBM file
    if not input_file.endswith(".webm"):
        print(f"The input file '{input_file}' is not a WEBM file.")
        return

    # Check if the output file is an M4A file
    if not output_file.endswith(".flac"):
        print(f"The output file '{output_file}' is not an M4A file.")
        return
    # Check if the output directory exists
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"The output directory '{os.path.dirname(output_file)}' does not exist.")
        return
    # Convert the WEBM file to M4A
    os.system(f"ffmpeg -i '{input_file}' '{output_file}'")

    # Check if the output file was created
    if not os.path.exists(output_file):
        print(f"The output file '{output_file}' was not created.")
        return

    # Return the output file path
    return output_file


def convert_wav_to_flac(input_file, output_file):
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"The input file '{input_file}' does not exist.")
        return

    # Check if the input file is a WAV file
    if not input_file.endswith(".wav"):
        print(f"The input file '{input_file}' is not a WAV file.")
        return

    # Check if the output file is an M4A file
    if not output_file.endswith(".flac"):
        print(f"The output file '{output_file}' is not an M4A file.")
        return
    # Check if the output directory exists
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"The output directory '{os.path.dirname(output_file)}' does not exist.")
        return
    # Convert the WAV file to M4A
    os.system(f"ffmpeg -i '{input_file}' '{output_file}'")

    # Check if the output file was created
    if not os.path.exists(output_file):
        print(f"The output file '{output_file}' was not created.")
        return

    # Return the output file path
    return output_file


def manage_conversations(
    path_1: str,
    path_2: str,
    output_path: str,
    key_field: str = "create_time",
    operation_mode: str = "update",
    strict_mode: bool = False,
    target_num: int = None,
    verbose: bool = True,
    save_result: bool = True,
) -> List[Dict[str, Any]]:
    def log(message: str):
        if verbose:
            print(message)

    data_1, keys_1 = load_and_preprocess_data(path_1, key_field, target_num, verbose)
    data_2, keys_2 = load_and_preprocess_data(path_2, key_field, target_num, verbose)

    if not data_1 or not data_2:
        log("Error: One or both input files are not loaded properly.")
        return []

    # Check for strict mode
    if strict_mode and (None in keys_1 or None in keys_2):
        log(f"Error: Missing '{key_field}' field in one or more entries.")
        return []

    # Initialize result variable
    result = []

    if operation_mode == "difference":
        difference_keys = keys_2 - keys_1
        if not difference_keys:
            log("No new entries found in the second file based on the provided key.")
            return []
        result = [item for item in data_2 if item.get(key_field) in difference_keys]
        log(f"Found {len(result)} new entries based on '{key_field}'.")

    elif operation_mode == "update":
        unique_to_data_1 = [
            item for item in data_1 if item.get(key_field) not in keys_2
        ]
        shared_keys = keys_1.intersection(keys_2)
        updated_shared_conversations = [
            item for item in data_1 if item.get(key_field) in shared_keys
        ]
        unique_to_data_2 = [
            item for item in data_2 if item.get(key_field) not in keys_1
        ]

        result = unique_to_data_1 + updated_shared_conversations + unique_to_data_2
        log(f"Total of {len(result)} entries after updating.")

    else:
        log(f"Error: Invalid operation mode '{operation_mode}'.")
        return []

    if save_result:
        save_json(output_path, result)
        log(f"Saved results to {output_path}.")

    return result


def load_json(source: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if not os.path.isfile(source):
        raise ValueError(f"{source} does not exist.")
    with open(source, "r") as f:
        data = json.load(f)
    return data


def save_json(path: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def combine_json_files(path1: str, path2: str, output_path: str) -> None:
    data1 = load_json(path1)
    data2 = load_json(path2)

    if not isinstance(data1, list) or not isinstance(data2, list):
        raise ValueError("Both input files should contain a list of JSON objects.")

    combined_data = data1 + data2

    save_json(output_path, combined_data)
    print(f"Combined data saved to {output_path}.")

    return combined_data


def backoff_handler(
    retries: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True
) -> float:
    """
    Calculate an exponential backoff time with optional jitter.

    Parameters:
    - retries (int): The current retry attempt.
    - base_delay (float): The base delay in seconds for the backoff. Default is 1.0 second.
    - max_delay (float): The maximum backoff limit in seconds to prevent excessive wait time. Default is 60 seconds.
    - jitter (bool): If True, add random jitter to the backoff time to help spread out retry attempts.

    Returns:
    - float: The calculated wait time in seconds.
    """

    # Validate input
    if retries < 0:
        raise ValueError("Retries must be a non-negative integer.")

    # Calculate exponential backoff
    backoff_time = min(base_delay * (2 ** retries), max_delay)

    # Apply jitter by randomizing the backoff time
    if jitter:
        backoff_time = random.uniform(0, backoff_time)

    return backoff_time


def concat_dirs(dir1: str, dir2: str) -> str:
    """

    Concat dir1 and dir2 while avoiding backslashes when running on windows.
    os.path.join(dir1,dir2) will add a backslash before dir2 if dir1 does not
    end with a slash, so we make sure it does.

    """
    dir1 += "/" if dir1[-1] != "/" else ""
    return os.path.join(dir1, dir2)


def get_file_paths(base_persist_dir, title):
    persist_dir = os.path.join(base_persist_dir, str(title))

    return (persist_dir,)


def to_unix_timestamp(date_str: str) -> int:
    """
    Convert a date string to a unix timestamp (seconds since epoch).

    Args:
        date_str: The date string to convert.

    Returns:
        The unix timestamp corresponding to the date string.

    If the date string cannot be parsed as a valid date format, returns the current unix timestamp and prints a warning.
    """
    # Try to parse the date string using arrow, which supports many common date formats
    try:
        date_obj = arrow.get(date_str)
        return int(date_obj.timestamp())
    except arrow.parser.ParserError:
        # If the parsing fails, return the current unix timestamp and print a warning
        logger.info(f"Invalid date format: {date_str}")
        return int(arrow.now().timestamp())


def filter_by_prefix(
    data: Union[List[str], List[dict]],
    phase: str,
    include_more: bool = False,
    case_sensitive: bool = False,
    match_strategy: str = "start",
) -> List[Union[str, dict]]:
    """
    Filter the given data based on the provided phase.

    Args:
        data (Union[List[str], List[dict]]): Data to filter. Accepts both string lists and dictionaries.
        phase (str): phase to match against each data item.
        include_more (bool, optional): Include data with content beyond the phase. Defaults to False.
        case_sensitive (bool, optional): Consider case in matching. Defaults to False.
        match_strategy (str, optional): Matching strategy ("start", "exact", "contains"). Defaults to "start".

    Returns:
        List[Union[str, dict]]: Filtered data.
    """

    # Convert the phase to lowercase if case sensitivity is not required.
    if not case_sensitive:
        phase = phase.lower()

    # Inner function to determine if an item matches the phase based on the specified match strategy.
    def match(item):
        # Convert the item to string for uniformity, and make it lowercase if case sensitivity is off.
        content = item if isinstance(item, str) else str(item)
        if not case_sensitive:
            content = content.lower()

        # Determine if the content matches the phase based on the match strategy.
        if match_strategy == "start":
            return content.startswith(phase)
        elif match_strategy == "exact":
            return content == phase
        elif match_strategy == "contains":
            return phase in content
        elif match_strategy == "end":
            return content.endswith(phase)
        elif match_strategy == "regex":
            import re

            return re.search(phase, content) is not None
        else:
            # If the match strategy is not recognized, return False.
            return False

    # Apply the match function to filter the data based on the phase.
    filtered_data = [item for item in data if match(item)]

    # If the include_more option is enabled, filter the data to include items with more content than the phase.
    if include_more:
        filtered_data = [
            item
            for item in filtered_data
            if len(str(item).strip()) > len(phase) and str(item).strip() != phase
        ]

    # Return the filtered data.
    return filtered_data
