from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Iterable,
    Union,
    TypeVar,
    Optional,
)
from pydub.silence import split_on_silence
from selenium import webdriver
from pydub import AudioSegment
from tqdm import tqdm, trange
from loguru import logger
from PIL import ImageGrab
import concurrent.futures
from PIL import Image
import pandas as pd
import numpy as np
import scrapetube
import subprocess
import pyautogui
import datetime
import random
import arrow
import time
import json
import glob
import cv2
import os
import re
import io


T = TypeVar("T")

DEFAULT_CONFIG_PATH = "/Users/mohameddiomande/Desktop/dlm_matrix/.config/data.json"
FILE_PATTERN_VARIABLE_MAP = ""


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


def get_shortcut(chain, shortcuts_dict):
    """
    Retrieves the shortcut from the nested dictionary structure by chaining together
    the provided keys.

    Args:
        chain (list): A list of keys representing the nested path to the shortcut.
                      Example: ['Action', 'left_deck', 'Play/Pause']
        shortcuts_dict (dict, optional): The dictionary containing the shortcuts.
                                         Defaults to the 'playback_shortcuts' example.

    Returns:
        str: The corresponding shortcut if found.
        str: "Shortcut not found" if the chain is invalid.
    """
    # lower case all words in the dictionary
    shortcuts_dict = {k.lower(): v for k, v in shortcuts_dict.items()}
    for k, v in shortcuts_dict.items():
        if isinstance(v, dict):
            shortcuts_dict[k] = {k.lower(): v for k, v in v.items()}
            for k2, v2 in shortcuts_dict[k].items():
                if isinstance(v2, dict):
                    shortcuts_dict[k][k2] = {k.lower(): v for k, v in v2.items()}
    # Set the current level to the shortcuts dictionary

    current_level = shortcuts_dict

    # Traverse the dictionary using the chain if the shortcut is found we try again with the next word in the chain
    # if the shortcut is not found we return "Shortcut not found"
    for word in chain:
        if word in current_level:
            current_level = current_level[word]
        else:
            return ""
    return current_level


def get_shortcut_from_string(user_input):
    """
    Processes a user input string into a chain list and retrieves the corresponding shortcut.

    Args:
        user_input (str): The string input from the user representing the shortcut path.
                          Example: "Action Left Play"
        shortcuts_dict (dict, optional): The dictionary containing the shortcuts
                                         Defaults to the 'playback_shortcuts' example.

    Returns:
        str: The corresponding shortcut if found.
        str: "Shortcut not found" if the chain is invalid.
    """
    # lower case the user input
    user_input = user_input.lower()
    chain = user_input.split(" ")
    # the core chain is the first two words of the input

    core_chain = chain[:2]
    # the rest of the chain is the rest of the words

    rest_of_chain = chain[2:]
    # join the rest of the chain into a single string
    rest_of_chain = " ".join(rest_of_chain)

    chain = core_chain + [rest_of_chain]

    shortcuts_dict = json.load(open("config/playback_shortcuts.json"))

    return get_shortcut(chain, shortcuts_dict)


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
    backoff_time = min(base_delay * (2**retries), max_delay)

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


def youtube_search_urls(query, max_results=1, api_key=""):
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", developerKey=api_key)

    search_response = (
        youtube.search()
        .list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
        )
        .execute()
    )

    youtube_urls = []
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            youtube_urls.append(
                f"https://www.youtube.com/watch?v={search_result['id']['videoId']}"
            )

    return youtube_urls


def markdown_to_dataframe(markdown_text, output_path=None):
    # Split the markdown text into sections based on the number of hashtags
    sections = re.split(r"\n#+\s", markdown_text)

    # Initialize an empty list to store the section titles and questions
    data = []

    # Iterate over the sections to extract the titles and questions
    for section in sections:
        # Split the section into title and questions based on newlines
        lines = section.split("\n")

        # Extract the title (first line) and questions (remaining lines)
        title = lines[0]
        questions = [
            re.sub(r"^\d+\.\s", "", q) for q in lines[1:] if q.strip()
        ]  # Remove numbers from questions

        # Append the title and questions as a dictionary to the data list
        data.append({"Title": title, "Questions": " ".join(questions)})

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file if an output path is provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def extract_synergy_chats(base_dir, sub_dir, file_patterns, context="synergy"):
    """Extracts synergy chats from a directory structure and saves them to a JSON file.

    Args:
        base_dir (str): The base directory containing the chain data.
        sub_dir (str): The subdirectory within each main directory where chats are located.
        file_patterns (list): A list of file patterns to match for synergy chats.

    Returns:
        None
    """

    main_directory = os.listdir(base_dir)
    main_directory = [
        x for x in main_directory if not x.startswith(".")
    ]  # Filter hidden files

    sub_directories = [os.path.join(directory, sub_dir) for directory in main_directory]
    synergy_chats = []

    for sub_directory in sub_directories:
        files = glob.glob(os.path.join(base_dir, sub_directory, "*"))
        for file in files:
            if any(file.endswith(pattern) for pattern in file_patterns):
                with open(file) as f:
                    synergy_chat = json.load(f)
                    prompt = synergy_chat.get("prompt")
                    response = synergy_chat.get("response")
                    messages = [
                        {"author": "user", "content": prompt},
                        {"author": "assistant", "content": "\n\n".join(response)},
                    ]

                    synergy_chat = {
                        "context": context,
                        "messages": messages,
                    }
                    synergy_chats.append(synergy_chat)

    with open("synts.json", "w") as f:
        json.dump(synergy_chats, f, indent=4)


def process_data(
    train_input,
    test_input,
    train_output,
    test_output,
    prompt_col,
    response_col,
    starts_from,
    stops_at,
):
    # Load the data pandas
    train_df = pd.read_json(train_input, lines=True)
    test_df = pd.read_json(test_input, lines=True)

    # Preprocess the data get the prompt and response
    train_df = train_df[[prompt_col, response_col]]
    test_df = test_df[[prompt_col, response_col]]
    train_df[prompt_col] = train_df[prompt_col] + " " + train_df[response_col]
    test_df[prompt_col] = test_df[prompt_col] + " " + test_df[response_col]

    # remove the text that do not start with starts_from
    train_df["text"] = train_df[prompt_col].apply(lambda x: x[x.find(starts_from) :])
    test_df["text"] = test_df[prompt_col].apply(lambda x: x[x.find(starts_from) :])

    # remove the text with row"."
    train_df["text"] = train_df["text"].apply(lambda x: x[: x.find(stops_at)])
    test_df["text"] = test_df["text"].apply(lambda x: x[: x.find(stops_at)])

    train_dict = train_df.to_dict(orient="records")
    test_dict = test_df.to_dict(orient="records")

    # add keys to the dictionary
    train_dict = [{"text": x["text"]} for x in train_dict]
    test_dict = [{"text": x["text"]} for x in test_dict]

    # remove the empty text, string
    train_dict = [x for x in train_dict if x["text"] != ""]
    test_dict = [x for x in test_dict if x["text"] != ""]

    train_df = pd.DataFrame(train_dict).to_json(
        train_output, orient="records", lines=True
    )
    test_df = pd.DataFrame(test_dict).to_json(test_output, orient="records", lines=True)

    return train_df, test_df


def save_webpage_as_png(url, output_folder, split_part=None, split_direction="row"):
    # Configure Selenium webdriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run headless to not open browser window
    driver = webdriver.Chrome(options=options)

    # Open the webpage
    driver.get(url)

    # Wait for the page to load
    time.sleep(2)  # Adjust waiting time as needed

    # Get webpage height
    total_height = driver.execute_script("return document.body.scrollHeight")

    os.makedirs(output_folder, exist_ok=True)

    # Set initial height and capture screenshot
    driver.set_window_size(
        1200, total_height
    )  # Set window size for capturing full page
    screenshot = driver.get_screenshot_as_png()
    full_screenshot = Image.open(io.BytesIO(screenshot))

    # Optionally split the image into smaller dimensions
    saved_files = []
    if split_part:
        width, height = full_screenshot.size
        if split_direction == "row":
            split_height = height // split_part
            for i in range(split_part):
                box = (0, i * split_height, width, (i + 1) * split_height)
                split_image = full_screenshot.crop(box)
                file_path = os.path.join(output_folder, f"split_{i}.png")
                split_image.save(file_path, "PNG")
                saved_files.append(file_path)
        elif split_direction == "column":
            split_width = width // split_part
            for i in range(split_part):
                box = (i * split_width, 0, (i + 1) * split_width, height)
                split_image = full_screenshot.crop(box)
                file_path = os.path.join(output_folder, f"split_{i}.png")
                split_image.save(file_path, "PNG")
                saved_files.append(file_path)
        else:
            raise ValueError(
                "Invalid value for split_direction. Use 'row' or 'column'."
            )

    else:
        # Save full screenshot as .png
        file_path = os.path.join(output_folder, "full_screenshot.png")
        full_screenshot.save(file_path, "PNG")
        saved_files.append(file_path)

    # Close the webdriver
    driver.quit()

    if split_part:
        return saved_files[: split_part - 1]
    else:
        return saved_files


def check_for_image(text):
    pattern = r"!\[\[(.*?)\.(png|jpg)\]\]"
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Return the filename without extension
    else:
        return None


def check_for_file(text):
    pattern = (
        r"Path: (.*)$"
        or r"Path:\n (.*)$"
        or r"Path:\n(.*)$"
        or r"Path:\n\n (.*)$"
        or r"Path:\n\n(.*)$"
    )
    match = re.search(pattern, text)
    if match:
        file_path = match.group(1)
        with open(file_path, "r") as file:
            lines = file.readlines()

        content = "\n".join(lines)
        return "\n\n".join(["#" + file_path, content])
    else:
        return None


def check_for_youtube(text):
    pattern = r"https://www.youtube.com/watch\?v=(\S+)"

    match = re.search(pattern, text)
    if match:
        return match.group(0)

    else:
        return None


def check_for_website(text):
    pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"

    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None


def check(text):
    image = check_for_image(text)
    youtube = check_for_youtube(text)
    website = check_for_website(text)
    file = check_for_file(text)
    return image, youtube, website, file


def extract_video_info(video_data):
    """
    Extract the video URL and cleaned title from the video data.

    Parameters:
    - video_data: The raw video data from YouTube.

    Returns:
    - dict: Dictionary with "url" and "title" keys.
    """
    base_video_url = "https://www.youtube.com/watch?v="
    video_id = video_data["videoId"]
    video_title = video_data["title"]["runs"][0]["text"]
    return {"url": f"{base_video_url}{video_id}", "title": video_title}


def get_video_title(video_url: str, path: bool = False) -> Dict[str, str]:
    """
    returns video data from a video url
    """
    video_id = video_url.split("?v=")[-1]

    video = scrapetube.get_video(video_id)

    # get the title of the video
    video_title = video["title"]["runs"][0]["text"]
    if path:
        return video_title + ".m4a"

    return video_title


def get_playlist_urls(playlist_id: str) -> List[str]:
    """
    Get video URLs from a playlist.
    """
    videos = scrapetube.get_playlist(playlist_id)
    return [extract_video_info(video)["url"] for video in videos]


def get_channel_urls(channel_id: str, pattern: str = None) -> List[str]:
    """
    Get video URLs from a channel with a specified pattern.
    """
    videos = scrapetube.get_channel(channel_id)
    if pattern:
        videos = [
            video
            for video in videos
            if re.search(pattern, video["title"]["runs"][0]["text"])
        ]
    return [extract_video_info(video)["url"] for video in videos]


def get_video_urls(
    channel_or_playlist_id: str, pattern: str = None
) -> List[Dict[str, str]]:
    """
    Get video URLs and cleaned video titles either from a channel with a specified pattern or from a playlist.
    """
    if "youtube.com/playlist" in channel_or_playlist_id:
        playlist_id = channel_or_playlist_id.split("?list=")[-1].split("&")[0]
        videos = scrapetube.get_playlist(playlist_id)
    else:
        videos = scrapetube.get_channel(channel_or_playlist_id)

    if pattern:
        videos = [
            video
            for video in videos
            if re.search(pattern, video["title"]["runs"][0]["text"])
        ]

    return videos


def get_video_urls_from_channel(channel_id: str, pattern: str = None) -> List[str]:
    """
    Get video URLs from a channel with a specified pattern.
    """
    return [video["url"] for video in get_video_urls(channel_id, pattern)]


def get_video_urls_from_playlist(playlist_id: str) -> List[str]:
    """
    Get video URLs from a playlist.
    """
    return [video["url"] for video in get_video_urls(playlist_id)]


def download_youtube_media(
    youtube_url: str,
    output_path: str,
    media_type: str = "video",
    custom_format: Optional[str] = None,
) -> Optional[str]:
    """
    Downloads audio or video from a YouTube video using yt-dlp and converts it to a QuickTime compatible format.
    """
    # Default format selectors for yt-dlp
    default_format_selector = {
        "audio": "bestaudio[ext=m4a]",
        "video": "best",
    }

    # Use custom format if provided, otherwise use default
    format_selector = (
        custom_format if custom_format else default_format_selector[media_type]
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Command setup for yt-dlp
    cmd = [
        "yt-dlp",
        "-f",
        format_selector,
        "-o",
        os.path.join(output_path, "%(title)s.%(ext)s"),
        youtube_url,
    ]

    try:
        # Run the yt-dlp command and capture the output
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        ) as process, tqdm(
            total=100, unit="%", desc="Downloading", leave=False
        ) as pbar:
            for line in process.stdout:
                if "download" in line.lower():
                    # Parse the percentage and update tqdm progress bar
                    percentage = re.findall(r"\d+\.\d+%", line)
                    if percentage:
                        progress = float(percentage[0].replace("%", ""))
                        pbar.n = progress
                        pbar.refresh()
        #

        # get the title of the video
        video_id = get_video_title(youtube_url, path=True)
        # Check if yt-dlp was successful
        if process.returncode == 0:
            file_path = os.path.join(
                output_path, f"{video_id}"
            )  # Example output file name
            print("Downloaded file path:", file_path)

            # Convert the video to a QuickTime compatible format using ffmpeg
            if media_type == "video":
                converted_file_path = os.path.splitext(file_path)[0] + "_converted.mp4"
                ffmpeg_command = [
                    "ffmpeg",
                    "-i",
                    file_path,
                    "-vcodec",
                    "h264",
                    "-acodec",
                    "aac",
                    "-strict",
                    "-2",  # This may be necessary if using an older ffmpeg version
                    converted_file_path,
                ]
                # Show a spinner while ffmpeg is running since we can't track progress easily
                with trange(
                    1, desc="Converting", bar_format="{desc}: {bar}", leave=False
                ) as t:
                    subprocess.run(ffmpeg_command, check=True)
                    for _ in t:
                        time.sleep(
                            0.1
                        )  # Sleep to slow down the spinner since ffmpeg is fast for small files

                print("Converted file path:", converted_file_path)
                return converted_file_path
            else:
                return file_path
        else:
            print("yt-dlp command was not successful.")
            print("Return code:", process.returncode)
            return None
    except subprocess.CalledProcessError as e:
        print("An exception occurred while running yt-dlp.")
        print("Error output:", e.output)
        print("Error code:", e.returncode)
        return None


def extract_audio_from_video(video_file: str, output_audio_file: str) -> None:
    """
    Extracts audio from the video and saves it as a new file.

    Parameters:
    - video_file (str): Path to the video file.
    - output_audio_file (str): Path to save the extracted audio.
    """
    try:
        video = AudioSegment.from_file(video_file)
        audio = video.set_channels(1).set_frame_rate(16000)
        audio.export(output_audio_file, format="wav")
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None


def download_multiple_youtube_media(
    youtube_urls: List[str],
    output_paths: List[str],
    media_types: List[str],
    custom_format: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Downloads audio or video from a list of YouTube videos using download_youtube_media.

    Parameters:
    - youtube_urls (List[str]): List of URLs of the YouTube videos.
    - output_path (str): Path to save the downloaded audio or video files.
    - media_type (str): Type of media to download, can be 'audio' or 'video'.

    Returns:
    - List[Tuple[str, Optional[str]]]: A list of tuples where the first element is the YouTube URL and
    the second element is the name of the downloaded file or None if unsuccessful.
    """

    downloaded_files = []

    # Use concurrent.futures for parallel downloads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                download_youtube_media,
                url,
                output_path,
                media_type,
                custom_format,
            )
            for url, output_path, media_type in zip(
                youtube_urls, output_paths, media_types
            )
        ]

        for future in concurrent.futures.as_completed(futures):
            url = youtube_urls[futures.index(future)]
            try:
                downloaded_file = future.result()
                downloaded_files.append((url, downloaded_file))
            except Exception as e:
                print(f"An error occurred while downloading {url}: {e}")
                downloaded_files.append((url, None))

    return downloaded_files


def download_videos_from_source(
    source_id: str,
    output_path: str,
    media_type: str = "audio",
    pattern: str = None,
    custom_format: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Download media from YouTube videos given a source, which can be a channel or playlist ID or URL.

    Parameters:
    - source_id (str): YouTube channel ID, playlist ID, or URL.
    - output_path (str): Path to save the downloaded media.
    - media_type (str): Type of media to download, 'audio' or 'video'.
    - pattern (str, optional): Pattern to match video titles (for channels).

    Returns:
    - List[Tuple[str, Optional[str]]]: List of tuples containing the video URL and the filename of the downloaded media.
    """
    if "youtube.com/playlist" in source_id:
        video_urls = get_video_urls_from_playlist(source_id)
    else:
        video_urls = get_video_urls_from_channel(source_id, pattern)

    return download_multiple_youtube_media(
        video_urls, output_path, media_type, custom_format
    )


def preprocess_audio(
    input_file: str,
    output_file: str,
    audio_format: str = "m4a",
    channels: int = 1,
    frame_rate: int = 16000,
    min_silence_len: int = 500,
    silence_thresh: int = -40,
    keep_silence: int = 200,
    supported_formats: List[str] = ["m4a", "mp3", "wav"],
) -> None:
    """
    Preprocesses audio by splitting on silence and exports to WAV.

    Parameters:
    - input_file (str): Path to the input audio file.
    - output_file (str): Path to save the processed audio file.
    - audio_format (str): Format of the input audio file.
    - channels (int): Desired number of channels for the processed audio.
    - frame_rate (int): Desired frame rate for the processed audio.
    - min_silence_len (int): Minimum length of silence to split on.
    - silence_thresh (int): Threshold value for silence detection.
    - keep_silence (int): Amount of silence to retain around detected non-silent chunks.
    """
    if audio_format not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: {audio_format}. Supported formats are: {', '.join(supported_formats)}"
        )
    try:
        audio = (
            AudioSegment.from_file(input_file, format=audio_format)
            .set_channels(channels)
            .set_frame_rate(frame_rate)
        )
        audio_segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        processed_audio = AudioSegment.empty()
        for segment in audio_segments:
            processed_audio += segment
        processed_audio.export(output_file, format="wav")
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        return None


def convert_mp4_to_m4a(input_file: str, output_file: str) -> None:
    """
    Converts an MP4 file to M4A.

    Parameters:
    - input_file (str): Path to the input MP4 file.
    - output_file (str): Path to save the converted M4A file.
    """
    try:
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-vn",  # No video
            "-c:a",
            "aac",  # Use AAC codec for audio
            "-b:a",
            "256k",  # Set audio bitrate
            output_file,
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"Error converting MP4 to M4A: {e}")
        return None


def split_audio(
    input_file: str,
    max_segment_length_ms: int = 60000,  # Default is 1 minute
    output_path: str = "podcast_transcription",
    convert: bool = False,
) -> List[str]:
    """
    Splits an audio file into segments of maximum length.

    Parameters:
    - input_file (str): Path to the input audio file.
    - max_segment_length_ms (int): Maximum length of each segment in milliseconds. Default is 10 minutes (600,000 ms).
    - output_path (str): Path to save the split segments.

    Returns:
    - List[str]: List of paths to the split audio segments.
    """
    audio = AudioSegment.from_file(input_file)
    duration_ms = len(audio)

    # Calculate number of segments needed
    num_segments = duration_ms // max_segment_length_ms
    if duration_ms % max_segment_length_ms != 0:
        num_segments += 1

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Split audio into segments
    segment_paths = []
    for i in range(num_segments):
        start_time = i * max_segment_length_ms
        end_time = min((i + 1) * max_segment_length_ms, duration_ms)
        segment = audio[start_time:end_time]

        # Save segment to file
        output_file = os.path.join(output_path, f"segment{i+1}.wav")
        segment.export(output_file, format="wav")
        segment_paths.append(output_file)

    # convert wav to m4a
    segment_m4a_paths = []
    for segment_path in segment_paths:
        m4a_file = os.path.splitext(segment_path)[0] + ".m4a"
        if convert:
            wavfile = convert_mp4_to_m4a(segment_path, m4a_file)

            segment_m4a_paths.append(wavfile)

            # remove mp4 file
            os.remove(m4a_file)

        else:
            segment_m4a_paths.append(m4a_file)

    # Get time markers for each segment
    segment_m4a_time_markers = []
    for i in range(num_segments):
        start_time = i * max_segment_length_ms
        end_time = min((i + 1) * max_segment_length_ms, duration_ms)
        segment_m4a_time_markers.append(f"{start_time}-{end_time}")

    return segment_m4a_paths, segment_m4a_time_markers


def motion_detection_and_capture(
    record_video=False,
    video_duration=None,
    capture_path="captures",
    video_path="videos",
):
    # Create directories if they don't exist
    if not os.path.exists(capture_path):
        os.makedirs(capture_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    # Initialize variables
    cap = cv2.VideoCapture(0)
    last_mean = 0
    detected_motion = False
    frame_rec_count = 0
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None
    start_time = None

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        last_mean = np.mean(gray)

        # Motion detection
        if result > 0.3:
            print("Motion detected!")
            if record_video:
                print("Started recording.")
                if out is None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(video_path, f"motion_{timestamp}.avi")
                    out = cv2.VideoWriter(
                        filename, fourcc, 20.0, (frame.shape[1], frame.shape[0])
                    )
                detected_motion = True
                start_time = time.time()
            else:
                print("Capturing picture.")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(capture_path, f"motion_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                break

        # Video recording
        if detected_motion and record_video:
            out.write(frame)
            frame_rec_count += 1
            if video_duration is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= video_duration:
                    print("Finished recording.")
                    detected_motion = False
                    out.release()
                    out = None
                    break

        # Check for user interruption or completion
        if cv2.waitKey(1) & 0xFF == ord("q") or frame_rec_count == 240:
            break

    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    return os.path.abspath(filename)


def record_screen(duration: int = 10, output_file: str = "screen_capture.mp4"):
    """
    Record the screen for a specified duration and save the video to a file.

    Args:
        duration (int): The duration of the screen recording in seconds.
        output_file (str): The name of the output video file.
    """
    # Get the screen resolution
    screen_width, screen_height = pyautogui.size()

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (screen_width, screen_height))

    # Record the screen for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        # Capture the screen
        img = pyautogui.screenshot()

        # Convert the image to a numpy array
        frame = np.array(img)

        # Convert the color from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object and destroy all OpenCV windows
    out.release()
    cv2.destroyAllWindows()

    # return the full path of the output file
    return os.path.abspath(output_file)


def take_picture():
    # Capture the screen and save it as an image using PIL (Pillow)
    img = ImageGrab.grab()
    img.save("screenshot.png")

    return os.path.abspath("screenshot.png")


# def find_node_with_condition(json_file_path: str):
#     """
#     Search for nodes in the coordinate tree that match user-specified conditions.

#     Args:
#         json_file_path (str): Path to the JSON file containing the coordinate tree data.
#     """
#     from prompt_toolkit import prompt
#     from prompt_toolkit.history import InMemoryHistory
#     from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
#     from dlm_matrix.transform.tree import CoordinateTree
#     from dlm_matrix.transform.traverse import CoordinateTreeTraverser

#     # Initialize history for prompt_toolkit
#     history = InMemoryHistory()

#     while True:
#         # Capture user input for the search condition
#         user_condition = prompt(
#             "Enter your condition (e.g., x == 1 or y == 7 and x == 4): ",
#             history=history,
#             auto_suggest=AutoSuggestFromHistory(),
#         )

#         try:
#             # Create a lambda function based on user input
#             condition_fn = lambda x: eval(user_condition, {}, x.__dict__)

#             # Load coordinate tree and initialize tree traverser
#             tree = CoordinateTree.from_json(json_file_path)
#             tree_traverser = CoordinateTreeTraverser(tree)

#             # Perform a depth-first search based on the condition
#             result = tree_traverser.traverse_depth_first_all(condition_fn)

#             if result:
#                 print("-" * 50)
#                 print("Depth-first search found node:")
#                 for res in result:
#                     print(res.text)
#             else:
#                 print("Node not found based on the given condition.")

#         except Exception as e:
#             print(f"An error occurred: {str(e)}")


# from scenedetect import ContentDetector, detect
# from imageio_ffmpeg import get_ffmpeg_exe
# from mmengine.logging import print_log
# from scenedetect import FrameTimecode
# import subprocess


# vide = "/Users/mohameddiomande/Desktop/YouTube/data/Create Your Own AI-Powered To-Do List Application With No Code (Bubble.io & ChatGPT).mp4"

# # config
# target_fps = 30  # int
# shorter_size = 512  # int
# min_seconds = 1  # float
# max_seconds = 5  # float
# assert max_seconds > min_seconds
# cfg = dict(
#     target_fps=target_fps,
#     min_seconds=min_seconds,
#     max_seconds=max_seconds,
#     shorter_size=shorter_size,
# )

# def split_video(
#     sample_path,
#     scene_list,
#     save_dir,
#     target_fps=30,
#     min_seconds=1,
#     max_seconds=10,
#     shorter_size=512,
#     verbose=False,
#     logger=None,
# ):
#     FFMPEG_PATH = get_ffmpeg_exe()

#     save_path_list = []
#     for idx, scene in enumerate(scene_list):
#         s, t = scene  # FrameTimecode
#         fps = s.framerate
#         max_duration = FrameTimecode(timecode="00:00:00", fps=fps)
#         max_duration.frame_num = round(fps * max_seconds)
#         duration = min(max_duration, t - s)
#         if duration.get_frames() < round(min_seconds * fps):
#             continue

#         # save path
#         fname = os.path.basename(sample_path)
#         fname_wo_ext = os.path.splitext(fname)[0]
#         # TODO: fname pattern
#         save_path = os.path.join(save_dir, f"{fname_wo_ext}_scene-{idx}.mp4")

#         # ffmpeg cmd
#         cmd = [FFMPEG_PATH]

#         # Only show ffmpeg output for the first call, which will display any
#         # errors if it fails, and then break the loop. We only show error messages
#         # for the remaining calls.
#         # cmd += ['-v', 'error']

#         # input path
#         cmd += ["-i", sample_path]

#         # clip to cut
#         cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-t", str(duration.get_seconds())]

#         # target fps
#         # cmd += ['-vf', 'select=mod(n\,2)']
#         cmd += ["-r", f"{target_fps}"]

#         # aspect ratio
#         cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]
#         # cmd += ['-vf', f"scale='if(gt(iw,ih),{shorter_size},trunc(ow/a/2)*2)':-2"]

#         cmd += ["-map", "0", save_path]

#         proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#         stdout, stderr = proc.communicate()
#         if verbose:
#             stdout = stdout.decode("utf-8")
#             print_log(stdout, logger=logger)

#         save_path_list.append(sample_path)
#         print_log(f"Video clip saved to '{save_path}'", logger=logger)

#     return save_path_list


# scene_list = detect(vide, ContentDetector(), start_in_scene=True)


# # split scenes
# save_path_list = split_video(vide, scene_list, save_dir='.', **cfg,)
