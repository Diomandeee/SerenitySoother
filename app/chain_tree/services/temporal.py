from typing import Optional, List, Union
from chain_tree.models import ChainMap
from datetime import datetime
import numpy as np
import logging


def time_diff_decay_function(
    time_diff: float, decay_type: str = "exponential", half_life: float = 60
) -> float:
    """
    Returns the time decay factor for a given time difference.

    Args:
        time_diff: The time difference in minutes.
        decay_type: The type of time decay function. It can be one of the following:
            - 'exponential': An exponential decay function.
            - 'logarithmic': A logarithmic decay function.
            - 'linear': A linear decay function.
        half_life: The half-life in minutes for the exponential decay function. Default is 60 minutes.

    Returns:
        The time decay factor. Returns 1.0 if an error occurs.

    Raises:
        ValueError: If an unsupported decay_type is given.
    """
    try:
        # Exponential decay
        if decay_type == "exponential":
            decay_factor = 0.5 ** (time_diff / half_life)
        # Logarithmic decay
        elif decay_type == "logarithmic":
            decay_factor = 1 / (1 + np.log1p(time_diff))
        # Linear decay
        elif decay_type == "linear":
            decay_factor = max(1 - time_diff / (time_diff + half_life), 0)
        else:
            raise ValueError(f"Unsupported decay type: {decay_type}")

        return decay_factor

    except Exception as e:
        logging.error(f"Error occurred while calculating time decay factor: {str(e)}")
        return 1.0  # Default value


def time_decay_factor(
    message: Optional[Union[ChainMap, float]] = None,
    sibling_time_differences: Optional[List[float]] = None,
    sub_thread_root: Optional[ChainMap] = None,
    root_timestamp: Optional[float] = None,
) -> float:
    """
    Returns the time decay factor for a given message timestamp or message ChainMap.

    Args:
        message: The ChainMap of the message or timestamp of the message for which the time decay factor is to be calculated.
        sibling_time_differences: A list of time differences between the given message and its siblings.
        sub_thread_root: The ChainMap of the root message in the sub-thread.
        root_timestamp: The timestamp of the root message.

    Returns:
        The time decay factor. Returns 1.0 if an error occurs.
    """
    try:
        # Handle timestamp inputs
        if root_timestamp is not None and isinstance(message, float):
            time_diff = (
                datetime.fromtimestamp(message) - datetime.fromtimestamp(root_timestamp)
            ).total_seconds() / 60

        # Handle ChainMap inputs
        elif sub_thread_root is not None and isinstance(message, ChainMap):
            if not message or not sub_thread_root:
                logging.error("Either message or sub_thread_root is invalid.")
                return 1.0

            time_diff = (
                datetime.fromtimestamp(message.message.create_time)
                - datetime.fromtimestamp(sub_thread_root.message.create_time)
            ).total_seconds() / 60

        else:
            logging.error("Invalid input arguments.")
            return 1.0

        # Calculate the time decay factor
        decay_factor = time_diff_decay_function(time_diff)

        # Adjust decay factor based on sibling time differences
        if isinstance(sibling_time_differences, list) and sibling_time_differences:
            decay_factor *= np.mean(sibling_time_differences)

        return decay_factor

    except Exception as e:
        logging.error(f"Error occurred while calculating time decay factor: {str(e)}")
        return 1.0
