from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
import pandas as pd
import logging


def build_graph_from_dataframe(
    df: pd.DataFrame, column_mapping: Dict[str, str]
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Constructs a directed graph from the provided DataFrame using a mapping of attributes to column names.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the conversational data.
    - column_mapping (Dict[str, str]): A dictionary mapping attributes to their respective column names in the DataFrame.

    Returns:
    - Tuple[nx.DiGraph, Dict[str, Any]]: A tuple containing the directed graph and a dictionary of attributes associated with the graph.
    """

    graph = nx.DiGraph()

    # Iterate over the DataFrame to add nodes and edges to the graph
    for _, row in df.iterrows():
        prompt_id = row[column_mapping["prompt_id"]]
        response_id = row[column_mapping["response_id"]]

        # Add nodes with attributes
        graph.add_node(
            prompt_id, **{k: row[v] for k, v in column_mapping.items() if "prompt" in k}
        )
        graph.add_node(
            response_id,
            **{k: row[v] for k, v in column_mapping.items() if "response" in k},
        )

        # Add an edge between the prompt and response
        graph.add_edge(prompt_id, response_id)

    # Attributes of the graph (e.g., number of nodes and edges)
    graph_attributes = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }

    return graph, graph_attributes


def _validate_and_trim_arrays(*arrays):
    """
    Validates and trims arrays to the length of the shortest non-empty array.
    Skips empty arrays.
    """
    # Filter out empty arrays
    non_empty_arrays = [arr for arr in arrays if len(arr) > 0]

    if not non_empty_arrays:
        raise ValueError("All provided arrays are empty")

    # Find the minimum length among non-empty arrays
    min_length = min(len(arr) for arr in non_empty_arrays)

    # Trim arrays to the minimum length
    trimmed_arrays = [arr[:min_length] for arr in arrays if len(arr) > 0]

    return tuple(trimmed_arrays)


def build_dataframe(
    collected_messages: tuple,
    conversation_id: str,
    filter_words: Optional[List[str]] = None,
    build_graph: bool = False,
) -> pd.DataFrame:
    """
    Constructs a structured DataFrame from provided conversational data and optionally builds a graph.

    Parameters:
    - collected_messages (Tuple): A packed tuple containing conversational data features.
    - conversation_id (str): Identifier for the conversation.
    - build_graph (bool): Flag to determine if a graph should be constructed from the DataFrame.
    - column_mapping (Optional[Dict[str, str]]): A dictionary mapping attributes to column names for graph construction.

    Returns:
    - pd.DataFrame: A DataFrame representing the conversation with optional graph attributes.
    """

    try:
        (
            prompts,
            responses,
            prompt_ids,
            response_ids,
            created_times,
            prompt_coordinates,
            response_coordinates,
            prompt_encodings,
            response_encodings,
        ) = collected_messages

    # Validate and trim arrays to ensure consistency
    except Exception as e:
        logging.error(f"Error unpacking collected messages: {e}")
        raise

    try:
        (
            prompts,
            responses,
            prompt_ids,
            response_ids,
            created_times,
            prompt_coordinates,
            response_coordinates,
            prompt_encodings,
            response_encodings,
        ) = _validate_and_trim_arrays(
            prompts,
            responses,
            prompt_ids,
            response_ids,
            created_times,
            prompt_coordinates,
            response_coordinates,
            prompt_encodings,
            response_encodings,
        )

    except ValueError as e:
        logging.error(f"Array validation error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in array trimming: {e}")
        raise

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "response_id": response_ids,
            "prompt": prompts,
            "response": responses,
            "prompt_embedding": prompt_encodings,
            "response_embedding": response_encodings,
            "created_time": created_times,
            "prompt_coordinate": prompt_coordinates,
            "response_coordinate": response_coordinates,
            "conversation_id": [conversation_id] * len(prompt_ids),
        }
    )

    # Optionally build a graph from the DataFrame
    if build_graph:
        if column_mapping is None:
            # Define a default column mapping if none provided
            column_mapping = {
                "prompt_id": "prompt_id",
                "response_id": "response_id",
                "prompt": "prompt",
                "response": "response",
                "prompt_embedding": "prompt_embedding",
                "response_embedding": "response_embedding",
                "prompt_coordinate": "prompt_coordinate",
                "response_coordinate": "response_coordinate",
                "created_time": "created_time",
                "conversation_id": "conversation_id",
            }

        # Construct the graph from the DataFrame
        graph, graph_attributes = build_graph_from_dataframe(df, column_mapping)

        # Store the graph and attributes in the DataFrame if needed
        df["graph"] = [graph] * len(df)
        df["graph_attributes"] = [graph_attributes] * len(df)

    return df


def split_dataframe_by_column(df, column_name: str, value):
    """
    Split a DataFrame into two parts based on a column value.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column used for splitting.
    - value: The value used as a criterion for splitting.

    Returns:
    - tuple: Two DataFrames. The first contains rows where the column has the specified value.
             The second contains all other rows.
    """
    df_match = df[df[column_name] == value]
    df_no_match = df[df[column_name] != value]
    return df_match, df_no_match


def split_dataframe_by_index(df, index):
    """
    Split a DataFrame into two parts based on a row index.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - tuple: Two DataFrames. The first contains rows up to the specified index.
             The second contains all other rows.
    """

    df_match = df.iloc[:index]
    df_no_match = df.iloc[index:]
    return df_match, df_no_match
