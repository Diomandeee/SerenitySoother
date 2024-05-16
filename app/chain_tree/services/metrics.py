from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy import stats
import pandas as pd
import numpy as np
import plotly.express as px

import torch
import ast
import os


def calculate_similarity(embeddings1, embeddings2):
    """
    Calculate semantic similarity between two sets of embeddings using cosine similarity.
    ... [rest of your docstring] ...
    """

    # Function to convert embeddings to numpy arrays
    def convert_to_numpy(embeddings):
        if isinstance(embeddings, torch.Tensor):
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


def convert_embedding(embedding):
    if isinstance(embedding, str):
        return ast.literal_eval(
            embedding
        )  # Convert string representation of list to actual list
    elif (
        isinstance(embedding, list)
        or isinstance(embedding, np.ndarray)
        or isinstance(embedding, torch.Tensor)
    ):
        return embedding
    else:
        raise TypeError(
            "Unsupported embedding type. Must be a string, list, numpy array, or PyTorch tensor."
        )


def visualize_bursts(df: pd.DataFrame, prompt_time_col: str, similarity_col: str):
    """
    Visualizes detected bursts using a scatter plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data and burst cluster labels.
    - prompt_time_col (str): Name of the column containing the prompt time.
    - similarity_col (str): Name of the column containing the similarity scores.
    """

    plt.figure(figsize=(10, 6))

    # Plot non-clustered points in black
    mask = df["burst_clusters"] == -1
    plt.scatter(
        df[mask][prompt_time_col],
        df[mask][similarity_col],
        color="black",
        s=20,
        label="Noise",
    )

    # Plot clustered points in different colors
    unique_labels = set(df["burst_clusters"])
    unique_labels.remove(-1)
    for label in unique_labels:
        mask = df["burst_clusters"] == label
        plt.scatter(
            df[mask][prompt_time_col],
            df[mask][similarity_col],
            s=50,
            label=f"Cluster {label}",
        )

    plt.xlabel(prompt_time_col)
    plt.ylabel(similarity_col)
    plt.title("Burst Detection Results")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def visualize_clusters(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Visualizes clustering results using a scatter plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data and cluster labels.
    - x_col (str): Name of the column to be used for x-axis.
    - y_col (str): Name of the column to be used for y-axis.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df[x_col],
        df[y_col],
        c=df["pattern_label"],
        cmap="rainbow",
        edgecolors="k",
        s=100,
    )
    plt.colorbar()
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("DBSCAN Clustering Results")
    plt.grid(True)
    plt.show()


def compute_cosine_similarity(relationship_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the cosine similarity between the embeddings of prompts and responses in the provided DataFrame.
    Cosine similarity measures the cosine of the angle between two non-zero vectors, making it a measure of
    similarity between these vectors. For embeddings in natural language processing, a higher cosine similarity
    indicates a greater degree of semantic similarity between the two compared text segments.

    The function aims to quantify the similarity between the embeddings of prompts and their corresponding responses
    and augments the original DataFrame with this newly calculated similarity measure.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame that contains at least two columns, 'prompt_embedding'
                                      and 'response_embedding'. Each entry in these columns should be an embedding
                                      (a vector) representing the prompt or the response, respectively.

    Workflow:
    1. Validate that the provided DataFrame contains the required columns: 'prompt_embedding' and 'response_embedding'.
    2. Iterate over each row in the DataFrame:
        a. Extract and reshape the embeddings for the prompt and response.
        b. Compute the cosine similarity between these embeddings.
        c. Append the computed similarity value to a list.
    3. Add the list of computed cosine similarities to the original DataFrame as a new column named 'cosine_similarity'.
    4. Return the updated DataFrame.

    Returns:
    - pd.DataFrame: The input DataFrame augmented with an additional column, 'cosine_similarity', which contains the
                    computed similarity scores between the corresponding prompts and responses in each row.

    By providing a measure of similarity between prompts and responses, this function aids in understanding
    how semantically close the responses are to their prompts, potentially serving as an indicator of the
    relevance or appropriateness of the response in the context of its prompt.
    """

    # Ensure the DataFrame contains the necessary columns
    if (
        "prompt_embedding" not in relationship_df.columns
        or "response_embedding" not in relationship_df.columns
    ):
        raise ValueError(
            "The DataFrame must contain 'prompt_embedding' and 'response_embedding' columns."
        )

    # Compute cosine similarity for each row and store it in a list
    cosine_similarities = []
    for _, row in relationship_df.iterrows():
        prompt_embedding = row["prompt_embedding"]
        response_embedding = row["response_embedding"]

        similarity = calculate_similarity(prompt_embedding, response_embedding)
        cosine_similarities.append(similarity)

    # Add the cosine similarity list as a new column to the DataFrame
    relationship_df["similarity"] = cosine_similarities

    return relationship_df


def add_ranking_and_flags(
    relationship_df: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    """
    Enhances the provided DataFrame by computing and adding several metrics related to cosine similarity.
    It evaluates the relationships based on their similarity scores and adds rankings, flags, and other
    metrics to help in further analysis and interpretation.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame that primarily contains a column 'similarity'
                                      indicating the similarity between two entities (e.g., prompts and responses).
    - threshold (float, optional): A predefined threshold value to determine if a similarity score
                                   is above a certain desirable level. Defaults to 0.8.

    Workflow:
    1. Rank the similarity scores:
        a. Rank entries based on their 'similarity' values in descending order, such that
           the highest similarity gets the top rank.
        b. Compute percentiles for each similarity score.
    2. Flag entries based on whether their similarity score surpasses the given threshold.
    3. Calculate the Z-score for the similarity scores. This standardizes the scores and indicates
       how many standard deviations a value is from the mean.
    4. Normalize the similarity scores between 0 and 1 using Min-Max scaling.
    5. Identify outliers based on the Interquartile Range (IQR) method.
    6. Calculate the running average of similarity scores.

    Returns:
    - pd.DataFrame: The enhanced DataFrame containing the following additional columns:
        a. 'similarity_rank': Rank of each entry based on its cosine similarity.
        b. 'similarity_percentile': Percentile rank of each entry's similarity score.
        c. 'above_threshold': A binary flag indicating if the similarity score is above the predefined threshold.
        d. 'z_score': The Z-score of the similarity.
        e. 'scaled_similarity': The normalized similarity score between 0 and 1.
        f. 'is_outlier': A binary flag indicating whether the similarity score is considered an outlier.
        g. 'running_avg_similarity': The running average of similarity scores.

    This function provides an analytical perspective on the relationship data, enabling better insights
    into patterns, exceptional cases, and general tendencies within the data. The generated metrics can be
    valuable for exploratory data analysis, data visualization, and decision-making processes.
    """

    # Basic Ranking and Flags
    relationship_df["similarity_rank"] = relationship_df["similarity"].rank(
        method="dense", ascending=False
    )
    relationship_df["similarity_percentile"] = (
        relationship_df["similarity"].rank(pct=True) * 100
    )
    relationship_df["above_threshold"] = (
        relationship_df["similarity"] > threshold
    ).astype(int)

    # Z-Score of similarity
    relationship_df["z_score"] = stats.zscore(relationship_df["similarity"])

    # Min-Max Scaling of similarity
    min_val = relationship_df["similarity"].min()
    max_val = relationship_df["similarity"].max()
    relationship_df["scaled_similarity"] = (relationship_df["similarity"] - min_val) / (
        max_val - min_val
    )

    # Outlier detection based on IQR
    Q1 = relationship_df["similarity"].quantile(0.25)
    Q3 = relationship_df["similarity"].quantile(0.75)
    IQR = Q3 - Q1
    relationship_df["is_outlier"] = (
        (relationship_df["similarity"] < (Q1 - 1.5 * IQR))
        | (relationship_df["similarity"] > (Q3 + 1.5 * IQR))
    ).astype(int)

    relationship_df["running_avg_similarity"] = (
        relationship_df["similarity"].expanding().mean()
    )

    return relationship_df


def add_categorical_and_time_series_metrics(
    relationship_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enhances the provided DataFrame by computing and adding time-series and categorical metrics. This function
    focuses on deriving insights from the similarity scores over a sequence, such as change in similarity from
    one entry to the next, and categorizing similarity scores into predefined groups.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame that primarily contains a column 'similarity'
                                      indicating the similarity between two entities (e.g., prompts and responses).

    Workflow:
    1. Time-Series Metrics:
        a. Compute the difference in 'similarity' between consecutive rows. This provides insights
           into the change in similarity from one relationship to the next, helping identify any trends or patterns.
    2. Categorical Metrics:
        a. Categorize each similarity score into one of several predefined categories. This simplifies
           the interpretation of scores by grouping them into broader buckets based on their magnitude.

    Returns:
    - pd.DataFrame: The enhanced DataFrame containing the following additional columns:
        a. 'similarity_change': Represents the change in similarity score from the previous row.
        b. 'similarity_category': Categorical representation of the similarity score. The categories
                                   are defined as follows:
            - 'Low' for similarity scores between 0 and 0.2.
            - 'Medium-Low' for scores between 0.2 and 0.5.
            - 'Medium-High' for scores between 0.5 and 0.7.
            - 'High' for scores between 0.7 and 1.

    This function aims to simplify and summarize the relationship data in ways that are easier to
    understand and visualize. By grouping similarity scores into categories and computing their change
    over sequences, users can quickly assess the nature and progression of relationships in the data.
    """

    relationship_df["similarity_change"] = relationship_df["similarity"].diff()
    bins = [0, 0.2, 0.5, 0.7, 1]
    labels = ["Low", "Medium-Low", "Medium-High", "High"]
    relationship_df["similarity_category"] = pd.cut(
        relationship_df["similarity"], bins=bins, labels=labels
    )
    return relationship_df


def get_top_k_prompt_ids(
    relationship_df: pd.DataFrame, similarity_matrix: np.array, k: int = 3
) -> pd.DataFrame:
    """
    Enhances the relationship DataFrame by identifying and adding the top k most similar prompts for each response.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame containing the relationship data. It is expected to have
                                      at least a column named 'prompt_id' representing the identifier for prompts.
    - similarity_matrix (np.array): A 2D array where the element at [i, j] represents the similarity between
                                    the i-th prompt and the j-th response. The dimensions of this matrix should be
                                    (number of prompts x number of responses).
    - k (int): The number of top similar prompts to retrieve for each response. This value defines how many
               of the most similar prompts will be added to the DataFrame for each response.

    Workflow:
    1. For each response (column in the similarity matrix):
        a. Identify the top k prompts (rows) that have the highest similarity scores with the response.
        b. Retrieve the identifiers of these prompts from the relationship DataFrame.
        c. Retrieve the corresponding similarity scores.
    2. Add two new columns to the relationship DataFrame:
        a. 'top_k_prompt_ids': Contains the list of the top k prompt IDs for each response.
        b. 'top_k_prompt_scores': Contains the list of the corresponding similarity scores.

    Returns:
    - pd.DataFrame: The enhanced DataFrame with added columns providing insights into which prompts are
                    most similar to each response.

    By using this function, users can quickly identify the most closely related prompts for each response
    in the dataset. This can be beneficial for numerous applications, including recommendation systems,
    content optimization, or simply to gain a deeper understanding of the relationships present in the data.
    """

    top_k_indices = np.argsort(-similarity_matrix, axis=0)[:k, :]
    top_k_scores = -np.sort(-similarity_matrix, axis=0)[:k, :]

    top_k_prompt_ids = [
        relationship_df["prompt_id"].iloc[top_k_indices[:, i]].tolist()
        for i in range(top_k_indices.shape[1])
    ]
    top_k_prompt_scores = [
        top_k_scores[:, i].tolist() for i in range(top_k_scores.shape[1])
    ]

    relationship_df["top_k_prompt_ids"] = top_k_prompt_ids
    relationship_df["top_k_prompt_scores"] = top_k_prompt_scores

    return relationship_df


def add_top_k_responses(relationship_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Augments the relationship DataFrame by identifying and adding the top k most similar responses for each prompt.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame containing the relationship data. It is expected to possess
                                      columns named 'prompt_embedding' for embeddings of the prompts and 'response_embedding'
                                      for embeddings of the responses, as well as 'response_id' for identifying each response.
    - k (int): The number of top similar responses to retrieve for each prompt. This parameter dictates how many
               of the most similar responses will be added to the DataFrame for each prompt.

    Workflow:
    1. Extract prompt and response embeddings from the DataFrame.
    2. Compute the similarity matrix between all prompts and responses using the cosine similarity measure.
    3. For each prompt (row in the similarity matrix):
        a. Identify the top k responses (columns) that have the highest similarity scores with the prompt.
        b. Retrieve the identifiers of these responses from the relationship DataFrame.
        c. Retrieve the corresponding similarity scores.
    4. Append two new columns to the relationship DataFrame:
        a. 'top_k_response_ids': Contains the list of the top k response IDs for each prompt.
        b. 'top_k_response_scores': Contains the list of the corresponding similarity scores.
    5. Utilize the previously defined 'get_top_k_prompt_ids' function to also add the top k prompts for each response.

    Returns:
    - pd.DataFrame: An enhanced DataFrame that now includes columns indicating which responses are most similar
                    to each prompt and vice versa.

    By employing this function, users can rapidly discern the most closely related responses for each prompt
    in the dataset. This can prove invaluable for a range of applications like enhancing user experience by
    suggesting relevant content, optimizing content presentation based on similarity, or simply gaining more
    profound insights into the inherent relationships in the data.
    """

    prompt_encodings = np.array(relationship_df["prompt_embedding"].tolist())
    response_encodings = np.array(relationship_df["response_embedding"].tolist())

    similarity_matrix = cosine_similarity(prompt_encodings, response_encodings)

    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
    top_k_scores = -np.sort(-similarity_matrix, axis=1)[
        :, :k
    ]  # Sort in descending order

    relationship_df["top_k_response_ids"] = [
        relationship_df["response_id"].iloc[top_k_indices[i]].tolist()
        for i in range(len(relationship_df))
    ]

    relationship_df["top_k_response_scores"] = [
        top_k_scores[i].tolist() for i in range(len(relationship_df))
    ]

    relationship_df = get_top_k_prompt_ids(relationship_df, similarity_matrix, k)

    return relationship_df


def calculate_additional_metrics(relationship_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and adds several similarity-based metrics to the relationship DataFrame.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame containing the relationship data. The DataFrame is expected
                                      to have 'top_k_response_scores' and 'top_k_prompt_scores' columns, which
                                      respectively contain lists of similarity scores between a prompt and its top k
                                      most similar responses, and a response and its top k most similar prompts.

    Workflow:
    1. Calculate the average similarity score for each list in the 'top_k_response_scores' and 'top_k_prompt_scores'
       columns and add them to the DataFrame as 'avg_top_k_response_scores' and 'avg_top_k_prompt_scores' respectively.
    2. Determine the maximum and minimum similarity scores in each list for both columns and add them to the DataFrame
       as 'max_top_k_response_scores', 'min_top_k_response_scores', 'max_top_k_prompt_scores', and 'min_top_k_prompt_scores'.
    3. Compute the range of similarity scores (difference between maximum and minimum) for both columns and append them
       to the DataFrame as 'range_top_k_response_scores' and 'range_top_k_prompt_scores'.

    Returns:
    - pd.DataFrame: The updated DataFrame now contains several new columns that offer a richer perspective on
                    the relationships and similarities between prompts and responses. Specifically, these metrics
                    furnish a more detailed view of how similar the top k responses or prompts are to each other
                    for any given prompt or response. This can be beneficial in scenarios where understanding
                    the distribution of similarity scores is crucial.

    By incorporating these metrics into the DataFrame, users can better gauge the overall relationship landscape
    within their dataset. This can be particularly useful for tasks such as anomaly detection, content optimization,
    and understanding user interactions.
    """
    # Average Similarity Score
    relationship_df["avg_top_k_response_scores"] = relationship_df[
        "top_k_response_scores"
    ].apply(np.mean)
    relationship_df["avg_top_k_prompt_scores"] = relationship_df[
        "top_k_prompt_scores"
    ].apply(np.mean)

    # Max/Min Similarity Score
    relationship_df["max_top_k_response_scores"] = relationship_df[
        "top_k_response_scores"
    ].apply(np.max)
    relationship_df["min_top_k_response_scores"] = relationship_df[
        "top_k_response_scores"
    ].apply(np.min)

    relationship_df["max_top_k_prompt_scores"] = relationship_df[
        "top_k_prompt_scores"
    ].apply(np.max)
    relationship_df["min_top_k_prompt_scores"] = relationship_df[
        "top_k_prompt_scores"
    ].apply(np.min)

    # Range of Similarity Scores
    relationship_df["range_top_k_response_scores"] = (
        relationship_df["max_top_k_response_scores"]
        - relationship_df["min_top_k_response_scores"]
    )
    relationship_df["range_top_k_prompt_scores"] = (
        relationship_df["max_top_k_prompt_scores"]
        - relationship_df["min_top_k_prompt_scores"]
    )

    return relationship_df


def calculate_density_of_scores(
    relationship_df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    Calculates and adds a density metric to the relationship DataFrame based on similarity scores.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame containing the relationship data. It's expected to have columns
                                      'top_k_response_scores' and 'top_k_prompt_scores', which respectively contain
                                      lists of similarity scores between a prompt and its top k most similar responses,
                                      and a response and its top k most similar prompts.
    - threshold (float): A similarity score threshold. The density is calculated as the proportion of scores
                         in the list that exceed this threshold.

    Workflow:
    1. For each list in 'top_k_response_scores', calculate the proportion of scores that exceed the given threshold.
    2. Append this proportion to the DataFrame as 'density_of_response_scores'.
    3. Repeat the above two steps for 'top_k_prompt_scores' to get 'density_of_prompt_scores'.

    Returns:
    - pd.DataFrame: The updated DataFrame now contains two new columns: 'density_of_response_scores' and
                    'density_of_prompt_scores'. These columns represent the proportion of top k similarity scores
                    that are above the provided threshold for each prompt and response, respectively.

    The density metric provides an insight into how many of the top k similarity scores are above a certain threshold.
    This can help in understanding the overall quality of relationships in the dataset. For instance, a high density
    might indicate that a prompt or response has strong relationships with many of its top k counterparts, whereas
    a low density might suggest that the relationships are weaker or less consistent.
    """

    relationship_df["density_of_response_scores"] = relationship_df[
        "top_k_response_scores"
    ].apply(lambda x: np.mean(np.array(x) > threshold))
    relationship_df["density_of_prompt_scores"] = relationship_df[
        "top_k_prompt_scores"
    ].apply(lambda x: np.mean(np.array(x) > threshold))
    return relationship_df


def calculate_popularity(relationship_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds columns to the relationship DataFrame indicating the popularity of each prompt and response.

    Parameters:
    - relationship_df (pd.DataFrame): A DataFrame containing the relationship data. It's expected to have columns
                                      'top_k_response_ids' and 'top_k_prompt_ids', which respectively contain lists of
                                      response_ids that are most similar to a given prompt, and prompt_ids that are
                                      most similar to a given response.

    Workflow:
    1. For the 'top_k_response_ids' column, count the occurrence of each response_id across all rows to determine
       its popularity.
    2. Add this count to the DataFrame as 'response_popularity', mapping the popularity count to the corresponding
       response_id in the 'response_id' column.
    3. Repeat the above two steps for 'top_k_prompt_ids' to get 'prompt_popularity'.

    Returns:
    - pd.DataFrame: The updated DataFrame now contains two new columns: 'response_popularity' and 'prompt_popularity'.
                    These columns represent the popularity count of each response and prompt, respectively.

    Popularity is defined as the number of times a particular response or prompt appears among the top k similar items
    across all rows. High popularity suggests that a prompt or response frequently appears as a top match for various
    items in the dataset, indicating it has a more generalized relationship. In contrast, low popularity suggests
    more specialized or unique relationships.
    """
    response_popularity = relationship_df["top_k_response_ids"].explode().value_counts()
    prompt_popularity = relationship_df["top_k_prompt_ids"].explode().value_counts()

    relationship_df["response_popularity"] = relationship_df["response_id"].map(
        response_popularity
    )
    relationship_df["prompt_popularity"] = relationship_df["prompt_id"].map(
        prompt_popularity
    )
    return relationship_df


def calculate_overlap(relationship_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the overlap between different prompts and responses based on their top similar matches.

    For each prompt, the overlap represents the count of its top k similar responses that are also
    top k similar responses for other prompts. Similarly, for each response, the overlap represents
    the count of its top k similar prompts that are also top k similar prompts for other responses.

    Parameters:
    - relationship_df (pd.DataFrame): DataFrame containing top k response_ids and prompt_ids.

    Returns:
    - pd.DataFrame: Updated DataFrame with two new columns: 'response_overlap' and 'prompt_overlap'.
                    These columns represent the overlap count of top similar items for each response and prompt, respectively.

    Overlap serves as a measure of the generality of relationships. High overlap suggests that an item's
    top similar matches are frequently considered top matches for various other items in the dataset.
    Conversely, low overlap indicates more specialized or unique relationships.
    """

    # Create dictionaries to map each prompt/response to its top k matches
    prompt_to_responses = dict(
        zip(relationship_df["prompt_id"], relationship_df["top_k_response_ids"])
    )
    response_to_prompts = dict(
        zip(relationship_df["response_id"], relationship_df["top_k_prompt_ids"])
    )

    # 1. Compute overlap for each prompt
    prompt_overlap = {}
    for prompt, top_responses in prompt_to_responses.items():
        overlap_count = sum(
            [
                sum(
                    [
                        (r in prompt_to_responses.get(p, []))
                        for p in relationship_df["prompt_id"]
                        if p != prompt
                    ]
                )
                for r in top_responses
            ]
        )
        prompt_overlap[prompt] = overlap_count

    # 2. Compute overlap for each response
    response_overlap = {}
    for response, top_prompts in response_to_prompts.items():
        overlap_count = sum(
            [
                sum(
                    [
                        (p in response_to_prompts.get(r, []))
                        for r in relationship_df["response_id"]
                        if r != response
                    ]
                )
                for p in top_prompts
            ]
        )
        response_overlap[response] = overlap_count

    # Map overlap values to dataframe
    relationship_df["response_overlap"] = relationship_df["response_id"].map(
        response_overlap
    )
    relationship_df["prompt_overlap"] = relationship_df["prompt_id"].map(prompt_overlap)

    return relationship_df


def calculate_distances(
    df: pd.DataFrame, prompt_coordinate_col: str, response_coordinate_col: str
) -> pd.DataFrame:
    """
    Calculate Euclidean distance for specific indices and entire array.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with coordinates.
    - prompt_coordinate_col (str): Column name containing prompt coordinates.
    - response_coordinate_col (str): Column name containing response coordinates.

    Returns:
    - pd.DataFrame: DataFrame with added columns for Euclidean distance (entire array and first 3 indices).
    """

    # convert coordinates to numpy arrays 1 D
    df[prompt_coordinate_col] = df[prompt_coordinate_col].apply(
        lambda x: np.array(ast.literal_eval(x))
    )
    df[response_coordinate_col] = df[response_coordinate_col].apply(
        lambda x: np.array(ast.literal_eval(x))
    )
    # just the first 4 indices
    df["euclidean_distance"] = df.apply(
        lambda row: distance.euclidean(
            row[prompt_coordinate_col][:4], row[response_coordinate_col][:4]
        ),
        axis=1,
    )

    df["running_avg_euclidean_distance"] = df["euclidean_distance"].expanding().mean()

    return df


def add_time_decay(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Adds a time decay feature to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - timestamp_col (str): Column name containing timestamps.

    Returns:
    - pd.DataFrame: DataFrame with added time decay feature.
    """
    df.sort_values(by=[timestamp_col], inplace=True)
    df["time_decay"] = np.exp(-df.index / df.shape[0])
    return df


def detect_anomalies(df: pd.DataFrame, columns_to_compare: List[str]) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with metrics.
    - columns_to_compare (List[str]): List of columns to use for anomaly detection.

    Returns:
    - pd.DataFrame: DataFrame with added anomaly flag.
    """
    clf = IsolationForest(contamination=0.01)
    df["anomaly_flag"] = clf.fit_predict(df[columns_to_compare])
    return df


def flag_high_density(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Flags entries with high density based on a given threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with similarity metrics.
    - threshold (float): Threshold to consider for high density flag.

    Returns:
    - pd.DataFrame: DataFrame with added high density flag.
    """
    df["high_density_flag"] = (df["similarity"] > threshold).astype(int)
    return df


def calculate_response_time(df: pd.DataFrame, prompt_time_col: str) -> pd.DataFrame:
    """
    Calculates the response time for each prompt-response pair.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with time data.
    - prompt_time_col (str): Column name containing the time each prompt was created.

    Returns:
    - pd.DataFrame: DataFrame with an added column for response time (in seconds).
    """

    # Shift the created_time column by one entry to compute the difference
    df["response_time"] = (df["created_time"].shift(-1) - df[prompt_time_col]).astype(
        "timedelta64[s]"
    )

    # Since there will be no response for the last prompt, we can set its response time as NaN
    df.loc[df.index[-1], "response_time"] = pd.NaT

    return df


def add_statistical_analysis(
    relationship_df: pd.DataFrame,
    prompt_coordinate_col: str,
    response_coordinate_col: str,
    columns_to_compare: List[str],
    timestamp_col: str = None,
) -> pd.DataFrame:
    """
    Augments the provided DataFrame with statistical and analytical metrics derived from the coordinates
    and other columns. This function acts as a master function, leveraging several helper functions to
    compute various metrics like distance calculations, time-decay weighting, anomaly detection, and
    data density flags.

    Parameters:
    - relationship_df (pd.DataFrame): The DataFrame with coordinates and potentially existing similarity scores.
    - prompt_coordinate_col (str): Name of the column containing the prompt coordinates.
    - response_coordinate_col (str): Name of the column containing the response coordinates.
    - columns_to_compare (List[str]): Columns to analyze for correlation or other statistical comparisons.
    - timestamp_col (str, optional): Column with timestamps, if present. Used for time-decay computations.

    Workflow:
    1. Compute distances or similarities between the prompt and response coordinates.
    2. If timestamps are provided, calculate a time-decay weighting for the data.
    3. Detect anomalies in the specified columns.
    4. Flag data points that lie in high-density regions.

    Returns:
    - pd.DataFrame: The input DataFrame augmented with the computed metrics.

    The provided dataframe is enhanced to provide deeper insights into the relationships
    between prompts and responses, as well as any temporal dynamics that may be present.
    """

    relationship_df = calculate_distances(
        relationship_df, prompt_coordinate_col, response_coordinate_col
    )

    if timestamp_col:
        relationship_df = add_time_decay(relationship_df, timestamp_col)

    relationship_df = detect_anomalies(relationship_df, columns_to_compare)
    relationship_df = flag_high_density(relationship_df)

    return relationship_df


def calculate_time_decay_similarity(
    df: pd.DataFrame, prompt_time_col: str, similarity_col: str
) -> pd.DataFrame:
    """
    Calculates the time decay in similarity for each prompt by computing the expanding mean similarity score.

    An expanding mean takes into account all the data points up to the current point and averages them. This method ensures that as more time passes (and more data becomes available), the similarity value is influenced by the aggregate effect of all previous similarity values. The result is that older similarity scores will have a reduced influence on the overall score, leading to the effect of a time decay.

    Parameters:
    - df (pd.DataFrame): The input DataFrame that contains data on prompt times and similarity scores.
    - prompt_time_col (str): The column name in the DataFrame which specifies when each prompt was created or made.
    - similarity_col (str): The column name in the DataFrame which provides the similarity scores.

    Workflow:
    1. Sort the DataFrame based on the prompt's time of creation.
    2. Calculate the expanding mean of similarity scores for each prompt. This gives the effect of a time decay since older values gradually have less impact on the average.

    Returns:
    - pd.DataFrame: An updated DataFrame containing a new column, 'time_decay_similarity', that indicates the decayed similarity for each prompt based on its time.

    Example:
    If the DataFrame contains similarity scores for a prompt over 5 days as [0.9, 0.8, 0.75, 0.8, 0.85], the 'time_decay_similarity' column will contain values that represent the average score up to that day, e.g., [0.9, 0.85, 0.8166, 0.8125, 0.81].
    """

    df.sort_values(by=[prompt_time_col], inplace=True)
    df["time_decay_similarity"] = (
        df.groupby("prompt_id")[similarity_col]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def calculate_temporal_density(df: pd.DataFrame, prompt_time_col: str) -> pd.DataFrame:
    """
    Calculates the temporal density based on prompt time.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with time data.
    - prompt_time_col (str): Column name containing the time each prompt was created.

    Returns:
    - pd.DataFrame: DataFrame with an added column for temporal density.
    """

    min_time = df[prompt_time_col].min()
    max_time = df[prompt_time_col].max()
    df["temporal_density"] = df[prompt_time_col].apply(
        lambda x: (x - min_time) / (max_time - min_time)
    )
    return df


def perform_pattern_detection(
    df: pd.DataFrame,
    similarity_col: str,
    eps: float = 0.5,
    min_samples: int = 5,
    visualize: bool = True,
) -> pd.DataFrame:
    """
    Detects patterns using DBSCAN clustering on similarity and temporal density.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with similarity and density data.
    - similarity_col (str): Column name containing the similarity scores.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - visualize (bool): If True, visualize the clustering result.

    Returns:
    - pd.DataFrame: DataFrame with an added column for pattern labels.
    """

    # Scaling data
    scaler = StandardScaler()
    data = df[["temporal_density", similarity_col]]
    scaled_data = scaler.fit_transform(data)

    # Applying DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["pattern_label"] = dbscan.fit_predict(scaled_data)

    # Visualization (optional)
    if visualize:
        visualize_clusters(df, "temporal_density", similarity_col)

    return df


def perform_burst_detection(
    df: pd.DataFrame,
    prompt_time_col: str,
    similarity_col: str,
    eps: float = 0.3,
    min_samples: int = 10,
    visualize: bool = True,
) -> pd.DataFrame:
    """
    Detects bursts in the data based on similarity and scaled time.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with similarity and time data.
    - prompt_time_col (str): Column name containing the time each prompt was created.
    - similarity_col (str): Column name containing the similarity scores.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - visualize (bool): If True, visualize the clustering result.

    Returns:
    - pd.DataFrame: DataFrame with an added column for burst cluster labels.
    """

    # Ensure the columns exist
    for col in [prompt_time_col, similarity_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Handle NaN values (optional)
    df = df.dropna(subset=[prompt_time_col, similarity_col])

    # Scaling time
    scaler = StandardScaler()
    df["scaled_time"] = scaler.fit_transform(df[prompt_time_col].values.reshape(-1, 1))

    # Applying DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    df["burst_clusters"] = db.fit_predict(df[["scaled_time", similarity_col]])

    # Visualization (optional)
    if visualize:
        visualize_bursts(df, prompt_time_col, similarity_col)

    return df


def perform_clustering(
    df: pd.DataFrame, embedding_col: str, eps: float = 0.5, min_samples: int = 5
) -> pd.DataFrame:
    """
    Applies the DBSCAN clustering algorithm on the embeddings present in the DataFrame and adds clustering information
    to it. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) works on the principle of identifying
    dense regions in the data space where many closely located data points exist, separated by less dense regions or
    regions with few or no data points.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing embeddings, where each row corresponds to a unique data point.
    - embedding_col (str): The name of the column in the DataFrame containing the embeddings. These embeddings are
      typically high-dimensional vector representations of data points.
    - eps (float, optional): The maximum distance between two data points for one to be considered as being in the
      neighborhood of the other. This parameter effectively controls the size of the neighborhoods. Default is 0.5.
    - min_samples (int, optional): The minimum number of data points required in a neighborhood to consider the central
      data point as a core point. Core points are central to forming clusters. If a data point is not a core point and
      is not part of a cluster, it's treated as noise. Default is 5.

    Returns:
    - pd.DataFrame: The input DataFrame augmented with a new column named "cluster_label". This column contains the
      cluster labels assigned by the DBSCAN algorithm. Points that are considered noise (i.e., not part of any cluster)
      by the algorithm are labeled as -1.

    Note:
    - The success of DBSCAN often depends on the appropriate setting of the `eps` and `min_samples` parameters. It's
      advisable to visualize the data or use domain-specific knowledge to help set these parameters.
    """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    embeddings = np.stack(df[embedding_col].to_numpy())
    labels = dbscan.fit_predict(embeddings)
    df["cluster_label"] = labels

    return df


def add_temporal_trends(
    df: pd.DataFrame,
    prompt_time_col: str,
    similarity_col: str,
    min_samples=5,
    perform_analysis=True,
) -> pd.DataFrame:
    """
    Incorporates various temporal trend metrics into the DataFrame based on similarity and time data.

    This function encompasses multiple stages of temporal analysis:
    1. Time Decay in Similarity: It computes an average similarity score for each prompt over time,
       representing how the similarity has decayed (or evolved) as new data points are added.
    2. Response Time: For each prompt-response pair, it calculates the time taken for the response to
       be generated after the prompt was created.
    3. Temporal Density: It calculates the density based on the creation time of each prompt. This is
       done by normalizing the prompt time with respect to the time range covered in the dataset.

    Optionally, if perform_analysis is set to True:
    4. Pattern Detection: It utilizes DBSCAN clustering on the similarity and temporal density metrics
       to detect patterns in the data. The resultant clusters can indicate groupings of prompts that
       have similar temporal and similarity characteristics.
    5. Burst Detection: The function scales the prompt time and applies DBSCAN clustering to detect
       bursts of activity based on similarity scores over time.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with similarity and time data.
    - prompt_time_col (str): Column name containing the time each prompt was created.
    - similarity_col (str): Column name containing the similarity scores.
    - min_samples (int, optional): The number of samples in a neighborhood for a point to be considered
       as a core point in DBSCAN clustering. Default is 5.
    - perform_analysis (bool, optional): Flag to decide whether to perform pattern and burst detection.
       Default is False.

    Returns:
    - pd.DataFrame: DataFrame with added columns related to time decay in similarity, response time,
       temporal density, and (optionally) pattern and burst detection.
    """

    df = calculate_time_decay_similarity(df, prompt_time_col, similarity_col)
    df = calculate_response_time(df, prompt_time_col)
    df = calculate_temporal_density(df, prompt_time_col)

    if perform_analysis:
        df = perform_pattern_detection(df, similarity_col, min_samples)
        df = perform_burst_detection(df, prompt_time_col, similarity_col)
    return df


def perform_anomaly_detection(df: pd.DataFrame, embedding_col: str) -> pd.DataFrame:
    """
    Applies the Isolation Forest algorithm on the embeddings present in the DataFrame to identify anomalies.

    Isolation Forest is an unsupervised anomaly detection method. It works by isolating anomalies in the data based on
    their attribute values. Since anomalies are "few and different", they are easier to isolate, meaning that in a
    random recursive partition of data, they tend to be isolated closer to the root of the chain_tree.tree. Hence, the number of
    partitions required to isolate anomalies is indicative of their anomalous nature.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing embeddings, where each row corresponds to a unique data point.
    - embedding_col (str): The name of the column in the DataFrame containing the embeddings. These embeddings are
      typically high-dimensional vector representations of data points.

    Returns:
    - pd.DataFrame: The input DataFrame augmented with a new column named "anomaly_label". This column contains the
      anomaly labels assigned by the Isolation Forest algorithm. Normal points are labeled as 1 and anomalies are
      labeled as -1.

    Note:
    - The `contamination` parameter in the Isolation Forest algorithm represents the proportion of outliers in the
      data set. Adjusting this value can influence the number of anomalies detected. The default value in this function
      is set to 0.1, indicating that 10% of the data points are expected to be anomalies. Fine-tuning this parameter
      based on the specific data distribution or domain knowledge can yield more accurate results.
    """

    iso_forest = IsolationForest(contamination=0.1)
    embeddings = np.stack(df[embedding_col].to_numpy())
    labels = iso_forest.fit_predict(embeddings)
    df["anomaly_label"] = labels

    return df


def calculate_correlation_metrics(
    relationship_df: pd.DataFrame, similarity_col: str = "similarity"
) -> pd.DataFrame:
    """
    Computes correlation metrics between various columns and the specified similarity metric in the DataFrame.

    This function calculates Pearson correlation coefficients between the 'response_time' column and the similarity
    metric specified by the user (default is 'similarity'). Additionally, it computes the Pearson correlation
    coefficient between the 'temporal_density' column and the similarity metric. The results are added as new columns
    in the DataFrame.

    Pearson correlation coefficient is a measure of linear correlation between two variables. It ranges from -1 (perfect
    inverse correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.

    Parameters:
    - relationship_df (pd.DataFrame): The input DataFrame containing relationship data, including the columns
      'response_time', 'temporal_density', and the user-specified similarity metric.
    - similarity_col (str): The name of the column in the DataFrame that contains the similarity metric to be correlated
      with. Default is 'similarity'.

    Returns:
    - pd.DataFrame: The input DataFrame augmented with two new columns:

      * 'correlation_temporal_density_similarity': Contains the Pearson correlation coefficient between 'temporal_density'
        and the specified similarity metric.

    Note:
    - The correlation values added to the DataFrame will be the same for all rows since it's a single value computed
      for the entire column. Consider extracting these values separately if you don't want repeated values in the
      DataFrame.
    - Ensure that the input DataFrame contains 'response_time' and 'temporal_density' columns before calling this
      function.
    """

    relationship_df["correlation_temporal_density_similarity"] = relationship_df[
        "temporal_density"
    ].corr(relationship_df[similarity_col])
    return relationship_df


def perform_metrics(
    relationship_df: pd.DataFrame, min_samples: int = 5
) -> pd.DataFrame:
    """
    Performs clustering and anomaly detection metrics on the embeddings within the DataFrame.

    This function applies DBSCAN clustering and Isolation Forest anomaly detection on both prompt and response
    embeddings. The output DataFrame is enhanced with cluster labels (from DBSCAN) and anomaly labels (from
    Isolation Forest) for both prompt and response embeddings.

    The clustering is achieved using the `perform_clustering` function, which uses DBSCAN (Density-Based Spatial
    Clustering of Applications with Noise). This method clusters core samples that are densely packed together and
    marks as outliers the samples that lie alone in low-density regions.

    The anomaly detection is achieved using the `perform_anomaly_detection` function, which employs the Isolation Forest
    algorithm. It isolates anomalies by randomly selecting a feature and then randomly selecting a split value between
    the maximum and minimum values of the selected feature. This process creates partitions and the anomalies get
    isolated quicker.

    Parameters:
    - relationship_df (pd.DataFrame): The input DataFrame containing relationship data and embeddings columns
      named "prompt_embedding" and "response_embedding".
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point
      in the DBSCAN clustering algorithm. Default is 5.

    Returns:
    - pd.DataFrame: The input DataFrame enhanced with additional columns:
      * 'cluster_label_prompt': Contains the cluster labels for the prompt embeddings.
      * 'cluster_label_response': Contains the cluster labels for the response embeddings.
      * 'anomaly_label_prompt': Contains the anomaly labels for the prompt embeddings.
      * 'anomaly_label_response': Contains the anomaly labels for the response embeddings.

    Note:
    - Before using this function, ensure that the input DataFrame contains columns named "prompt_embedding" and
      "response_embedding" which have the embeddings data.
    - The values in 'anomaly_label' columns will be -1 for anomalies and 1 for normal data points.
    """

    for embedding_type in ["prompt_embedding", "response_embedding"]:
        relationship_df = perform_clustering(
            relationship_df, embedding_type, min_samples=min_samples
        )
        relationship_df = perform_anomaly_detection(relationship_df, embedding_type)

    return relationship_df


def add_relation_metrics(
    relationship_df: pd.DataFrame, min_samples: int
) -> pd.DataFrame:
    """
    Enriches the provided DataFrame with various metrics related to relationships.

    This function aggregates a comprehensive set of features and metrics to enhance the insights from the data.
    It applies rankings, flags, statistical analyses, temporal trends, and retrieves top K responses, among
    other operations.

    Here's a brief breakdown of the steps:
    1. Rankings and Flags: Using the `add_ranking_and_flags` function, it ranks and flags the data based on
       certain conditions.
    2. Categorical and Time Series Metrics: This uses `add_categorical_and_time_series_metrics` to derive metrics
       that provide insights into categorical variables and time series data present in the DataFrame.
    3. Top K Responses: The `add_top_k_responses` function retrieves the top K responses for each prompt.
    4. Statistical Analysis: With `add_statistical_analysis`, it calculates various statistics and analyses
       between the embeddings and similarity scores. Correlations, distances, and other metrics are determined in this step.
    5. Temporal Trends: The `add_temporal_trends` function analyzes the temporal behavior of the data, looking at how
       the similarity and other metrics trend over time. This can provide insights into periods of high activity or
       relevance.

    Parameters:
    - relationship_df (pd.DataFrame): The input DataFrame containing relationship data. It is expected to have columns
      like "prompt_coordinate", "response_coordinate", "similarity", "euclidean_distance", and "created_time".
    - min_samples (int): A parameter passed to functions that require it for algorithms like DBSCAN clustering.

    Returns:
    - pd.DataFrame: The enhanced DataFrame with a richer set of columns capturing the various metrics and analyses.

    Note:
    - Ensure that the necessary columns are present in the input DataFrame. Functions being called within might
      expect specific columns to be available.
    - This function acts as an aggregator, calling multiple other utility functions to gather a comprehensive set
      of metrics. Ensure the prerequisite functions are correctly defined and available in the environment.
    """

    relationship_df = add_ranking_and_flags(relationship_df)
    relationship_df = add_categorical_and_time_series_metrics(relationship_df)
    relationship_df = add_top_k_responses(relationship_df)
    relationship_df = add_statistical_analysis(
        relationship_df,
        "prompt_coordinate",
        "response_coordinate",
        ["similarity", "euclidean_distance"],
        "created_time",
    )
    relationship_df = add_temporal_trends(
        relationship_df, "created_time", "similarity", min_samples
    )
    return relationship_df


def calculate_metrics(relationship_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Enriches the provided DataFrame with metrics such as density, popularity, overlap, and others.

    The function operates in several stages to analyze and generate insights from the relationship data:
    1. Additional Metrics: General metrics such as deviation, spread, and other statistical measurements are derived.
    2. Density of Scores: Using a specified threshold, this stage determines the density or concentration of scores around
       this threshold value.
    3. Popularity: Calculates the popularity metric based on frequency, relevance, or other criteria defined in the
       `calculate_popularity` function.
    4. Overlap: Determines the overlap metric which can indicate shared characteristics or commonalities between different entities.

    Parameters:
    - relationship_df (pd.DataFrame): The input DataFrame containing relationship data. This DataFrame should have
      columns that the internal utility functions expect, such as similarity scores, frequency counts, or other
      relevant metrics.
    - threshold (float): A reference value used to calculate the density of scores. Scores around this threshold
      will be analyzed for their distribution or concentration.

    Returns:
    - pd.DataFrame: The enhanced DataFrame with new columns for the derived metrics.

    Note:
    - The function depends on other utility functions (`calculate_additional_metrics`, `calculate_density_of_scores`,
      `calculate_popularity`, `calculate_overlap`) to perform its tasks. Ensure these functions are available and
      correctly implemented in the environment.
    - Ensure that the input DataFrame contains the necessary columns that these utility functions expect. Otherwise,
      errors or incorrect results may arise.
    """

    relationship_df = calculate_additional_metrics(relationship_df)
    relationship_df = calculate_density_of_scores(relationship_df, threshold)
    relationship_df = calculate_popularity(relationship_df)
    relationship_df = calculate_overlap(relationship_df)
    return relationship_df


def relation_metrics(
    relationship_df: pd.DataFrame,
    threshold: float = 0.8,
    relation_clusters: int = 5,
    min_samples: int = 5,
    _perform_metrics: bool = True,
    _add_metrics: bool = False,
    _calculate_metrics: bool = False,
    path: str = None,
) -> pd.DataFrame:
    """
    Consolidates various metric calculations into a singular function and applies them based on provided flags.

    This function provides a comprehensive analysis of relationships within the DataFrame. By toggling specific flags,
    users can decide which metric analyses to perform. It begins with a cosine similarity computation and can proceed
    to various analyses like clustering, anomaly detection, statistical analyses, overlap computations, and
    correlation metrics.

    Parameters:
    - relationship_df (pd.DataFrame): The input DataFrame containing relationship data.
    - threshold (float, default=0.8): A reference value used to calculate the density of scores.
      Scores around this threshold will be analyzed for their distribution or concentration.
    - relation_clusters (int, default=5): The number of clusters to use for specific clustering algorithms.
    - min_samples (int, default=5): The number of samples in a neighborhood for clustering algorithms like DBSCAN.
    - _perform_metrics (bool, default=True): Flag to decide whether to perform clustering and anomaly detection metrics.
    - _add_metrics (bool, default=True): Flag to decide whether to add various rankings, flags, and statistical analyses.
    - _calculate_metrics (bool, default=True): Flag to decide whether to calculate metrics such as density, popularity, and overlap.
    - _calculate_correlation_metrics (bool, default=True): Flag to decide whether to compute correlation metrics between different features.

    Returns:
    - pd.DataFrame: The enhanced DataFrame with new columns based on the metrics applied.

    """

    # check if the response_coordinate and prompt_coordinate columns are present
    if "response_coordinate" or "prompt_coordinate" not in relationship_df.columns:
        columns_to_convert = ["prompt_embedding", "response_embedding"]
    else:
        columns_to_convert = [
            "prompt_embedding",
            "response_embedding",
            "prompt_coordinate",
            "response_coordinate",
        ]

    for column in columns_to_convert:
        relationship_df[column] = relationship_df[column].apply(convert_embedding)

    relationship_df = compute_cosine_similarity(relationship_df)

    if _perform_metrics:
        relationship_df = perform_metrics(relationship_df, min_samples)

    if _add_metrics:
        relationship_df = add_relation_metrics(relationship_df, relation_clusters)

    if _calculate_metrics:
        relationship_df = calculate_metrics(relationship_df, threshold)

    if path:
        if not os.path.exists(path):
            os.makedirs(path)

        # create the full path to save the file
        path = os.path.join(path, "metrics_df.csv")

        relationship_df.to_csv(path, index=False)
    return relationship_df


def create_parallel_coordinates_plot(
    df,
    color_column=None,
    labels=None,
    color_scale=px.colors.diverging.Tealrose,
    mid_point=None,
):
    """
    Generates a Parallel Coordinates Plot for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to plot.
    - color_column (str): Optional. The name of the column to use for coloring the lines in the plot.
    - labels (dict): Optional. A dictionary mapping original DataFrame column names to the labels desired in the plot.
    - color_scale (list or str): The color scale to use for the plot. Default is a diverging teal-rose color scale.
    - mid_point (float): Optional. The midpoint of the color scale if a diverging color scale is used.

    Returns:
    - plotly.graph_objs._figure.Figure: The Parallel Coordinates Plot figure object.
    """
    # Check if a color column is provided and adjust parameters accordingly

    if labels is None:
        # Custom labels for better readability (adjust according to your data)
        labels = {
            "similarity": "Similarity Score",
            "euclidean_distance": "Euclidean Distance",
            "anomaly_label": "Anomaly Score",
            "cluster_label": "Cluster",
        }

    if color_column is None:
        color_column = "cluster_label"

    if color_column and df[color_column].dtype == "object":
        color_scale = px.colors.qualitative.Plotly

    fig = px.parallel_coordinates(
        df,
        color=color_column,
        labels=labels,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=mid_point,
    )
    fig.update_layout(
        autosize=False,
        width=1800,  # Increased width
        height=900,  # Increased height
        margin=dict(t=0, b=0, l=0, r=0),
        font=dict(size=9),  # Reduced font size
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green",
    )
    fig.update_yaxes(tickangle=90)  # Make labels vertical
    return fig


def create_time_series_plot(df, x_col, y_col, title="Time Series Plot", labels=None):
    """
    Generates an interactive time series plot from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the time series data.
    - x_col (str): The column in df which contains the datetime values.
    - y_col (str or list of str): The column(s) in df which contains the values to plot.
    - title (str): Title of the plot.
    - labels (dict): Optional. A dictionary mapping DataFrame column names to labels for the plot.

    Returns:
    - plotly.graph_objs._figure.Figure: The time series plot figure object.
    """
    if labels is None:
        labels = {y_col: y_col}

    # Create the plot
    if isinstance(y_col, list):
        fig = px.line(df, x=x_col, y=y_col, title=title, labels=labels)
    else:
        fig = px.line(
            df, x=x_col, y=y_col, title=title, labels={y_col: labels.get(y_col, y_col)}
        )

    # Update layout for a cleaner look
    fig.update_layout(xaxis_title="Date", yaxis_title="Value", legend_title="Variable")
    fig.update_xaxes(rangeslider_visible=True)  # Optional: add a range slider

    return fig
