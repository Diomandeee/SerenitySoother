from typing import List, Tuple, Optional, Union, Any, Dict, Callable
from sklearn.metrics.pairwise import cosine_similarity
from chain_tree.type import ElementType
from collections import OrderedDict
from abc import abstractmethod, ABC
from functools import wraps
from loguru import logger
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import tiktoken
import warnings
import logging
import os


with warnings.catch_warnings():
    from numba.core.errors import NumbaWarning

    warnings.simplefilter("ignore", category=NumbaWarning)
    from umap import UMAP


tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # The encoding scheme to use for tokenization


# Constants
CHUNK_SIZE = 200  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = int(
    os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", 128)
)  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text


def get_text_chunks(
    text: str,
    n_chunks: Optional[int] = None,
    chunk_token_size: Optional[int] = None,
    use_advanced_tokenization: bool = True,
    split_by_line: bool = False,
) -> List[str]:
    """
    Splits the text into chunks, either by simple splitting or tokenization.

    Args:
        text: The text to split into chunks.
        n_chunks: The number of chunks to split the text into (used only if use_advanced_tokenization is False).
        chunk_token_size: The target size of each chunk in tokens (used only if use_advanced_tokenization is True).
        use_advanced_tokenization: Whether to use advanced tokenization or simple splitting.
        split_by_line: Whether to split by line if the number of paragraphs is less than n_chunks.

    Returns:
        A list of text chunks.
    """

    # Validate input type
    if not isinstance(text, str):
        raise ValueError("The input text must be a string.")
    # Check for empty or whitespace-only strings
    if not text or text.isspace():
        return []

    if not use_advanced_tokenization:
        text_chunks = text.split("\n\n")

        if split_by_line:
            if n_chunks is not None and len(text_chunks) < n_chunks:
                text_chunks = text.split("\n")
        else:
            if n_chunks is not None:
                text_chunks = text_chunks[:n_chunks]

        return text_chunks

    else:
        # Perform advanced tokenization
        tokens = tokenizer.encode(text, disallowed_special=())
        chunks = []
        chunk_size = chunk_token_size or CHUNK_SIZE
        num_chunks = 0

        while tokens and num_chunks < MAX_NUM_CHUNKS:
            chunk = tokens[:chunk_size]
            chunk_text = tokenizer.decode(chunk)

            last_punctuation = max(
                chunk_text.rfind("."),
                chunk_text.rfind("?"),
                chunk_text.rfind("!"),
                chunk_text.rfind("\n"),
            )

            if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
                chunk_text = chunk_text[: last_punctuation + 1]

            chunk_text_to_append = chunk_text.replace("\n", " ").strip()

            if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
                chunks.append(chunk_text_to_append)

            tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())) :]
            num_chunks += 1

        if tokens:
            remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
            if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
                chunks.append(remaining_text)

        return chunks


        


def apply_umap(
    combined_features: np.ndarray,
    n_neighbors: int,
    n_components: int,
    verbose: bool = True,
    parametric: bool = False,
    create_encoder: Optional[Callable] = None,
    create_decoder: Optional[Callable] = None,
    unique: bool = False,
    ParametricUMAP: Optional[Callable] = None,
) -> np.ndarray:
    """
    Apply UMAP (Uniform Manifold Approximation and Projection) on the given features.

    Parameters:
        combined_features (np.ndarray): The feature matrix.
        n_neighbors (int): The number of neighbors to consider for each point.
        n_components (int): The number of components (dimensions) for the output.
        verbose (bool, optional): Whether to display progress. Default is True.

    Returns:
        np.ndarray: The UMAP-embedded feature matrix.
    """

    # Check if n_neighbors is larger than the dataset size
    if n_neighbors > len(combined_features):
        n_neighbors = len(combined_features)
        print(
            f"Warning: n_neighbors was larger than the dataset size; truncating to {n_neighbors}"
        )

    elif n_neighbors < 2:
        n_neighbors = 2

    if parametric:
        # Create an encoder model
        input_dim = combined_features.shape[1]
        output_dim = n_components

        if encoder is None:
            encoder = create_encoder(input_dim, output_dim)

        if decoder is None:
            decoder = create_decoder(output_dim, input_dim)

        umap_model = ParametricUMAP(
            n_neighbors=int(n_neighbors),
            n_components=n_components,
            n_epochs=6000,
            min_dist=1,
            low_memory=False,
            verbose=verbose,
            metric="cosine",
            encoder=encoder,
            decoder=decoder,
        )

        umap_embedding = umap_model.fit_transform(combined_features)

        return umap_embedding, umap_model

    else:
        #
        # Use the regular UMAP without encoder and decoder
        umap_model = UMAP(
            n_neighbors=int(n_neighbors) if n_neighbors is not None else 15,
            n_components=n_components,
            n_epochs=6000,
            min_dist=1,
            low_memory=False,
            learning_rate=0.5,
            verbose=verbose,
            metric="cosine",
            random_state=42,
            unique=unique,
        )

        # check if we need to reshape the input
        if len(combined_features.shape) == 1:
            combined_features = combined_features.reshape(-1, 1)

        umap_embedding = umap_model.fit_transform(combined_features)

        return umap_embedding, None


def process_message_dict(message_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract the message text and ID from the message dictionary.
    """
    message_texts = []
    message_ids = []
    for message_id, message in message_dict.items():
        if isinstance(message, str):
            message_texts.append(message)
            message_ids.append(message_id)
        elif message.message and message.message.author.role != "system":
            message_texts.append(message.message.content.parts[0])
            message_ids.append(message.id)
    return message_texts, message_ids


def generate_message_to_embedding_dict(
    message_ids: List[str], embeddings: List[np.array]
) -> Dict[str, np.array]:
    """
    Generate a dictionary mapping message IDs to embeddings.
    """
    return dict(zip(message_ids, embeddings))


def compute_neighbors(estimator, message_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    For each message, determine the number of neighbors.
    """
    n_neighbors_dict = {}
    for message_id in message_dict:
        n_neighbors_dict[message_id] = estimator.determine_n_neighbors(message_id)
    return n_neighbors_dict


def update_message_dict_with_embeddings(
    message_dict: Dict[str, Any],
    embeddings: Any,
) -> None:
    """
    Update the 'embedding' or 'umap_embedding' fields of each Message object
    in the provided message_dict based on the embedding_type.

    Parameters:
        message_dict: A dictionary containing Message objects.
        embeddings: A dictionary mapping message IDs to their embeddings.

    Returns:
        None

    """

    for message_id, embedding in embeddings.items():

        if message_id in message_dict:
            message_dict[message_id].message.embedding = embedding

        else:
            raise ValueError(
                f"Message ID {message_id} not found in message_dict. "
                f"Please check that the message_dict contains the correct message IDs."
            )


def generate_reduced_embeddings(
    embeddings: Union[Dict[str, np.array], List[np.array]],
    options: dict,
) -> Dict[str, np.array]:
    """
    Generate reduced embeddings for the messages in the conversation tree using UMAP.
    """
    if isinstance(embeddings, list):
        embeddings = dict(enumerate(embeddings))

    # Extract the embeddings and message IDs from the message dictionary
    message_ids = list(embeddings.keys())
    embeddings = list(embeddings.values())

    # Generate the reduced embeddings
    reduced_embeddings = apply_umap(
        np.array(embeddings),
        n_neighbors=options["n_neighbors"],
        n_components=options["n_components"],
        verbose=options["verbose"],
        parametric=options["parametric"],
    )
    # Create a dictionary mapping message IDs to reduced embeddings
    message_embeddings = generate_message_to_embedding_dict(
        message_ids, reduced_embeddings
    )
    return message_embeddings


def set_coordinates(main_df: pd.DataFrame, umap_array: np.array) -> pd.DataFrame:
    """
    Set the UMAP coordinates in the main DataFrame.

    Parameters:
    - main_df (pd.DataFrame): The main DataFrame.
    - umap_array (np.array): UMAP coordinates as a numpy array.

    Returns:
    - pd.DataFrame: Updated DataFrame with UMAP coordinates.
    """
    main_df["x"] = umap_array[:, 0]
    main_df["y"] = umap_array[:, 1]
    main_df["z"] = umap_array[:, 2]
    return main_df


def set_umap_embeddings(main_df: pd.DataFrame, umap_embeddings: List) -> pd.DataFrame:
    """
    Set the UMAP coordinates in the main DataFrame.

    Parameters:
    - main_df (pd.DataFrame): The main DataFrame.
    - umap_array (np.array): UMAP coordinates as a numpy array.

    Returns:
    - pd.DataFrame: Updated DataFrame with coordinates.
    """
    main_df["umap_embeddings"] = umap_embeddings
    return main_df


def drop_columns(main_df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from the main DataFrame.

    Parameters:
    - main_df (pd.DataFrame): The main DataFrame.
    - columns_to_drop (List[str]): List of column names to drop.

    Returns:
    - pd.DataFrame: Updated DataFrame without the specified columns.
    """
    main_df.drop(columns_to_drop, axis=1, inplace=True)
    return main_df


def set_id_column(main_df: pd.DataFrame, custom_id_column: str = None) -> pd.DataFrame:
    """
    Determine and set the appropriate ID column in the main DataFrame.

    Parameters:
    - main_df (pd.DataFrame): The main DataFrame.
    - custom_id_column (str, optional): Custom ID column to set, if any. Defaults to None.

    Returns:
    - pd.DataFrame: Updated DataFrame with the ID column set.
    """
    if custom_id_column:
        if custom_id_column in main_df.columns:
            main_df.rename(columns={custom_id_column: "id"}, inplace=True)
        else:
            raise ValueError(
                f"The custom ID column '{custom_id_column}' does not exist in the DataFrame."
            )
    else:
        # If no custom ID is provided, then use "doc_id" if it exists; otherwise, use "id"
        if "doc_id" in main_df.columns:
            main_df.rename(columns={"doc_id": "id"}, inplace=True)
        elif "id" not in main_df.columns:
            # Create a default 'id' column if none exists
            main_df["id"] = range(1, len(main_df) + 1)

    return main_df


def calculate_scores(
    query_vector: np.ndarray, corpus_vectors: np.ndarray, indices: List[int]
) -> np.ndarray:
    """
    Calculate cosine similarity scores between the query and corpus vectors.

    Parameters:
        query_vector: The encoded query vector.
        corpus_vectors: The encoded corpus vectors.
        indices: List of indices for the most relevant similar keywords in the corpus.

    Returns:
        An array of cosine similarity scores.
    """
    most_relevant_similar_vectors = corpus_vectors[indices]
    return cosine_similarity(query_vector, most_relevant_similar_vectors).flatten()


def filter_corpus(corpus: List[str], query: str):
    return [kw for kw in corpus if kw.strip().lower() != query.strip().lower()]


class BaseEmbedding(ABC):
    def __init__(
        self,
        api_key: str = None,
        batch_size: int = 128,
        reduce_dimensions=True,
        n_components=3,
        weights=None,
        model_name="all-mpnet-base-v2",
        verbose: bool = True,
        parametric: bool = False,
        engine: str = "text-embedding-3-small",
        sentence_transformer_model: object = None,
    ):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        self.batch_size = batch_size
        self.show_progress_bar = True
        self.engine = engine
        self.max_retries = 3
        self.api_key = api_key
        self._model_name = model_name
        self.verbose = verbose
        self.parametric = parametric
        self.weights = weights if weights is not None else {}
        if sentence_transformer_model is None:
            self._model = None
        else:
            self._model = sentence_transformer_model
        self.default_options = {
            "n_components": n_components,
            "reduce_dimensions": reduce_dimensions,
            "n_neighbors": None,
            "verbose": verbose,
            "parametric": parametric,
        }
        self._semantic_vectors = []
        self.keywords = []

    @abstractmethod
    def _process_batch(self, items: List[str]) -> List[List[float]]:
        pass

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        return self._process_batch(text)

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def generate_umap_projections(
        self,
        embeddings: List[np.array],
        n_neighbors: int = 15,
        n_components: int = 5,
        verbose: bool = True,
        parametric: bool = False,
        create_encoder: Optional[Callable] = None,
        create_decoder: Optional[Callable] = None,
        ParametricUMAP: Optional[Callable] = None,
    ) -> np.array:
        """
        Generate UMAP projections for the embeddings.
        """
        umap_projections, _ = apply_umap(
            np.array(embeddings),
            n_neighbors=n_neighbors,
            n_components=n_components,
            verbose=verbose,
            parametric=parametric,
            create_encoder=create_encoder,
            create_decoder=create_decoder,
            ParametricUMAP=ParametricUMAP,
        )
        return umap_projections

    def generate_message_embeddings(
        self,
        grid: Any,  # Add appropriate type annotation
        message_dict: Dict[str, Any],
        embeddings: Dict[str, np.array],
        message_ids: List[str],
        message_embeddings: Dict[str, np.array],
        options: Optional[dict] = None,
    ) -> Dict[str, Union[np.array, Tuple[str, str]]]:
        """
        Generate semantic embeddings for the messages in the conversation tree
        and clusters them.

        Parameters:
        - grid (Any): The grid for computing neighbors. Type should be specified.
        - message_dict (Dict[str, Any]): Dictionary containing message data.
        - embeddings (Dict[str, np.array]): Original embeddings for the messages.
        - message_ids (List[str]): List of message IDs.
        - message_embeddings (Dict[str, np.array]): Dictionary to store generated message embeddings.
        - options (Optional[dict], default=None): Configuration options for embedding generation.
          If None, default options will be used.

        Returns:
        - Dict[str, Union[np.array, Tuple[str, str]]]: A dictionary mapping message IDs to embeddings and cluster labels.
        """

        if options is not None:
            self.default_options.update(options)

        if len(message_dict) > 1:
            n_neighbors_dict = compute_neighbors(grid, message_dict)

            n_neighbors_mean = np.mean(list(n_neighbors_dict.values()))

            self.default_options["n_neighbors"] = n_neighbors_mean

            return message_embeddings

        else:
            return message_embeddings, 0

    def compute_similarity_scores_dataframe(
        self,
        df: pd.DataFrame,
    ) -> List[float]:
        """
        Gets the similarity scores between a prompt and a list of responses using the provided model.

        Args:
            model (SentenceTransformer): The Sentence Transformer model.
            prompt (str): The prompt.
            responses (List[str]): The list of responses.

        Returns:
            List[float]: A list of similarity scores.
        """

        try:

            prompt_embedding = self.embed(df["prompt"])
            response_embeddings = self.embed(df["response"])
            similarity_scores = cosine_similarity(
                prompt_embedding, response_embeddings
            ).tolist()
            return similarity_scores
        except Exception as e:
            raise ValueError("Failed to get similarity scores.")

    def embed_text_batch(
        self,
        prompts: List[str],
        responses: List[str],
        encode_corpus=None,
        compute_similarity_scores=False,
        convert_to_numpy=False,
        save_to_file=False,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Converts a batch of textual messages into their corresponding numerical encodings using a text embedding model.

        Parameters:
        - prompts (List[str]): A compilation of primary messages or questions which commonly originate from the user
        or another source. They serve as the starting point or trigger for any conversation or interaction.
        - responses (List[str]): A compilation of follow-up messages or answers to the aforementioned prompts. These
        often emanate from systems like chat assistants, addressing the content or query raised in the prompts.

        Returns:
        - Tuple: A dual-list tuple, with the first list containing the vector encodings for the prompts and the second
        list containing the vector encodings for the responses.

        """
        # First, find all the unique texts across both prompts and responses
        unique_texts = list(OrderedDict.fromkeys(prompts + responses).keys())

        # Encode these unique texts
        all_encoded_texts = (
            self.embed(unique_texts)
            if encode_corpus is None
            else encode_corpus(unique_texts)
        )
        # Create a mapping from text to encoding
        encoding_lookup = dict(zip(unique_texts, all_encoded_texts))

        # Now populate the return lists, re-using the unique encodings where possible
        encoded_prompts = [list(map(float, encoding_lookup[p])) for p in prompts]
        encoded_responses = [list(map(float, encoding_lookup[r])) for r in responses]

        if compute_similarity_scores:
            similarity_scores = cosine_similarity(encoded_prompts, encoded_responses)
            return encoded_prompts, encoded_responses, similarity_scores

        if convert_to_numpy:
            return np.array(encoded_prompts), np.array(encoded_responses)

        if save_to_file:
            np.save("encoded_prompts.npy", np.array(encoded_prompts))
            np.save("encoded_responses.npy", np.array(encoded_responses))

        return encoded_prompts, encoded_responses

    def compute_similar_keywords_per_keyword(
        self, keywords: List[str], embeddings: List[List[float]], num_keywords: int
    ) -> List[List[str]]:
        # Compute the similarity matrix for all vectors
        similarity_matrix = cosine_similarity(embeddings).reshape(1, -1)

        # Sort the indices based on similarity score for each row
        sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]

        # Pick top num_keywords for each keyword
        top_indices = sorted_indices[:, :num_keywords]

        # Create list of most similar keywords for each keyword
        similar_keywords_list = [
            [keywords[i] for i in row_indices] for row_indices in top_indices
        ]

        return similar_keywords_list

    def get_most_relevant_similar_keywords(
        self,
        filtered_corpus: List[str],
        most_relevant_keyword: str,
        corpus_vectors: np.ndarray,
        num_results: int,
    ) -> List[str]:
        """
        Get the most relevant similar keywords to a given keyword.

        Parameters:
            filtered_corpus: List of keywords in corpus, filtered to exclude the query.
            most_relevant_keyword: The most relevant keyword to the query.
            corpus_vectors: The vectors for each keyword in the corpus.
            num_results: Number of results to return.

        Returns:
            A list of the most relevant similar keywords.
        """
        similar_keywords = self.compute_similar_keywords_per_keyword(
            filtered_corpus, corpus_vectors, num_keywords=num_results
        )
        most_relevant_idx = filtered_corpus.index(most_relevant_keyword)
        most_relevant_similar_keywords = similar_keywords[most_relevant_idx]
        return most_relevant_similar_keywords

    def compute_similar_keywords_query(
        self,
        keywords: List[str],
        query_vector: List[float],
        use_argmax: bool,
        query: str,
    ) -> List[Tuple[str, float]]:
        """
        Compute similarity scores for keywords against a query vector.

        Args:
            keywords (List[str]): List of keywords to compute similarity for.
            query_vector (List[float]): Vector representing the query.
            use_argmax (bool): Whether to use argmax for similarity scores.

        Returns:
            List[Tuple[str, float]]: List of tuples containing keyword and similarity score.
        """
        # Remove the query keyword from the list of keywords
        filtered_keywords = [
            keyword.strip()
            for keyword in keywords
            if keyword.strip().lower() != query.strip().lower()
        ]

        # Use the model to encode all keywords in a single batch
        keyword_vectors = self.embed(filtered_keywords)

        # Compute cosine similarity in a vectorized manner
        similarity_scores = cosine_similarity([query_vector], keyword_vectors)[0]

        # Create a list of (keyword, similarity) tuples
        filtered_keywords_with_scores = list(zip(filtered_keywords, similarity_scores))

        # Sort the list based on similarity scores
        if use_argmax:
            filtered_keywords_with_scores = sorted(
                filtered_keywords_with_scores, key=lambda x: x[1], reverse=True
            )[:1]
        else:
            filtered_keywords_with_scores = sorted(
                filtered_keywords_with_scores, key=lambda x: x[0]
            )

        return filtered_keywords_with_scores

    def get_most_relevant(
        self,
        filtered_corpus: List[str],
        query_vector: np.ndarray,
        use_argmax: bool,
        query: str,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Gets the most relevant keyword from the corpus based on the query vector.

        Parameters:
            filtered_corpus: List of keywords in corpus, filtered to exclude the query.
            query_vector: The query vector.
            model: The SentenceTransformer model.
            use_argmax: Whether to use argmax for finding the most similar keyword.
            query: The query keyword.

        Returns:
            A tuple containing the most relevant keyword and its similarity score.
            Returns (None, None) if no relevant keyword is found.
        """
        most_relevant = self.compute_similar_keywords_query(
            filtered_corpus,
            query_vector.flatten(),
            use_argmax=use_argmax,
            query=query,
        )
        return most_relevant[0] if most_relevant else (None, None)

    def get_similar_keywords(
        self,
        query: str,
        corpus: List[str],
        num_results: int = 10,
        use_argmax: bool = False,
    ) -> List[str]:
        """
        Get the most relevant similar keywords to a given keyword.

        Parameters:
            query: The query keyword.
            corpus: The list of corpus strings.
            num_results: Number of results to return.
            use_argmax: Whether to use argmax for finding the most similar keyword.
            model: The SentenceTransformer model used for encoding.

        Returns:
            A list of the most relevant similar keywords.
        """
        query_vector = self.embed_query(query)
        corpus_vectors = self.embed(corpus)

        most_relevant_keyword, _ = self.get_most_relevant(
            corpus, query_vector, use_argmax, query
        )

        if most_relevant_keyword is None:
            return []

        most_relevant_similar_keywords = self.get_most_relevant_similar_keywords(
            corpus, most_relevant_keyword, corpus_vectors, num_results
        )

        return most_relevant_similar_keywords

    def get_query_and_corpus_vectors(
        self, query: str, corpus: List[str], model=None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate vectors for the query and the corpus.

        Parameters:
            model: The SentenceTransformer model used for encoding.
            query: The query string.
            corpus: The list of corpus strings.

        Returns:
            A tuple containing the query vector, corpus vectors, and the filtered corpus.
        """
        query_vector = self.embed_query(query)
        filtered_corpus = filter_corpus(corpus, query)
        corpus_vectors = self.embed(corpus)
        return query_vector, corpus_vectors, filtered_corpus

    def get_relevant_keywords(
        self,
        query: str,
        corpus: List[str],
        num_results: int = 10,
        model=None,
        use_argmax: bool = False,
    ) -> List[str]:

        query_vector, corpus_vectors, filtered_corpus = (
            self.get_query_and_corpus_vectors(model, query, corpus)
        )

        most_relevant_keyword, _ = self.get_most_relevant(
            filtered_corpus, query_vector, model, use_argmax, query
        )

        if most_relevant_keyword is None:
            return []

        most_relevant_similar_keywords = self.get_most_relevant_similar_keywords(
            filtered_corpus, most_relevant_keyword, corpus_vectors, num_results
        )

        return most_relevant_similar_keywords

    def compute_keywords(
        self,
        keywords: List[str],
        embeddings: Optional[List[List[float]]] = None,
        per_keyword: bool = True,
        num_keywords: Optional[int] = None,
        return_with_scores: bool = False,
    ) -> Union[List[List[str]], List[List[Tuple[str, float]]]]:

        if embeddings is None:
            embeddings = self.embed(keywords)

        similarity_matrix = cosine_similarity(embeddings)

        if per_keyword:
            similar_keywords = []
            for i, keyword in enumerate(keywords):
                similar_keyword_scores = [
                    (keywords[j], similarity_matrix[i, j]) for j in range(len(keywords))
                ]
                sorted_similar_keywords = sorted(
                    similar_keyword_scores, key=lambda x: x[1], reverse=True
                )

                if return_with_scores:
                    similar_keywords.append(
                        sorted_similar_keywords[1 : num_keywords + 1]
                    )
                else:
                    similar_keywords.append(
                        [x[0] for x in sorted_similar_keywords[1 : num_keywords + 1]]
                    )

            return similar_keywords

        else:
            similar_keywords_global = [
                [
                    (
                        keywords[idx]
                        if not return_with_scores
                        else (keywords[idx], similarity_matrix[i, idx])
                    )
                    for idx in np.argsort(similarity_matrix[i])[::-1][
                        1 : num_keywords + 1
                    ]
                ]
                for i in range(len(keywords))
            ]

            return similar_keywords_global

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        num_results: int = 10,
        return_with_scores: bool = False,
        compute_keywords: bool = False,
        model=None,
    ) -> Union[List[Tuple[str, float]], List[str], List[List[Tuple[str, float]]]]:
        query_vector = self.embed_query(query)
        corpus_vectors = self.embed(corpus)

        similarity_scores = calculate_scores(
            query_vector, corpus_vectors, indices=range(len(corpus))
        )
        results_with_scores = [
            (corpus[i], score) for i, score in enumerate(similarity_scores)
        ]
        sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)

        if compute_keywords:
            similar_keywords = self.compute_keywords(
                corpus,
                model=model,
                per_keyword=True,
                num_keywords=num_results,
                return_with_scores=return_with_scores,
            )
            return similar_keywords

        if return_with_scores:
            return sorted_results[:num_results]
        else:
            return [result[0] for result in sorted_results[:num_results]]

    def compute_embeddings(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        element_type: ElementType = ElementType.Segment,
        separate_columns: bool = False,
    ) -> pd.DataFrame:
        """
        Compute embeddings for each step.

        Args:
            df (pd.DataFrame): The DataFrame containing the steps.
            element_type (ElementType): The type of the elements (e.g., ElementType.STEP, ElementType.CHAPTER, ElementType.PAGE).
            separate_columns (bool, optional): Whether to separate the columns (Prefix and Steps)
                during the embedding process. Defaults to True.

        Returns:
            pd.DataFrame: The DataFrame with added columns for embeddings.
        """
        try:
            # Prepare the data for embedding
            if separate_columns:
                # Separate the columns (Prefix and Elements) and drop NaN values
                element_columns = [
                    col
                    for col in dataframe.columns
                    if col.startswith(element_type.value)
                ]
                all_elements = pd.concat(
                    [dataframe[col] for col in element_columns], ignore_index=True
                )
                prefix = pd.Series(dtype=str)
            else:
                # Combine all columns (Prefix and Elements) into a single column and drop NaN values
                all_elements = dataframe.stack().dropna()
                prefix = pd.Series(
                    dtype=str
                )  # Explicitly specify the dtype of the empty Series

            # Compute embeddings for all elements
            embeddings = self.embed(
                all_elements.tolist() + prefix.tolist(),
            )

            embedding_dict = {i: embeddings[i] for i in range(len(embeddings))}

            # Add embeddings to the DataFrame for each element
            df_copy = (
                dataframe.copy()
            )  # Create a copy of the DataFrame to avoid potential warnings
            for col in df_copy.columns:
                if col.startswith(element_type.value):
                    element_len = len(df_copy[col])
                    embedding_col = f"{col} embedding"
                    if separate_columns:
                        # Separate columns: Add embeddings for each element separately
                        df_copy[embedding_col] = (
                            pd.Series(embedding_dict).loc[: element_len - 1].tolist()
                        )
                    else:
                        # Combined columns: Add embeddings for all elements in a single column
                        df_copy[embedding_col] = pd.Series(
                            list(embedding_dict.values())
                        )

                    embedding_dict = {
                        k - element_len: v
                        for k, v in embedding_dict.items()
                        if k >= element_len
                    }

            return df_copy

        except Exception as e:
            print(f"Error computing embeddings: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of errors

    def _compute_semantic_vectors(
        self, keywords: List[str]
    ) -> List[Tuple[str, List[float]]]:
        """
        Compute semantic vectors for a list of keywords.

        Parameters:
        - keywords (List[str]): A list of keywords for which semantic vectors need to be computed.

        Returns:
        - List[Tuple[str, List[float]]]: A list of tuples where each tuple contains a keyword and its corresponding semantic vector.
        """
        try:
            # Compute semantic vectors
            semantic_vectors = [(keyword, self.embed(keyword)) for keyword in keywords]

            return semantic_vectors

        except ValueError as e:
            logging.error(f"An error occurred while computing semantic vectors: {e}")
            return []

    def compute_similarity_scores(
        self,
        query: Union[str, List[str]],
        keywords: List[str],
        batches: bool = False,
        top_k: int = 1,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Compute similarity scores between a query or a list of queries and a list of keywords.

        Parameters:
        - query (Union[str, List[str]]): A single query or a list of queries for which similarity scores need to be computed.
        - keywords (List[str]): A list of keywords for which similarity scores need to be computed.
        - batches (bool, optional): If True, computes similarity scores in batches. Defaults to False.
        - top_k (int, optional): The top K similarity scores to consider. Defaults to 1.

        Returns:
        - Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]: A list of tuples where each tuple contains a keyword and its corresponding similarity score with the query.
        """
        try:
            # Compute semantic vectors
            semantic_vectors = self._compute_semantic_vectors(keywords)

            # Compute similarity scores
            if batches:
                # Compute similarity scores in batches
                similarity_scores = []
                for i in range(0, len(query), self.batch_size):
                    batch_query = query[i : i + self.batch_size]
                    embeddings = self.embed(batch_query)

                    # Calculate cosine similarity for each query against all keywords
                    batch_similarity_scores = [
                        [
                            (keywords[j], np.dot(query_vector, keyword_vector[0]))
                            for j, (keyword, keyword_vector) in enumerate(
                                semantic_vectors
                            )
                        ]
                        for query_vector in embeddings
                    ]
                    similarity_scores.extend(batch_similarity_scores)
            else:
                # Compute similarity scores for a single query
                embeddings = self.embed(query)

                # Calculate cosine similarity for the query against all keywords
                similarity_scores = [
                    [
                        (keyword, np.dot(query_vector, keyword_vector[0]))
                        for keyword, keyword_vector in semantic_vectors
                    ]
                    for query_vector in embeddings
                ]

            # Sort and return top K similarity scores
            sorted_similarity_scores = [
                sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
                for scores in similarity_scores
            ]

            return sorted_similarity_scores

        except ValueError as e:
            logging.error(f"An error occurred while computing similarity scores: {e}")
            return []

    def compute_similar_keywords(
        self,
        keywords: List[str],
        num_keywords: int = 10,
        top_k: int = 3,
        use_argmax: bool = False,
        per_keyword: bool = False,
        query: Optional[str] = None,
        group_terms: Optional[Callable] = None,
    ) -> List[str]:
        """
        Compute a list of similar keywords using a specified language model.

        Parameters:
            keywords (List[str]): A list of keywords for which similarity needs to be computed.
            num_keywords (int, optional): The maximum number of similar keywords to return. Defaults to 10.
            use_argmax (bool, optional): Determines the grouping approach. If True, uses argmax approach.
                If False, uses a broader approach selecting multiple similar keywords per group.
                Defaults to True.
            per_keyword (bool, optional): If True, returns a list of similar keywords for each keyword.
                If False, returns a flattened list of top similar keywords across all keywords.
                Defaults to True.
            query (str, optional): If provided, computes similarity scores between the query and each keyword.

        Returns:
            List[str]: A list of similar keywords or similarity scores, based on the provided parameters.

        Note:
            When `per_keyword` is True, the returned list of similar keywords will be in the format:
                - If query is None: [[sim_keyword_1, sim_keyword_2, ...], [sim_keyword_1, sim_keyword_2, ...], ...]
                - If query is provided: [(sim_keyword, similarity_score), ...]

            When `per_keyword` is False, the returned list will be a flattened list of top similar keywords.
        """
        try:
            # Compute semantic vectors
            semantic_vectors = self._compute_semantic_vectors(keywords)
            embeddings = self.embed(keywords)

            # Compute similarity scores
            if query is not None:
                similarity_scores = self.compute_similarity_scores(
                    query=query, keywords=keywords, top_k=top_k
                )

            # Compute similar keywords
            else:
                if per_keyword:

                    similarity_scores = [
                        [
                            (keyword, np.dot(query_vector, keyword_vector[0]))
                            for keyword, keyword_vector in semantic_vectors
                        ]
                        for query_vector in embeddings
                    ]

                else:
                    if group_terms is None:
                        similarity_scores = self.compute_similarity_scores(
                            query=keywords, keywords=keywords, top_k=top_k
                        )
                    else:
                        clusters = group_terms(semantic_vectors, use_argmax=use_argmax)
                        sorted_clusters = sorted(
                            clusters.values(), key=len, reverse=True
                        )[:num_keywords]
                        similarity_scores = [
                            term for cluster in sorted_clusters for term, _ in cluster
                        ]

            return similarity_scores

        except ValueError as e:
            logging.error(f"An error occurred while computing similar keywords: {e}")
            return []

    def predict_semantic_similarity(
        self, query: str, keywords: List[str], threshold: float = 0.5, top_k: int = None
    ) -> List[str]:
        """
        This method takes a single query and a list of keywords, and returns a list of keywords that are semantically similar
        to the query based on a given threshold. Essentially, it predicts which keywords are relevant to a given query
        by comparing their semantic meanings.

        Parameters:
        - query (str): A text query for which we want to find semantically similar keywords.
        - keywords (List[str]): A list of keywords to compare with the query for semantic similarity.
        - threshold (float, optional): The threshold for semantic similarity score. Keywords that have a similarity
          score above or equal to this threshold are considered similar. Default is 0.5.

        Returns:
        - List[str]: A list of keywords that are predicted to be semantically similar to the query.

        Exceptions:
        - Logs a ValueError and returns an empty list if an error occurs during processing.

        Example:
        >>> predict_semantic_similarity("apple", ["fruit", "company", "color"])
        ["fruit"]

        """

        try:
            # Compute similarity scores using the modified compute_similarity_scores
            similarity_scores = self.compute_similarity_scores(
                query=query, keywords=keywords, top_k=top_k
            )

            # Create a list of predicted keywords based on the similarity threshold
            predicted_keywords = [
                keyword
                for score_list in similarity_scores
                for keyword, similarity in score_list
                if similarity >= threshold
            ]

            return predicted_keywords

        except ValueError as e:
            logging.error(
                f"An error occurred while predicting semantic similarity: {e}"
            )
            return []

    def predict_semantic_similarity_batch(
        self,
        queries: List[str],
        keywords: List[str],
        threshold: float = 0.5,
        top_k: int = None,
        with_scores: bool = True,
    ) -> List[Union[List[str], List[Tuple[str, float]]]]:
        """
        This method takes a list of queries and a list of keywords, and for each query, it returns a list of keywords
        that are semantically similar based on a given threshold.

        Parameters:
        - queries (List[str]): A list of text queries for which we want to find semantically similar keywords.
        - keywords (List[str]): A list of keywords to compare with each query for semantic similarity.
        - threshold (float, optional): The threshold for semantic similarity score. Keywords that have a similarity
        score above or equal to this threshold are considered similar. Default is 0.5.
        - top_k (int, optional): The top K similar keywords to consider. Defaults to None (i.e., all similar keywords).
        - with_scores (bool, optional): If True, includes similarity scores in the results. If False, returns only keywords.
        Default is True.

        Returns:
        - List[Union[List[str], List[Tuple[str, float]]]]: A list of lists where each inner list contains keywords that are
        predicted to be semantically similar to the corresponding query in the 'queries' input list. If with_scores is True,
        each inner list will contain tuples (keyword, similarity_score).

        Exceptions:
        - Logs a ValueError and returns an empty list if an error occurs during processing.

        Example:
        >>> predict_semantic_similarity_batch(["apple", "orange"], ["fruit", "company", "color"])
        [["fruit"], [("fruit", 0.8), ("color", 0.6)]]

        """
        try:
            # Compute similarity scores using the modified compute_similarity_scores
            similarity_scores = self.compute_similarity_scores(
                query=queries, keywords=keywords, batches=True, top_k=top_k
            )

            # Create a list of predicted keywords for each query based on the similarity threshold
            predicted_keywords = []
            for score_list in similarity_scores:
                if with_scores:
                    keywords_with_scores = [
                        (keyword, similarity)
                        for keyword, similarity in score_list
                        if similarity >= threshold
                    ]
                    predicted_keywords.append(keywords_with_scores)
                else:
                    keywords_above_threshold = [
                        keyword
                        for keyword, similarity in score_list
                        if similarity >= threshold
                    ]
                    predicted_keywords.append(keywords_above_threshold)

            return predicted_keywords

        except ValueError as e:
            logging.error(
                f"An error occurred while predicting semantic similarity: {e}"
            )
            return []

    def semantic_similarity_interface(
        self,
        query: Union[str, List[str]] = None,
        keywords: List[str] = None,
        top_k: int = 1,
        threshold: float = 0.5,
        num_keywords: int = 10,
        use_argmax: bool = False,
        per_keyword: bool = False,
        with_scores: bool = True,  # Optional parameter with default value True
    ) -> Union[List[str], List[List[str]], List[List[Tuple[str, float]]]]:
        """
        An interface for various semantic similarity methods.

        Parameters:
        - query (Union[str, List[str]]): A single query or a list of queries to compare with the keywords.
        - keywords (List[str]): A list of keywords to compare with the query/queries for similarity.
        - threshold (float, optional): The similarity threshold for predicting semantically similar keywords. Defaults to 0.5.
        - top_k (int, optional): The top K similarity scores to consider when predicting or computing. Defaults to 1.
        - num_keywords (int, optional): The maximum number of similar keywords to return when using the compute_similar_keywords method. Defaults to 10.
        - use_argmax (bool, optional): Determines the grouping approach for compute_similar_keywords. Defaults to False.
        - per_keyword (bool, optional): If True, returns a list of similar keywords for each keyword when using compute_similar_keywords. Defaults to False.
        - with_scores (bool, optional): If True, includes similarity scores in the results. If False, returns only keywords.
        Default is True.

        Returns:
        - Union[List[str], List[List[str]], List[List[Tuple[str, float]]]]:
            - When predicting, returns a list of keywords similar to the query or a list of lists for each query when in batch mode.
            - When computing, returns a list of tuples where each tuple contains a keyword and its similarity score with the query.

        Example:
        - Single query, predicting similar keywords:
        >>> semantic_similarity_interface("apple", ["fruit", "company", "color"], threshold=0.5)
        ["fruit"]

        - Batch of queries, predicting similar keywords:
        >>> semantic_similarity_interface(["apple", "orange"], ["fruit", "company", "color"], threshold=0.5)
        [["fruit"], ["fruit"]]

        - Single query, computing similarity scores (assumes compute_similarity_scores function exists):
        >>> semantic_similarity_interface("apple", ["fruit", "company", "color"], threshold=0.5, top_k=2)
        [("fruit", 0.8), ("company", 0.2)]

        - Batch of queries, computing similarity scores (assumes compute_similarity_scores function exists):
        >>> semantic_similarity_interface(["apple", "orange"], ["fruit", "company", "color"], threshold=0.5, top_k=2)
        [[("fruit", 0.8), ("company", 0.2)], [("fruit", 0.9), ("color", 0.7)]]

        - Compute similar keywords based on given list of keywords (assumes compute_similar_keywords function exists):
        >>> semantic_similarity_interface(None, ["fruit", "company"], num_keywords=2, use_argmax=True, per_keyword=True)
        [["fruit", "company"], ["company", "fruit"]]

        """

        try:
            is_batch = isinstance(query, list)  # Determine if it's a batch operation

            # If keywords are provided, then compute semantic similarity, otherwise use compute_similar_keywords
            compute_similar_kw = bool(keywords)

            if is_batch:
                # Handle batch operations
                if compute_similar_kw:
                    return self.predict_semantic_similarity_batch(
                        query, keywords, threshold, top_k=top_k, with_scores=with_scores
                    )
                else:
                    # Batch operation using compute_similarity_scores
                    return self.compute_similarity_scores(
                        query, keywords, batches=True, top_k=top_k
                    )
            else:
                # Handle single query operations
                if compute_similar_kw:
                    if query:  # make sure query is not None or empty
                        return self.predict_semantic_similarity(
                            query, keywords, threshold, top_k=top_k
                        )
                    else:
                        # Here, query is None or empty, so we fall back to compute_similar_keywords
                        return self.compute_similar_keywords(
                            keywords,
                            num_keywords=num_keywords,
                            top_k=top_k,
                            use_argmax=use_argmax,
                            per_keyword=per_keyword,
                            query=query,
                        )
                elif (
                    num_keywords and use_argmax is not None and per_keyword is not None
                ):
                    # Single query using compute_similar_keywords
                    return self.compute_similar_keywords(
                        keywords,
                        num_keywords=num_keywords,
                        top_k=top_k,
                        use_argmax=use_argmax,
                        per_keyword=per_keyword,
                        query=query,
                    )
                else:
                    # Single query using compute_similarity_scores
                    return self.compute_similarity_scores(query, keywords, top_k=top_k)

        except ValueError as e:
            logging.error(f"An error occurred while computing semantic similarity: {e}")
            return []


class OpenAIEmbedding(BaseEmbedding):

    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)

    def _set_openai_api_key(self):
        if not self.api_key:
            raise ValueError("Groq API key not set")

    @staticmethod
    def _retry_decorator():
        def decorator(func):
            @wraps(func)
            def wrapped_func(*args, **kwargs):

                from openai import (
                    APITimeoutError,
                    APIError,
                    APIConnectionError,
                    RateLimitError,
                )

                min_seconds = 1
                max_seconds = 60

                retry_decorator = retry(
                    reraise=True,
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(
                        multiplier=1, min=min_seconds, max=max_seconds
                    ),
                    retry=(
                        retry_if_exception_type(APITimeoutError)
                        | retry_if_exception_type(APIError)
                        | retry_if_exception_type(APIConnectionError)
                        | retry_if_exception_type(RateLimitError)
                    ),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                )
                return retry_decorator(func)(*args, **kwargs)

            return wrapped_func

        return decorator

    @staticmethod
    def _handle_openai_response(response) -> List[List[float]]:
        return [item.embedding for item in response.data]

    @_retry_decorator()
    def _create_embeddings(self, items: str) -> List[float]:
        self._set_openai_api_key()
        response = self.client.embeddings.create(input=items, model=self.engine)
        return self._handle_openai_response(response)

    def _process_batch(self, items: List[str]) -> List[List[float]]:
        # Create batches from the full list of items
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        embeddings = []
        # Process each batch and update the progress bar
        for batch in tqdm(batches, desc="Processing Embeddings", leave=False):
            embeddings.extend(self._create_embeddings(batch))

        return embeddings

    def embed(self, items: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(items, str):
            items = [items]
        return self._process_batch(items)

    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)[0]

    def __call__(self, items: List[str]) -> List[List[float]]:
        return self.embed(items)

    def prepare_texts_for_embedding(
        self,
        input_data: Union[List[str], Dict[str, Any]],
        n_chunks: Optional[int] = None,
        chunk_token_size: Optional[int] = None,
        use_advanced_tokenization: bool = True,
    ) -> Union[Tuple[List[str], List[str]], List[str]]:
        """
        Process the input data (either list of texts or message dictionary) and optionally chunk the message texts.

        Parameters:
        - input_data (Union[List[str], Dict[str, Any]]): Either a list of message texts or a dictionary containing message data.
        - n_chunks (Optional[int], default=None): Number of chunks to split the message text into. If None, no splitting is performed.
        - chunk_token_size (Optional[int], default=None): The maximum token size for each chunk. Only used if n_chunks is specified.
        - use_advanced_tokenization (bool, default=True): Whether to use advanced tokenization techniques.

        Returns:
        - Union[Tuple[List[str], List[str]], List[str]]:
          1. If input_data is a dictionary:
             - List of message IDs.
             - Flattened list of message text chunks or original message texts.
          2. If input_data is a list:
             - Flattened list of message text chunks or original message texts.
        """

        # Determine if input_data is a list or dictionary
        if isinstance(input_data, list):
            message_texts = input_data
            message_ids = None
        elif isinstance(input_data, dict):
            # Extract the message text and ID from the message dictionary
            message_texts, message_ids = process_message_dict(input_data)
        else:
            raise ValueError(
                "Input data must be either a list of strings or a message dictionary."
            )

        # Optionally chunk the message texts
        message_texts_chunks = [
            get_text_chunks(
                text,
                n_chunks=n_chunks,
                chunk_token_size=chunk_token_size,
                use_advanced_tokenization=use_advanced_tokenization,
            )
            for text in message_texts
        ]

        # Flatten the list of lists into a single list
        message_texts_flattened = [
            chunk for chunks in message_texts_chunks for chunk in chunks
        ]

        # Return values based on the type of input_data
        if message_ids is not None:
            return message_ids, message_texts_flattened
        else:
            return message_texts_flattened

    def _generate_embeddings(
        self,
        message_dict: Dict[str, Any],
        message_texts_flattened: List[str],
        message_ids: List[str],
    ) -> Tuple[Dict[str, np.array], List[str], Dict[str, np.array]]:
        """
        Generate basic semantic embeddings for the provided message texts.

        Parameters:
        - message_dict (Dict[str, Any]): Original dictionary containing message data.
        - message_texts_flattened (List[str]): Flattened list of message text chunks or original message texts.
        - message_ids (List[str]): List of message IDs.

        Returns:
        - Tuple[Dict[str, np.array], List[str], Dict[str, np.array]]:
        1. Dictionary mapping message IDs to their embeddings.
        2. List of message IDs.
        3. Dictionary mapping message IDs to their generated embeddings.
        """

        embeddings = self.embed(message_texts_flattened)

        # Generate a dictionary mapping message IDs to embeddings
        message_embeddings = generate_message_to_embedding_dict(message_ids, embeddings)

        # Update the original message dictionary with these embeddings
        update_message_dict_with_embeddings(message_dict, message_embeddings)

        return embeddings, message_ids, message_embeddings

    def generate_embeddings(
        self,
        input_data: Union[List[str], Dict[str, Any]],
        n_chunks: Optional[int] = None,
        chunk_token_size: Optional[int] = None,
        use_advanced_tokenization: bool = True,
    ) -> Tuple[Dict[str, np.array], List[str], Dict[str, np.array]]:
        """
        Generate basic semantic embeddings for the provided message texts.

        Parameters:
        - input_data (Union[List[str], Dict[str, Any]]): Either a list of message texts or a dictionary containing message data.
        - n_chunks (Optional[int], default=None): Number of chunks to split the message text into. If None, no splitting is performed.
        - chunk_token_size (Optional[int], default=None): The maximum token size for each chunk. Only used if n_chunks is specified.
        - use_advanced_tokenization (bool, default=True): Whether to use advanced tokenization techniques.

        Returns:
        - Tuple[Dict[str, np.array], List[str], Dict[str, np.array]]:
        1. Dictionary mapping message IDs to their embeddings.
        2. List of message IDs.
        3. Dictionary mapping message IDs to their generated embeddings.
        """

        # Prepare the message texts for embedding generation
        message_ids, message_texts_flattened = self.prepare_texts_for_embedding(
            input_data, n_chunks, chunk_token_size, use_advanced_tokenization
        )

        # Generate embeddings for the message texts
        embeddings, message_ids, message_embeddings = self._generate_embeddings(
            input_data, message_texts_flattened, message_ids
        )

        return embeddings, message_ids, message_embeddings

    def get_global_embedding(
        self,
        main_df: pd.DataFrame,
        use_embeddings: bool = False,
        text_column: str = "text",
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        """
        Retrieve or calculate global embeddings for the text.

        Parameters:
        - main_df (pd.DataFrame): The main DataFrame.
        - use_embeddings (bool): Whether to use existing embeddings.
        - text_column (str): The column name where text data is stored.

        Returns:
        - np.ndarray: The embeddings in NumPy array format.
        """

        if use_embeddings:
            embeddings = np.array(main_df["embedding"].tolist())
            return embeddings
        else:
            embeddings = self.embed(main_df[text_column].tolist())

            embeddings = np.array(embeddings)
            return embeddings

    def get_embeddings(
        self,
        main_df: pd.DataFrame,
        use_embeddings: bool = False,
        text_column: str = "text",
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Retrieve or calculate embeddings for the text.

        Parameters:
        - main_df (pd.DataFrame): The main DataFrame.
        - use_embeddings (bool): Whether to use existing embeddings.
        - text_column (str): The column name where text data is stored.

        Returns:
        - pd.DataFrame: The DataFrame with added columns for embeddings.
        """

        if use_embeddings:
            return main_df
        else:
            embeddings = self.embed(main_df[text_column].tolist())

            main_df["embedding"] = embeddings
            return main_df
