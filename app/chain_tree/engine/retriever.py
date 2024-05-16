from typing import List, Tuple, Union
from chain_tree.engine.embedder import OpenAIEmbedding, BaseEmbedding, cosine_similarity
from chain_tree.engine.manipulator import DataManipulator
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import logging
import json
import re


class DataRetriever:
    def __init__(
        self, manipulator: DataManipulator, embedder: BaseEmbedding = OpenAIEmbedding()
    ):
        """Initializes the DataRetriever with a given DataHelper."""
        self.manipulator = manipulator
        self.embedder = embedder
        self.data = manipulator.finalize()
        self.prompt_col = manipulator.prompt_col
        self.response_col = manipulator.response_col
        self.prompts = None
        self.responses = None
        self._validate_columns()
        self._validate_data()

    def _validate_data(self) -> None:
        """Validates if the data is not empty."""
        if self.data.empty:
            logging.error("Data is empty.")
            raise ValueError("Data is empty.")

        self.prompts = self.data[self.prompt_col].tolist()
        self.responses = self.data[self.response_col].tolist()

    def _validate_columns(self) -> None:
        """Validates if the specified columns exist in the dataset."""
        for col in [self.prompt_col, self.response_col]:
            if col not in self.data.columns:
                logging.error(f"Column '{col}' not found in data.")
                raise ValueError(f"Column '{col}' not found in data.")

    def _validate_pair_type(self, pair_type: str) -> None:
        """Validates if the provided pair_type is valid."""
        valid_pair_types = ["both", self.prompt_col, self.response_col]
        if pair_type not in valid_pair_types:
            logging.error(
                f"Invalid pair_type. Choose from {', '.join(valid_pair_types)}"
            )
            raise ValueError(
                f"Invalid pair_type. Choose from {', '.join(valid_pair_types)}"
            )

    def _get_data_by_pair_type(
        self, data_subset: pd.DataFrame, pair_type: str
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Returns data based on the pair_type from a given data subset."""
        self._validate_pair_type(pair_type)

        if pair_type == "both":
            return list(
                zip(
                    data_subset[self.prompt_col].tolist(),
                    data_subset[self.response_col].tolist(),
                )
            )
        return data_subset[pair_type].tolist()

    def get_steps(self, step_num: int) -> List[str]:
        data = self.manipulator.dataset_loader.filter_responses()
        steps = []
        for _, row in data.iterrows():
            steps.append(str(row[f"Step {step_num}"]).strip())
        return steps

    def get_examples(
        self, pair_type: str = "both"
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Gets examples of the specified type from the data."""
        return self._get_data_by_pair_type(self.data, pair_type)

    def get_random_examples(
        self, n: int, pair_type: str = "both"
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Gets n random examples of the specified type from the data."""
        return self._get_data_by_pair_type(self.data.sample(n), pair_type)

    def get_first_n_examples(
        self, n: int, pair_type: str = "both"
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Gets the first n examples of the specified type from the data."""
        return self._get_data_by_pair_type(self.data.head(n), pair_type)

    def search_examples(
        self, keywords: Union[str, List[str]], pair_type: str = "both"
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Searches examples containing the keyword(s) of the specified type from the data."""
        if isinstance(keywords, str):
            keywords = [keywords]

        mask = self.data[self.prompt_col].str.contains(
            "|".join(map(re.escape, keywords))
        ) | self.data[self.response_col].str.contains(
            "|".join(map(re.escape, keywords))
        )

        filtered_data = self.data[mask]
        return self._get_data_by_pair_type(filtered_data, pair_type)

    def filter_data(self, word: str, pair_type: str = None) -> List[str]:
        """Returns the data that contain a specific word"""
        if pair_type not in [self.prompt_col, self.response_col]:
            raise ValueError(
                f"Invalid pair_type. Choose from '{self.prompt_col}', '{self.response_col}'"
            )
        data_column = (
            self.prompt_col if pair_type == self.prompt_col else self.response_col
        )
        data = self.data[data_column].tolist()
        return [text for text in data if word in text]

    def _convert_embeddings_to_sparse_matrix(self, embeddings: List[str]) -> csr_matrix:
        """Converts a list of stringified embeddings into a sparse matrix."""
        numerical_embeddings = [
            json.loads(emb) if isinstance(emb, str) else emb for emb in embeddings
        ]
        return csr_matrix(numerical_embeddings)

    def create_sparse_matrix(self, pair_type: str = "both") -> csr_matrix:
        """Creates a sparse matrix of prompts and responses using precomputed embeddings."""

        # Assuming 'prompt_embedding' and 'response_embedding' are columns with stringified JSON embeddings
        if pair_type == "both":
            prompt_embeddings = self.data["prompt_embedding"].apply(json.loads).tolist()
            response_embeddings = (
                self.data["response_embedding"].apply(json.loads).tolist()
            )
            combined_embeddings = prompt_embeddings + response_embeddings
            return self._convert_embeddings_to_sparse_matrix(combined_embeddings)
        elif pair_type == "prompt":
            prompt_embeddings = self.data["prompt_embedding"].apply(json.loads).tolist()
            return self._convert_embeddings_to_sparse_matrix(prompt_embeddings)
        elif pair_type == "response":
            response_embeddings = (
                self.data["response_embedding"].apply(json.loads).tolist()
            )
            return self._convert_embeddings_to_sparse_matrix(response_embeddings)
        else:
            raise ValueError(
                "Inva lid pair_type. Choose from 'prompt', 'response', 'both'"
            )

    def _fetch_data_for_index(self, index: int) -> Tuple:
        """Fetch data for a given index from the DataFrame."""
        data_row = self.data.iloc[index]
        return (
            data_row["prompt"],
            data_row["response"],
            data_row["created_time"],
            data_row["prompt_id"],
            data_row["response_id"],
            data_row["prompt_coordinate"],
            data_row["response_coordinate"],
            data_row["prompt_embedding"],
            data_row["response_embedding"],
        )

    def find_similar(
        self, text: str, step_num: int, top_n: int = 1, pair_type: str = "response"
    ) -> Union[str, List[str]]:
        """Finds the top_n most similar data to the input text based on TF-IDF cosine similarity with weighted sum and propagation"""

        if pair_type not in ["prompt", "response"]:
            raise ValueError("Invalid pair_type. Choose from 'prompt', 'response'")

        data = self.get_steps(step_num) if pair_type == "response" else self.prompts

        if not data:
            raise ValueError("No data available for the specified step number")

        vectorizer = self.embedder.embed(data + [text])

        cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer).flatten()[
            :-1
        ]

        if not any(cosine_similarities):
            raise ValueError("No similarity found for the input text")

        # Compute the weighted sum of previous steps
        weighted_sum = np.cumsum(cosine_similarities) + 1

        # Calculate the weighted cosine similarity with propagation
        weighted_cosine_similarities = cosine_similarities / weighted_sum

        # Propagate the weighted cosine similarities to subsequent steps
        for i in range(step_num + 1, len(data)):
            if weighted_sum[i - 1] != 0:
                weighted_cosine_similarities[i] *= weighted_cosine_similarities[i - 1]

        # Select the top_n most similar data based on weighted cosine similarities
        weighted_similar_indices = weighted_cosine_similarities.argsort()[::-1][:top_n]
        weighted_similar_data = [data[i] for i in weighted_similar_indices]

        # Calculate the scores for the top_n most similar data
        similarity_scores = weighted_cosine_similarities[weighted_similar_indices]

        # Normalize the similarity scores
        max_score = np.max(similarity_scores)
        if max_score == 0:  # Handle division by zero
            normalized_scores = np.zeros_like(similarity_scores)
        else:
            normalized_scores = similarity_scores / max_score

        if top_n == 1:
            return {"data": weighted_similar_data[0], "score": normalized_scores[0]}

        return [
            {"data": d, "score": s}
            for d, s in zip(weighted_similar_data, normalized_scores)
        ]

    def get_similar_examples(
        self,
        prompt: str,
        n: int = 10,
        pair_type: str = "response",
        return_scores: bool = True,
        return_df: bool = True,
        return_pair_type: bool = False,
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, str, float]]]:
        self._validate_pair_type(pair_type)

        # Create a sparse matrix of prompts and/or responses
        sparse_matrix = self.create_sparse_matrix(pair_type)

        # Encode the prompt
        encoded_prompt = self.embedder.embed(text=prompt)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(encoded_prompt, sparse_matrix).flatten()

        # Get the n most similar examples
        top_n_indices = np.argsort(similarity_scores)[-n:][::-1]

        # Extract the most similar prompts and responses (and scores if required)
        results = []
        for i in top_n_indices:
            index = i % len(self.data)
            result = {
                "prompt": self.data[self.prompt_col][index],
                "response": self.data[self.response_col][index],
            }
            optional_columns = [
                "created_time",
                "prompt_id",
                "response_id",
                "prompt_coordinate",
                "response_coordinate",
                "prompt_embedding",
                "response_embedding",
            ]
            for col in optional_columns:
                if col in self.data:
                    result[col] = self.data[col][index]

            if return_scores:
                result["similarity"] = similarity_scores[i]
            results.append(result)

        # Create a DataFrame from the results if return_df is True
        if return_df:
            results_df = pd.DataFrame(results)
            return results_df

        if return_pair_type:
            # if return_pair_type is True, return a list of tuples with (prompt, response)
            return [(result["prompt"], result["response"]) for result in results]

        # Convert dictionary to tuple if not returning as DataFrame
        return [tuple(v for v in result.values()) for result in results]
