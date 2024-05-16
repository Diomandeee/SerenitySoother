from typing import Dict, List, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from chain_tree.utils import log_handler
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


class ScenarioHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def identify(self, df: pd.DataFrame) -> List[Dict[str, int]]:
        """Identify scenarios in the dataframe."""
        pass

    @abstractmethod
    def handle(self, df: pd.DataFrame, pairs: List[Dict[str, int]]) -> None:
        """Handle identified scenarios."""
        pass


class PhaseHandler(ScenarioHandler):

    @staticmethod
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

    def identify(
        self,
        df: pd.DataFrame,
        phase: str,
        match_strategy: str = "start",
        include_more: bool = False,
        case_sensitive: bool = False,
    ) -> List[Dict[str, int]]:
        """
        Identify scenarios that match the specified phase.

        Parameters:
            df (pd.DataFrame): The data frame containing the conversation data.
            phase (str): The keyword or phrase to look for in the user's text to identify a particular phase.
            match_strategy (str, optional): The strategy used for matching phase in user texts.
                - "start": Match from the start of the user text
                - Other strategies can be defined. Default is "start".
            include_more (bool, optional): Whether to include more than just the exact match of 'phase'. Default is False.
            case_sensitive (bool, optional): Whether the match should be case-sensitive. Default is False.

        Returns:
            List[Dict[str, int]]: A list of dictionaries, each containing a pair of message IDs that match the condition.
            Each dictionary has two keys: 'user_message_id' and 'assistant_message_id'.

        Raises:
            Logs an error if an exception occurs.
        """
        try:
            if df.empty:
                return []

            # Filtering user rows based on the phase, match_strategy, include_more, and case_sensitive
            user_texts = df[df["author"] == "user"]["text"].tolist()
            filtered_user_texts = PhaseHandler.filter_by_prefix(
                user_texts,
                phase,
                include_more=include_more,
                case_sensitive=case_sensitive,
                match_strategy=match_strategy,
            )

            user_rows_with_continue = df[df["text"].isin(filtered_user_texts)]

            next_rows = df.shift(-1)
            valid_next_rows = next_rows[next_rows["author"] == "assistant"]

            continue_pairs = [
                {"user_message_id": user_id, "assistant_message_id": assistant_id}
                for user_id, assistant_id in zip(
                    user_rows_with_continue["message_id"], valid_next_rows["message_id"]
                )
            ]

            return continue_pairs

        except Exception as e:
            log_handler(f"Error identifying continue scenarios: {str(e)}")
            return []

    def handle(
        self,
        df: pd.DataFrame,
        continue_pairs: List[Dict[str, int]],
        default_response: str = "Take it to the next level!",
    ) -> None:
        """
        Handle the scenarios identified by the 'identify' method.

        Parameters:
            df (pd.DataFrame): The data frame containing the conversation data.
            continue_pairs (List[Dict[str, int]]): A list of dictionaries with pairs of message IDs.
                Each pair consists of a 'user_message_id' and an 'assistant_message_id'.
            default_response (str, optional): The default response to insert into the data frame when a match is found.
                Default is "Take it to the next level!".

        Raises:
            Logs an error if an exception occurs.
        """
        try:
            for pair in continue_pairs:
                user_message_id = pair["user_message_id"]
                matching_rows = df[df["message_id"] == user_message_id]

                if matching_rows.empty:
                    continue

                idx = matching_rows.index[0]
                df.loc[idx, "text"] = default_response

        except Exception as e:
            log_handler(f"Error handling continue scenarios: {str(e)}")


class UnwantedHandler(ScenarioHandler):
    def __init__(self, unwanted_phrases: List[str]):
        """
        Initialize the UnwantedHandler class.

        Parameters:
            unwanted_phrases (List[str]): A list of phrases that are considered unwanted in assistant responses.
        """
        super().__init__()
        self.unwanted_phrases = unwanted_phrases

    def identify(self, df: pd.DataFrame) -> List[Dict[str, int]]:
        """
        Identify scenarios where the assistant's response includes any of the unwanted phrases.

        Parameters:
            df (pd.DataFrame): The data frame containing the conversation data.

        Returns:
            List[Dict[str, int]]: A list of dictionaries, each containing a pair of message IDs where unwanted phrases were used.
            Each dictionary has two keys: 'assistant_message_id' and 'user_message_id'.

        Raises:
            Logs an error if an exception occurs.
        """
        try:
            masks = [
                df["text"].str.contains(phrase, case=False, na=False)
                for phrase in self.unwanted_phrases
            ]
            combined_mask = pd.concat(masks, axis=1).any(axis=1)

            unwanted_assistant_rows = df[combined_mask & (df["author"] == "assistant")]

            previous_rows = df.shift(1)
            valid_previous_rows = previous_rows[previous_rows["author"] == "user"]

            message_pairs = [
                {"assistant_message_id": assistant_id, "user_message_id": user_id}
                for assistant_id, user_id in zip(
                    unwanted_assistant_rows["message_id"],
                    valid_previous_rows["message_id"],
                )
            ]

            return message_pairs

        except Exception as e:
            log_handler(f"Error identifying unwanted response scenarios: {str(e)}")
            return []

    def handle(
        self,
        df: pd.DataFrame,
        message_pairs: List[Dict[str, int]],
        replacement_phrase: str = "I challenge you to make it better!",
    ) -> None:
        """
        Handle the scenarios where unwanted phrases are found in the assistant's responses.

        Parameters:
            df (pd.DataFrame): The data frame containing the conversation data.
            message_pairs (List[Dict[str, int]]): A list of dictionaries with pairs of message IDs.
                Each pair consists of an 'assistant_message_id' and a 'user_message_id'.
            replacement_phrase (str, optional): The phrase to replace the unwanted text with.
                Default is "I challenge you to make it better!".

        Raises:
            Logs an error if an exception occurs.
        """
        try:
            for pair in message_pairs:
                assistant_message_id = pair["assistant_message_id"]
                matching_rows = df[df["message_id"] == assistant_message_id]

                if matching_rows.empty:
                    continue

                idx = matching_rows.index[0]
                df.loc[idx, "text"] = replacement_phrase

        except Exception as e:
            log_handler(f"Error handling unwanted response scenarios: {str(e)}")


class ChainHandler:
    @staticmethod
    def count_starting_phrase(
        df: pd.DataFrame, phrase: str, author: str = "user"
    ) -> int:
        """
        Count the number of messages by a specific author that start with a given phrase.

        Args:
            df (pd.DataFrame): The DataFrame containing the message data.
            re
            phrase (str): The phrase to search for at the start of messages.
            author (str): The author to search messages from. Default is 'assistant'.

        Returns:
            int: The number of messages by the author that start with the given phrase.
        """
        user_messages = df[df["author"] == author]["text"]
        count = user_messages.str.startswith(phrase).sum()
        return count

    @staticmethod
    def filter_df(
        relation_df: pd.DataFrame, text: str, phrase: str = None, pattern: str = None
    ) -> pd.DataFrame:
        if pattern:
            return relation_df[relation_df[text].str.contains(pattern, regex=True)]
        else:
            return relation_df[relation_df[text].str.startswith(phrase)]

    @staticmethod
    def starting_with_phrase(
        relation_df: pd.DataFrame,
        phrases: list,
        texts: list = ["prompt", "response"],
        na_action: str = "fill",
        include_more: bool = False,
        basic: bool = False,
    ) -> pd.DataFrame:
        """
        Filter messages in a DataFrame that start with any given phrases and append the results.
        Optionally include rows that start with the phrases followed by more characters.
        The 'basic' flag allows for simple str.startswith() checks.
        """

        column_mapping = [
            "prompt_id",
            "response_id",
            "prompt",
            "response",
            "prompt_embedding",
            "response_embedding",
            "prompt_coordinate",
            "response_coordinate",
            "created_time",
            "conversation_id",
        ]

        # Handle NA values as specified
        if na_action == "drop":
            relation_df = relation_df.dropna(subset=texts)
        else:
            fill_value = "" if na_action == "fill" else na_action
            relation_df.update(relation_df[texts].fillna(fill_value))

        concatenated_df = pd.DataFrame()

        if basic:
            for text in texts:
                for phrase in phrases:
                    filtered_df = ChainHandler.filter_df(
                        relation_df, text, phrase=phrase
                    )
                    concatenated_df = pd.concat(
                        [concatenated_df, filtered_df], ignore_index=True
                    )
        else:
            pattern = r"^(" + "|".join([re.escape(phrase) for phrase in phrases]) + ")"
            if include_more:
                pattern += ".*"
            else:
                pattern += r"( |$)"

            for text in texts:
                filtered_df = ChainHandler.filter_df(relation_df, text, pattern=pattern)
                concatenated_df = pd.concat(
                    [concatenated_df, filtered_df], ignore_index=True
                )

        concatenated_df = concatenated_df.drop_duplicates().reindex(
            columns=column_mapping
        )
        return concatenated_df

    @staticmethod
    def message_contains_phrase(df: pd.DataFrame, phrase: str) -> pd.DataFrame:
        """
        Filter a DataFrame to return rows where the message text contains the given phrase.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the message data.
            phrase (str): The phrase to search for within the messages.
        
        Returns:
            pd.DataFrame: A DataFrame with rows where messages contain the given phrase.
        """
        return df[df['text'].str.contains(phrase, na=False, regex=False)]
    @staticmethod
    def get_top_ngrams(
        df: pd.DataFrame,
        n: int = 3,
        top_k: int = 10,
        text_column: str = "text",
        return_type: str = "list",
    ):
        """
        Get top K n-grams from a DataFrame's text column.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
            text_column (str): The name of the column containing the text data.
            n (int): The length of the n-grams.
            top_k (int): The number of top n-grams to return.
            return_type (str): The type of object to return. Can be 'list', 'dataframe', or 'series'.

        Returns:
            The top K n-grams and their counts, either as a list of tuples, a DataFrame, or a Series.
        """
        vec = CountVectorizer(ngram_range=(n, n)).fit(df[text_column])
        bag_of_words = vec.transform(df[text_column])
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_ngrams = words_freq[:top_k]

        if return_type == "dataframe":
            return pd.DataFrame(top_ngrams, columns=["Ngram", "Count"])
        elif return_type == "series":
            return pd.Series(dict(top_ngrams), name="Count")
        else:
            return top_ngrams

    @staticmethod
    def calculate_user_message_metrics(
        df: pd.DataFrame,
        user_author: str = "user",
        time_column: str = "create_time",
        content_column: str = "text",
        author_column: str = "author",
        keywords: list = None,
    ) -> dict:
        # Safety checks
        if (
            df is None
            or time_column not in df.columns
            or content_column not in df.columns
        ):
            return {}

        # Convert time_column to datetime if it's not
        if not np.issubdtype(df[time_column].dtype, np.datetime64):
            df[time_column] = pd.to_datetime(df[time_column], unit="s")

        # Sort by time column
        df = df.sort_values(by=[time_column])

        # Filter only user messages
        user_df = df[df[author_column] == user_author].copy()

        # Calculate number of messages sent by user per day
        user_df["date"] = user_df[time_column].dt.date
        total_days = user_df["date"].nunique()
        total_messages = len(user_df)
        messages_per_user_per_day = (
            total_messages / total_days if total_days > 0 else total_messages
        )

        # Average number of messages sent by user per hour of the day
        user_df["hour"] = user_df[time_column].dt.hour
        total_hours = user_df["hour"].nunique()
        messages_per_user_per_hour = (
            total_messages / (total_days * total_hours)
            if total_days > 0 and total_hours > 0
            else total_messages
        )

        # Average length of user's messages
        user_df["message_length"] = user_df[content_column].apply(len)
        avg_message_length_user = user_df["message_length"].mean()

        # Standard deviation of message length
        std_message_length_user = user_df["message_length"].std()

        # Most frequent starting phrase in user's messages
        def get_starting_phrase(message, n=3):
            return " ".join(message.split()[:n])

        user_df["starting_phrase"] = user_df[content_column].apply(get_starting_phrase)
        most_common_starting_phrase = (
            user_df["starting_phrase"].mode()[0]
            if not user_df["starting_phrase"].mode().empty
            else None
        )

        # Longest time period without a message from the user
        user_df["time_diff"] = user_df[time_column].diff()
        longest_silence_user = user_df["time_diff"].max()

        # Most active day
        most_active_day = (
            user_df["date"].mode()[0] if not user_df["date"].mode().empty else None
        )

        # Most active hour
        most_active_hour = (
            user_df["hour"].mode()[0] if not user_df["hour"].mode().empty else None
        )

        # Find frequency of certain key words
        keyword_frequencies = {}
        if keywords:
            keywords = [word.lower() for word in keywords]
            keyword_frequencies = {
                word: user_df[content_column]
                .apply(lambda x: x.lower().count(word))
                .sum()
                for word in keywords
            }

        # Calculate the total active hours spent by the user
        df_sorted = df.sort_values(by=time_column)
        df_sorted["time_diff"] = (
            df_sorted[time_column].diff().dt.total_seconds().fillna(0)
        )

        # Set a threshold for considering two successive messages as part of the same session (1 hour = 3600 seconds)
        session_threshold = 3600

        # Sum the intervals where the time difference is less than the threshold
        # and where at least one of the messages is from the user
        active_seconds = df_sorted[
            (df_sorted["time_diff"] < session_threshold)
            & (
                df_sorted[author_column].shift(-1).isin([user_author, "assistant"])
                | df_sorted[author_column].isin([user_author, "assistant"])
            )
        ]["time_diff"].sum()

        total_active_hours = active_seconds / 3600

        # Calculate messages per hour based on active hours
        total_messages = len(user_df)
        messages_per_active_hour = (
            total_messages / total_active_hours if total_active_hours > 0 else 0
        )

        return {
            "messages_per_user_per_day": messages_per_user_per_day,
            "messages_per_user_per_hour": messages_per_user_per_hour,
            "avg_message_length_user": avg_message_length_user,
            "std_message_length_user": std_message_length_user,
            "most_common_starting_phrase": most_common_starting_phrase,
            "longest_silence_user": longest_silence_user,
            "most_active_day": most_active_day,
            "most_active_hour": most_active_hour,
            "keyword_frequencies": keyword_frequencies,
            "total_active_hours": total_active_hours,
            "messages_per_active_hour": messages_per_active_hour,
        }

    @staticmethod
    def plot_hourly_activity_heatmap(
        df: pd.DataFrame,
        time_column: str = "create_time",
        content_column: str = "text",
        user_author: str = "user",
    ):
        """
        Generate a heat map showing user activity by hour and day of the week.

        Args:
            df (pd.DataFrame): DataFrame containing the messages.
            time_column (str): The name of the column containing the timestamp of messages.
            content_column (str): The name of the column containing the content of messages.
        """
        if (
            df is None
            or time_column not in df.columns
            or content_column not in df.columns
        ):
            raise ValueError("Invalid DataFrame or missing columns.")

        df = df[df["author"] == user_author].copy()

        # Convert to datetime format if it's not
        if not np.issubdtype(df[time_column].dtype, np.datetime64):
            df[time_column] = pd.to_datetime(
                df[time_column], unit="s"
            )  # Assuming the time is in Unix format

        try:
            pivot = (
                df.groupby([df[time_column].dt.hour, df[time_column].dt.day_name()])
                .count()[content_column]
                .unstack()
            )

            fig, ax = plt.subplots(figsize=(10, 8))

            cax = ax.matshow(pivot, cmap="YlGnBu")
            fig.colorbar(cax)

            plt.title("Hourly activity heat map (days of the week)", pad=20)
            plt.xlabel("Day of the week")
            plt.ylabel("Hour of the day")

            plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
            plt.yticks(np.arange(len(pivot.index)), pivot.index)

            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def plot_activity_over_time(
        df: pd.DataFrame,
        time_column: str = "create_time",
        content_column: str = "text",
        user_author: str = "user",
    ):
        if (
            df is None
            or time_column not in df.columns
            or content_column not in df.columns
        ):
            raise ValueError("Invalid DataFrame or missing columns.")

        df = df[df["author"] == user_author].copy()

        # Convert to datetime format if it's not
        if not np.issubdtype(df[time_column].dtype, np.datetime64):
            df[time_column] = pd.to_datetime(df[time_column], unit="s")

        plt.figure(figsize=(10, 6))
        user_activity_trend = df.resample("D", on=time_column).count()[content_column]
        plt.plot(user_activity_trend.index, user_activity_trend.values, color="skyblue")
        plt.title("User activity over time")
        plt.xlabel("Time")
        plt.ylabel("Number of messages")
        plt.show()

    @staticmethod
    def plot_activity_by_hour(
        df: pd.DataFrame,
        time_column: str = "create_time",
        content_column: str = "text",
        user_author: str = "user",
    ):
        if (
            df is None
            or time_column not in df.columns
            or content_column not in df.columns
        ):
            raise ValueError("Invalid DataFrame or missing columns.")

        df = df[df["author"] == user_author].copy()

        # Convert to datetime format if it's not
        if not np.issubdtype(df[time_column].dtype, np.datetime64):
            df[time_column] = pd.to_datetime(df[time_column], unit="s")

        user_messages_by_hour = df.groupby(df[time_column].dt.hour)[
            content_column
        ].count()
        user_messages_by_hour = user_messages_by_hour / user_messages_by_hour.max()

        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111, polar=True)
        ax.fill(np.linspace(0, 2 * np.pi, 24), user_messages_by_hour, color="skyblue")
        ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 24))
        ax.set_xticklabels(range(24))
        ax.set_theta_offset(np.pi / 2 - np.pi / 24)
        plt.title("User Activity by Hour of Day")
        plt.show()

    @staticmethod
    def display_top_words_by_cluster(
        df: pd.DataFrame,
        message_column="text",
        n_clusters=10,
        top_n=10,
        user_author="user",
    ):
        """
        Display the top words for each cluster based on their TF-IDF scores.

        Args:
            df (pd.DataFrame): DataFrame containing the messages.
            message_column (str): The name of the column containing the messages.
            n_clusters (int): Number of clusters used in KMeans.
            top_n (int): Number of top words to display for each cluster.
        """
        if df is None or message_column not in df.columns:
            raise ValueError("Invalid DataFrame or missing columns.")

        # Filter messages only from the user of interest
        user_messages = df[df["author"] == user_author][message_column].values

        if len(user_messages) == 0:
            print(f"No messages from {user_author} found.")
            return

        # Compute tf-idf scores
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_scores = vectorizer.fit_transform(user_messages)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans_labels = kmeans.fit_predict(tfidf_scores)

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Calculate and display top words for each cluster
        for i in range(n_clusters):
            # Extract tf-idf scores of messages in this cluster
            messages_in_cluster = tfidf_scores[kmeans_labels == i]

            # If no messages in this cluster, continue
            if messages_in_cluster.shape[0] == 0:
                print(f"Cluster {i} has no messages.")
                continue

            # Compute average tf-idf scores across all messages in this cluster
            avg_tfidf_scores = np.mean(
                messages_in_cluster, axis=0
            ).A1  # Convert to 1D array

            # Find top N words in this cluster
            top_indices = avg_tfidf_scores.argsort()[-top_n:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]

            print(f"Cluster {i}: {top_words}")

        return kmeans_labels
