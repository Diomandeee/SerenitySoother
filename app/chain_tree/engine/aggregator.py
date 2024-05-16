from typing import List, Set, Optional
from chain_tree.utils import log_handler
import random


class ChainAggregator:
    def __init__(
        self,
        strategy=None,
        weight=0.5,
        threshold=0.5,
        probability_threshold=0.5,
        use_semantic_similarity=False,
        verbose=False,
        phrases: Optional[List[str]] = None,
        message_contains: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.use_semantic_similarity = use_semantic_similarity
        self.weight = weight
        self.threshold = threshold
        self.probability_threshold = probability_threshold
        self.verbose = verbose
        self.phrases = phrases
        self.message_contains = message_contains

    def apply_strategy(
        self, traditional_indexes: Set[int], semantic_indexes: Set[int]
    ) -> Set[int]:
        """
        This method combines traditional and semantic indexes based on a predefined strategy. The
        strategies available are:
        - 'union': Returns all unique indexes from both sets.
        - 'intersection': Returns only the indexes common to both sets.
        - 'difference': Returns indexes present in the traditional set but not in the semantic set.
        - 'symmetric_difference': Returns indexes that are in either of the sets but not both.
        - 'weighted': Combines indexes using weights. For each index, a weight is assigned based on
                    which set it belongs to. The resultant set will contain indexes that have a
                    combined weight equal to or greater than a threshold.
        - 'custom': Applies a custom combination logic defined in the 'probabilistic_combiner' function.
        - 'random': Each index from the union of the two sets has a 50% chance of being included in the
                    resultant set.
        - None: If no strategy is specified, it defaults to returning semantic indexes if
                'use_semantic_similarity' is True, else it returns traditional indexes.

        Args:
            traditional_indexes (Set[int]): A set containing traditional indexes.
            semantic_indexes (Set[int]): A set containing semantic indexes.

        Returns:
            Set[int]: A combined set of indexes based on the selected strategy.

        Note:
            - For the 'weighted' strategy, the method assumes that 'self.weight' (a float between 0 and 1)
            and 'self.threshold' are defined.
            - For the 'custom' strategy, a 'probabilistic_combiner' function must be defined elsewhere with
            appropriate logic and 'self.probability_threshold' is considered.
            - For the 'random' strategy, a random decision is made for each index, hence the results will
            vary on different calls.
            - If the strategy is not recognized or is None, and 'use_semantic_similarity' is not set, it
            defaults to returning traditional indexes.
        """

        # Union strategy: Return the union of both sets
        if self.strategy == "union":
            return traditional_indexes | semantic_indexes

        # Intersection strategy: Return only indexes that are common in both sets
        elif self.strategy == "intersection":
            return traditional_indexes & semantic_indexes

        # Difference strategy: Return indexes from the traditional set that are not in the semantic set
        elif self.strategy == "difference":
            return traditional_indexes - semantic_indexes

        # Symmetric difference strategy: Return indexes that are unique to each set
        elif self.strategy == "symmetric_difference":
            return traditional_indexes ^ semantic_indexes

        # Weighted strategy: Combine indexes based on assigned weights and a threshold
        elif self.strategy == "weighted":
            wt = self.weight
            weighted_traditional = {idx: wt for idx in traditional_indexes}
            weighted_semantic = {idx: 1 - wt for idx in semantic_indexes}

            # Calculate combined weights for all indexes
            weighted_total = {
                idx: weighted_traditional.get(idx, 0) + weighted_semantic.get(idx, 0)
                for idx in traditional_indexes | semantic_indexes
            }

            # Only include indexes that meet the weight threshold
            return {
                idx
                for idx, weight in weighted_total.items()
                if weight >= self.threshold
            }

        # Custom strategy: Apply custom logic to combine sets
        elif self.strategy == "custom":
            return self.probabilistic_combiner(
                traditional_indexes, semantic_indexes, self.probability_threshold
            )

        # Random strategy: Make a random decision for each index
        elif self.strategy == "random":
            return {
                idx
                for idx in traditional_indexes | semantic_indexes
                if random.random() >= 0.5
            }

        # If no strategy is defined, default to returning semantic or traditional indexes based on a flag
        elif self.strategy is None:
            return (
                semantic_indexes
                if self.use_semantic_similarity
                else traditional_indexes
            )

        # For unrecognized strategies, return an empty set
        return set()

    def get_indexes_by_phrases_similarity(
        self,
        titles: List[str],
        phrases: List[str],
        top_k: int,
        start: int = 0,
        semantic_search=None,
    ) -> List[int]:
        """
        Get conversation indexes whose title has semantic similarity with any of the specified phrases within a specified range.

        Args:
            phrases (List[str]): List of phrases to look for. Defaults to None.
            threshold (float): Minimum semantic similarity required to consider a match.
            top_k (int): The top K most similar titles to return.
            start (int): Start index for the range to consider.
            end (int): End index for the range to consider.

        Returns:
            List[int]: List of conversation indexes.
        """

        # If phrases is None, return an empty list
        if phrases is None:
            log_handler(
                "No phrases provided, returning an empty list of indexes.",
                verbose=self.verbose,
            )
            return []

        # Initialize an empty list to store indexes
        matched_indexes = []

        for phrase in phrases:
            # Get semantic similarity scores
            similarities = semantic_search(
                query=phrase,
                corpus=titles,
                num_results=top_k,
            )

            log_handler(
                f"Similarities for phrase '{phrase}': {similarities}",
                verbose=self.verbose,
            )

            # Find the indexes of the titles that are similar to the phrase
            indexes = [i for i, title in enumerate(titles) if title in similarities]

            # Update the indexes to match the global range
            indexes = [i + start for i in indexes]

            # Append to the matched_indexes list
            matched_indexes.extend(indexes)

        # Log the result
        log_handler(
            f"Found {len(matched_indexes)} indexes that have semantic similarity with the phrases.",
            verbose=self.verbose,
        )

        return matched_indexes

    def probabilistic_combiner(
        self,
        traditional_indexes: Set[int],
        semantic_indexes: Set[int],
        probability_threshold: float = 0.5,
    ) -> Set[int]:
        """
        Combines traditional and semantic indexes based on a probability threshold.

        Args:
            traditional_indexes (Set[int]): Set of traditional indexes.
            semantic_indexes (Set[int]): Set of semantic indexes.
            probability_threshold (float): A threshold for random selection. Defaults to 0.5.

        Returns:
            Set[int]: A set of indexes combined based on custom logic.
        """

        combined_set = set()

        for idx in traditional_indexes:
            if random.random() < probability_threshold:
                combined_set.add(idx)

        for idx in semantic_indexes:
            if random.random() >= probability_threshold:
                combined_set.add(idx)

        return combined_set
