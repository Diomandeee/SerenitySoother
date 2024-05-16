from typing import List, Optional, Tuple, Dict, Callable
from chain_tree.models import (
    ChainMap,
    ChainTree,
    Chain,
)
from datetime import datetime


class MessageFilter:
    """
    A class used to filter messages based on various criteria such as date range,
    presence of keywords, case sensitivity, message length, and specific content within the message.

    Attributes:
        date_range (Optional[Tuple[str, str]]): A tuple containing start and end dates as strings in the format 'mm/dd/yyyy'.
        keyword_filter (Optional[List[str]]): A list of keywords to filter the messages.
        case_sensitive (bool): A flag to determine if the keyword search should be case sensitive.
        message_length (Optional[int]): The maximum length of messages to include in the filter.
        message_contains (Optional[List[str]]): A list of substrings that the message must contain.

    Methods:
        filter_messages(messages, keywords): Filters a list of messages based on the instance's criteria.
    """

    def __init__(
        self,
        date_range: Optional[Tuple[str, str]] = None,
        keyword_filter: Optional[List[str]] = None,
        case_sensitive: bool = False,
        message_length: Optional[int] = None,
        message_contains: Optional[List[str]] = None,
    ):
        """
        Initializes the MessageFilter with the specified filtering criteria.
        """
        try:
            self.date_range = (
                self._convert_date_range(date_range) if date_range else None
            )
        except ValueError as e:
            raise ValueError(f"Invalid date format in date_range: {e}")

        self.keyword_filter = keyword_filter
        self.case_sensitive = case_sensitive
        self.message_length = message_length
        self.message_contains = message_contains

    @staticmethod
    def _convert_date_range(date_range: Tuple[str, str]) -> Tuple[float, float]:
        """
        Converts a tuple of date strings to a tuple of timestamp floats.
        """
        try:
            start_date_str, end_date_str = date_range
            start_date = datetime.strptime(start_date_str, "%m/%d/%Y")
            end_date = datetime.strptime(end_date_str, "%m/%d/%Y")
            return (
                start_date.timestamp(),
                end_date.timestamp(),
            )
        except ValueError as e:
            raise ValueError(f"Invalid date string in date_range: {e}")

    def filter_messages(
        self, messages: List[ChainMap], keywords: Optional[List[str]] = None
    ) -> List[ChainMap]:
        """
        Filters a list of ChainMap messages based on the initialized criteria of the MessageFilter instance.
        """
        filtered_messages = []
        for message in messages:
            try:
                if not self._within_date_range(message):
                    continue
                if not self._within_message_contains(
                    message, keywords or self.keyword_filter, self.case_sensitive
                ):
                    continue
                if not self._within_message_length(message):
                    continue
                filtered_messages.append(message)
            except Exception as e:
                print(f"Error processing message {message}: {e}")
        return filtered_messages

    def _within_message_contains(
        self,
        mapping: ChainMap,
        keywords: Optional[List[str]],
        case_sensitive: bool = False,
    ) -> bool:
        """
        Checks if the message content contains any of the specified keywords.
        """
        try:
            if not keywords:
                return True
            if (
                mapping.message is None
                or mapping.message.content is None
                or mapping.message.content.text is None
            ):
                return False  # Or handle this in some other way

            text_to_search = mapping.message.content.text
            if not case_sensitive:
                text_to_search = text_to_search.lower()
                keywords = [keyword.lower() for keyword in keywords]

            return any(keyword in text_to_search for keyword in keywords)
        except Exception as e:
            print(f"Error checking message content: {e}")
            return False

    def _within_date_range(self, mapping: ChainMap) -> bool:
        """
        Checks if the message creation time falls within the specified date range.
        """
        try:
            if not self.date_range:
                return True
            if mapping.message is None or mapping.message.create_time is None:
                return False  # Or handle this in some other way
            start_date, end_date = self.date_range
            return start_date <= mapping.message.create_time <= end_date
        except Exception as e:
            print(f"Error checking date range: {e}")
            return False

    def _within_message_length(self, mapping: ChainMap) -> bool:
        """
        Checks if the message content is within the specified length.
        """
        try:
            if not self.message_length:
                return True
            if (
                mapping.message is None
                or mapping.message.content is None
                or mapping.message.content.text is None
            ):
                return False  # Or handle this in some other way
            return len(mapping.message.content.text) <= self.message_length
        except Exception as e:
            print(f"Error checking message length: {e}")
            return False


class DepthFilter:
    """
    This class is responsible for filtering based on the depth of a chain_tree.tree.

    Attributes:
        depth_range (Optional[Tuple[int, int]]): A tuple indicating the minimum and maximum depth range.
    """

    def __init__(self, depth_range: Optional[Tuple[int, int]]):
        """
        Initializes the DepthFilter with an optional depth range.

        Parameters:
            depth_range (Optional[Tuple[int, int]]): The range of depth to be used for filtering.
        """
        # if the end is None, it means there is no upper limit
        if depth_range:
            min_depth, max_depth = depth_range
            if max_depth is None:
                max_depth = float("inf")
            if min_depth > max_depth:
                raise ValueError("Minimum depth cannot be greater than maximum")

        self.depth_range = (min_depth, max_depth) if depth_range else None

    def filter_tree_by_depth(self, depth: int) -> bool:
        """
        Determines if the given tree depth is within the depth range.

        Parameters:
            depth (int): The depth of the tree to be checked.

        Returns:
            bool: True if the tree depth is within the range, False otherwise.
        """
        tree_depth = depth

        if self.depth_range:
            min_depth, max_depth = self.depth_range
            return self._within_depth_range(tree_depth, min_depth, max_depth)

        return True

    def _within_depth_range(
        self, tree_depth: int, min_depth: int, max_depth: int
    ) -> bool:
        """
        Checks if the tree depth is within the specified minimum and maximum depth.

        Parameters:
            tree_depth (int): The depth of the chain_tree.tree.
            min_depth (int): The minimum depth to filter by.
            max_depth (int): The maximum depth to filter by.

        Returns:
            bool: True if the depth is within range, False otherwise.
        """
        return min_depth <= tree_depth <= max_depth


class TreeFilter:
    """
    This class is responsible for filtering based on the message count within a conversation chain_tree.tree.

    Attributes:
        message_range (Optional[Tuple[int, int]]): A tuple indicating the minimum and maximum message count range.
    """

    def __init__(self, message_range: Optional[Tuple[int, int]]):
        """
        Initializes the TreeFilter with an optional message range.

        Parameters:
            message_range (Optional[Tuple[int, int]]): The range of message count to be used for filtering.
        """

        self.message_range = message_range

    def filter_tree_by_message_count(self, conversation_tree) -> bool:
        """
        Determines if the given conversation tree has a message count within the message range.

        Parameters:
            conversation_tree (ChainTree): The conversation tree to be checked.

        Returns:
            bool: True if the message count is within the range, False otherwise.
        """
        if self.message_range:
            min_messages, max_messages = self.message_range
            return self._within_message_range(
                conversation_tree, min_messages, max_messages
            )

        return True

    def _within_message_range(
        self, conversation_tree: ChainTree, min_messages: int, max_messages: int
    ) -> bool:
        """
        Checks if the conversation tree's message count is within the specified minimum and maximum message range.

        Parameters:
            conversation_tree (ChainTree): The conversation tree to check.
            min_messages (int): The minimum message count to filter by.
            max_messages (int): The maximum message count to filter by.

        Returns:
            bool: True if the message count is within range, False otherwise.
        """
        num_messages = len(conversation_tree.mapping.values())
        return min_messages <= num_messages <= max_messages


class RangeFilter:
    def __init__(
        self,
        title_range: Optional[Tuple[int, int]] = None,
        index_range: Optional[Tuple[int, int]] = None,
        custom_filter: Optional[Dict[str, Tuple[int, int]]] = None,
        content_filter: Optional[Callable[[str], bool]] = None,
    ):

        self.title_range = title_range
        self.index_range = index_range
        self.custom_filter = custom_filter if custom_filter else {}
        self.content_filter = content_filter

    def filter_by_title(self, conversation: ChainTree) -> bool:
        """
        Filters a conversation based on the title range.

        Parameters:
            conversation (ChainTree): The conversation to be filtered.

        Returns:
            bool: True if the conversation's title is within the range, False otherwise.
        """
        if self.title_range is None:
            return True

        try:
            title = int(conversation.title)
            return self.title_range[0] <= title <= self.title_range[1]
        except ValueError:
            return False

    def filter_by_index(self, idx: int, total: int) -> bool:
        """
        Filters based on the index range.

        Parameters:
            idx (int): The current index to be checked.
            total (int): The total count of items.

        Returns:
            bool: True if the index is within the range, False otherwise.
        """
        if self.index_range is None:
            return True

        start, end = self.index_range
        if end is None:
            end = total

        return start <= idx <= end

    def filter_custom(self, conversation) -> bool:
        """
        Filters a conversation based on custom attribute ranges.

        Parameters:
            conversation (ChainTree): The conversation to be filtered.

        Returns:
            bool: True if all custom attributes are within their respective ranges, False otherwise.
        """
        for attribute, (lower, upper) in self.custom_filter.items():
            attr_value = getattr(conversation, attribute, None)
            if attr_value is None or not (lower <= attr_value <= upper):
                return False
        return True

    def filter_content(self, conversation: ChainTree):
        """
        Filters a conversation based on the content filter.

        Parameters:
            conversation (ChainTree): The conversation to be filtered.

        Returns:
            bool: True if the conversation's content passes the filter, False otherwise.
        """
        if self.content_filter is None:
            return True

        return self.content_filter(conversation.mapping)

    def filter_chain(self, chain: Chain) -> bool:
        """
        Filters a chain based on the content filter.

        Parameters:
            chain (Chain): The chain to be filtered.

        Returns:
            bool: True if the chain's content passes the filter, False otherwise.
        """
        if self.content_filter is None:
            return True

        return self.content_filter(chain.content.text)


class ChainFilter:
    def __init__(
        self,
        message_range: Optional[Tuple[int, int]] = None,
        depth_range: Optional[Tuple[int, int]] = None,
        date_range: Optional[Tuple[float, float]] = None,
        keyword_filter: Optional[List[str]] = None,
        range_filter: Optional[RangeFilter] = None,
        custom_filter: Optional[Callable[[ChainTree], bool]] = None,
    ):
        """
        Initializes the ChainFilter instance with various filtering options.

        Args:
            message_range (Optional[Tuple[int, int]]): Message range filter. Default is None.
            depth_range (Optional[Tuple[int, int]]): Depth range filter. Default is None.
            date_range (Optional[Tuple[float, float]]): Date range filter. Default is None.
            keyword_filter (Optional[List[str]]): Keyword filter. Default is None.
            range_filter (Optional[RangeFilter]): Range filter instance. Default is None.
            custom_filter (Optional[Callable]): Custom filter function. Default is None.
        """
        self.tree_filter = TreeFilter(
            message_range if message_range is not None else None
        )
        self.depth_filter = DepthFilter(
            depth_range if depth_range is not None else None
        )
        self.message_filter = MessageFilter(
            date_range if date_range is not None else None,
            keyword_filter,
        )
        self.range_filter = range_filter
        self.custom_filter = custom_filter

    def is_valid(
        self, idx, total: int, conversation_tree: ChainTree, tree_depth: int = None
    ) -> bool:
        """
        Checks if a conversation tree is valid based on the filters set.

        Args:
            idx (int): The index of the conversation chain_tree.tree.
            total (int): The total number of conversation trees.
            conversation_tree (ChainTree): The conversation tree to be validated.
            tree_depth (int): The depth of the conversation chain_tree.tree.

        Returns:
            bool: True if the conversation tree is valid, False otherwise.
        """

        if not self.tree_filter.filter_tree_by_message_count(conversation_tree):
            return False

        if not self.depth_filter.filter_tree_by_depth(tree_depth):
            return False

        if self.range_filter is not None:
            if not self.range_filter.filter_by_index(idx, total):
                return False
            if not self.range_filter.filter_by_title(conversation_tree):
                return False
            if not self.range_filter.filter_custom(conversation_tree):
                return False
            if not self.range_filter.filter_content(conversation_tree):
                return False

        if self.custom_filter is not None and not self.custom_filter(conversation_tree):
            return False

        valid_messages = [
            message
            for message in conversation_tree.mapping.values()
            if self.message_filter.filter_messages([message])
        ]

        return bool(valid_messages)

    def get_filtered_tree(self, conversation_tree: ChainTree) -> ChainTree:
        """Generate a filtered tree based on the valid messages."""
        valid_messages = [
            message
            for message in conversation_tree.mapping.values()
            if self.message_filter.filter_messages([message])
        ]

        valid_message_ids = {m.id for m in valid_messages}
        new_mapping = {}
        for message in valid_messages:
            new_mapping[message.id] = ChainMap(
                id=message.id,
                message=message.message,
                parent=message.parent if message.parent in valid_message_ids else None,
                children=[
                    child for child in message.children if child in valid_message_ids
                ],
                references=message.references,
            )

        return ChainTree(
            conversation=ChainTree(
                title=conversation_tree.title,
                create_time=conversation_tree.create_time,
                update_time=conversation_tree.update_time,
                mapping=new_mapping,
                moderation_results=conversation_tree.moderation_results,
                current_node=conversation_tree.current_node,
            )
        )
