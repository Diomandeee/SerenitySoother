from chain_tree.models import Content, ChainCoordinate
from chain_tree.response.factory import ChainFactory
from chain_tree.response.links import ChainTreeLink
from chain_tree.interface import ChainBuilder
from chain_tree.utils import log_handler


class ReplyChainBuilder(ChainBuilder):
    def __init__(self):
        self.chain_tree = ChainTreeLink(ChainFactory())

    def validate_content(self, content: Content):
        """
        Validates the provided content.

        This method checks if the content is not empty and if it is an instance
        of the `Content` class.

        Parameters:
        - content: The content to be validated.

        Raises:
        - ValueError: If the content is empty or not an instance of the `Content` class.
        """
        try:
            if not content:
                raise ValueError("Content cannot be empty.")
            if not isinstance(content, Content):
                raise ValueError("Content should be an instance of the Content class.")
        except ValueError as e:
            log_handler(f"Error in validate_content: {e}")
            raise

    def build_chain(
        self, chain_type, content: Content, coordinate: ChainCoordinate, parent=None
    ):
        """
        Builds a chain based on the provided chain type, content, and coordinate.

        Parameters:
        - chain_type (str): Type of the chain. Accepted values are 'system', 'assistant', and 'user'.
        - content (Content): The content of the chain.
        - coordinate (Coordinate): The spatial node associated with the content.
        - parent (Optional): An optional parent for the chain.

        Raises:
        - ValueError: If the provided chain_type is unrecognized, or if the parent doesn't exist in the chain_tree.tree.
        """
        try:
            if chain_type not in ["system", "assistant", "user"]:
                raise ValueError(
                    "Unrecognized chain type. Accepted types are 'system', 'assistant', and 'user'."
                )

            if parent and parent not in self.chain_tree.chains:
                raise ValueError(f"Parent chain with id {parent} does not exist.")

            self.validate_content(content)

            chain_id = self.chain_tree.chain_factory.generate_id()
            self.chain_tree.add_chain(chain_type, chain_id, content, coordinate, parent)
            log_handler(f"Added {chain_type} chain with id {chain_id}.")

        except ValueError as e:
            log_handler(f"Error in build_chain: {e}")
            raise

    def build_system_chain(
        self, content: Content, coordinate: ChainCoordinate, parent=None
    ):
        """
        Builds a system chain with the provided content and coordinate.

        Parameters:
        - content (Content): The content of the chain.
        - coordinate (Coordinate): The spatial node associated with the content.
        - parent (Optional): An optional parent for the system chain.
        """
        self.build_chain("system", content, coordinate, parent)

    def build_assistant_chain(
        self, content: Content, coordinate: ChainCoordinate, parent=None
    ):
        """
        Builds an assistant chain with the provided content and coordinate.

        Parameters:
        - content (Content): The content of the chain.
        - coordinate (Coordinate): The spatial node associated with the content.
        - parent (Optional): An optional parent for the assistant chain.
        """
        self.build_chain("assistant", content, coordinate, parent)

    def build_user_chain(
        self, content: Content, coordinate: ChainCoordinate, parent=None
    ):
        """
        Builds a user chain with the provided content and coordinate.

        Parameters:
        - content (Content): The content of the chain.
        - coordinate (Coordinate): The spatial node associated with the content.
        - parent (Optional): An optional parent for the user chain.
        """
        self.build_chain("user", content, coordinate, parent)

    def build_custom_chain(
        self,
        chain_type: str,
        content: Content,
        coordinate: ChainCoordinate,
        metadata: dict = None,
    ):
        """
        Builds a custom chain with the provided type, content, and coordinate.

        This method allows for more flexibility by allowing any chain type to be
        provided as a string.

        Parameters:
        - chain_type (str): The custom type of the chain.
        - content (Content): The content of the chain.
        - coordinate (Coordinate): The spatial node associated with the content.
        - metadata (dict, optional): Additional metadata for the chain. Defaults to None.

        Note:
        - This function internally uses the build_chain method.
        """
        self.build_chain(chain_type, content, coordinate, metadata)

    def get_result(self):
        """
        Retrieves the current state of the chain chain_tree.tree.

        Returns:
        - The constructed chain chain_tree.tree.
        """
        return self.chain_tree
