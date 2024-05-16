from typing import Optional, Dict, Any, Type
from chain_tree.interface import IChainFactory
from chain_tree.models import Content, Chain, ChainCoordinate
from chain_tree.utils import (
    InvalidChainTypeException,
    InvalidIdException,
    InvalidContentException,
    InvalidCoordinateException,
)
from chain_tree.base import (
    AssistantChain,
    UserChain,
    SystemChain,
)
import threading
import logging
import uuid


class ChainFactory(IChainFactory):
    """
    Factory class for creating different types of chains.
    """

    _instance = None
    _lock = threading.Lock()
    chain_classes = {}

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of ChainFactory if it doesn't exist.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialize_chain_classes()
        return cls._instance

    def generate_id(self) -> str:
        """
        Generate a unique ID for the chain.

        Returns:
            str: The generated unique ID.
        """
        return str(uuid.uuid4())

    def initialize_chain_classes(self):
        """
        Initialize the chain classes by registering the default chain classes.
        """
        self.register_chain_class("system", SystemChain)
        self.register_chain_class("assistant", AssistantChain)
        self.register_chain_class("user", UserChain)

    @classmethod
    def register_chain_class(cls, chain_type: str, chain_class: Type[Chain]):
        """
        Register a chain class with a given chain type.

        Args:
            chain_type (str): The type of the chain.
            chain_class (Type[Chain]): The class representing the chain.

        Raises:
            ValueError: If the chain type is already registered.
        """
        if chain_type in cls.chain_classes:
            logging.warning(f"Overwriting existing chain type: {chain_type}")
        cls.chain_classes[chain_type] = chain_class

    @classmethod
    def unregister_chain_class(cls, chain_type: str):
        """
        Unregister a chain class with a given chain type.

        Args:
            chain_type (str): The type of the chain.

        Raises:
            ValueError: If the chain type is not registered.
        """
        if chain_type not in cls.chain_classes:
            raise ValueError(f"Chain type {chain_type} is not registered")
        del cls.chain_classes[chain_type]

    def create_chain(
        self,
        chain_type: str,
        id: str,
        content: Content,
        coordinate: ChainCoordinate,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chain:
        """
        Create a chain of the specified type.

        Args:
            chain_type (str): The type of the chain.
            id (str): The ID of the chain.
            content (Content): The content of the chain.
            coordinate (Coordinate): The coordinate of the chain.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the chain. Defaults to None.

        Returns:
            Chain: The created chain.

        Raises:
            InvalidChainTypeException: If the chain type is invalid.
            InvalidIdException: If the ID is invalid.
            InvalidContentException: If the content is invalid.
            InvalidCoordinateException: If the coordinate is invalid.
            Exception: If an error occurs while creating the chain.
        """
        if (
            chain_type is None
            or not isinstance(chain_type, str)
            or chain_type.strip() == ""
        ):
            raise InvalidChainTypeException("Chain type must be a non-empty string")

        chain_class = self.chain_classes.get(chain_type)
        if chain_class is None:
            message = f"Invalid chain type: {chain_type}"
            logging.error(message)
            raise InvalidChainTypeException(message)

        if id is None or not isinstance(id, str) or id.strip() == "":
            raise InvalidIdException("Id must be a non-empty string")

        if not isinstance(content, Content):
            raise InvalidContentException(
                "Content must be an instance of the Content class"
            )

        if not isinstance(coordinate, ChainCoordinate):
            raise InvalidCoordinateException(
                "Coordinate must be an instance of the Coordinate class"
            )

        try:
            return chain_class(
                id=id, content=content, coordinate=coordinate, metadata=metadata
            )
        except Exception as e:
            message = f"Error occurred while creating chain: {str(e)}"
            logging.error(message)
            raise

    def create_system_chain(
        self,
        id: str,
        content: Content,
        coordinate: ChainCoordinate,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SystemChain:
        """
        Create a system chain.

        Args:
            id (str): The ID of the chain.
            content (Content): The content of the chain.
            coordinate (Coordinate): The coordinate of the chain.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the chain. Defaults to None.

        Returns:
            SystemChain: The created system chain.
        """
        return self.create_chain(
            chain_type="system",
            id=id,
            content=content,
            coordinate=coordinate,
            metadata=metadata,
        )

    def create_assistant_chain(
        self,
        id: str,
        content: Content,
        coordinate: ChainCoordinate,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssistantChain:
        """
        Create an assistant chain.

        Args:
            id (str): The ID of the chain.
            content (Content): The content of the chain.
            coordinate (Coordinate): The coordinate of the chain.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the chain. Defaults to None.

        Returns:
            AssistantChain: The created assistant chain.
        """
        return self.create_chain(
            chain_type="assistant",
            id=id,
            content=content,
            coordinate=coordinate,
            metadata=metadata,
        )

    def create_user_chain(
        self,
        id: str,
        content: Content,
        coordinate: ChainCoordinate,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserChain:
        """
        Create a user chain.

        Args:
            id (str): The ID of the chain.
            content (Content): The content of the chain.
            coordinate (Coordinate): The coordinate of the chain.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the chain. Defaults to None.

        Returns:
            UserChain: The created user chain.
        """
        return self.create_chain(
            chain_type="user",
            id=id,
            content=content,
            coordinate=coordinate,
            metadata=metadata,
        )
