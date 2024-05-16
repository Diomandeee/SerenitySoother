from typing import Any, Dict, Optional, List, Union
from chain_tree.utils import filter_none_values
from chain_tree.models.content import Content
from chain_tree.models.author import Author
from chain_tree.type import ContentType
from pydantic import BaseModel, Field
from pydantic import BaseModel
from uuid import uuid4
import pandas as pd
import json


class ChainCoordinate(BaseModel):
    id: str = Field(
        str(uuid4()),
        description="The ID of the node.",
    )

    x: Any = Field(0, description="The x-coordinate of the coordinate.")

    y: Any = Field(0, description="The y-coordinate of the coordinate.")

    z: Any = Field(0, description="The z-coordinate of the coordinate.")

    t: Any = Field(0, description="The t-coordinate of the coordinate.")

    n_parts: int = Field(0, description="The number of parts of the coordinate.")

    parent: Optional[str] = Field(
        None,
        description="The ID of the parent node.",
    )

    children: List["ChainCoordinate"] = Field(
        [],
        description="The list of children nodes.",
    )

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "id": "1234",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "t": 0.0,
            "n_parts": 0,
            "parent": "5678",
            "children": [],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ChainCoordinate to a dictionary."""
        return self.dict(exclude_none=True)


class ChainMessage(BaseModel):
    content: Optional[Content] = Field(
        default=None, description="The content of the message."
    )

    author: Optional[Union[Author]] = Field(
        default=None, description="The author of the message."
    )
    create_time: float = Field(
        default=None, description="Timestamp when the message was created."
    )
    end_turn: Optional[bool] = Field(
        default=None, description="Whether the message ends the current turn."
    )
    weight: int = Field(
        default=1,
        description="Weight indicating the message's importance or relevance.",
    )
    metadata: Optional[Any] = Field(
        default=None, description="Metadata associated with the message."
    )
    recipient: Optional[str] = Field(
        default=None, description="Recipient of the message, if applicable."
    )

    coordinate: Optional[Union[ChainCoordinate, Any]] = Field(
        default=None,
        description="Coordinate of the message in the conversation.",
    )


class Chain(ChainMessage):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the message.",
    )
    cluster_label: Optional[int] = Field(
        default=None,
        description="Cluster label for cluster-based analysis or navigation.",
    )
    n_neighbors: Optional[int] = Field(
        default=None, description="Number of nearest neighbors for graph construction."
    )

    embedding: Optional[Any] = Field(
        default=None, description="Embedding of the document for similarity search."
    )

    parent: Optional[str] = Field(default=None, description="Parent of the message.")

    children: Optional[Any] = Field(
        default=None, description="Children of the message."
    )

    depth: Optional[int] = Field(
        default=None, description="Depth of the message in the conversation."
    )

    next: Optional[str] = Field(None, description="The ID of the next message.")

    prev: Optional[str] = Field(None, description="The ID of the previous message.")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        schema_extra = (
            {
                "id": "1234",
                "content": {
                    "text": "Hello, how are you?",
                    "content_type": [
                        {
                            "type": "image",
                            "url": "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
                        }
                    ],
                    "parts": ["Hello, how are you?"],
                    "part_lengths": 1,
                },
                "author": {
                    "name": "John Doe",
                },
                "create_time": 1234567890,
                "end_turn": True,
                "weight": 1,
                "metadata": {"key": "value"},
                "recipient": "Jane Doe",
                "coordinate": {"x": 0.0, "y": 0.0, "z": 0.0, "t": 0.0, "n_part": 0},
                "embedding": [0.0, 0.0, 0.0, 0.0],
                "umap_embeddings": [0.0, 0.0],
                "n_neighbors": 10,
                "cluster_label": 1,
                "children": [],
            },
        )

    def compute_depth(self, message_data: List["Chain"]) -> None:
        """Computes the depth of the Chain."""
        if self.parent is None:
            self.depth = 0
        else:
            parent = next(
                (message for message in message_data if message.id == self.parent), None
            )
            if parent is not None:
                self.depth = parent.depth + 1
            else:
                self.depth = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Chain to a dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chain":
        """Converts a dictionary to a Chain."""
        return cls(
            id=data["id"],
            content=data["content"],
            author=data["author"],
            create_time=data["create_time"],
            end_turn=data["end_turn"],
            weight=data["weight"],
            metadata=data["metadata"],
            recipient=data["recipient"],
            coordinate=data["coordinate"],
            embedding=data["embedding"],
            next=data["next"],
            prev=data["prev"],
        )

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> List["Chain"]:
        """Converts a dictionary to a Chain."""
        return [cls.from_dict(row) for row in data]

    def to_dict(self) -> Dict[str, Any]:
        """Flattens the Chain."""
        return {
            "id": self.id,
            **filter_none_values(self.content.to_dict()),
            **filter_none_values(self.author.to_dict()),
            "coordinate": self.coordinate,
            "create_time": self.create_time,
            "end_turn": self.end_turn,
            "weight": self.weight,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "next": self.next,
            "prev": self.prev,
        }

    @classmethod
    def from_dataframe_row(cls, row: Dict[str, Any]) -> "Chain":
        """Converts a DataFrame row to a Chain."""
        return cls.from_dict(row)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["Chain"]:
        """Converts a DataFrame to a list of Chains."""
        return [cls.from_dataframe_row(row) for _, row in df.iterrows()]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Chain to a DataFrame.

        Returns:
            pd.DataFrame: The DataFrame representation of the Chain.
        """
        return pd.DataFrame([self.to_dict()])

    @classmethod
    def from_text(cls, text: str):
        """Creates a Chain object from text."""
        return cls(content=Content.from_text(text))

    @classmethod
    def from_content_type(cls, content_type: ContentType, parts: List[str]):
        """Creates a Chain object from a content type and a list of parts."""
        return cls(content=Content.from_content_type(content_type, parts))

    @classmethod
    def from_content(cls, content: Content):
        """Creates a Chain object from a content object."""
        return cls(content=content)

    @classmethod
    def from_author(cls, author: Author):
        """Creates a Chain object from an author object."""
        return cls(author=author)

    @staticmethod
    def flatten_all_chain_trees(chains: List["Chain"]) -> List["Chain"]:
        """
        Recursively flatten a list of root Chain objects (each representing a tree) into a single list using DFS.

        Args:
            mappin (List[Chain]): List of root Chain objects, each a root of a chain tree.

        Returns:
            List[Chain]: A list containing all the Chain objects from all the trees.
        """

        def dfs(node: "Chain", flat_list: List["Chain"]) -> None:
            """
            Helper function to perform DFS and flatten the tree.

            Args:
                node (Chain): Current node in the DFS traversal.
                flat_list (List[Chain]): Accumulator for all nodes visited.
            """
            if node is not None:
                if node.content and node.content.text:
                    # Strip whitespace from the content text
                    node.content.text = node.content.text.strip()
                if hasattr(node.content, "parts"):
                    # Completely remove content parts
                    del node.content.parts

                if node.children is not None:
                    for child in node.children:
                        flat_list.append(
                            child
                        )  # Append the current node before its children
                        dfs(child, flat_list)

        all_chains_flat = []
        for root_chain in chains:
            dfs(
                root_chain, all_chains_flat
            )  # Flatten each chain tree and extend the main list

        return all_chains_flat

    @staticmethod
    def flatten_chain_tree(root_chain: "Chain") -> List["Chain"]:
        """
        Flatten a tree of Chain objects into a list using DFS.

        Args:
            root_chain (Chain): The root of the chain tree.

        Returns:
            List[Chain]: A list containing all the Chain objects in the tree.
        """
        flat_list = []

        def dfs(node: "Chain") -> None:
            """
            Helper function to perform DFS and flatten the tree.

            Args:
                node (Chain): Current node in the DFS traversal.
            """
            if node is not None:
                if node.content and node.content.text:
                    # Strip whitespace from the content text
                    node.content.text = node.content.text.strip()
                if hasattr(node.content, "parts"):
                    # Completely remove content parts
                    del node.content.parts

                flat_list.append(node)  # Append the current node before its children
                if node.children is not None:
                    for child in node.children:
                        dfs(child)

        dfs(root_chain)
        return flat_list

    @staticmethod
    def get_chain_tree_depth(root_chain: "Chain") -> int:
        """
        Get the depth of a chain tree.

        Args:
            root_chain (Chain): The root of the chain tree.

        Returns:
            int: The depth of the chain tree.
        """

        def dfs(node: "Chain") -> int:
            """
            Helper function to perform DFS and calculate the depth of the tree.

            Args:
                node (Chain): Current node in the DFS traversal.

            Returns:
                int: The depth of the tree.
            """
            if node is not None:
                if node.children is not None:
                    return 1 + max(dfs(child) for child in node.children)
            return 1

        return dfs(root_chain)


@staticmethod
def save_chains_to_csv(chains: List[Chain], path: str) -> None:
    """
    Save a list of Chain objects to a CSV file.

    Args:
        chains (List[Chain]): List of Chain objects.
        path (str): Path to save the CSV file.
    """
    df = pd.DataFrame([chain.to_dict() for chain in chains])
    df.to_csv(path, index=False)


@staticmethod
def load_chains_from_csv(path: str) -> List[Chain]:
    """
    Load a list of Chain objects from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        List[Chain]: List of Chain objects.
    """
    df = pd.read_csv(path)
    return Chain.from_dataframe(df)


@staticmethod
def save_chains_to_json(chains: List[Chain], path: str) -> None:
    """
    Save a list of Chain objects to a JSON file.

    Args:
        chains (List[Chain]): List of Chain objects.
        path (str): Path to save the JSON file.
    """
    with open(path, "w") as f:
        json.dump([chain.to_dict() for chain in chains], f, indent=4)
