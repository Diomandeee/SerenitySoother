from typing import Any, Dict, List, Optional
from chain_tree.models.metadata import Metadata
from chain_tree.models.mapping import ChainMap
from chain_tree.models.content import Content
from chain_tree.models.author import Author
from chain_tree.models.chain import Chain
from pydantic import BaseModel, Field
from uuid import uuid4
import pandas as pd
import json


class ChainTree(BaseModel):
    """

    Represents a conversation as a tree of messages.
    """

    title: str = Field(None, description="The title of the conversation.")

    id: str = Field(default_factory=lambda: str(uuid4()))

    create_time: float = Field(
        None, description="The timestamp for when the conversation was created."
    )
    update_time: float = Field(
        None, description="The timestamp for when the conversation was last updated."
    )
    mapping: Dict[str, ChainMap] = Field(
        None,
        description="A dictionary mapping node IDs to their corresponding message nodes.",
    )

    moderation_results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Moderation results associated with the conversation."
    )
    current_node: Optional[str] = Field(None, description="The ID of the current node.")

    conversation_template_id: Optional[str] = Field(
        None, description="The ID of the conversation template."
    )

    plugin_ids: Optional[List[str]] = Field(
        None, description="The IDs of the plugins associated with the conversation."
    )

    gizmo_id: Optional[str] = Field(
        None, description="The ID of the Gizmo associated with the conversation."
    )

    def __init__(self, **data):
        super().__init__(**data)

        if self.mapping is None:
            self.mapping = {}

    class Config:
        arbitrary_types_allowed = True

    def to_json(self) -> str:
        """
        Convert the ChainTree to a JSON string.

        Returns:
            str: The JSON string representation of the Chainchain_tree.tree.
        """

        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "id": self.id,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "mapping": {k: v.to_dict() for k, v in self.mapping.items()},
            "moderation_results": self.moderation_results,
            "current_node": self.current_node,
            "conversation_template_id": self.conversation_template_id,
            "plugin_ids": self.plugin_ids,
            "gizmo_id": self.gizmo_id,
        }

    def save(self, path: str) -> None:
        """
        Save the ChainTree to a file.

        Args:
            path (str): The path to save the ChainTree to.
        """
        with open(path, "w") as f:
            f.write(self.to_json())

    def __str__(self) -> str:
        return f"ChainTree(title={self.title}, id={self.id})"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChainTree":
        return cls(
            title=data["title"],
            id=data.get("id", str(uuid4())),
            create_time=data["create_time"],
            update_time=data["update_time"],
            mapping=data["mapping"],
            moderation_results=data["moderation_results"],
            current_node=data["current_node"],
            conversation_template_id=data.get("conversation_template_id", None),
            plugin_ids=data.get("plugin_ids", None),
            gizmo_id=data.get("gizmo_id", None),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ChainTree":
        return cls.from_dict(json.loads(json_str))

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the ChainTree to a DataFrame.

        Returns:
            pd.DataFrame: The DataFrame representation of the Chainchain_tree.tree.
        """
        columns_to_extract = ["content", "author", "metadata", "id"]

        chain_maps = []

        for chain_map in self.mapping.values():
            if chain_map.message is None:
                continue

            flattened_chain_map = chain_map.flatten()
            flattened_message = chain_map.message.flatten()
            flattened_message_data = {
                key: flattened_message.pop(key) for key in columns_to_extract
            }

            # Merge the extracted message data with 'id'
            flattened_message_data["id"] = {"id": flattened_message_data["id"]}
            flattened_message_data.update(flattened_message)

            flattened_chain_map.pop("message")
            flattened_chain_map.update(flattened_message_data)
            chain_maps.append(flattened_chain_map)

        df = pd.DataFrame(chain_maps)

        return df

    @staticmethod
    def process_conversation_dataframe(conversation_df):
        # Create a list to store the updated relationships
        updated_conversation_data = []

        # Add the root message (index 0) with no parent and an empty children list
        updated_conversation_data.append(
            {"id": conversation_df.loc[0, "id"], "parent": None, "children": []}
        )

        # Loop through the DataFrame starting from index 1
        for i in range(1, len(conversation_df)):
            current_message = conversation_df.loc[i, "id"]
            parent_message = conversation_df.loc[i - 1, "id"]

            # Update the parent for the current message
            updated_message = {
                "id": current_message,
                "parent": parent_message,
                "children": [],
            }

            # Update the children for the parent message
            parent_index = next(
                (
                    index
                    for index, item in enumerate(updated_conversation_data)
                    if item["id"] == parent_message
                ),
                None,
            )
            if parent_index is not None:
                updated_conversation_data[parent_index]["children"].append(
                    current_message
                )

            updated_conversation_data.append(updated_message)

        # Create the updated DataFrame
        updated_conversation_df = pd.DataFrame(updated_conversation_data)

        # Update the conversation_df DataFrame with the new parent and children id
        conversation_df["parent"] = updated_conversation_df["parent"]
        conversation_df["children"] = updated_conversation_df["children"]
        conversation_df["cluster_label"] = 0
        conversation_df["n_neighbors"] = 0

        # Drop the 'coordinate' column
        if "coordinate" in conversation_df.columns:
            conversation_df = conversation_df.drop(columns=["coordinate"])

        return conversation_df

    @staticmethod
    def create_metadata(row: pd.Series) -> Metadata:
        return Metadata(links=row["links"])

    @staticmethod
    def create_content(row: pd.Series, content_metadata: Metadata) -> Content:
        text = row["text"]
        return Content(
            text=text,
            content_type=row.get("content_type", "text"),
            parts=[text],
            part_lengths=[len(text)],
            content_metadata=content_metadata,
        )

    @staticmethod
    def create_author(row: pd.Series) -> Author:
        return Author(role=row["role"])

    @staticmethod
    def create_chain(
        row: pd.Series, message_id: str, content: Content, author: Author
    ) -> Chain:
        return Chain(
            id=message_id,
            content=content,
            author=author,
            create_time=row["create_time"],
            end_turn=row.get("end_turn", False),
            weight=row.get("weight", 1.0),
            metadata=row.get("metadata", {}),
            recipient=row.get("recipient", "all"),
            coordinate=row.get("coordinate", []),
            embedding=row.get("embedding", []),
            umap_embeddings=row.get("umap_embeddings"),
            n_neighbors=row.get("n_neighbors", 0),
            cluster_label=row.get("cluster_label", 0),
        )

    @staticmethod
    def create_chain_map(row: pd.Series, message_id: str, chain: Chain) -> ChainMap:
        return ChainMap(
            id=message_id,
            parent=row["parent"],
            children=row["children"],
            references=row.get("references", []),
            relationships=row.get("relationships", []),
            message=chain,
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ChainTree":
        """
        Create a ChainTree from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to use.

        Returns:
            ChainTree: The ChainTree object.
        """
        df = cls.process_conversation_dataframe(df)
        chain_tree = cls()
        chain_tree.mapping = {}

        for _, row in df.iterrows():
            message_id = row["id"]
            if message_id is None:
                raise ValueError("ID is missing or None in the DataFrame.")

            content_metadata = cls.create_metadata(row)
            content = cls.create_content(row, content_metadata)
            author = cls.create_author(row)
            chain = cls.create_chain(row, message_id, content, author)
            chain_map = cls.create_chain_map(row, message_id, chain)

            chain_tree.mapping[message_id] = chain_map

        return chain_tree

    def get_node(self, node_id: str) -> ChainMap:
        """
        Get a node from the Chainchain_tree.tree.

        Args:
            node_id (str): The ID of the node to get.

        Returns:
            ChainMap: The ChainMap object.
        """
        # If the node_id is not in the mapping, raise a KeyError
        if node_id not in self.mapping:
            raise KeyError(f"Node with ID '{node_id}' not found.")

        # Return the ChainMap object
        return self.mapping[node_id]

    def get_current_node(self) -> ChainMap:
        """
        Get the current node from the Chainchain_tree.tree.

        Returns:
            ChainMap: The ChainMap object.
        """
        # If the current_node is not set, raise a KeyError
        if not self.current_node:
            raise KeyError("Current node not set.")

        # Return the ChainMap object
        return self.get_node(self.current_node)

    def add_node(self, node: ChainMap) -> None:
        """
        Add a node to the Chainchain_tree.tree.

        Args:
            node (ChainMap): The ChainMap object to add.
        """
        # If the node already exists, raise a KeyError
        if node.id in self.mapping:
            raise KeyError(f"Node with ID '{node.id}' already exists.")

        # Add the node to the mapping
        self.mapping[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the Chainchain_tree.tree.

        Args:
            node_id (str): The ID of the node to remove.
        """
        # If the node does not exist, raise a KeyError
        if node_id not in self.mapping:
            raise KeyError(f"Node with ID '{node_id}' not found.")

        # Remove the node from the mapping
        del self.mapping[node_id]

    def set_current_node(self, node_id: str) -> None:
        """
        Set the current node in the Chainchain_tree.tree.

        Args:
            node_id (str): The ID of the node to set as current.
        """
        # If the node does not exist, raise a KeyError
        if node_id not in self.mapping:
            raise KeyError(f"Node with ID '{node_id}' not found.")

        # Set the current_node
        self.current_node = node_id
