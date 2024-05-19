from app.chain_tree.schemas.mapping import ChainMap
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4
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
