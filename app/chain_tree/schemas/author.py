from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from app.enums import RoleType


class Author(BaseModel):
    """
    Represents an author in the conversation.
    """

    role: RoleType = Field(..., description="The role of the author.")

    name: Optional[str] = Field(None, description="The name of the author.")

    metadata: Any = Field(None, description="Additional metadata about the author.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Author to a dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Author":
        """Converts a dictionary to an Author."""
        return cls(**data)

    def flatten(self) -> Dict[str, Any]:
        """
        Flatten the Author object to a dictionary.

        Returns:
            Dict[str, Any]: The flattened Author object.
        """
        return self.dict(exclude_none=True)


class User(Author):
    """Represents a user."""

    role = RoleType.USER


class Chat(Author):
    """Represents a chat."""

    role = RoleType.CHAT


class Assistant(Author):
    """Represents an assistant."""

    role = RoleType.ASSISTANT


class System(Author):
    """Represents a system."""

    role = RoleType.SYSTEM
