from typing import List, Optional, Any, Dict
from chain_tree.type import ContentType
from pydantic import BaseModel, Field


class Content(BaseModel):
    """
    The base class for all content types.
    """

    text: Optional[str] = Field(
        None, description="The text content of the message (if any)."
    )
    content_type: Any = Field(
        ContentType.TEXT, description="The type of content (text, image, audio, etc.)"
    )
    parts: Optional[List[Any]] = Field(
        None, description="The parts of the content (text, image, audio, etc.)"
    )

    part_lengths: Optional[Any] = Field(
        None, description="The lengths of the parts of the content."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "content_type": "text",
                "parts": ["Hello, how are you?"],
                "part_lengths": 1,
            }
        }

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.parts:
            self.text = self.parts[0]
            self.part_lengths = [
                len(part.text.split("\n\n")) if isinstance(part, Content) else 1
                for part in self.parts
            ]
        else:
            self.part_lengths = 0  # If parts are not provided, set part_lengths to 0.

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Content to a dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Content":
        """Converts a dictionary to a Content."""
        return {
            "text": data["text"],
            "content_type": data["content_type"],
            "parts": data["parts"],
            "part_lengths": data["part_lengths"],
        }

    @classmethod
    def from_text(cls, text: str):
        return cls(content_type=ContentType.TEXT, parts=[text])

    @classmethod
    def from_content_type(cls, content_type: ContentType, parts: list):
        return cls(content_type=content_type, parts=parts)

    def flatten(self) -> Dict[str, Any]:
        """Flattens the Content."""
        return {
            "text": self.text,
            "content_type": self.content_type,
            "parts": self.parts,
            "part_lengths": self.part_lengths,
        }
