from typing import Any, Optional, Union
from app.chain_tree.schemas.content import Content
from app.chain_tree.schemas.author import Author
from app.chain_tree.schemas.chain_coordinate import ChainCoordinate
from pydantic import BaseModel, Field


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
