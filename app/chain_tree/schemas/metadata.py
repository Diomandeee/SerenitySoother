from app.chain_tree.schemas.finish_details import FinishDetails
from app.chain_tree.schemas.attachment import Attachment
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json


class Metadata(BaseModel):
    """
    Represents metadata associated with a message.
    Attributes:
        timestamp (Optional[str]): The timestamp for when the metadata was created.
        finish_details (Optional[MetadataFinishDetails]): Details about the model's finish process.
        metadata_details  (Optional[Dict[str, MetadataDetails]]): Any additional custom metadata.

    """

    finish_details: Optional[FinishDetails] = Field(
        None, description="Details on how the model finished processing the content."
    )

    attachments: Optional[List[Attachment]] = Field(
        None, description="List of attachments included in the message."
    )

    model_slug: Optional[str] = Field(
        None,
        description="The slug or identifier for the model used.",
        alias="model_slug",
    )

    parent_id: Optional[str] = Field(None, description="The id of the parent message.")

    timestamp_: Optional[str] = Field(
        None, description="The timestamp for when the finish details were generated."
    )

    links: Optional[Any] = Field(
        None, description="List of links extracted from the message content."
    )

    message_type: Optional[str] = Field(None, description="The type of message.")

    is_complete: Optional[bool] = Field(
        None, description="Whether the message is complete."
    )

    command: Optional[str] = Field(
        None, description="The command executed, related to this metadata."
    )

    args: Optional[List[int]] = Field(
        None, description="The arguments related to the command."
    )

    status: Optional[str] = Field(
        None, description="The status of the command execution."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "finish_details": {
                    "type": "text_generation",
                    "stop": "max_length",
                },
                "model_slug": "gpt2",
                "parent_id": "1234",
                "timestamp_": "2021-06-01T12:00:00.000Z",
                "links": ["https://example.com", "https://example2.com"],
            }
        }

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.metadata_details = self.metadata_details or {}

    def to_dict(self) -> Dict[str, Any]:
        return self.dict(exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
