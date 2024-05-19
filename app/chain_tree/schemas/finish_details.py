from typing import Optional
from pydantic.fields import Field
from pydantic import BaseModel


class FinishDetails(BaseModel):
    """
    Represents finish details for a message.
    Attributes:
        type (str): The type of finish details (text generation, classification, etc.).
        stop (str): Information on when the finish detail process stopped.
    """

    type: str = Field(
        ...,
        description="The type of finish details (text generation, classification, etc.).",
    )
    stop_tokens: str = Field(
        ..., description="Information on when the finish detail process stopped."
    )

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "type": "text_generation",
                "stop_tokens": "max_length",
            }
        }


class Attachment(BaseModel):
    name: str = Field(..., description="The display name of the attachment.")
    id: str = Field(..., description="A unique identifier for the attachment.")
    size: int = Field(..., description="The file size of the attachment in bytes.")
    mime_type: str = Field(
        ...,
        alias="mimeType",
        description="The MIME type of the attachment, indicating the file format.",
    )
    width: int = Field(
        ..., description="The width of the image or video attachment in pixels."
    )
    height: int = Field(
        ..., description="The height of the image or video attachment in pixels."
    )
    checksum: Optional[str] = Field(
        None, description="A checksum hash to verify the integrity of the attachment."
    )
    description: Optional[str] = Field(
        None, description="A brief description of the attachment."
    )
    uploaded_timestamp: Optional[str] = Field(
        None, description="The timestamp for when the attachment was uploaded."
    )
    url: Optional[str] = Field(
        None, description="A URL where the attachment can be accessed or downloaded."
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "Screenshot.png",
                "id": "file-123abc",
                "size": 204800,
                "mimeType": "image/png",
                "width": 1920,
                "height": 1080,
                "checksum": "e99a18c428cb38d5f260853678922e03",
                "description": "A screenshot of the latest UI design.",
                "uploaded_timestamp": "2023-11-14T15:29:00Z",
                "url": "http://example.com/download/screenshot.png",
            }
        }
