from typing import Dict, Any, Optional, List
from pydantic.fields import Field
from pydantic import BaseModel
import json


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


class CitationFormat(BaseModel):
    name: str = Field(..., description="The name of the citation format.")


class MetadataItem(BaseModel):
    type: str = Field(..., description="The type of the metadata item.")
    title: str = Field(..., description="The title of the metadata item.")
    url: str = Field(..., description="The URL associated with the metadata item.")
    text: Optional[str] = Field(
        None, description="Any text associated with the metadata item."
    )
    pub_date: Optional[str] = Field(
        None, description="The publication date of the metadata item."
    )
    extra: Optional[str] = Field(
        None, description="Any extra information associated with the metadata item."
    )


class CiteMetadata(BaseModel):
    citation_format: CitationFormat = Field(
        ..., description="The citation format details."
    )
    metadata_list: List[MetadataItem] = Field(
        ..., description="A list of metadata items."
    )
    original_query: Optional[str] = Field(
        None, description="The original query leading to this metadata."
    )


class JupyterMessage(BaseModel):
    msg_type: str = Field(..., description="The type of the message.")
    parent_header: Dict[str, str] = Field(
        ..., description="The header information of the parent message."
    )
    version: str = Field(..., description="The protocol version.")
    content: Dict[str, str] = Field(
        ..., description="The content of the Jupyter message."
    )


class AggregateResult(BaseModel):
    status: str = Field(..., description="The status of the execution.")
    run_id: str = Field(..., description="The unique identifier for the run.")
    start_time: float = Field(..., description="The start time of the execution.")
    update_time: float = Field(
        ..., description="The last update time of the execution."
    )
    code_time: float = Field(..., description="The code execution time.")
    end_time: float = Field(..., description="The end time of the execution.")
    final_expression_output: str = Field(
        ..., description="The final output of the executed expression."
    )
    in_kernel_exception: Optional[str] = Field(
        None, description="Any exception from within the kernel."
    )
    system_exception: Optional[str] = Field(
        None, description="Any system-level exception."
    )
    messages: List[JupyterMessage] = Field(
        ..., description="List of messages related to the Jupyter execution."
    )
    timeout_triggered: Optional[bool] = Field(
        None, description="Whether a timeout was triggered during execution."
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "run_id": "example_id",
                "start_time": 1667923781.7307215,
                "update_time": 1667923788.2035695,
                "code_time": 1667923788.2035695,
                "end_time": 1667923788.2035695,
                "final_expression_output": "",
                "in_kernel_exception": None,
                "system_exception": None,
                "messages": [
                    {
                        "msg_type": "status",
                        "parent_header": {"msg_id": "example_msg_id", "version": "5.3"},
                        "content": {"execution_state": "busy"},
                    },
                ],
                "timeout_triggered": None,
            }
        }


class Metadata(BaseModel):
    """
    Represents metadata associated with a message.
    Attributes:
        timestamp (Optional[str]): The timestamp for when the metadata was created.
        finish_details (Optional[MetadataFinishDetails]): Details about the model's finish process.
        metadata_details  (Optional[Dict[str, MetadataDetails]]): Any additional custom metadata.

    """

    _cite_metadata: CiteMetadata = Field(..., description="Citation metadata details.")

    finish_details: Optional[FinishDetails] = Field(
        None, description="Details on how the model finished processing the content."
    )

    attachments: Optional[List[Attachment]] = Field(
        None, description="List of attachments included in the message."
    )

    aggregate_result: Optional[AggregateResult] = Field(
        None, description="The result of an aggregate execution."
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
