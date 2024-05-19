from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field


class Attachment(BaseModel):
    file_name: str
    file_data: Union[str, bytes]
    mime_type: str


class SecurityDetails(BaseModel):
    is_encrypted: bool
    digital_signature: Optional[str]


class MessageMetaData(BaseModel):
    priority_level: int
    type_of_message: str
    period_of_validity: Optional[timedelta]
    current_status: str
    conversation_thread_id: Optional[str]


class SenderReceiverInfo(BaseModel):
    sender: str
    receiver: str


class MessageContent(BaseModel):
    content: str
    list_of_attachments: List[Attachment]
    timestamp: datetime


class Message(BaseModel):
    id: str
    sender_receiver_info: SenderReceiverInfo
    metadata: MessageMetaData
    security_details: SecurityDetails
    content_details: MessageContent
    additional_metadata: Dict[str, Union[str, int, float]] = Field(
        {}, description="Additional metadata for the message."
    )

    @property
    def is_still_valid(self) -> bool:
        """Check if the message is still within its validity period."""
        if self.metadata.period_of_validity is None:
            return True
        expiration_time = (
            self.content_details.timestamp + self.metadata.period_of_validity
        )
        return datetime.utcnow() <= expiration_time

    @property
    def is_encrypted(self) -> bool:
        """Check if the message is encrypted."""
        return self.security_details.is_encrypted

    @property
    def is_signed(self) -> bool:
        """Check if the message is signed."""
        return self.security_details.digital_signature is not None
