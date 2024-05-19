from typing import Optional, Dict, Any, List, Union, Callable, NamedTuple, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from app.chain_tree.schemas import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import numpy as np
import threading
import uuid
import time


@dataclass
class EntityAction:
    """A full description of an action for an ActionEntity to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""


class EntityFinish(NamedTuple):
    """The final return value of an ActionEntity."""

    return_values: dict
    """Dictionary of return values."""
    log: str
    """Additional information to log about the return value"""


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""


class Generation(BaseModel):
    """Output of a single generation."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""


class ChainGeneration(Generation):
    """Output of a single generation."""

    text = ""
    message: Chain

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["text"] = values["message"].content
        return values


class ChainResult(BaseModel):
    """Class that contains all relevant information for a Chat Result."""

    generations: List[ChainGeneration]
    """List of the things generated."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_chain(self) -> List[Chain]:
        """Return prompt as messages."""


class DistanceStrategy(ABC):
    @abstractmethod
    def compute(self, chain_a: "RoleChain", chain_b: "RoleChain") -> float:
        pass


class EuclideanDistance(DistanceStrategy):
    def compute(self, chain_a: "RoleChain", chain_b: "RoleChain") -> float:

        return np.linalg.norm(np.array(chain_a.embedding) - np.array(chain_b.embedding))


class CosineDistance(DistanceStrategy):
    def compute(self, chain_a: "RoleChain", chain_b: "RoleChain") -> float:
        return np.dot(chain_a.embedding, chain_b.embedding) / (
            np.linalg.norm(chain_a.embedding) * np.linalg.norm(chain_b.embedding)
        )


class ComponentModel(BaseModel):
    messages_received: List[Message] = Field(
        default_factory=list,
        description="List of messages that this component has received.",
    )
    messages_sent: List[Message] = Field(
        default_factory=list,
        description="List of messages that this component has sent.",
    )
    message_queue: deque = Field(
        default_factory=deque,
        description="Queue for messages that are yet to be processed.",
    )

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Start the automated processing of messages.
        self.start_message_processing()

    def communicate(
        self,
        other: "ComponentModel",
        content: str,
        priority: int = 1,
        message_type: str = "default",
        meta: Dict[str, Union[str, int, float]] = {},
        is_encrypted: bool = False,
    ) -> None:
        if not hasattr(other, "messages_received"):
            raise AttributeError(
                f"The other model {other.__class__.__name__} does not support messaging."
            )

        # Create the Message object with sub-models
        message = Message(
            id=str(uuid.uuid4()),
            sender_receiver_info=SenderReceiverInfo(
                sender=self.__class__.__name__, receiver=other.__class__.__name__
            ),
            metadata=MessageMetaData(
                priority_level=priority,
                type_of_message=message_type,
                current_status="unread",
            ),
            security_details=SecurityDetails(is_encrypted=is_encrypted),
            content_details=MessageContent(
                content=content, timestamp=datetime.utcnow(), list_of_attachments=[]
            ),
            additional_metadata=meta,
        )

        # Append to sent and received message lists
        other.messages_received.append(message)
        self.messages_sent.append(message)

        # Handle the received message in the other component
        other.handle_message(message)

    def handle_message(self, message: Message) -> None:
        """Handle received messages."""
        if message.metadata.priority_level < 5:
            self.message_queue.append(message)
            return
        self.process_message(message)

    def process_message(self, message: Message):
        """Process a message immediately based on its type."""
        if message.metadata.type_of_message == "alert":
            print(
                f"[ALERT][{message.content_details.timestamp}] From {message.sender_receiver_info.sender}: {message.content_details.content}"
            )
        elif message.metadata.type_of_message == "update":
            print(
                f"[UPDATE][{message.content_details.timestamp}] From {message.sender_receiver_info.sender}: {message.content_details.content}"
            )

        elif message.metadata.type_of_message == "default":
            print(
                f"[DEFAULT][{message.content_details.timestamp}] From {message.sender_receiver_info.sender}: {message.content_details.content}"
            )

        elif message.metadata.type_of_message == "error":
            print(
                f"[ERROR][{message.content_details.timestamp}] From {message.sender_receiver_info.sender}: {message.content_details.content}"
            )

        self.update_message_status(message, "read")

    def process_queued_messages(self):
        """Process all queued messages."""
        while self.message_queue:
            message = self.message_queue.popleft()
            self.process_message(message)

    def start_message_processing(self):
        """Starts a background thread that processes messages from the queue."""
        self.message_processing_thread = threading.Thread(
            target=self.run_message_processing
        )
        self.message_processing_thread.daemon = True  # Daemonize thread
        self.message_processing_thread.start()

    def run_message_processing(self):
        """Method to run on the background thread."""
        while True:
            if self.message_queue:
                message = self.message_queue.popleft()
                self.process_message(message)
            else:
                # Sleep briefly to avoid busy waiting if the queue is empty.
                time.sleep(0.1)

    def update_message_status(self, message: Message, new_status: str) -> None:
        """Update the status of a message."""
        message.metadata.current_status = new_status

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Get a message by its ID."""
        for message in self.messages_received:
            if message.id == message_id:
                return message
        return None

    def get_read_messages(self) -> List[Message]:
        """Get all read messages."""
        return [
            m for m in self.messages_received if m.metadata.current_status == "read"
        ]

    def get_unread_messages(self) -> List[Message]:
        """Get all unread messages."""
        return [
            m for m in self.messages_received if m.metadata.current_status == "unread"
        ]

    def __del__(self):
        if self.message_processing_thread:
            self.message_processing_thread.join()


class SynthesisTechnique(ComponentModel):
    model: Any = Field(None, description="The model used for synthesis")
    system_prompt: str = Field(None, description="The model used for synthesis")
    epithet: str = Field(..., description="The epithet describing the technique")
    name: str = Field(..., description="The name of the synthesis technique")
    technique_name: str = Field(..., description="The detailed name of the technique")
    description: str = Field(..., description="The description of the technique")
    imperative: str = Field(
        ..., description="The imperative command related to the technique"
    )
    prompts: Dict[str, Any] = Field(
        ..., description="Additional prompts for the technique"
    )
    tokenizer: Any = None

    message_processing_thread: Optional[threading.Thread] = Field(
        None, description="Thread for processing messages."
    )

    embedding: Optional[Any] = Field(None, description="Embeddings of the technique.")

    class Config(ComponentModel.Config):
        arbitrary_types_allowed = True

    @validator("name")
    def validate_name(cls, value: str):
        if not value:
            raise ValueError("Name cannot be empty.")
        return value

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        pass

    def get_options(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "epithet": self.epithet,
            "name": self.name,
            "technique_name": self.technique_name,
            "imperative": self.imperative,
            "prompts": self.prompts,
            "description": self.description,
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_str(self) -> str:
        similarity_string = (
            str(self.get_options())[1:-1]
            .replace("'", "")
            .replace(",", "\n")
            .replace("[", "")
            .replace("]", "")
        )
        return similarity_string

    def compute(self, other: "SynthesisTechnique", embedder: Callable) -> float:
        a = self.convert_to_str()
        b = other.convert_to_str()
        a = embedder(a)
        b = embedder(b)

        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return similarity

    def compute_similarity(
        self, techniques: List["SynthesisTechnique"], embedder: Callable, k: int
    ) -> List[Tuple["SynthesisTechnique", float]]:
        """
        Compute the similarity between a target technique and a list of techniques.

        Args:
            techniques (List[SynthesisTechnique]): List of SynthesisTechnique instances.
            target_technique (SynthesisTechnique): The target technique to compare against.
            embedder (Callable): Embedder function to convert technique descriptions to vectors.
            k (int): Number of top similar techniques to return.

        Returns:
            List[Tuple[SynthesisTechnique, float]]: List of tuples containing similar techniques and their similarity scores.
        """
        # Compute similarity scores for all techniques
        similarity_scores = [
            (technique, self.compute(technique, embedder)) for technique in techniques
        ]

        # Sort techniques based on similarity scores
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k similar techniques
        return similarity_scores[:k]


class RoleChain(ComponentModel):
    id: str = Field(..., description="Unique identifier for the chain.")
    author: Author = Field(..., description="The author of the chain.")
    content: Content = Field(..., description="The content of the chain.")
    coordinate: BaseOperations = Field(
        ...,
        description="The coordinate of the chain in the conversation chain_tree.tree.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Any additional metadata about the chain."
    )
    embedding: Optional[List[float]] = Field(
        None, description="Embeddings of the chain."
    )
    children: Optional[List["RoleChain"]] = Field(
        [], description="The children of the chain in the conversation chain_tree.tree."
    )

    message_processing_thread: Optional[threading.Thread] = Field(
        None, description="Thread for processing messages."
    )

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "id": "Chain1",
                "author": {"role": "Role.USER", "metadata": {}},
                "content": {"content_type": "text", "parts": ["Hello"]},
                "coordinate": {"x": 0, "y": 0, "z": 0, "w": 0},
                "metadata": {},
                "embedding": [],
                "children": [],
            }
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_child(cls, child: "RoleChain"):
        if not isinstance(child, RoleChain):
            raise TypeError("Child must be an instance of RoleChain.")

    def add_child(self, child: "RoleChain"):
        """Add a child to the current chain."""
        self.validate_child(child)
        self.children.append(child)

    def remove_child(self, child: "RoleChain"):
        """Remove a child from the current chain."""
        self.validate_child(child)
        if child in self.children:
            self.children.remove(child)

    def find_child(
        self, condition: Callable[["RoleChain"], bool]
    ) -> Optional["RoleChain"]:
        """Find a child that satisfies the condition."""
        for child in self.children:
            if condition(child):
                return child
        return None

    def broadcast(self, message: Message):
        """Send a message to all children in the RoleChain hierarchy."""
        for child in self.children:
            child.handle_message(message)
            child.broadcast(message)  # Recursively broadcast to children's children

    def update_metadata(self, key: str, value: Any):
        """Update the metadata of the current chain."""
        self.metadata[key] = value

    def delete_metadata(self, key: str):
        """Delete a metadata key-value pair."""
        if key in self.metadata:
            del self.metadata[key]

    def get_children(self) -> List["RoleChain"]:
        """Retrieve the list of children of the current chain."""
        return self.children

    def get_distance(
        self, other_chain: "RoleChain", strategy: DistanceStrategy
    ) -> Optional[float]:
        """Calculate the distance between two chains based on a specified strategy.
        Assumes embeddings and coordinates are lists of floats.
        """
        if not self.embedding or not other_chain.embedding:
            print("One or both chains don't have embeddings.")
            return None

        if len(self.embedding) != len(other_chain.embedding):
            print("Embeddings don't have the same dimension.")
            return None

        try:
            return strategy.compute(self, other_chain)

        except TypeError:
            print(
                "One of the properties (embedding/coordinate) contains non-numeric values."
            )
            return None

        except Exception as e:
            print(f"Error computing distance: {str(e)}")
            return None

    def __len__(self):
        return len(self.children)

    def get_text(self):
        return self.content.parts.raw


class AssistantChain(RoleChain):
    def __init__(
        self,
        id: str = str(uuid.uuid4()),
        content: Content = Content(),
        coordinate: BaseOperations = BaseOperations(),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            id=id,
            author=Assistant(),
            content=content,
            coordinate=coordinate,
            metadata=metadata,
        )


class UserChain(RoleChain):
    def __init__(
        self,
        id: str = str(uuid.uuid4()),
        content: Content = Content(),
        coordinate: BaseOperations = BaseOperations(),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            id=id,
            author=User(),
            content=content,
            coordinate=coordinate,
            metadata=metadata,
        )


class SystemChain(RoleChain):
    def __init__(
        self,
        id: str = str(uuid.uuid4()),
        content: Content = Content(),
        coordinate: BaseOperations = BaseOperations(),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            id=id,
            author=System(),
            content=content,
            coordinate=coordinate,
            metadata=metadata,
        )


class RoleChainTrees:
    """A collection of RoleChain trees."""

    def __init__(self, trees: List[RoleChain]):
        self.trees = trees

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index: int) -> RoleChain:
        return self.trees[index]

    def __iter__(self):
        return iter(self.trees)

    def __str__(self):
        return "\n".join(map(str, self.trees))

    def __repr__(self):
        return f"RoleChainTrees({self.trees})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RoleChainTrees) and self.trees == other.trees

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def add_tree(self, tree: RoleChain):
        self.trees.append(tree)

    def remove_tree(self, tree: RoleChain):
        if tree in self.trees:
            self.trees.remove(tree)

    def find_tree(self, condition: Callable[[RoleChain], bool]) -> Optional[RoleChain]:
        return next((tree for tree in self.trees if condition(tree)), None)

    def get_tree(self, condition: Callable[[RoleChain], bool]) -> Optional[RoleChain]:
        return self.find_tree(condition)


class ChatMessage(RoleChain):
    """Type of message with arbitrary speaker."""

    role: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"


def _convert_dict_to_message(message_dict: dict) -> RoleChain:
    role = message_dict["role"]
    content = Content(raw=message_dict["content"])
    coordinate = BaseOperations(x=0, y=0, z=0, t=0)
    if role == "user":
        return UserChain(
            id=str(uuid.uuid4()),
            content=content,
            coordinate=coordinate,
        )
    elif role == "assistant":
        return AssistantChain(
            id=str(uuid.uuid4()),
            content=content,
            coordinate=coordinate,
        )
    elif role == "system":
        return SystemChain(
            id=str(uuid.uuid4()),
            content=content,
            coordinate=coordinate,
        )
    else:
        raise ValueError(f"Got unknown role {role}")


def _convert_message_to_dict(message: Dict[str, Any]) -> dict:
    if "role" in message and "content" in message:
        message_dict = {"role": message["role"], "content": message["content"]}
    else:
        raise ValueError(f"Got unknown type {message}")

    if "name" in message:
        message_dict["name"] = message["name"]
    return message_dict


# from chain_tree.models import (
#     Content,
#     Author,
#     User,
#     Assistant,
#     System,
#     Message,
#     SenderReceiverInfo,
#     SecurityDetails,
#     MessageContent,
#     MessageMetaData,
#     ChainCoordinate,
# )
