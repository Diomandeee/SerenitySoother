from typing import Optional, Dict, Any, List, Union, Callable, NamedTuple, Tuple
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from chain_tree.models import *
from datetime import datetime
from collections import deque
import networkx as nx
import numpy as np
import threading
import torch
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

    def compute_cross_entropy_loss(
        self, other: "SynthesisTechnique", embedder: Callable
    ) -> float:
        a = self.convert_to_str()
        b = other.convert_to_str()
        a = embedder(a)
        b = embedder(b)

        loss = torch.nn.functional.kl_div(
            torch.log_softmax(a, dim=0), b, reduction="sum"
        )
        return loss.item()

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


class CustomSynthesisTechnique(SynthesisTechnique):
    """
    A generalized template for creating SynthesisTechnique subclasses with dynamic prompts.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def generate_prompts(
        cls,
        technique_name: str,
        prompts_data: Dict[str, Dict[str, Union[List[str], Dict[str, List[str]]]]],
    ) -> Dict[str, Any]:
        """
        Generate prompts dynamically based on the provided parameters.

        Args:
        - technique_name (str): The name of the synthesis technique.
        - prompts_data (Dict[str, Dict[str, Union[List[str], Dict[str, List[str]]]]]): A dictionary containing prompt data.

        Returns:
        - prompts (Dict[str, Any]): A dictionary containing prompts.
        """
        prompts = {}

        for prompt, data in prompts_data.items():
            branching_options = data.get("branching_options", [])
            dynamic_prompts = data.get("dynamic_prompts", [])
            complex_diction = data.get("complex_diction", [])

            prompt_data = {
                "branching_options": branching_options,
                "dynamic_prompts": dynamic_prompts,
                "complex_diction": complex_diction,
            }

            prompts[prompt] = prompt_data

        return prompts

    def execute(self, *args, **kwargs) -> None:
        """
        Implement the execution logic of the synthesis technique.
        """
        pass

    def execute(self, *args, **kwargs) -> None:
        """
        [Execution details of the technique]
        """
        return super().execute(*args, **kwargs)


class BaseOperations(ChainCoordinate):
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        t: float = 0.0,
        n_parts: float = 0.0,
        **kwargs,
    ):
        super().__init__(x=x, y=y, z=z, t=t, n_parts=n_parts, **kwargs)

    def __getitem__(self, key: str) -> float:
        return self.fetch_value(key)

    def __setitem__(self, key: str, value: float) -> None:
        setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.dict())

    def __contains__(self, key: str) -> bool:
        return key in self.dict()

    def __iter__(self):
        return iter(self.dict().values())

    def fetch_value(self, field: str) -> float:
        """Fetch a value from the coordinate fields."""
        return getattr(self, field, 0.0)

    @classmethod
    def get_coordinate_fields(cls) -> List[str]:
        """Return names of the coordinate fields."""
        return [
            "x",
            "y",
            "z",
            "t",
            "n_parts",
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseOperations":
        """Initialize Coordinate from a dictionary."""
        return cls(**data)

    @staticmethod
    def get_coordinate_names() -> List[str]:
        """Return names of the coordinate dimensions."""
        return [
            "depth_x",
            "sibling_y",
            "sibling_count_z",
            "time_t",
            "n_parts",
        ]

    @classmethod
    def get_default_coordinates(cls) -> Dict[str, Any]:
        """Get the default values for the coordinates."""
        return {
            "depth_x": 0.0,
            "sibling_y": 0.0,
            "sibling_count_z": 0.0,
            "time_t": 0.0,
            "n_parts": 0.0,
        }

    @staticmethod
    def from_tuple(values: tuple) -> "BaseOperations":
        """Initialize Coordinate from a tuple."""
        return BaseOperations(
            x=values[0],
            y=values[1],
            z=values[2],
            t=values[3] if len(values) > 3 else 0.0,
            n_parts=values[4] if len(values) > 4 else 0.0,
        )

    @staticmethod
    def unflatten(values: np.ndarray) -> "BaseOperations":
        """Convert a flattened array back into a Coordinate."""
        return BaseOperations(
            x=values[0],
            y=values[1],
            z=values[2],
            t=values[3] if values.shape[0] > 3 else 0.0,
            n_parts=values[4] if values.shape[0] > 4 else 0.0,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "BaseOperations":
        """Initialize Coordinate from a PyTorch tensor."""
        return cls.unflatten(tensor.cpu().numpy())

    def to_reduced_array(self, exclude_t: bool = False) -> np.ndarray:
        """
        Convert the Coordinate object into a reduced numpy array representation, excluding n_parts.

        Args:
            exclude_t (bool, optional): If set to True, the 't' value will not be included in the array.
                                       Defaults to False.

        Returns:
            A numpy array representation of the Coordinate object.
        """
        if exclude_t:
            return np.array([self.x, self.y, self.z])
        else:
            return np.array([self.x, self.y, self.z, self.t])

    @staticmethod
    def to_tensor(
        coordinate: "BaseOperations", device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Convert Coordinate to a PyTorch tensor."""
        return torch.tensor(coordinate.to_reduced_array(), device=device)

    @staticmethod
    def flatten(coordinate: "BaseOperations") -> np.ndarray:
        """Flatten the Coordinate instance into a numpy array."""
        return np.array(
            [
                coordinate.text,
                coordinate.x,
                coordinate.y,
                coordinate.z,
                coordinate.t,
                coordinate.n_parts,
                coordinate.create_time,
            ]
        )

    def to_list(self) -> List[float]:
        """Convert Coordinate to list."""
        return [self.x, self.y, self.z, self.t, self.n_parts]

    def to_dict(self) -> dict:
        """Convert Coordinate to dict."""
        return self.dict()

    def tuple(self) -> tuple:
        """Convert Coordinate to tuple."""
        return tuple(self.dict().values())

    @staticmethod
    def flatten_list(coordinates: List["BaseOperations"]) -> np.ndarray:
        """Flatten a list of Coordinates."""
        return np.array([BaseOperations.flatten(c) for c in coordinates])

    @staticmethod
    def flatten_list_of_lists(coordinates: List[List["BaseOperations"]]) -> np.ndarray:
        """Flatten a list of lists of Coordinates."""

        flattened_coordinates = []
        for c in coordinates:
            flattened_coordinates.extend(c)

        return np.array([BaseOperations.flatten(c) for c in flattened_coordinates])

    @staticmethod
    def create_sequence_from_coordinates(
        coordinates: list, convert_to_string: bool = False
    ):
        sequence = []

        # Flatten the list of coordinates
        flattened_coordinates = BaseOperations.flatten_list(coordinates)

        def coordinate_to_string(
            coordinate: "BaseOperations", separator: str = ","
        ) -> str:
            """Convert Coordinate to string representation."""
            return np.array2string(coordinate, separator=separator)[1:-1]

        # Convert each flattened coordinate to string format, if required
        if convert_to_string:
            str_coordinates = [
                coordinate_to_string(fc, separator=",") for fc in flattened_coordinates
            ]
        else:
            str_coordinates = flattened_coordinates  # Keep original coordinates

        # Create the sequence of key-value pairs
        sequence = [(c.id, sc) for c, sc in zip(coordinates, str_coordinates)]

        return sequence

    @staticmethod
    def stack_coordinates(
        coordinates_dict: Dict[str, Union["BaseOperations", np.array]]
    ) -> np.array:
        """Stack coordinates from a dictionary."""
        return np.stack(list(coordinates_dict.values()), axis=0)

    @classmethod
    def batch_to_tensor(cls, batch: List["BaseOperations"]) -> torch.Tensor:
        """Convert a batch of Coordinates to a single tensor."""
        return torch.stack([coordinate.to_tensor() for coordinate in batch])

    def serialize(self) -> str:
        """Serialize the Coordinate object to a string."""
        return ",".join([str(x) for x in self])

    @classmethod
    def deserialize(cls, data: str) -> "BaseOperations":
        """Deserialize a string to a Coordinate object."""
        values = list(map(float, data.split(",")))
        return cls(
            x=values[0], y=values[1], z=values[2], t=values[3], n_parts=values[4]
        )

    def save(self, filename: str) -> None:
        """Save the serialized Coordinate object to a file."""
        with open(filename, "w") as f:
            f.write(self.serialize())

    @classmethod
    def load(cls, filename: str) -> "BaseOperations":
        """Load a Coordinate object from a file."""
        with open(filename, "r") as f:
            data = f.read().strip()
        return cls.deserialize(data)

    @staticmethod
    def from_tensor(coordinates_tensor: torch.Tensor) -> Dict[str, "BaseOperations"]:
        """
        Converts a PyTorch tensor into a dictionary of Coordinate objects.

        Args:
            coordinates_tensor: The PyTorch tensor to convert.

        Returns:
            A dictionary of Coordinate objects.
        """
        # Convert the PyTorch tensor to a numpy array.
        coordinates_array = coordinates_tensor.numpy()

        # Convert the numpy array to a dictionary of Coordinate objects.
        coordinates_dict = BaseOperations.from_array(coordinates_array)

        return coordinates_dict

    @staticmethod
    def from_array(coordinates_array: np.array) -> Dict[str, "BaseOperations"]:
        """
        Converts a numpy array into a dictionary of Coordinate objects.

        Args:
            coordinates_array: The numpy array to convert.

        Returns:
            A dictionary of Coordinate objects.
        """
        # Convert the numpy array to a list of Coordinate objects.
        coordinates_list = BaseOperations.from_array_to_list(coordinates_array)

        # Convert the list of Coordinate objects to a dictionary.
        coordinates_dict = BaseOperations.from_list(coordinates_list)

        return coordinates_dict

    @staticmethod
    def from_array_to_list(coordinates_array: np.array) -> List["BaseOperations"]:
        """
        Converts a numpy array into a list of Coordinate objects.

        Args:
            coordinates_array: The numpy array to convert.

        Returns:
            A list of Coordinate objects.
        """

        # Convert the numpy array to a list of Coordinate objects.
        coordinates_list = [
            BaseOperations.from_tuple(tuple(coord)) for coord in coordinates_array
        ]

        return coordinates_list

    @staticmethod
    def from_list(
        coordinates_list: List["BaseOperations"],
    ) -> Dict[str, "BaseOperations"]:
        """
        Converts a list of Coordinate objects into a dictionary.

        Args:
            coordinates_list: The list of Coordinate objects.

        Returns:
            A dictionary where the keys are the IDs of the Coordinate objects and the values are the Coordinate objects.
        """
        return {coordinate.id: coordinate for coordinate in coordinates_list}

    @staticmethod
    def list_to_dict(
        coordinates: List["BaseOperations"], flatten: bool = False
    ) -> Dict[str, Union["BaseOperations", np.array]]:
        """
        Convert a list of Coordinate objects into a dictionary.

        Args:
            coordinates: The list of Coordinate objects.
            flatten: A flag to determine if the Coordinate objects should be flattened.

        Returns:
            A dictionary where the keys are the IDs of the Coordinate objects and the values are the Coordinate objects
            or their flattened representations.
        """
        if flatten:
            return {
                coordinate.id: BaseOperations.flatten(coordinate)
                for coordinate in coordinates
            }
        else:
            return {coordinate.id: coordinate for coordinate in coordinates}

    @classmethod
    def flatten_coordinates_to_graph(
        cls,
        coordinates: List["BaseOperations"],
        edges: Optional[List[Tuple[str, str, float]]] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        depth: Optional[Dict[str, Any]] = None,
        siblings: Optional[Dict[str, Any]] = None,
    ) -> nx.Graph:
        """
        Flatten a list of coordinates and adds each as a node to a NetworkX graph.
        Adds edges, labels, metadata, depth and siblings information between the nodes in the graph.

        Args:
            coordinates: A list of coordinates.
            edges: A list of edges between the coordinates. Each edge is represented as a tuple (node1, node2, weight).
            labels: A dictionary with node labels.
            metadata: A dictionary with additional metadata for each node.
            depth: A dictionary with depth information for each node.
            siblings: A dictionary with siblings information for each node.

        Returns:
            A NetworkX graph with the flattened coordinates as nodes and edges, labels, metadata, depth and siblings information between the nodes.
        """
        # Create graph
        graph = nx.Graph()

        # Add nodes from the flattened coordinates
        for coordinate in coordinates:
            graph.add_node(coordinate.id, coordinate=coordinate)

        # Add edges
        if edges:
            graph.add_weighted_edges_from(edges)

        # Add labels
        if labels:
            nx.set_node_attributes(graph, labels, "label")

        # Add metadata
        if metadata:
            nx.set_node_attributes(graph, metadata, "metadata")

        # Add depth
        if depth:
            nx.set_node_attributes(graph, depth, "depth")

        # Add siblings
        if siblings:
            nx.set_node_attributes(graph, siblings, "siblings")

        return graph

    @staticmethod
    def create_tree(
        root: str, connections: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Creates a tree structure.

        Args:
            root (str): The root node.
            connections (Dict[str, List[str]]): Dictionary representing the connections between nodes.

        Returns:
            Dict[str, List[str]]: A tree structure.
        """
        tree = {root: []}

        for parent, children in connections.items():
            tree[parent] = children
            for child in children:
                if child not in tree:
                    tree[child] = []

        return tree

    @staticmethod
    def create_graph(
        root: str,
        connections: Dict[str, List[str]],
        coordinates: List["BaseOperations"],
        edges: Optional[List[Tuple[str, str, float]]] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        depth: Optional[Dict[str, Any]] = None,
        siblings: Optional[Dict[str, Any]] = None,
    ) -> nx.Graph:
        """
        Creates a NetworkX graph.

        Args:
            root (str): The root node.
            connections (Dict[str, List[str]]): Dictionary representing the connections between nodes.
            coordinates: The list of Coordinate objects.
            edges: A list of edges between the coordinates. Each edge is represented as a tuple (node1, node2, weight).
            labels: A dictionary with node labels.
            metadata: A dictionary with additional metadata for each node.
            depth: A dictionary with depth information for each node.
            siblings: A dictionary with siblings information for each node.

        Returns:
            A NetworkX graph.
        """
        # Create tree
        tree = BaseOperations.create_tree(root, connections)

        graph = BaseOperations.flatten_coordinates_to_graph(
            coordinates, edges, labels, metadata, depth, siblings
        )
        return graph, tree


class RoleChain(ComponentModel):
    id: str = Field(..., description="Unique identifier for the chain.")
    author: Author = Field(..., description="The author of the chain.")
    content: Content = Field(..., description="The content of the chain.")
    coordinate: ChainCoordinate = Field(
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

    def __str__(self):
        return f"{self.author.role} Chain: {self.content.parts[0]}"

    def __len__(self):
        return len(self.children)

    def get_text(self):
        return self.content.parts.raw


class AssistantChain(RoleChain):
    def __init__(
        self,
        id: str,
        content: Content,
        coordinate: BaseOperations,
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
        id: str,
        content: Content,
        coordinate: BaseOperations,
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
        id: str,
        content: Content,
        coordinate: BaseOperations,
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
