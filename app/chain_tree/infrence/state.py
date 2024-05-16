from typing import Optional, Dict, Any, Callable, List, Union
from chain_tree.models import (
    Chain,
    ChainMap,
    Author,
    ChainCoordinate,
    Content,
)
from pydantic import BaseModel, Field
from collections import defaultdict
from chain_tree.utils import (
    filter_none_values,
    load_json,
    save_json,
)
import datetime
import random
import uuid
import json
import os


class BaseState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    mappings: Union[Dict[str, ChainMap], Dict[str, Any]] = Field(default_factory=dict)
    ancestors = defaultdict(set)
    title: Optional[str] = Field(default=None)

    root: Optional[str] = Field(default=None)
    timeout: Optional[int] = Field(default=None)
    last_interaction_time: float = Field(
        default_factory=lambda: datetime.datetime.now().timestamp()
    )

    history: List[Chain] = Field(default_factory=list)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.mappings = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateMachine":
        """Create a StateMachine from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "StateMachine":
        """Create a StateMachine from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, file_path: str) -> "StateMachine":
        """Create a StateMachine from a JSON file."""
        with open(file_path, "r") as file:
            return cls.from_json(file.read())

    def _validate_relationship(self, relationship: str) -> bool:
        return relationship in {"before", "after", "child"}

    def _validate_offset(self, parent: Optional[str], offset: Optional[int]) -> bool:
        if parent is None:
            return offset is None
        else:
            return offset is not None

    def create_conversation(
        self,
        title,
        convo: Dict[str, Any],
        folder_path: str = "/Users/mohameddiomande/Desktop/conversation",
    ) -> Dict[str, Any]:
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save to json file
        save_json(f"{folder_path}/{title}.json", [convo])

        return convo

    def delete_conversation(self, title) -> None:
        """Delete a conversation from json file."""
        os.remove(f"/Users/mohameddiomande/Desktop/conversation/{title}.json")

    def update_interaction_time(self) -> None:
        date = datetime.datetime.now()
        # make date json serializable
        date = date.isoformat()
        self.last_interaction_time = date

    def reset_timeout(self) -> None:
        self.update_interaction_time()

    def get_history(self) -> List[Chain]:
        return [
            mapping.message.content.parts
            for message_id, mapping in self.mappings.items()
            if message_id != self.root
        ]

    def _get_message_node(self, message_id: str) -> ChainMap:
        """Retrieve the Mapping corresponding to a given message ID."""
        if message_id not in self.mappings:
            raise ValueError(
                f"Message ID '{message_id}' not found in the conversation."
            )
        return self.mappings[message_id]

    def get_message(self, message_id: str) -> Chain:
        return self._get_message_node(message_id).message

    def _remove_message_from_parent(self, message_id: str) -> None:
        """Remove a message from its parent's children list."""
        message_node = self._get_message_node(message_id)
        parent_id = message_node.parent

        if parent_id is None:
            raise ValueError(f"Cannot modify the root message.")

        siblings = self.mappings[parent_id].children

        if message_id not in siblings:
            raise ValueError(
                f"Message ID '{message_id}' is not a sibling of the parent '{parent_id}'."
            )

        siblings.remove(message_id)

    def _get_parent_id(self) -> str:
        parent_id = self.root
        while True:
            parent_node = self.mappings[parent_id]
            if len(parent_node.children) == 0:
                return parent_id
            parent_id = parent_node.children[-1]

    def get_last_message_id(self) -> str:
        """Return the ID of the last message in the conversation."""
        return self._get_parent_id()

    def get_last_message(self) -> Chain:
        """Return the last message in the conversation."""
        return self.get_message(self.get_last_message_id())

    def is_finished(self) -> bool:
        """Return whether the conversation is finished."""
        return self.get_last_message().end_turn

    def end_conversation(self) -> None:
        """Remove all messages from the conversation."""
        self.mappings = {self.root: self.mappings[self.root]}
        self.last_interaction_time = datetime.datetime.now()

    def restart_conversation(self) -> None:
        """Remove all messages from the conversation except the root."""
        self.mappings = {
            self.root: self.mappings[self.root],
            self.mappings[self.root].children[-1]: self.mappings[
                self.mappings[self.root].children[-1]
            ],
        }
        self.last_interaction_time = datetime.datetime.now()

    def get_truncated_history(
        self, max_history_length: int, include_current_state: bool = True
    ) -> List[Chain]:
        """Return a truncated version of the conversation history."""
        history = self.get_history()
        if not include_current_state:
            history = history[:-1]
        if len(history) > max_history_length:
            history = history[-max_history_length:]
        return history

    def rewind_conversation(self, steps: int = 1) -> None:
        """Removes the last n messages from the conversation."""
        for _ in range(steps):
            last_message_id = self.get_last_message_id()
            if last_message_id == self.root:
                break
            self._remove_message_from_parent(last_message_id)
            del self.mappings[last_message_id]
        self.update_interaction_time()
        self.get_last_message().end_turn = False
        self.update_conversation(self.title)

    def print_conversation(self) -> None:
        """Print the conversation history."""
        for message in self.get_history():
            print(message)

    def get_messages(self) -> List[str]:
        """Return all text content of messages in the conversation."""
        return [
            mapping.message.content.text
            for mapping in self.mappings.values()
            if mapping.message.content and mapping.message.content.text
        ]

    def delete_message(self, message_id: str) -> None:
        self._remove_message_from_parent(message_id)
        del self.mappings[message_id]
        self.update_interaction_time()

    def update_message(
        self,
        message_id: str,
        content: Optional[Content] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        message_node = self._get_message_node(message_id)

        if content is not None:
            message_node.message.content = content

        if metadata_updates is not None:
            message_node.message.metadata.update(metadata_updates)

        self.update_interaction_time()

    def move_message(
        self,
        message_id: str,
        new_parent_id: str,
        new_sibling_index: Optional[int] = None,
    ) -> None:
        message_node = self._get_message_node(message_id)
        self._remove_message_from_parent(message_id)
        message_node.parent = new_parent_id

        if new_sibling_index is None:
            new_sibling_index = len(self.mappings[new_parent_id].children)

        self.mappings[new_parent_id].children.insert(new_sibling_index, message_id)
        self.update_interaction_time()

    def load_conversation(self, title: str, conversation_id: str) -> None:
        """Load a conversation from json file using the title and ensure conversation_id is not in the title."""

        def sanitize_filename(filename: str) -> str:
            """Remove unwanted characters and strip whitespace from a filename."""
            return "".join(
                char for char in filename if char.isalnum() or char in "._- "
            ).strip()

        # If the conversation_id is part of the title, remove it
        if conversation_id in title:
            title = title.replace(conversation_id, "").strip()

        # Sanitize the title to create a valid filename
        sanitized_title = sanitize_filename(title)
        file_path = (
            f"/Users/mohameddiomande/Desktop/conversation/{sanitized_title}.json"
        )

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The conversation file {file_path} does not exist."
            )

        # Load the conversation
        convo = load_json(file_path)[0]
        # Here, check if the loaded conversation_id matches the provided one
        if convo["conversation_id"] != conversation_id:
            raise ValueError(
                f"The loaded conversation ID does not match the provided ID: {conversation_id}"
            )

        self.conversation_id = convo["conversation_id"]
        self.mappings = {
            message_id: ChainMap(
                id=message_id,
                message=Chain(**message_data["message"]),
                parent=message_data["parent"],
                children=message_data["children"],
                references=message_data["references"],
                relationships=message_data["relationships"],
            )
            for message_id, message_data in convo["mappings"].items()
        }
        self.root = convo["root"]
        self.last_interaction_time = convo["last_interaction_time"]

        return self

    def save_conversation(
        self,
        title: str = None,
        folder_path: str = "/Users/mohameddiomande/Desktop/conversation",
    ) -> None:
        """Return a dictionary representation of the conversation."""
        if title is None:
            title = self.title

        else:
            self.title = title

        convo = {
            "title": title,
            "id": self.conversation_id,
            "created_time": self.mappings[self.root].message.create_time,
            "updated_time": self.last_interaction_time,
            "mappings": {
                message_id: {
                    "message": filter_none_values(mapping.message.to_dict()),
                    "parent": mapping.parent,
                    "children": mapping.children,
                    "references": mapping.references,
                    "relationships": mapping.relationships,
                }
                for message_id, mapping in self.mappings.items()
            },
            "current_node": self.mappings[self.root].message.id,
            "moderation": [],
        }

        # Create conversation json file
        convo = self.create_conversation(title, convo, folder_path)

        return convo

    def update_conversation(self, title: str) -> None:
        """Update a conversation in json file."""
        convo = self.save_conversation(title)


class StateMachine(BaseState):
    @classmethod
    def from_conversation(
        cls,
        conversation_id: str,
        conversation: "StateMachine",
        timeout: Optional[int] = None,
    ) -> "StateMachine":
        """Create a StateMachine from an existing conversation."""
        state_machine = cls(conversation_id, timeout)
        state_machine.mappings = conversation.mappings
        state_machine.root = conversation.root
        state_machine.last_interaction_time = conversation.last_interaction_time
        return state_machine

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the StateMachine."""
        return {
            "conversation_id": self.conversation_id,
            "mappings": {
                message_id: {
                    "message": filter_none_values(mapping.message.dict()),
                    "parent": mapping.parent,
                    "children": mapping.children,
                    "references": mapping.references,
                    "relationships": mapping.relationships,
                }
                for message_id, mapping in self.mappings.items()
            },
            "root": self.root,
            "timeout": self.timeout,
            "last_interaction_time": self.last_interaction_time,
        }

    def add_conversation(
        self,
        conversation: "StateMachine",
        relationship: str = "child",
        parent: Optional[str] = None,
        offset: Optional[int] = None,
    ) -> None:
        """Add a conversation to the current conversation."""

        # Validate parent ID, relationship, and offset
        if parent is not None and parent not in self.mappings:
            raise ValueError(f"Parent ID '{parent}' not found in the conversation.")
        if not self._validate_relationship(relationship):
            raise ValueError(f"Invalid relationship '{relationship}'.")
        if (
            parent
            and offset is not None
            and (offset < 0 or offset > len(self.mappings[parent]["children"]))
        ):
            raise ValueError(f"Invalid offset '{offset}' for parent ID '{parent}'.")

        # Validate that the conversation can be added without ID conflicts or creating cycles
        for message_id in conversation.mappings:
            if message_id in self.mappings:
                raise ValueError(
                    f"Message ID '{message_id}' already exists in the current conversation."
                )
            if self._creates_cycle(message_id, parent):
                raise ValueError("Adding this conversation would create a cycle.")

        # Establish the relationship between the new conversation's root and the current conversation
        if parent is None:
            if self.root is not None:
                raise ValueError("The current conversation already has a root.")
            self.root = conversation.root
        else:
            self._insert_into_parent(parent, conversation.root, relationship, offset)

        # Merge the mappings from the new conversation into the current one, updating parent IDs where necessary
        for message_id, conv_mapping in conversation.mappings.items():
            new_mapping = conv_mapping.copy()
            if new_mapping["parent"] is not None:
                new_mapping["parent"] = self._update_parent_id(
                    new_mapping["parent"], conversation
                )
            self.mappings[message_id] = new_mapping

        self.last_interaction_time = datetime.datetime.now().timestamp()

    def _insert_into_parent(
        self,
        parent_id: str,
        child_id: str,
        relationship: str,
        offset: Optional[int] = None,
    ) -> None:
        """
        Inserts a child node into the parent node's children list based on the relationship and offset.
        :param parent_id: The ID of the parent node.
        :param child_id: The ID of the child node to insert.
        :param relationship: The relationship of the child to the parent ("child", "before", "after").
        :param offset: The position at which to insert the child node relative to the parent's other children.
        """
        # Ensure the parent exists
        if parent_id not in self.mappings:
            raise ValueError(f"Parent ID '{parent_id}' not found in the conversation.")

        # Retrieve the parent's children list
        parent_children = self.mappings[parent_id].get("children", [])

        # Determine the insertion index based on the relationship and offset
        if relationship == "child":
            # If no offset is provided or offset is greater than the number of children,
            # insert the child at the end of the children list.
            index = len(parent_children) if offset is None else offset
        elif relationship in ("before", "after"):
            if parent_id not in self.mappings:
                raise ValueError(
                    f"Parent ID '{parent_id}' not found in the conversation."
                )

            # Find the index of the parent in its own parent's children list
            grandparent_id = self.mappings[parent_id].get("parent")
            if grandparent_id is None:
                raise ValueError(f"No grandparent found for parent ID '{parent_id}'.")

            parent_index = (
                self.mappings[grandparent_id].get("children", []).index(parent_id)
            )
            index = parent_index + (1 if relationship == "after" else 0)
        else:
            raise ValueError(f"Invalid relationship '{relationship}'.")

        # Adjust the index if an offset is provided for "before" or "after" relationships
        if relationship in ("before", "after") and offset is not None:
            index += offset

        # Insert the child ID at the determined index
        parent_children.insert(index, child_id)
        self.mappings[parent_id]["children"] = parent_children

    def _update_parent_id(self, parent_id: str, conversation: "StateMachine") -> str:
        """
        Updates the parent ID of a node in the new conversation to the corresponding ID in the current conversation.
        :param parent_id: The parent ID to update.
        :param conversation: The new conversation being added.
        :return: The updated parent ID.
        """
        # If the parent ID is the root of the new conversation, return the root of the current conversation
        if parent_id == conversation.root:
            return self.root

        # Otherwise, find the parent ID in the new conversation's mappings and return the corresponding ID in the current conversation
        for message_id, mapping in conversation.mappings.items():
            if message_id == parent_id:
                return self.mappings[message_id]["id"]

        raise ValueError(f"Parent ID '{parent_id}' not found in the new conversation.")

    def _creates_cycle(self, new_child_id: str, parent_id: Optional[str]) -> bool:
        """
        Checks if adding a new message with the given ID under the specified parent ID would create a cycle.
        :param new_child_id: The ID of the new message to be added.
        :param parent_id: The ID of the parent under which the new message would be added.
        :return: True if a cycle would be created; False otherwise.
        """
        visited = set()

        def visit(node_id: str) -> bool:
            """Depth-first search to detect a cycle."""
            if node_id in visited:
                return False  # Already visited, no cycle detected in this path

            if node_id == new_child_id:
                return True  # Cycle detected

            visited.add(node_id)

            node_children = self.mappings.get(node_id, {}).get("children", [])
            for child_id in node_children:
                if visit(child_id):
                    return True  # Cycle detected in children

            visited.remove(node_id)
            return False

        # Start the cycle check with the intended parent
        return visit(parent_id)

    def _get_parent_id(self) -> str:
        """Return the ID of the last message in the conversation."""
        parent_id = self.root
        while True:
            parent_node = self.mappings[parent_id]
            if len(parent_node.children) == 0:
                return parent_id
            parent_id = parent_node.children[-1]

    def _create_message(
        self,
        id: str,
        author: Optional[Author],
        content: Content,
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[List[float]],
        weight: int,
        end_turn: Optional[bool],
        recipient: Optional[str],
        coordinate: Optional[ChainCoordinate] = None,
    ) -> Chain:
        return Chain(
            id=id,
            author=author,
            create_time=get_current_timestamp(),
            content=content,
            metadata=metadata,
            embedding=embedding,
            weight=weight,
            end_turn=end_turn,
            recipient=recipient,
            coordinate=coordinate,
        )

    def add_message(
        self,
        message_id: str,
        author: Author,
        content: Content,
        relationship: str = "child",
        parent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: int = 1,
        end_turn: Optional[bool] = None,
        recipient: Optional[str] = None,
        coordinate: Optional[ChainCoordinate] = None,
        offset: Optional[int] = 0,  # default value for offset
    ) -> None:
        # Validate weight and relationship
        if weight <= 0:
            raise ValueError("Weight must be a positive integer.")
        if not self._validate_relationship(relationship):
            raise ValueError(f"Invalid relationship '{relationship}'.")

        # Create the message and its mapping
        message = self._create_message(
            message_id,
            author,
            content,
            metadata,
            embedding,
            weight,
            end_turn,
            recipient,
            coordinate,
        )

        mapping = ChainMap(id=message_id, message=message, parent=parent, children=[])

        # Set as root if no parent is provided
        if parent is None:
            if self.root is not None:
                raise ValueError("Cannot set multiple root messages in a DAG.")
            self.root = message_id

        else:
            # Check if parent exists
            if parent not in self.mappings:
                raise ValueError(f"Parent ID '{parent}' not found in the conversation.")

            # Ensure no cycles would be created
            if message_id in self.ancestors[parent]:
                raise ValueError(f"Cannot add message as it would create a cycle.")

            # Add message based on relationship
            if parent is not None:
                self._insert_message(parent, message_id, relationship, offset)

            else:
                self.mappings[parent].children.append(message_id)
            # Update ancestors for the new message
            self.ancestors[message_id].add(parent)
            for ancestor in self.ancestors[parent]:
                self.ancestors[message_id].add(ancestor)

        self.mappings[message_id] = mapping
        self.last_interaction_time = datetime.datetime.now().timestamp()

    def _insert_message(self, parent, message_id, relationship, offset):
        if relationship == "child":
            # check if offset is within bounds
            if offset is None or offset > len(self.mappings[parent].children):
                self.mappings[parent].children.append(message_id)
            else:
                self.mappings[parent].children.insert(offset, message_id)
        elif relationship == "before":
            parent_index = self.mappings[self.mappings[parent].parent].children.index(
                parent
            )
            self.mappings[self.mappings[parent].parent].children.insert(
                parent_index + offset, message_id
            )
        elif relationship == "after":
            parent_index = self.mappings[self.mappings[parent].parent].children.index(
                parent
            )
            self.mappings[self.mappings[parent].parent].children.insert(
                parent_index + offset + 1, message_id
            )

    def regenerate_message(
        self,
        message_id: str,
        content: Optional[Content] = None,
        new_weight: Optional[int] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        message_node = self._get_message_node(message_id)
        content = content if content is not None else message_node.message.content
        new_weight = (
            new_weight if new_weight is not None else message_node.message.weight
        )
        metadata_updates = (
            metadata_updates
            if metadata_updates is not None
            else message_node.message.metadata
        )

        self.add_message(
            message_id,
            message_node.message.author,
            content,
            message_node.parent,
            metadata_updates,
            new_weight,
            message_node.message.end_turn,
            message_node.message.recipient,
        )

        self.delete_message(message_id)

    def merge(
        self,
        other: "StateMachine",
        offset: Optional[int] = None,
        merge_operation: Optional[Callable[[Chain, Chain], Chain]] = None,
    ) -> None:
        if other is self:
            raise ValueError("Cannot merge a conversation with itself.")

        if other.root in self.mappings:
            raise ValueError(
                f"Conversation already contains a message with ID '{other.root}'."
            )

        if not self._validate_offset(self.root, offset):
            raise ValueError(f"Invalid offset '{offset}' for parent ID '{self.root}'.")

        root_mapping = self.mappings[self.root]
        other_root_mapping = other.mappings[other.root]

        self.mappings.update(other.mappings)

        if offset is None:
            root_mapping.children.append(other.root)
        else:
            root_mapping.children.insert(offset, other.root)

        for message_id, mapping in other.mappings.items():
            if message_id != self.root:
                mapping.children = [
                    other.mappings[child_id].id for child_id in mapping.children
                ]

        if other_root_mapping.message.author.role == "user":
            self.reset_timeout()

        self.last_interaction_time = datetime.datetime.now()

        if merge_operation is not None:
            merge_operation(root_mapping.message, other_root_mapping.message)

        return self

    def split(self, message_id: str, offset: Optional[int] = None) -> "StateMachine":
        if message_id not in self.mappings:
            raise ValueError(
                f"Message ID '{message_id}' not found in the conversation."
            )

        if not self._validate_offset(message_id, offset):
            raise ValueError(f"Invalid offset '{offset}' for parent ID '{message_id}'.")

        message_node = self.mappings[message_id]
        parent_id = message_node.parent

        if parent_id is None:
            raise ValueError(f"Cannot split the root message.")

        parent_node = self.mappings[parent_id]
        siblings = parent_node.children

        if message_id not in siblings:
            raise ValueError(
                f"Message ID '{message_id}' is not a sibling of the parent '{parent_id}'."
            )

        sibling_index = siblings.index(message_id)

        new_conversation = StateMachine(
            conversation_id=str(uuid.uuid4()), timeout=self.timeout
        )
        new_conversation.root = message_id

        for sibling_id in reversed(siblings[sibling_index:]):
            sibling_node = self.mappings.pop(sibling_id)
            new_conversation.mappings[sibling_id] = sibling_node
            if sibling_node.parent == message_id:
                sibling_node.parent = None
            sibling_node.children = [
                child_id
                for child_id in sibling_node.children
                if child_id not in siblings
            ]

        new_conversation.mappings[message_id].children = siblings[sibling_index + 1 :]
        del siblings[sibling_index:]

        self.last_interaction_time = datetime.datetime.now()

        return new_conversation

    def randomize(
        self, message_id: str, offset: Optional[int] = None
    ) -> "StateMachine":
        if message_id not in self.mappings:
            raise ValueError(
                f"Message ID '{message_id}' not found in the conversation."
            )

        if not self._validate_offset(message_id, offset):
            raise ValueError(f"Invalid offset '{offset}' for parent ID '{message_id}'.")

        message_node = self.mappings[message_id]
        parent_id = message_node.parent

        if parent_id is None:
            raise ValueError(f"Cannot randomize the root message.")

        parent_node = self.mappings[parent_id]

        siblings = parent_node.children

        if message_id not in siblings:
            raise ValueError(
                f"Message ID '{message_id}' is not a sibling of the parent '{parent_id}'."
            )

        random.shuffle(siblings)
        parent_node.children = siblings

        self.last_interaction_time = datetime.datetime.now()
        return self


def get_current_timestamp() -> float:
    """Return the current timestamp."""
    return datetime.datetime.now().timestamp()
