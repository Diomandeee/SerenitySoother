from typing import Optional, Dict, Any, Callable, List, Union
from concurrent.futures import ThreadPoolExecutor, Future
from sqlalchemy.ext.asyncio import AsyncSession
from app.helper import filter_none_values
from pydantic import BaseModel, Field
from app.dependencies import get_db
from collections import defaultdict
from app.models import ChainTree
from app.chain_tree.schemas import (
    Chain,
    ChainMap,
    Author,
    ChainCoordinate,
    Content,
)
from fastapi import Depends
import datetime
import aiofiles
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
    async def from_dict(cls, data: Dict[str, Any]) -> "StateMachine":
        """Create a StateMachine from a dictionary."""
        return cls(**data)

    @classmethod
    async def from_json(cls, json_str: str) -> "StateMachine":
        """Create a StateMachine from a JSON string."""
        return await cls.from_dict(json.loads(json_str))

    @classmethod
    async def from_file(cls, file_path: str) -> "StateMachine":
        """Create a StateMachine from a JSON file."""
        async with aiofiles.open(file_path, "r") as file:
            content = await file.read()
            return await cls.from_json(content)

    def _validate_relationship(self, relationship: str) -> bool:
        return relationship in {"before", "after", "child"}

    def _validate_offset(self, parent: Optional[str], offset: Optional[int]) -> bool:
        if parent is None:
            return offset is None
        else:
            return offset is not None

    async def create_conversation(
        self, session: AsyncSession, title: str
    ) -> Dict[str, Any]:
        """Create a conversation and save it to the database."""
        self.title = title
        self.last_interaction_time = datetime.datetime.now().timestamp()

        conversation = ChainTree(
            id=self.conversation_id,
            title=self.title,
            create_time=self.last_interaction_time,
            update_time=self.last_interaction_time,
            mapping={},
            current_node=None,
        )
        session.add(conversation)
        await session.commit()

        return {
            "id": self.conversation_id,
            "title": self.title,
            "create_time": self.last_interaction_time,
        }

    def update_interaction_time(self) -> None:
        self.last_interaction_time = datetime.datetime.now().timestamp()

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
        self.update_conversation()

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

    async def load_conversation(
        self, session: AsyncSession, conversation_id: str
    ) -> None:
        """Load a conversation from the database using SQLAlchemy."""
        async with session.begin():
            conversation = await session.get(ChainTree, conversation_id)
            if not conversation:
                raise ValueError(f"Conversation with ID '{conversation_id}' not found.")

            self.conversation_id = conversation.id
            self.title = conversation.title
            self.last_interaction_time = conversation.update_time
            self.mappings = {
                message.id: ChainMap(
                    id=message.id,
                    message=Chain.from_orm(message),
                    parent=message.parent_id,
                    children=[child.id for child in message.children],
                    references=message.references,
                    relationships=message.relationships,
                )
                for message in conversation.mapping.values()
            }
            self.root = conversation.current_node

    async def save_conversation(self, session: AsyncSession) -> ChainTree:
        """Save the current state of the conversation to the database using SQLAlchemy."""
        mapping_dict = {
            message_id: {
                "message": filter_none_values(mapping.message.dict()),
                "parent": mapping.parent,
                "children": mapping.children,
                "references": mapping.references,
                "relationships": mapping.relationships,
            }
            for message_id, mapping in self.mappings.items()
        }

        async with session.begin():
            conversation = await session.get(ChainTree, self.conversation_id)
            if not conversation:
                conversation = ChainTree(
                    id=self.conversation_id,
                    title=self.title,
                    create_time=self.mappings[self.root].message.create_time,
                    update_time=self.last_interaction_time,
                    mapping=mapping_dict,
                    current_node=self.root,
                )
                session.add(conversation)
            else:
                conversation.title = self.title
                conversation.update_time = self.last_interaction_time
                conversation.mapping = mapping_dict
                conversation.current_node = self.root

            await session.commit()
            return conversation

    async def update_conversation(self, session: AsyncSession) -> None:
        """Update a conversation in the database using SQLAlchemy."""
        await self.save_conversation(session)


class StateMachine(BaseState):
    @classmethod
    async def from_conversation(
        cls,
        conversation_id: str,
        conversation: "StateMachine",
        timeout: Optional[int] = None,
    ) -> "StateMachine":
        """Create a StateMachine from an existing conversation."""
        state_machine = cls(conversation_id=conversation_id, timeout=timeout)
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

    async def add_conversation(
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

    async def add_message(
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

    async def regenerate_message(
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

        await self.add_message(
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

    async def merge(
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

    async def split(
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

    async def randomize(
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


class ChainManager(BaseModel):
    conversations: Dict[str, StateMachine] = Field(
        {}, description="A dictionary mapping conversation IDs to conversations."
    )
    engine: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def conversation_exists(self, conversation_id: str) -> bool:
        return conversation_id in self.conversations

    async def start_conversation(self, initial_message: str) -> str:
        conversation_id = await self.create_conversation()
        await self.handle_system_message(conversation_id, initial_message)
        return conversation_id

    async def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        conversation = StateMachine(conversation_id=conversation_id)
        self.conversations[conversation_id] = conversation
        return conversation_id

    async def add_conversation(self, conversation: StateMachine) -> None:
        if not isinstance(conversation, StateMachine):
            raise TypeError(
                f"Expected 'StateMachine' object, got '{type(conversation).__name__}'."
            )

        if conversation.conversation_id in self.conversations:
            raise ValueError(
                f"Conversation with ID '{conversation.conversation_id}' already exists."
            )

        self.conversations[conversation.conversation_id] = conversation

    async def get_conversation(self, conversation_id: str) -> StateMachine:
        if conversation_id not in self.conversations:
            raise ValueError(
                f"Conversation with ID '{conversation_id}' does not exist."
            )

        return self.conversations[conversation_id]

    async def rewind_conversation(self, conversation_id: str, steps: int = 1) -> None:
        conversation = await self.get_conversation(conversation_id)
        await conversation.rewind_conversation(steps)

    async def print_conversation(self, conversation_id: str) -> None:
        conversation = await self.get_conversation(conversation_id)
        await conversation.print_conversation()

    async def end_conversation(self, conversation_id: str) -> None:
        conversation = await self.get_conversation(conversation_id)
        await conversation.end_conversation()

    async def restart_conversation(self, conversation_id: str) -> None:
        conversation = await self.get_conversation(conversation_id)
        await conversation.restart_conversation()

    async def get_conversation_history(
        self, conversation_id: str
    ) -> List[Dict[str, str]]:
        conversation = await self.get_conversation(conversation_id)
        return conversation.get_history()

    async def add_message(
        self,
        conversation_id: str,
        message_id: str,
        content: Content,
        author: Author,
        coordinate: Optional[ChainCoordinate] = None,
        embedding: List[float] = None,
        parent: str = None,
        save: bool = False,
        db: AsyncSession = Depends(get_db),
    ) -> None:
        conversation = await self.get_conversation(conversation_id)
        conversation.add_message(
            message_id=message_id,
            content=content,
            author=author,
            embedding=embedding,
            parent=parent,
            coordinate=coordinate,
        )
        if save:
            await conversation.save_conversation(db)

    async def update_message(
        self, conversation_id: str, message_id: str, new_message: Chain
    ) -> None:
        conversation = await self.get_conversation(conversation_id)
        conversation.update_message(message_id, new_message)

    async def delete_message(self, conversation_id: str, message_id: str) -> bool:
        conversation = await self.get_conversation(conversation_id)
        return conversation.delete_message(message_id)

    async def get_message(self, conversation_id: str, message_id: str) -> Chain:
        conversation = await self.get_conversation(conversation_id)
        return conversation.get_message(message_id)

    async def move_message(
        self, conversation_id: str, message_id: str, new_parent_id: str
    ) -> None:
        conversation = await self.get_conversation(conversation_id)
        conversation.move_message(message_id, new_parent_id)

    async def merge_conversations(
        self, conversation_id_1: str, conversation_id_2: str, db: AsyncSession
    ) -> None:
        conversation_1 = await self.get_conversation(conversation_id_1)
        conversation_2 = await self.get_conversation(conversation_id_2)

        conversation_1.merge(conversation_2)
        await self.delete_conversation(conversation_id_2)

    async def get_conversations(self) -> List[StateMachine]:
        return list(self.conversations.values())

    async def get_conversation_ids(self) -> List[str]:
        return list(self.conversations.keys())

    async def get_conversation_titles(self) -> List[str]:
        return [conv.title for conv in self.conversations.values()]

    async def get_conversation_titles_and_ids(self) -> List[Dict[str, str]]:
        return [
            {"title": conv.title, "id": conv.conversation_id}
            for conv in self.conversations.values()
        ]

    async def delete_conversation(self, conversation_id: str) -> None:
        if conversation_id not in self.conversations:
            raise ValueError(
                f"Conversation with ID '{conversation_id}' does not exist."
            )
        del self.conversations[conversation_id]

    async def delete_all_conversations(self) -> None:
        self.conversations = {}

    async def cleanup_inactive_conversations(
        self, inactivity_threshold_in_hours: int = 1
    ) -> None:
        current_time = datetime.datetime.now().timestamp()
        inactive_conversations = []

        # identify inactive conversations
        for conversation_id, conversation in self.conversations.items():
            time_since_last_interaction = (
                current_time - conversation.last_interaction_time
            )
            if time_since_last_interaction > inactivity_threshold_in_hours * 60 * 60:
                inactive_conversations.append(conversation_id)

        # remove inactive conversations
        for conversation_id in inactive_conversations:
            del self.conversations[conversation_id]

    async def export_conversations_to_json(self) -> str:
        conversations_data = [conv.to_dict() for conv in self.conversations.values()]
        return json.dumps(conversations_data, indent=2)

    async def import_conversations_from_json(self, json_data: str) -> None:
        try:
            conversations_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")

        if not isinstance(conversations_data, list):
            raise ValueError("JSON data should be a list of conversation dictionaries.")

        for conv_data in conversations_data:
            try:
                conversation = StateMachine.from_dict(conv_data)
            except ValueError as e:
                raise ValueError(f"Invalid conversation data: {e}")

            await self.add_conversation(conversation)

    async def _add_message(
        self,
        conversation_ids: Union[str, List[str]],
        messages: Union[str, List[str]],
        author_type: str,
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        if isinstance(conversation_ids, str):
            conversation_ids = [conversation_ids]

        if isinstance(messages, str):
            messages = [messages]

        if isinstance(parent_ids, str):
            parent_ids = [parent_ids]
        elif parent_ids is None:
            parent_ids = [None] * len(messages)

        for conversation_id in conversation_ids:
            for message, parent_id in zip(messages, parent_ids):
                message_id = str(uuid.uuid4())

                content = Content(text=message)

                author = Author(role=author_type)

                await self.add_message(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    content=content,
                    author=author,
                    parent=parent_id,
                )

    async def handle_user_input(
        self,
        conversation_ids: Union[str, List[str]],
        user_input: Union[str, List[str]],
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        await self._add_message(conversation_ids, user_input, "user", parent_ids)

    async def handle_agent_response(
        self,
        conversation_ids: Union[str, List[str]],
        agent_response: Union[str, List[str]],
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        await self._add_message(conversation_ids, agent_response, "agent", parent_ids)

    async def handle_system_message(
        self,
        conversation_ids: Union[str, List[str]],
        system_message: Union[str, List[str]],
        parent_ids: Union[str, List[str], None] = None,
    ) -> None:
        await self._add_message(conversation_ids, system_message, "system", parent_ids)

    async def get_messages(self, conversation_id: str) -> List[Chain]:
        conversation = await self.get_conversation(conversation_id)
        return conversation.get_messages()

    async def load_conversation(self, conversation_id: str, title: str = "Untitled"):
        """Load a conversation from json file"""
        conversation = await self.get_conversation(conversation_id)
        await conversation.load_conversation(title)

    async def save_conversation(self, conversation_id: str):
        """Save a conversation to json file"""
        conversation = await self.get_conversation(conversation_id)
        await conversation.save_conversation()

    async def delete_conversation(self, conversation_id: str) -> None:
        await self.delete_conversation(conversation_id)

    async def _generic_creation(
        self,
        prompt: Optional[str],
        generation_function: Callable[[Optional[str], Optional[Dict[str, Any]]], Any],
        creation_function: Callable[[str], Any],
        upload: bool = False,
        response: Optional[str] = None,
        **kwargs,
    ) -> str:
        conversation_id = await self.create_conversation()

        generated_parts = await generation_function(
            prompt, response, conversation_id=conversation_id, **kwargs
        )
        # If the generation_function returns a Chain object, access the content
        text = generated_parts.content.raw

        # Create the prompt object with the prompt parts and embedding
        await creation_function(text, prompt, conversation_id, upload, **kwargs)

        return text

    async def _generic_prompt_creation(
        self,
        prompt: Optional[str],
        generation_function: Callable[[Optional[str], Optional[Dict[str, Any]]], Any],
        creation_function: Callable[[str], Any],
        **kwargs,
    ) -> str:
        conversation_id = await self.create_conversation()

        # Use the generation_function to get the processed conversation parts
        text = await generation_function(prompt, **kwargs)

        # Create the prompt object with the prompt parts and embedding
        await creation_function(text, prompt, conversation_id, **kwargs)

        return text

    async def submit_task(
        self,
        executor: ThreadPoolExecutor,
        generate_prompt: Callable,
        message_data: Optional[str],
        response: Optional[str] = None,
        use_process_conversations: Optional[bool] = False,
    ) -> Future:
        return await executor.submit(
            generate_prompt, message_data, response, use_process_conversations
        ).result()

    async def is_conversation_finished(self, conversation_id: str) -> bool:
        return await self.get_conversation(conversation_id).is_finished()

    async def process_message_data(
        self, conversation_id: str, message_data: str, message_data_user: str
    ):
        last_message_id = await self.get_conversation(
            conversation_id
        ).get_last_message_id()

        await self.handle_user_input(
            conversation_id, message_data_user, last_message_id
        )
        await self.handle_agent_response(conversation_id, message_data, last_message_id)

    def handle_feedback(self, chat):
        feedback_input = (
            input("Are you satisfied with the responses? (yes/no): ").strip().lower()
        )
        if feedback_input == "no":
            self.adjust_chat_temperature(chat)

    def adjust_chat_temperature(self, chat):
        adjustment = (
            input(
                "Would you like the responses to be more 'random' or 'deterministic'? "
            )
            .strip()
            .lower()
        )
        if adjustment == "random":
            chat.temperature += 0.1
        elif adjustment == "deterministic":
            chat.temperature -= 0.1
        chat.temperature = min(max(chat.temperature, 0.2), 2.0)
        print(f"model_name temperature adjusted to: {chat.temperature}")

    def display_results_and_feedback(
        self, chat, message_data, display_results, feedback
    ):
        if display_results:
            for key, future in message_data.items():
                print(f"{key}: {future.result()}")
                print("\n")

            if feedback:
                self.handle_feedback(chat)

    async def run_chat(
        self,
        generate_prompt: Callable,
        chat,
        directory,
        subdirectory,
        initial_prompt: str = "",
        end_letter: str = "A",
        end_roman_numeral: Optional[str] = None,
        feedback: bool = False,
        display_results: bool = False,
        conversation_path: str = None,
    ):
        # create subdirectory if it does not exist
        conversation_path = os.path.join(directory, subdirectory, "conversations")

        if not os.path.exists(conversation_path):
            os.makedirs(conversation_path)

        conversation_id = await self.start_conversation(initial_prompt)
        saved_messages = []
        user_message = ""
        title = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        # Define a list of Roman numerals
        roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

        # Adjust the max_roman_index calculation
        max_roman_index = None
        if end_roman_numeral:
            max_roman_index = roman_numerals.index(end_roman_numeral) + 1

        while True:
            message_data = {}
            user_message = input()
            print("\n")
            # Command Handling

            with ThreadPoolExecutor(max_workers=4) as executor:
                prev_future = None

                # Adjusting the outer loop based on the provided end_letter
                for current_char in map(chr, range(ord("A"), ord(end_letter) + 1)):
                    # If end_roman_numeral is provided, iterate over Roman numerals
                    if end_roman_numeral:
                        for roman in roman_numerals[:max_roman_index]:
                            if prev_future:
                                result_list = prev_future.result().content.raw
                                joined_result = "".join(map(str, result_list))
                                future = executor.submit(generate_prompt, joined_result)
                            else:
                                future = executor.submit(
                                    generate_prompt,
                                    user_message,
                                )

                            key = f"{current_char}-{roman}"
                            message_data[key] = future.result().content.raw
                            prev_future = future
                    else:
                        # If end_roman_numeral is not provided, skip the inner loop
                        if prev_future:
                            result_list = prev_future.result().content.raw
                            joined_result = "".join(map(str, result_list))
                            future = executor.submit(
                                generate_prompt, "Continue", joined_result
                            )
                        else:
                            future = executor.submit(
                                generate_prompt,
                                user_message,
                            )

                        key = current_char
                        message_data[key] = future.result().content.raw
                        prev_future = future

                saved_messages.append(message_data)
                # Process the message data
                await self.process_message_data(
                    conversation_id, message_data, user_message
                )

                # Save the conversation to a JSON file
                await self.save_conversation(
                    conversation_id, title, folder_path=conversation_path
                )

                # Display the results and ask for feedback
                self.display_results_and_feedback(
                    chat, message_data, display_results, feedback
                )


# Initialize the ChainManager
chain_manager = ChainManager()
