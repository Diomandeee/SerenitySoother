from enum import Enum


class Status(Enum):
    NOT_STARTED = 1
    IN_PROGRESS = 2
    SUCCESS = 3
    FAILURE = 4
    TIMEOUT = 5
    ROLLED_BACK = 6
    CANCELLED = 7


class PromptStatus(Enum):
    SUCCESS = "Success"
    FAILURE = "Failure"
    NOT_FOUND = "Not Found"


class NodeRelationship(str, Enum):
    SOURCE = "source"
    PREVIOUS = "previous"
    NEXT = "next"
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"


class ElementType(Enum):
    STEP = "Step"
    CHAPTER = "Chapter"
    PAGE = "Page"
    SECTION = "Section"
    Question = "Question"
    Part = "Part"
    Segment = "Segment"
    Definition = "Definition"
