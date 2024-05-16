from enum import Enum


class DistanceMode(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


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


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    CONTACT = "contact"
    MESSAGE = "message"
    LINK = "link"
    EVENT = "event"
    DIRECTORY = "directory"
    OTHER = "other"
    EMAIL = "email"
    CODE = "code"
    TETHER_QUOTE = "tether_quote"
    MULTIMODALTEXT = "multimodal_text"


class RoleType(str, Enum):
    USER = "user"
    CHAT = "chat"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ADMIN = "admin"
    GUEST = "guest"
    ANONYMOUS = "anonymous"
    MODERATOR = "moderator"
    OWNER = "owner"
    DEVELOPER = "developer"
    CREATOR = "creator"
    BROWSER = "browser"
    TOOL = "tool"


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
