from sqlalchemy.orm import relationship, declarative_base, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from app.config import settings
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    JSON,
    Enum,
    Text,
    DateTime,
    BigInteger,
    Table,
)
from app.enums import (
    RoleType,
    ContentType,
    SessionType,
    ScriptType,
    SceneType,
    ElementType,
    EmotionType,
    GoalType,
    NotificationType,
    ThoughtType,
    MemoryType,
)
import datetime
import uuid

Base = declarative_base()

# Association Table for Scene-Element combinations
scene_element_association = Table(
    "scene_element_association",
    Base.metadata,
    Column("scene_id", Integer, ForeignKey("scenes.id")),
    Column("element_id", Integer, ForeignKey("elements.id")),
)

# User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    profile_information = Column(Text)
    bio = Column(Text)
    join_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    last_login_date = Column(DateTime)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False)

    achievements = relationship("Achievement", back_populates="user")
    rewards = relationship("Reward", back_populates="user")
    hypnotherapy_scripts = relationship("HypnotherapyScript", back_populates="user")
    hypnotherapy_sessions = relationship("HypnotherapySession", back_populates="user")
    sessions = relationship("Session", back_populates="user")
    emotions = relationship("Emotion", back_populates="user")
    goals = relationship("Goal", back_populates="user")
    progress_entries = relationship("Progress", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    thoughts = relationship("Thought", back_populates="user")
    memories = relationship("Memory", back_populates="user")
    settings = relationship("Setting", back_populates="user")
    trading_cards = relationship("TradingCard", back_populates="user")


# ChainTree related models
class Author(Base):
    __tablename__ = "authors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role = Column(Enum(RoleType), nullable=False)
    name = Column(String, nullable=True)
    author_metadata = Column(JSON, nullable=True)


class Content(Base):
    __tablename__ = "contents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(String, nullable=True)
    content_type = Column(Enum(ContentType), default=ContentType.TEXT)
    parts = Column(MutableList.as_mutable(JSON), nullable=True)
    part_lengths = Column(JSON, nullable=True)


class FinishDetails(Base):
    __tablename__ = "finish_details"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False)
    stop_tokens = Column(String, nullable=False)


class Attachment(Base):
    __tablename__ = "attachments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metadata_id = Column(UUID(as_uuid=True), ForeignKey("metadata.id"), nullable=False)
    name = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    checksum = Column(String, nullable=True)
    description = Column(String, nullable=True)
    uploaded_timestamp = Column(String, nullable=True)
    url = Column(String, nullable=True)


class Metadata(Base):
    __tablename__ = "metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finish_details_id = Column(
        UUID(as_uuid=True), ForeignKey("finish_details.id"), nullable=True
    )
    finish_details = relationship("FinishDetails")
    attachments = relationship(
        "Attachment",
        primaryjoin="Metadata.id == Attachment.metadata_id",
        backref="metadata",
    )
    model_slug = Column(String, nullable=True)
    parent_id = Column(String, nullable=True)
    timestamp_ = Column(String, nullable=True)
    links = Column(JSON, nullable=True)
    message_type = Column(String, nullable=True)
    is_complete = Column(Boolean, nullable=True)
    command = Column(String, nullable=True)
    args = Column(MutableList.as_mutable(JSON), nullable=True)
    status = Column(String, nullable=True)


class ChainCoordinate(Base):
    __tablename__ = "chain_coordinates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    x = Column(JSON, nullable=False, default=0)
    y = Column(JSON, nullable=False, default=0)
    z = Column(JSON, nullable=False, default=0)
    t = Column(JSON, nullable=False, default=0)
    n_parts = Column(Integer, nullable=False, default=0)
    parent_id = Column(
        UUID(as_uuid=True), ForeignKey("chain_coordinates.id"), nullable=True
    )
    parent = relationship("ChainCoordinate", remote_side=[id], backref="children")


class ChainMessage(Base):
    __tablename__ = "chain_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), ForeignKey("contents.id"), nullable=True)
    content = relationship("Content")
    author_id = Column(UUID(as_uuid=True), ForeignKey("authors.id"), nullable=True)
    author = relationship("Author")
    create_time = Column(Float, nullable=True)
    end_turn = Column(Boolean, nullable=True)
    weight = Column(Integer, nullable=False, default=1)
    message_metadata_id = Column(
        UUID(as_uuid=True), ForeignKey("metadata.id"), nullable=True
    )
    message_metadata = relationship("Metadata")
    recipient = Column(String, nullable=True)
    coordinate_id = Column(
        UUID(as_uuid=True), ForeignKey("chain_coordinates.id"), nullable=True
    )
    coordinate = relationship("ChainCoordinate")


class Chain(ChainMessage):
    __tablename__ = "chains"

    id = Column(UUID(as_uuid=True), ForeignKey("chain_messages.id"), primary_key=True)
    cluster_label = Column(Integer, nullable=True)
    n_neighbors = Column(Integer, nullable=True)
    embedding = Column(JSON, nullable=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("chains.id"), nullable=True)
    parent = relationship(
        "Chain", remote_side=[id], backref="children", foreign_keys=[parent_id]
    )
    depth = Column(Integer, nullable=True)
    next = Column(String, nullable=True)
    prev = Column(String, nullable=True)


class ChainMap(Base):
    __tablename__ = "chain_maps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("chains.id"), nullable=True)
    message = relationship("Chain", backref="chain_map")
    parent_id = Column(String, nullable=True)
    children = Column(MutableList.as_mutable(JSON), nullable=True)
    references = Column(MutableList.as_mutable(JSON), nullable=True)
    relationships = Column(JSON, nullable=True)
    depth = Column(Integer, nullable=True)
    chain_tree_id = Column(
        UUID(as_uuid=True), ForeignKey("chain_trees.id"), nullable=True
    )


class ChainTree(Base):
    __tablename__ = "chain_trees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    create_time = Column(Float, nullable=True)
    update_time = Column(Float, nullable=True)
    mapping = relationship("ChainMap", backref="chain_tree")
    moderation_results = Column(MutableList.as_mutable(JSON), nullable=True)
    current_node = Column(String, nullable=True)
    conversation_template_id = Column(String, nullable=True)
    plugin_ids = Column(MutableList.as_mutable(JSON), nullable=True)
    gizmo_id = Column(String, nullable=True)
    hypnotherapy_scripts = relationship(
        "HypnotherapyScript", back_populates="chain_tree"
    )


# HypnotherapyScript model
class HypnotherapyScript(Base):
    __tablename__ = "hypnotherapy_scripts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    chain_tree_id = Column(UUID(as_uuid=True), ForeignKey("chain_trees.id"))
    title = Column(String, index=True)
    content = Column(Text)

    user = relationship("User", back_populates="hypnotherapy_scripts")
    chain_tree = relationship("ChainTree", back_populates="hypnotherapy_scripts")
    sessions = relationship("HypnotherapySession", back_populates="script")


# HypnotherapySession model
class HypnotherapySession(Base):
    __tablename__ = "hypnotherapy_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    script_id = Column(Integer, ForeignKey("hypnotherapy_scripts.id"))
    session_notes = Column(Text)
    session_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    content = Column(Text)

    user = relationship("User", back_populates="hypnotherapy_sessions")
    script = relationship("HypnotherapyScript", back_populates="sessions")


class NFT(Base):
    __tablename__ = "nfts"
    id = Column(Integer, primary_key=True, index=True)
    nft_hash = Column(String, unique=True, index=True)
    tickets = Column(BigInteger)
    proof = Column(Text)
    size = Column(BigInteger)
    type = Column(String)


# Session model
class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    session_type = Column(Enum(SessionType), nullable=False)
    session_status = Column(String, nullable=False)
    session_duration = Column(Integer)
    session_description = Column(Text)

    user = relationship("User", back_populates="sessions")
    scripts = relationship("Script", back_populates="session")
    nft_id = Column(Integer, ForeignKey("nfts.id"))


# Script model
class Script(Base):
    __tablename__ = "scripts"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    script_type = Column(Enum(ScriptType), nullable=False)
    script_content = Column(Text, nullable=False, default="")
    script_rating = Column(Integer)
    script_description = Column(Text)
    script_image = Column(String)

    session = relationship("Session", back_populates="scripts")
    scenes = relationship("Scene", back_populates="script")
    sections = relationship("Section", back_populates="script")
    nft_id = Column(Integer, ForeignKey("nfts.id"))


# Section model
class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True)
    script_id = Column(Integer, ForeignKey("scripts.id"), nullable=False)
    part_title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    section_embedding = mapped_column(Vector(1536))

    script = relationship("Script", back_populates="sections")
    nft_id = Column(Integer, ForeignKey("nfts.id"))


# Scene model
class Scene(Base):
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True)
    script_id = Column(Integer, ForeignKey("scripts.id"), nullable=False)
    scene_type = Column(Enum(SceneType), nullable=False)
    scene_description = Column(Text)
    scene_embedding = mapped_column(Vector(1536))
    scene_image = Column(String)
    scene_audio = Column(String)

    script = relationship("Script", back_populates="scenes")
    elements = relationship(
        "Element", secondary=scene_element_association, back_populates="scenes"
    )
    nft_id = Column(Integer, ForeignKey("nfts.id"))


# Element model
class Element(Base):
    __tablename__ = "elements"

    id = Column(Integer, primary_key=True)
    scene_id = Column(Integer, ForeignKey("scenes.id"), nullable=True)
    element_type = Column(Enum(ElementType), nullable=False)
    element_description = Column(Text)
    element_embedding = mapped_column(Vector(1536))
    element_image = Column(String)
    element_audio = Column(String)

    scenes = relationship(
        "Scene", secondary=scene_element_association, back_populates="elements"
    )
    nft_id = Column(Integer, ForeignKey("nfts.id"))


# Emotion model
class Emotion(Base):
    __tablename__ = "emotions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    emotion_type = Column(Enum(EmotionType), nullable=False)
    emotion_intensity = Column(String, nullable=False)
    emotion_description = Column(Text)
    emotion_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="emotions")


# Goal model
class Goal(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    goal_type = Column(Enum(GoalType), nullable=False)
    goal_description = Column(Text)
    goal_status = Column(String, nullable=False)
    goal_deadline = Column(DateTime)

    user = relationship("User", back_populates="goals")
    progress_entries = relationship("Progress", back_populates="goal")


# Progress model
class Progress(Base):
    __tablename__ = "progress"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)
    progress_status = Column(String, nullable=False)
    progress_description = Column(Text)
    progress_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="progress_entries")
    goal = relationship("Goal", back_populates="progress_entries")


# Notification model
class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    notification_type = Column(Enum(NotificationType), nullable=False)
    notification_message = Column(Text)
    notification_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="notifications")


# Thought model
class Thought(Base):
    __tablename__ = "thoughts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    thought_type = Column(Enum(ThoughtType), nullable=False)
    thought_description = Column(Text)
    thought_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="thoughts")


# Memory model
class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    memory_type = Column(Enum(MemoryType), nullable=False)
    memory_description = Column(Text)
    memory_intensity = Column(String, nullable=False)
    memory_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="memories")


# Setting model
class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    setting_type = Column(String, nullable=False)
    setting_value = Column(String)

    user = relationship("User", back_populates="settings")


# TradingCard model
class TradingCard(Base):
    __tablename__ = "trading_cards"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    card_type = Column(String, nullable=False)  # Replace with Enum as needed
    card_design = Column(Text)
    realm_access_url = Column(String)
    qr_code_url = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="trading_cards")


# Achievement model
class Achievement(Base):
    __tablename__ = "achievements"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, index=True)

    user = relationship("User", back_populates="achievements")


# Reward model
class Reward(Base):
    __tablename__ = "rewards"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, index=True)
    description = Column(String)
    redeemed = Column(Boolean, default=False)

    user = relationship("User", back_populates="rewards")


# PromptMetadata model
class PromptMetadata(Base):
    __tablename__ = "prompt_metadata"

    id = Column(String, primary_key=True, index=True)
    create_time = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    prompt_num = Column(Integer, unique=True, index=True)
    prompt = Column(String)
    revised_prompt = Column(String)
    embedding = mapped_column(Vector(1536))
    image = Column(String)


# Database setup
engine = create_async_engine(settings.DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    import asyncio

    asyncio.run(init_db())
    print("Database tables created successfully.")
