# models.py

import datetime 
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, Enum, Table, Boolean
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.enums import (
    EmotionType, GoalType, SceneType, ElementType, NotificationType, 
    SessionType, ThoughtType, MemoryType, ScriptType, ProgressType
)

Base = declarative_base()

# Association Table for Scene-Element combinations
scene_element_association = Table(
    'scene_element_association', Base.metadata,
    Column('scene_id', Integer, ForeignKey('scenes.id')),
    Column('element_id', Integer, ForeignKey('elements.id'))
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

    achievements = relationship("Achievement", back_populates="user")
    rewards = relationship("Reward", back_populates="user")
    hypnotherapy_scriepts = relationship("HypnotherapyScript", back_populates="user")
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

# Script model
class Script(Base):
    __tablename__ = "scripts"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    script_type = Column(Enum(ScriptType), nullable=False)
    script_content = Column(Text, nullable=False, default='')
    script_rating = Column(Integer)
    script_description = Column(Text)
    script_image = Column(String)

    session = relationship("Session", back_populates="scripts")
    scenes = relationship("Scene", back_populates="script")
    sections = relationship("Section", back_populates="script")

# Section model
class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True)
    script_id = Column(Integer, ForeignKey("scripts.id"), nullable=False)
    part_title = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    script = relationship("Script", back_populates="sections")

# Scene model
class Scene(Base):
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True)
    script_id = Column(Integer, ForeignKey("scripts.id"), nullable=False)
    scene_type = Column(Enum(SceneType), nullable=False)
    scene_description = Column(Text)
    scene_image = Column(String)
    scene_audio = Column(String)

    script = relationship("Script", back_populates="scenes")
    elements = relationship("Element", secondary=scene_element_association, back_populates="scenes")

# Element model
class Element(Base):
    __tablename__ = "elements"

    id = Column(Integer, primary_key=True)
    scene_id = Column(Integer, ForeignKey("scenes.id"), nullable=True)
    element_type = Column(Enum(ElementType), nullable=False)
    element_description = Column(Text)
    element_image = Column(String)
    element_audio = Column(String)

    scenes = relationship("Scene", secondary=scene_element_association, back_populates="elements")

# HypnotherapyScript model
class HypnotherapyScript(Base):
    __tablename__ = "hypnotherapy_scripts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, index=True)
    content = Column(Text)

    user = relationship("User", back_populates="hypnotherapy_scripts")
    sessions = relationship("HypnotherapySession", back_populates="script")

# HypnotherapySession model
class HypnotherapySession(Base):
    __tablename__ = "hypnotherapy_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    script_id = Column(Integer, ForeignKey("hypnotherapy_scripts.id"))
    session_notes = Column(Text)
    session_date = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    user = relationship("User", back_populates="hypnotherapy_sessions")
    script = relationship("HypnotherapyScript", back_populates="sessions")

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

# Database setup
DATABASE_URL = "sqlite+aiosqlite:///./serenity_soother.db"
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
    print("Database tables created successfully.")
