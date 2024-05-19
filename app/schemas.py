from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.enums import (
    EmotionType,
    GoalType,
    SceneType,
    ElementType,
    NotificationType,
    SessionType,
    ThoughtType,
    MemoryType,
    ScriptType,
)


class UserBase(BaseModel):
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    profile_information: Optional[str] = None
    bio: Optional[str] = None
    is_admin: Optional[bool] = False


class UserCreate(UserBase):
    password: str
    is_admin: Optional[bool] = False


class UserUpdate(UserBase):
    password: Optional[str] = None
    is_admin: Optional[bool] = None


class User(UserBase):
    id: int
    join_date: datetime
    is_admin: bool

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class UserLogin(BaseModel):
    username: str
    password: str


class TokenData(BaseModel):
    username: Optional[str] = None


class SessionBase(BaseModel):
    user_id: int
    session_date: Optional[datetime] = None
    session_type: SessionType
    session_status: str
    session_duration: Optional[int] = None
    session_description: Optional[str] = None


class SessionCreate(SessionBase):
    pass


class SessionUpdate(BaseModel):
    session_date: Optional[datetime] = None
    session_type: Optional[SessionType] = None
    session_status: Optional[str] = None
    session_duration: Optional[int] = None
    session_description: Optional[str] = None


class Session(SessionBase):
    id: int

    class Config:
        orm_mode = True


class ScriptBase(BaseModel):
    session_id: int
    script_type: ScriptType
    script_content: str
    script_rating: Optional[int] = None
    script_description: Optional[str] = None
    script_image: Optional[str] = None


class ScriptCreate(ScriptBase):
    pass


class ScriptUpdate(BaseModel):
    script_type: Optional[ScriptType] = None
    script_content: Optional[str] = None
    script_rating: Optional[int] = None
    script_description: Optional[str] = None
    script_image: Optional[str] = None


class Script(ScriptBase):
    id: int

    class Config:
        orm_mode = True


class HypnotherapyScriptBase(BaseModel):
    user_id: int
    title: str
    content: str


class HypnotherapyScriptCreate(HypnotherapyScriptBase):
    pass


class HypnotherapyScript(HypnotherapyScriptBase):
    id: int

    class Config:
        orm_mode = True


class HypnotherapySessionBase(BaseModel):
    user_id: int
    script_id: int
    session_notes: str


class HypnotherapySessionCreate(HypnotherapySessionBase):
    pass


class HypnotherapySession(HypnotherapySessionBase):
    id: int
    session_date: datetime

    class Config:
        orm_mode = True


class SceneBase(BaseModel):
    script_id: int
    scene_type: SceneType
    scene_description: Optional[str] = None
    scene_image: Optional[str] = None
    scene_audio: Optional[str] = None


class SceneCreate(SceneBase):
    pass


class SceneUpdate(BaseModel):
    scene_type: Optional[SceneType] = None
    scene_description: Optional[str] = None
    scene_image: Optional[str] = None
    scene_audio: Optional[str] = None


class Scene(SceneBase):
    id: int

    class Config:
        orm_mode = True


class ElementBase(BaseModel):
    scene_id: int
    element_type: ElementType
    element_description: Optional[str] = None
    element_image: Optional[str] = None
    element_audio: Optional[str] = None


class ElementCreate(ElementBase):
    pass


class ElementUpdate(BaseModel):
    element_type: Optional[ElementType] = None
    element_description: Optional[str] = None
    element_image: Optional[str] = None
    element_audio: Optional[str] = None


class Element(ElementBase):
    id: int

    class Config:
        orm_mode = True


class EmotionBase(BaseModel):
    user_id: int
    emotion_type: EmotionType
    emotion_intensity: str
    emotion_description: Optional[str] = None
    emotion_date: Optional[datetime] = None


class EmotionCreate(EmotionBase):
    pass


class EmotionUpdate(BaseModel):
    emotion_type: Optional[EmotionType] = None
    emotion_intensity: Optional[str] = None
    emotion_description: Optional[str] = None
    emotion_date: Optional[datetime] = None


class Emotion(EmotionBase):
    id: int

    class Config:
        orm_mode = True


class GoalBase(BaseModel):
    user_id: int
    goal_type: GoalType
    goal_description: Optional[str] = None
    goal_status: str
    goal_deadline: Optional[datetime] = None


class GoalCreate(GoalBase):
    pass


class GoalUpdate(BaseModel):
    goal_type: Optional[GoalType] = None
    goal_description: Optional[str] = None
    goal_status: Optional[str] = None
    goal_deadline: Optional[datetime] = None


class Goal(GoalBase):
    id: int

    class Config:
        orm_mode = True


class ProgressBase(BaseModel):
    user_id: int
    goal_id: int
    progress_status: str
    progress_description: Optional[str] = None
    progress_date: Optional[datetime] = None


class ProgressCreate(ProgressBase):
    pass


class ProgressUpdate(BaseModel):
    progress_status: Optional[str] = None
    progress_description: Optional[str] = None
    progress_date: Optional[datetime] = None


class Progress(ProgressBase):
    id: int

    class Config:
        orm_mode = True


class NotificationBase(BaseModel):
    user_id: int
    notification_type: NotificationType
    notification_message: Optional[str] = None
    notification_date: Optional[datetime] = None


class NotificationCreate(NotificationBase):
    pass


class NotificationUpdate(BaseModel):
    notification_type: Optional[NotificationType] = None
    notification_message: Optional[str] = None
    notification_date: Optional[datetime] = None


class Notification(NotificationBase):
    id: int

    class Config:
        orm_mode = True


class MemoryBase(BaseModel):
    user_id: int
    memory_type: MemoryType
    memory_description: Optional[str] = None
    memory_intensity: str
    memory_date: Optional[datetime] = None


class MemoryCreate(MemoryBase):
    pass


class MemoryUpdate(BaseModel):
    memory_type: Optional[MemoryType] = None
    memory_description: Optional[str] = None
    memory_intensity: Optional[str] = None
    memory_date: Optional[datetime] = None


class Memory(MemoryBase):
    id: int

    class Config:
        orm_mode = True


class SettingBase(BaseModel):
    user_id: int
    setting_type: str
    setting_value: Optional[str] = None


class SettingCreate(SettingBase):
    pass


class SettingUpdate(BaseModel):
    setting_type: Optional[str] = None
    setting_value: Optional[str] = None


class Setting(SettingBase):
    id: int

    class Config:
        orm_mode = True


class TradingCardBase(BaseModel):
    user_id: int
    card_type: str
    card_design: str
    realm_access_url: str
    qr_code_url: Optional[str] = None
    box_size: Optional[int] = 10
    border: Optional[int] = 4
    fill_color: Optional[str] = "black"
    back_color: Optional[str] = "white"
    logo_path: Optional[str] = None
    background_image_path: Optional[str] = None


class TradingCardCreate(TradingCardBase):
    pass


class TradingCardUpdate(BaseModel):
    card_design: Optional[str] = None
    realm_access_url: Optional[str] = None
    box_size: Optional[int] = None
    border: Optional[int] = None
    fill_color: Optional[str] = None
    back_color: Optional[str] = None
    logo_path: Optional[str] = None
    background_image_path: Optional[str] = None
    card_type: Optional[str] = None
    qr_code_url: Optional[str] = None


class TradingCard(TradingCardBase):
    id: int

    class Config:
        orm_mode = True


class ThoughtBase(BaseModel):
    user_id: int
    thought_type: ThoughtType
    thought_description: Optional[str] = None
    thought_date: Optional[datetime] = None


class ThoughtCreate(ThoughtBase):
    pass


class ThoughtUpdate(BaseModel):
    thought_type: Optional[ThoughtType] = None
    thought_description: Optional[str] = None
    thought_date: Optional[datetime] = None


class Thought(ThoughtBase):
    id: int

    class Config:
        orm_mode = True


class AchievementBase(BaseModel):
    user_id: int
    title: str


class AchievementCreate(AchievementBase):
    pass


class Achievement(AchievementBase):
    id: int

    class Config:
        orm_mode = True


class RewardBase(BaseModel):
    user_id: int
    title: str
    description: str


class RewardCreate(RewardBase):
    pass


class Reward(RewardBase):
    id: int

    class Config:
        orm_mode = True
