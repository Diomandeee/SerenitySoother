from enum import Enum

# Enum for Emotion Types
class EmotionType(str, Enum):
    """Enumeration for different types of emotions."""

    ANXIETY = "Anxiety"
    FEAR = "Fear"
    JOY = "Joy"
    SADNESS = "Sadness"
    ANGER = "Anger"
    SURPRISE = "Surprise"
    CONFUSION = "Confusion"
    DISGUST = "Disgust"
    SHAME = "Shame"
    GUILT = "Guilt"
    EUPHORIA = "Euphoria"


# Enum for Goal Types
class GoalType(str, Enum):
    """Enumeration for different types of goals."""

    RELAXATION = "Relaxation"
    SELF_IMPROVEMENT = "Self-improvement"
    STRESS_RELIEF = "Stress relief"
    MINDFULNESS = "Mindfulness"
    FOCUS_ENHANCEMENT = "Focus enhancement"
    EMOTIONAL_BALANCE = "Emotional balance"
    CONFIDENCE_BUILDING = "Confidence building"
    CREATIVITY_ENHANCEMENT = "Creativity enhancement"
    RESILIENCE_DEVELOPMENT = "Resilience development"
    EMPATHY_GROWTH = "Empathy growth"


# Enum for Scene Types
class SceneType(str, Enum):
    """Enumeration for different types of scenes."""

    NATURE = "Nature"
    FANTASY = "Fantasy"
    ABSTRACT = "Abstract"
    URBAN = "Urban"
    FUTURISTIC = "Futuristic"
    HISTORICAL = "Historical"
    DREAMSCAPES = "Dreamscapes"
    SURREALISM = "Surrealism"
    STEAMPUNK = "Steampunk"
    CYBERPUNK = "Cyberpunk"


class ElementType(str, Enum):
    """Enumeration for different types of elements."""

    OBJECTS = "Objects"
    CREATURES = "Creatures"
    PLANTS = "Plants"
    LANDSCAPES = "Landscapes"
    ARCHITECTURE = "Architecture"
    ABSTRACT_SHAPES = "Abstract shapes"
    WATER_FEATURES = "Water features"
    CELESTIAL_BODIES = "Celestial bodies"
    MUSICAL_INSTRUMENTS = "Musical instruments"
    FRACTALS = "Fractals"
    MYTHICAL_BEINGS = "Mythical Beings"
    TIME_CONCEPTS = "Time Concepts"
    COSMIC_PHENOMENA = "Cosmic Phenomena"
    DREAMSCAPES = "Dreamscapes"
    ELEMENTAL_FORCES = "Elemental Forces"
    HISTORICAL_ERAS = "Historical Eras"
    SYMBOLIC_ICONS = "Symbolic Icons"
    FANTASY_REALMS = "Fantasy Realms"
    PSYCHEDELIC_EXPERIENCES = "Psychedelic Experiences"
    DIGITAL_WORLDS = "Digital Worlds"


# Enum for Notification Types
class NotificationType(str, Enum):
    """Enumeration for different types of notifications."""

    REMINDERS = "Reminders"
    ACHIEVEMENTS = "Achievements"
    MOTIVATIONAL_QUOTES = "Motivational quotes"
    INSPIRATIONAL_STORIES = "Inspirational stories"
    GUIDED_MEDITATION_SESSIONS = "Guided meditation sessions"
    PERSONALIZED_ENCOURAGEMENT = "Personalized encouragement"
    WELLNESS_TIPS = "Wellness tips"
    MINDFULNESS_EXERCISES = "Mindfulness exercises"
    QUOTE_OF_THE_DAY = "Quote of the day"
    DAILY_CHALLENGES = "Daily challenges"


# Enum for Session Types
class SessionType(str, Enum):
    """Enumeration for different types of sessions."""

    RELAXATION = "Relaxation"
    ANXIETY = "Anxiety"
    TRAUMA_HEALING = "Trauma healing"
    SLEEP_IMPROVEMENT = "Sleep improvement"
    FOCUS_ENHANCEMENT = "Focus enhancement"
    EMOTIONAL_BALANCE = "Emotional balance"
    CONFIDENCE_BOOSTING = "Confidence boosting"
    PAIN_MANAGEMENT = "Pain management"
    ADDICTION_RECOVERY = "Addiction recovery"
    MINDFUL_MOVEMENT = "Mindful movement"


# Enum for Thought Types
class ThoughtType(str, Enum):
    """Enumeration for different types of thoughts."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    INTRUSIVE = "Intrusive"
    REFLECTIVE = "Reflective"
    IMAGINATIVE = "Imaginative"
    RATIONAL = "Rational"
    IRRATIONAL = "Irrational"
    OBSESSIVE = "Obsessive"
    AMBIVALENT = "Ambivalent"


# Enum for Memory Types
class MemoryType(str, Enum):
    """Enumeration for different types of memories."""

    TRAUMATIC = "Traumatic"
    JOYFUL = "Joyful"
    NOSTALGIC = "Nostalgic"
    EDUCATIONAL = "Educational"
    FORGOTTEN = "Forgotten"
    VIVID = "Vivid"
    REMOTE = "Remote"
    RECENT = "Recent"
    REPRESSED = "Repressed"
    INSIGNIFICANT = "Insignificant"


# Enum for Script Types
class ScriptType(str, Enum):
    """Enumeration for different types of scripts."""

    GUIDED_MEDITATION = "Guided Meditation"
    HYPNOTHERAPY = "Hypnotherapy"
    STORYTELLING = "Storytelling"
    AFFIRMATIONS = "Affirmations"
    SOUND_HEALING = "Sound Healing"
    BREATHWORK = "Breathwork"
    VISUAL_IMAGERY = "Visual Imagery"
    PROGRESSIVE_RELAXATION = "Progressive Relaxation"
    AUTOGENIC_TRAINING = "Autogenic Training"
    MINDFULNESS_MEDITATION = "Mindfulness Meditation"


# Enum for Progress Types
class ProgressType(str, Enum):
    """Enumeration for different types of progress updates."""

    INITIAL_ASSESSMENT = "Initial Assessment"
    WEEKLY_CHECK_IN = "Weekly Check-In"
    MILESTONE_ACHIEVEMENT = "Milestone Achievement"
    PRE_SESSION_EVALUATION = "Pre-Session Evaluation"
    POST_SESSION_REFLECTION = "Post-Session Reflection"
    BI_MONTHLY_REVIEW = "Bi-Monthly Review"
    YEARLY_RECAP = "Yearly Recap"
    GOAL_ACCOMPLISHMENT = "Goal Accomplishment"
    RELAPSE_REVIEW = "Relapse Review"
    NEW_LEARNINGS = "New Learnings"


# Enum for Audio Types
class AudioType(str, Enum):
    """Enumeration for different types of audio tracks."""

    NATURE_SOUNDS = "Nature Sounds"
    INSTRUMENTAL = "Instrumental"
    BINAURAL_BEATS = "Binaural Beats"
    GUIDED_VOCAL = "Guided Vocal"
    SILENCE = "Silence"
    URBAN_SOUNDS = "Urban Sounds"
    HISTORICAL_SOUNDSCAPES = "Historical Soundscapes"
    FUTURISTIC_DRONES = "Futuristic Drones"
    ETHNIC_BEATS = "Ethnic Beats"
    FRACTAL_FREQUENCIES = "Fractal Frequencies"


# Enum for Visualization Types
class VisualizationType(str, Enum):
    """Enumeration for different types of visualizations."""

    SINGLE_FOCUS = "Single-Focus"
    JOURNEY = "Journey"
    ENERGY_FLOW = "Energy Flow"
    BODY_SCAN = "Body Scan"
    COLOR_HEALING = "Color Healing"
    SYMBOLIC_INTERACTION = "Symbolic Interaction"
    TIME_TRAVEL = "Time Travel"
    ALTERNATE_REALITY = "Alternate Reality"
    ASTRAL_PROJECTION = "Astral Projection"
    MICROCOSMIC_ORBIT = "Microcosmic Orbit"


# Enum for Content Types
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


# Enum for Role Types
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
