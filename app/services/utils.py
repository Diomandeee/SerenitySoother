from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import User, Session, Script, TradingCard, Scene, Goal, Emotion
from fastapi import HTTPException
import numpy as np
from collections import Counter
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

async def get_user(user_id: int, db: AsyncSession) -> User:
    try:
        result = await db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching the user.")

async def get_user_sessions(user_id: int, db: AsyncSession) -> List[Session]:
    try:
        result = await db.execute(select(Session).filter(Session.user_id == user_id).order_by(Session.session_date.desc()))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting sessions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user sessions.")

async def get_user_scripts(session_ids: List[int], db: AsyncSession) -> List[Script]:
    try:
        result = await db.execute(select(Script).filter(Script.session_id.in_(session_ids)))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting scripts for sessions {session_ids}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user scripts.")

async def get_user_trading_cards(user_id: int, db: AsyncSession) -> List[TradingCard]:
    try:
        result = await db.execute(select(TradingCard).filter(TradingCard.user_id == user_id))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting trading cards for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user trading cards.")

async def get_user_goals(user_id: int, db: AsyncSession) -> List[Goal]:
    try:
        result = await db.execute(select(Goal).filter(Goal.user_id == user_id))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting goals for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user goals.")

async def get_user_emotions(user_id: int, db: AsyncSession) -> List[Emotion]:
    try:
        result = await db.execute(select(Emotion).filter(Emotion.user_id == user_id).order_by(Emotion.emotion_date.desc()))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting emotions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user emotions.")

def calculate_similarity(user_ratings: Dict[int, float], other_ratings: Dict[int, float]) -> float:
    try:
        common_items = set(user_ratings.keys()) & set(other_ratings.keys())
        if not common_items:
            return 0
        user_vector = np.array([user_ratings[item] for item in common_items])
        other_vector = np.array([other_ratings[item] for item in common_items])
        return np.dot(user_vector, other_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(other_vector))
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise

async def calculate_user_similarity(user_id: int, user_trading_cards: List[TradingCard], db: AsyncSession) -> Dict[int, float]:
    try:
        user_card_types = [card.card_type for card in user_trading_cards]
        result = await db.execute(select(User))
        all_users = result.scalars().all()

        user_similarity = {}
        for other_user in all_users:
            if other_user.id == user_id:
                continue
            other_user_cards = await get_user_trading_cards(other_user.id, db)
            other_user_card_types = [card.card_type for card in other_user_cards]
            common_types = set(user_card_types) & set(other_user_card_types)
            if not common_types:
                continue
            user_similarity[other_user.id] = len(common_types) / len(set(user_card_types) | set(other_user_card_types))

        return user_similarity
    except Exception as e:
        logger.error(f"Error calculating user similarity for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while calculating user similarity.")

def recommend_goals(goals: List[Goal]) -> List[Dict[str, str]]:
    try:
        if not goals:
            return []
        achieved_goals = [goal for goal in goals if goal.goal_status == "achieved"]
        if not achieved_goals:
            return []
        common_goal_types = Counter([goal.goal_type for goal in achieved_goals])
        most_common_goal_type = common_goal_types.most_common(1)[0][0]
        return [{"goal_type": most_common_goal_type, "goal_description": "Based on your interest in past goals"}]
    except Exception as e:
        logger.error(f"Error recommending goals: {e}")
        raise

async def recommend_based_on_mood(emotions: List[Emotion], db: AsyncSession) -> Dict[str, List]:
    try:
        if not emotions:
            return {}
        recent_emotion = emotions[0]
        if recent_emotion.emotion_intensity == "high" and recent_emotion.emotion_type in ["stress", "anxiety"]:
            calming_scripts = await db.execute(select(Script).filter(Script.script_type == "calming").order_by(Script.script_rating.desc()).limit(3))
            calming_scenes = await db.execute(select(Scene).filter(Scene.scene_type == "calming").order_by(Scene.id.desc()).limit(3))
            return {"calming_scripts": calming_scripts.scalars().all(), "calming_scenes": calming_scenes.scalars().all()}
        return {}
    except Exception as e:
        logger.error(f"Error recommending based on mood: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while recommending based on mood.")
