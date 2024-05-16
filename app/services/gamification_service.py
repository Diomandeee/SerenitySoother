from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, Goal, Achievement, Reward
from app.services.utils import get_user, get_user_goals
from fastapi import HTTPException
from sqlalchemy.future import select
from typing import List
import logging

logger = logging.getLogger(__name__)

async def track_achievements(user_id: int, db: AsyncSession):
    try:
        user = await get_user(user_id, db)
        goals = await get_user_goals(user_id, db)
        achievements = await evaluate_achievements(user, goals, db)
        return {
            "message": f"Achievements tracked for {user.username}",
            "achievements": achievements
        }
    except Exception as e:
        logger.error(f"Error tracking achievements for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while tracking achievements.")

async def evaluate_rewards(user: User, db: AsyncSession):
    try:
        result = await db.execute(select(Reward).filter(Reward.user_id == user.id, Reward.redeemed == False))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error evaluating rewards for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while evaluating rewards.")
    
async def evaluate_achievements(user: User, goals: List[Goal], db: AsyncSession):
    try:
        achieved_goals = [goal for goal in goals if goal.goal_status == "achieved"]
        achievements = []

        if len(achieved_goals) >= 1:
            achievement = await award_achievement(user, "First Goal Achieved", db)
            achievements.append(achievement)

        if len(achieved_goals) >= 5:
            achievement = await award_achievement(user, "Five Goals Achieved", db)
            achievements.append(achievement)

        if len(achieved_goals) >= 10:
            achievement = await award_achievement(user, "Ten Goals Achieved", db)
            achievements.append(achievement)

        return achievements
    except Exception as e:
        logger.error(f"Error evaluating achievements for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while evaluating achievements.")

async def award_achievement(user: User, title: str, db: AsyncSession):
    try:
        result = await db.execute(select(Achievement).filter(Achievement.user_id == user.id, Achievement.title == title))
        existing_achievement = result.scalars().first()
        if existing_achievement:
            return existing_achievement

        new_achievement = Achievement(user_id=user.id, title=title)
        db.add(new_achievement)
        await db.commit()
        await db.refresh(new_achievement)
        return new_achievement
    except Exception as e:
        logger.error(f"Error awarding achievement {title} to user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while awarding achievement.")

async def award_reward(user: User, title: str, description: str, db: AsyncSession):
    try:
        new_reward = Reward(user_id=user.id, title=title, description=description)
        db.add(new_reward)
        await db.commit()
        await db.refresh(new_reward)
        return new_reward
    except Exception as e:
        logger.error(f"Error awarding reward {title} to user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while awarding reward.")

async def redeem_reward(user: User, reward_id: int, db: AsyncSession):
    try:
        result = await db.execute(select(Reward).filter(Reward.id == reward_id))
        reward = result.scalars().first()
        if not reward:
            raise HTTPException(status_code=404, detail="Reward not found")

        if reward.user_id != user.id:
            raise HTTPException(status_code=403, detail="You do not have permission to redeem this reward")

        reward.redeemed = True
        await db.commit()
        await db.refresh(reward)
        return reward
    except Exception as e:
        logger.error(f"Error redeeming reward {reward_id} for user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while redeeming reward.")
    
async def track_rewards(user_id: int, db: AsyncSession):
    try:
        user = await get_user(user_id, db)
        rewards = await evaluate_rewards(user, db)
        return {
            "message": f"Rewards tracked for {user.username}",
            "rewards": rewards
        }
    except Exception as e:
        logger.error(f"Error tracking rewards for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while tracking rewards.")
    
async def get_user_achievements(user_id: int, db: AsyncSession):
    try:
        result = await db.execute(select(Achievement).filter(Achievement.user_id == user_id))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting achievements for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user achievements.")

async def get_user_rewards(user_id: int, db: AsyncSession):
    try:
        result = await db.execute(select(Reward).filter(Reward.user_id == user_id))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting rewards for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user rewards.")
