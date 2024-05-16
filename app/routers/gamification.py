from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies import get_db
from app.services import gamification_service
from typing import List
from app.schemas import Achievement, Reward

router = APIRouter(
    prefix="/gamification",
    tags=["gamification"],
    responses={404: {"description": "Not found"}},
)

@router.get("/{user_id}/achievements", response_model=List[Achievement])
async def get_achievements(user_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.get_user_achievements(user_id, db)

@router.get("/{user_id}/rewards", response_model=List[Reward])
async def get_rewards(user_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.get_user_rewards(user_id, db)

@router.post("/{user_id}/track_achievements", response_model=dict)
async def track_achievements(user_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.track_achievements(user_id, db)

@router.post("/{user_id}/track_rewards", response_model=dict)
async def track_rewards(user_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.track_rewards(user_id, db)

@router.post("/{user_id}/redeem_reward", response_model=Reward)
async def redeem_reward(user_id: int, reward_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.redeem_reward(user_id, reward_id, db)

@router.post("/{user_id}/award_reward", response_model=Reward)
async def award_reward(user_id: int, reward_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.award_reward(user_id, reward_id, db)

@router.post("/{user_id}/award_achievement", response_model=Achievement)
async def award_achievement(user_id: int, achievement_id: int, db: AsyncSession = Depends(get_db)):
    return await gamification_service.award_achievement(user_id, achievement_id, db)