from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Emotion as EmotionModel, User as UserModel
from app.schemas import Emotion, EmotionCreate, EmotionUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/emotions",
    tags=["emotions"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Emotion)
async def create_emotion(emotion: EmotionCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == emotion.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_emotion = EmotionModel(
            user_id=emotion.user_id,
            emotion_type=emotion.emotion_type,
            emotion_intensity=emotion.emotion_intensity,
            emotion_description=emotion.emotion_description,
            emotion_date=emotion.emotion_date,
        )
        db.add(db_emotion)
        await db.commit()
        await db.refresh(db_emotion)
        return db_emotion


@router.get("/", response_model=List[Emotion])
async def read_emotions(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(EmotionModel).offset(skip).limit(limit))
        emotions = result.scalars().all()
        return emotions


@router.get("/{emotion_id}", response_model=Emotion)
async def read_emotion(emotion_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(EmotionModel).filter(EmotionModel.id == emotion_id)
        )
        emotion = result.scalars().first()
        if emotion is None:
            raise HTTPException(status_code=404, detail="Emotion not found")
        return emotion


@router.put("/{emotion_id}", response_model=Emotion)
async def update_emotion(
    emotion_id: int, emotion: EmotionUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(EmotionModel).filter(EmotionModel.id == emotion_id)
        )
        db_emotion = result.scalars().first()
        if db_emotion is None:
            raise HTTPException(status_code=404, detail="Emotion not found")

        if emotion.emotion_type is not None:
            db_emotion.emotion_type = emotion.emotion_type
        if emotion.emotion_intensity is not None:
            db_emotion.emotion_intensity = emotion.emotion_intensity
        if emotion.emotion_description is not None:
            db_emotion.emotion_description = emotion.emotion_description
        if emotion.emotion_date is not None:
            db_emotion.emotion_date = emotion.emotion_date

        await db.commit()
        await db.refresh(db_emotion)
        return db_emotion


@router.delete("/{emotion_id}", response_model=Emotion)
async def delete_emotion(emotion_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(EmotionModel).filter(EmotionModel.id == emotion_id)
        )
        db_emotion = result.scalars().first()
        if db_emotion is None:
            raise HTTPException(status_code=404, detail="Emotion not found")

        await db.delete(db_emotion)
        await db.commit()
        return db_emotion
