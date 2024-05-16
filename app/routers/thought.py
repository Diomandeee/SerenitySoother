from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Thought as ThoughtModel, User as UserModel
from app.schemas import Thought, ThoughtCreate, ThoughtUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/thoughts",
    tags=["thoughts"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Thought)
async def create_thought(thought: ThoughtCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == thought.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_thought = ThoughtModel(
            user_id=thought.user_id,
            thought_type=thought.thought_type,
            thought_description=thought.thought_description,
            thought_date=thought.thought_date,
        )
        db.add(db_thought)
        await db.commit()
        await db.refresh(db_thought)
        return db_thought


@router.get("/", response_model=List[Thought])
async def read_thoughts(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(ThoughtModel).offset(skip).limit(limit))
        thoughts = result.scalars().all()
        return thoughts


@router.get("/{thought_id}", response_model=Thought)
async def read_thought(thought_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ThoughtModel).filter(ThoughtModel.id == thought_id)
        )
        thought = result.scalars().first()
        if thought is None:
            raise HTTPException(status_code=404, detail="Thought not found")
        return thought


@router.put("/{thought_id}", response_model=Thought)
async def update_thought(
    thought_id: int, thought: ThoughtUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(ThoughtModel).filter(ThoughtModel.id == thought_id)
        )
        db_thought = result.scalars().first()
        if db_thought is None:
            raise HTTPException(status_code=404, detail="Thought not found")

        if thought.thought_type is not None:
            db_thought.thought_type = thought.thought_type
        if thought.thought_description is not None:
            db_thought.thought_description = thought.thought_description
        if thought.thought_date is not None:
            db_thought.thought_date = thought.thought_date

        await db.commit()
        await db.refresh(db_thought)
        return db_thought


@router.delete("/{thought_id}", response_model=Thought)
async def delete_thought(thought_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ThoughtModel).filter(ThoughtModel.id == thought_id)
        )
        db_thought = result.scalars().first()
        if db_thought is None:
            raise HTTPException(status_code=404, detail="Thought not found")

        await db.delete(db_thought)
        await db.commit()
        return db_thought
