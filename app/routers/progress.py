from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Progress as ProgressModel, User as UserModel, Goal as GoalModel
from app.schemas import Progress, ProgressCreate, ProgressUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/progress",
    tags=["progress"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Progress)
async def create_progress(progress: ProgressCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        user_result = await db.execute(
            select(UserModel).filter(UserModel.id == progress.user_id)
        )
        db_user = user_result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        goal_result = await db.execute(
            select(GoalModel).filter(GoalModel.id == progress.goal_id)
        )
        db_goal = goal_result.scalars().first()
        if not db_goal:
            raise HTTPException(status_code=400, detail="Goal not found")

        db_progress = ProgressModel(
            user_id=progress.user_id,
            goal_id=progress.goal_id,
            progress_status=progress.progress_status,
            progress_description=progress.progress_description,
            progress_date=progress.progress_date,
        )
        db.add(db_progress)
        await db.commit()
        await db.refresh(db_progress)
        return db_progress


@router.get("/", response_model=List[Progress])
async def read_progress(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(ProgressModel).offset(skip).limit(limit))
        progress_entries = result.scalars().all()
        return progress_entries


@router.get("/{progress_id}", response_model=Progress)
async def read_progress_entry(progress_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ProgressModel).filter(ProgressModel.id == progress_id)
        )
        progress_entry = result.scalars().first()
        if progress_entry is None:
            raise HTTPException(status_code=404, detail="Progress entry not found")
        return progress_entry


@router.put("/{progress_id}", response_model=Progress)
async def update_progress(
    progress_id: int, progress: ProgressUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(ProgressModel).filter(ProgressModel.id == progress_id)
        )
        db_progress = result.scalars().first()
        if db_progress is None:
            raise HTTPException(status_code=404, detail="Progress entry not found")

        if progress.progress_status is not None:
            db_progress.progress_status = progress.progress_status
        if progress.progress_description is not None:
            db_progress.progress_description = progress.progress_description
        if progress.progress_date is not None:
            db_progress.progress_date = progress.progress_date

        await db.commit()
        await db.refresh(db_progress)
        return db_progress


@router.delete("/{progress_id}", response_model=Progress)
async def delete_progress(progress_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ProgressModel).filter(ProgressModel.id == progress_id)
        )
        db_progress = result.scalars().first()
        if db_progress is None:
            raise HTTPException(status_code=404, detail="Progress entry not found")

        await db.delete(db_progress)
        await db.commit()
        return db_progress
