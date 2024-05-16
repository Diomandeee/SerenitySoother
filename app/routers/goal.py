from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Goal as GoalModel, User as UserModel
from app.schemas import Goal, GoalCreate, GoalUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/goals",
    tags=["goals"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Goal)
async def create_goal(goal: GoalCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == goal.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_goal = GoalModel(
            user_id=goal.user_id,
            goal_type=goal.goal_type,
            goal_description=goal.goal_description,
            goal_status=goal.goal_status,
            goal_deadline=goal.goal_deadline,
        )
        db.add(db_goal)
        await db.commit()
        await db.refresh(db_goal)
        return db_goal


@router.get("/", response_model=List[Goal])
async def read_goals(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(GoalModel).offset(skip).limit(limit))
        goals = result.scalars().all()
        return goals


@router.get("/{goal_id}", response_model=Goal)
async def read_goal(goal_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(select(GoalModel).filter(GoalModel.id == goal_id))
        goal = result.scalars().first()
        if goal is None:
            raise HTTPException(status_code=404, detail="Goal not found")
        return goal


@router.put("/{goal_id}", response_model=Goal)
async def update_goal(
    goal_id: int, goal: GoalUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(GoalModel).filter(GoalModel.id == goal_id))
        db_goal = result.scalars().first()
        if db_goal is None:
            raise HTTPException(status_code=404, detail="Goal not found")

        if goal.goal_type is not None:
            db_goal.goal_type = goal.goal_type
        if goal.goal_description is not None:
            db_goal.goal_description = goal.goal_description
        if goal.goal_status is not None:
            db_goal.goal_status = goal.goal_status
        if goal.goal_deadline is not None:
            db_goal.goal_deadline = goal.goal_deadline

        await db.commit()
        await db.refresh(db_goal)
        return db_goal


@router.delete("/{goal_id}", response_model=Goal)
async def delete_goal(goal_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(select(GoalModel).filter(GoalModel.id == goal_id))
        db_goal = result.scalars().first()
        if db_goal is None:
            raise HTTPException(status_code=404, detail="Goal not found")

        await db.delete(db_goal)
        await db.commit()
        return db_goal
