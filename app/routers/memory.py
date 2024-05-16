from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Memory as MemoryModel, User as UserModel
from app.schemas import Memory, MemoryCreate, MemoryUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/memories",
    tags=["memories"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Memory)
async def create_memory(memory: MemoryCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == memory.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_memory = MemoryModel(
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            memory_description=memory.memory_description,
            memory_intensity=memory.memory_intensity,
            memory_date=memory.memory_date,
        )
        db.add(db_memory)
        await db.commit()
        await db.refresh(db_memory)
        return db_memory


@router.get("/", response_model=List[Memory])
async def read_memories(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(MemoryModel).offset(skip).limit(limit))
        memories = result.scalars().all()
        return memories


@router.get("/{memory_id}", response_model=Memory)
async def read_memory(memory_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(MemoryModel).filter(MemoryModel.id == memory_id)
        )
        memory = result.scalars().first()
        if memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory


@router.put("/{memory_id}", response_model=Memory)
async def update_memory(
    memory_id: int, memory: MemoryUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(MemoryModel).filter(MemoryModel.id == memory_id)
        )
        db_memory = result.scalars().first()
        if db_memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        if memory.memory_type is not None:
            db_memory.memory_type = memory.memory_type
        if memory.memory_description is not None:
            db_memory.memory_description = memory.memory_description
        if memory.memory_intensity is not None:
            db_memory.memory_intensity = memory.memory_intensity
        if memory.memory_date is not None:
            db_memory.memory_date = memory.memory_date

        await db.commit()
        await db.refresh(db_memory)
        return db_memory


@router.delete("/{memory_id}", response_model=Memory)
async def delete_memory(memory_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(MemoryModel).filter(MemoryModel.id == memory_id)
        )
        db_memory = result.scalars().first()
        if db_memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        await db.delete(db_memory)
        await db.commit()
        return db_memory
