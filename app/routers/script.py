from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Script as ScriptModel, Session as SessionModel
from app.schemas import Script, ScriptCreate, ScriptUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/scripts",
    tags=["scripts"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Script)
async def create_script(script: ScriptCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(SessionModel).filter(SessionModel.id == script.session_id)
        )
        db_session = result.scalars().first()
        if not db_session:
            raise HTTPException(status_code=400, detail="Session not found")

        db_script = ScriptModel(
            session_id=script.session_id,
            script_type=script.script_type,
            script_content=script.script_content,
            script_rating=script.script_rating,
            script_description=script.script_description,
            script_image=script.script_image,
        )
        db.add(db_script)
        await db.commit()
        await db.refresh(db_script)
        return db_script


@router.get("/", response_model=List[Script])
async def read_scripts(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(ScriptModel).offset(skip).limit(limit))
        scripts = result.scalars().all()
        return scripts


@router.get("/{script_id}", response_model=Script)
async def read_script(script_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ScriptModel).filter(ScriptModel.id == script_id)
        )
        script = result.scalars().first()
        if script is None:
            raise HTTPException(status_code=404, detail="Script not found")
        return script


@router.put("/{script_id}", response_model=Script)
async def update_script(
    script_id: int, script: ScriptUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(ScriptModel).filter(ScriptModel.id == script_id)
        )
        db_script = result.scalars().first()
        if db_script is None:
            raise HTTPException(status_code=404, detail="Script not found")

        if script.script_type is not None:
            db_script.script_type = script.script_type
        if script.script_content is not None:
            db_script.script_content = script.script_content
        if script.script_rating is not None:
            db_script.script_rating = script.script_rating
        if script.script_description is not None:
            db_script.script_description = script.script_description
        if script.script_image is not None:
            db_script.script_image = script.script_image

        await db.commit()
        await db.refresh(db_script)
        return db_script


@router.delete("/{script_id}", response_model=Script)
async def delete_script(script_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ScriptModel).filter(ScriptModel.id == script_id)
        )
        db_script = result.scalars().first()
        if db_script is None:
            raise HTTPException(status_code=404, detail="Script not found")

        await db.delete(db_script)
        await db.commit()
        return db_script
