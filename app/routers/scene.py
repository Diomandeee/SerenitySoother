from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Scene as SceneModel, Script as ScriptModel
from app.schemas import Scene, SceneCreate, SceneUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/scenes",
    tags=["scenes"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Scene)
async def create_scene(scene: SceneCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ScriptModel).filter(ScriptModel.id == scene.script_id)
        )
        db_script = result.scalars().first()
        if not db_script:
            raise HTTPException(status_code=400, detail="Script not found")

        db_scene = SceneModel(
            script_id=scene.script_id,
            scene_type=scene.scene_type,
            scene_description=scene.scene_description,
            scene_image=scene.scene_image,
            scene_audio=scene.scene_audio,
        )
        db.add(db_scene)
        await db.commit()
        await db.refresh(db_scene)
        return db_scene


@router.get("/", response_model=List[Scene])
async def read_scenes(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(SceneModel).offset(skip).limit(limit))
        scenes = result.scalars().all()
        return scenes


@router.get("/{scene_id}", response_model=Scene)
async def read_scene(scene_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(select(SceneModel).filter(SceneModel.id == scene_id))
        scene = result.scalars().first()
        if scene is None:
            raise HTTPException(status_code=404, detail="Scene not found")
        return scene


@router.put("/{scene_id}", response_model=Scene)
async def update_scene(
    scene_id: int, scene: SceneUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(SceneModel).filter(SceneModel.id == scene_id))
        db_scene = result.scalars().first()
        if db_scene is None:
            raise HTTPException(status_code=404, detail="Scene not found")

        if scene.scene_type is not None:
            db_scene.scene_type = scene.scene_type
        if scene.scene_description is not None:
            db_scene.scene_description = scene.scene_description
        if scene.scene_image is not None:
            db_scene.scene_image = scene.scene_image
        if scene.scene_audio is not None:
            db_scene.scene_audio = scene.scene_audio

        await db.commit()
        await db.refresh(db_scene)
        return db_scene


@router.delete("/{scene_id}", response_model=Scene)
async def delete_scene(scene_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(select(SceneModel).filter(SceneModel.id == scene_id))
        db_scene = result.scalars().first()
        if db_scene is None:
            raise HTTPException(status_code=404, detail="Scene not found")

        await db.delete(db_scene)
        await db.commit()
        return db_scene
