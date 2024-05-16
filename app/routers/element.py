from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Element as ElementModel, Scene as SceneModel
from app.schemas import Element, ElementCreate, ElementUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/elements",
    tags=["elements"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Element)
async def create_element(element: ElementCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(SceneModel).filter(SceneModel.id == element.scene_id)
        )
        db_scene = result.scalars().first()
        if not db_scene:
            raise HTTPException(status_code=400, detail="Scene not found")

        db_element = ElementModel(
            scene_id=element.scene_id,
            element_type=element.element_type,
            element_description=element.element_description,
            element_image=element.element_image,
            element_audio=element.element_audio,
        )
        db.add(db_element)
        await db.commit()
        await db.refresh(db_element)
        return db_element


@router.get("/", response_model=List[Element])
async def read_elements(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(ElementModel).offset(skip).limit(limit))
        elements = result.scalars().all()
        return elements


@router.get("/{element_id}", response_model=Element)
async def read_element(element_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ElementModel).filter(ElementModel.id == element_id)
        )
        element = result.scalars().first()
        if element is None:
            raise HTTPException(status_code=404, detail="Element not found")
        return element


@router.put("/{element_id}", response_model=Element)
async def update_element(
    element_id: int, element: ElementUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(ElementModel).filter(ElementModel.id == element_id)
        )
        db_element = result.scalars().first()
        if db_element is None:
            raise HTTPException(status_code=404, detail="Element not found")

        if element.element_type is not None:
            db_element.element_type = element.element_type
        if element.element_description is not None:
            db_element.element_description = element.element_description
        if element.element_image is not None:
            db_element.element_image = element.element_image
        if element.element_audio is not None:
            db_element.element_audio = element.element_audio

        await db.commit()
        await db.refresh(db_element)
        return db_element


@router.delete("/{element_id}", response_model=Element)
async def delete_element(element_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(ElementModel).filter(ElementModel.id == element_id)
        )
        db_element = result.scalars().first()
        if db_element is None:
            raise HTTPException(status_code=404, detail="Element not found")

        await db.delete(db_element)
        await db.commit()
        return db_element
