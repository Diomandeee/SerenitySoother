from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Setting as SettingModel, User as UserModel
from app.schemas import Setting, SettingCreate, SettingUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Setting)
async def create_setting(setting: SettingCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == setting.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_setting = SettingModel(
            user_id=setting.user_id,
            setting_type=setting.setting_type,
            setting_value=setting.setting_value,
        )
        db.add(db_setting)
        await db.commit()
        await db.refresh(db_setting)
        return db_setting


@router.get("/", response_model=List[Setting])
async def read_settings(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(SettingModel).offset(skip).limit(limit))
        settings = result.scalars().all()
        return settings


@router.get("/{setting_id}", response_model=Setting)
async def read_setting(setting_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(SettingModel).filter(SettingModel.id == setting_id)
        )
        setting = result.scalars().first()
        if setting is None:
            raise HTTPException(status_code=404, detail="Setting not found")
        return setting


@router.put("/{setting_id}", response_model=Setting)
async def update_setting(
    setting_id: int, setting: SettingUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(SettingModel).filter(SettingModel.id == setting_id)
        )
        db_setting = result.scalars().first()
        if db_setting is None:
            raise HTTPException(status_code=404, detail="Setting not found")

        if setting.setting_type is not None:
            db_setting.setting_type = setting.setting_type
        if setting.setting_value is not None:
            db_setting.setting_value = setting.setting_value

        await db.commit()
        await db.refresh(db_setting)
        return db_setting


@router.delete("/{setting_id}", response_model=Setting)
async def delete_setting(setting_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(SettingModel).filter(SettingModel.id == setting_id)
        )
        db_setting = result.scalars().first()
        if db_setting is None:
            raise HTTPException(status_code=404, detail="Setting not found")

        await db.delete(db_setting)
        await db.commit()
        return db_setting
