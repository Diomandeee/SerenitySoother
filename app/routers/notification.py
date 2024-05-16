from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Notification as NotificationModel, User as UserModel
from app.schemas import Notification, NotificationCreate, NotificationUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/notifications",
    tags=["notifications"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Notification)
async def create_notification(
    notification: NotificationCreate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == notification.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_notification = NotificationModel(
            user_id=notification.user_id,
            notification_type=notification.notification_type,
            notification_message=notification.notification_message,
            notification_date=notification.notification_date,
        )
        db.add(db_notification)
        await db.commit()
        await db.refresh(db_notification)
        return db_notification


@router.get("/", response_model=List[Notification])
async def read_notifications(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(NotificationModel).offset(skip).limit(limit))
        notifications = result.scalars().all()
        return notifications


@router.get("/{notification_id}", response_model=Notification)
async def read_notification(notification_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(NotificationModel).filter(NotificationModel.id == notification_id)
        )
        notification = result.scalars().first()
        if notification is None:
            raise HTTPException(status_code=404, detail="Notification not found")
        return notification


@router.put("/{notification_id}", response_model=Notification)
async def update_notification(
    notification_id: int,
    notification: NotificationUpdate,
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        result = await db.execute(
            select(NotificationModel).filter(NotificationModel.id == notification_id)
        )
        db_notification = result.scalars().first()
        if db_notification is None:
            raise HTTPException(status_code=404, detail="Notification not found")

        if notification.notification_type is not None:
            db_notification.notification_type = notification.notification_type
        if notification.notification_message is not None:
            db_notification.notification_message = notification.notification_message
        if notification.notification_date is not None:
            db_notification.notification_date = notification.notification_date

        await db.commit()
        await db.refresh(db_notification)
        return db_notification


@router.delete("/{notification_id}", response_model=Notification)
async def delete_notification(notification_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(NotificationModel).filter(NotificationModel.id == notification_id)
        )
        db_notification = result.scalars().first()
        if db_notification is None:
            raise HTTPException(status_code=404, detail="Notification not found")

        await db.delete(db_notification)
        await db.commit()
        return db_notification
