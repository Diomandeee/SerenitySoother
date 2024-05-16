from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
import bcrypt
from datetime import datetime
from app.models import User as UserModel
from app.schemas import User, UserCreate, UserUpdate
from app.dependencies import get_db
from app.services import personalization_service


router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=User)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel).filter(UserModel.email == user.email))
    db_user = result.scalars().first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt())
    db_user = UserModel(
        username=user.username,
        email=user.email,
        password=hashed_password.decode("utf-8"),
        profile_information=user.profile_information,
        bio=user.bio,
        join_date=datetime.now(),
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


@router.get("/", response_model=List[User])
async def read_users(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserModel).offset(skip).limit(limit))
    users = result.scalars().all()
    return users


@router.get("/{user_id}", response_model=User)
async def read_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel).filter(UserModel.id == user_id))
    user = result.scalars().first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int, user: UserUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserModel).filter(UserModel.id == user_id))
    db_user = result.scalars().first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.username:
        db_user.username = user.username
    if user.email:
        db_user.email = user.email
    if user.password:
        hashed_password = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt())
        db_user.password = hashed_password.decode("utf-8")
    if user.profile_information:
        db_user.profile_information = user.profile_information
    if user.bio:
        db_user.bio = user.bio

    await db.commit()
    await db.refresh(db_user)
    return db_user


@router.delete("/{user_id}", response_model=User)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserModel).filter(UserModel.id == user_id))
    db_user = result.scalars().first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(db_user)
    await db.commit()
    return db_user


@router.get("/{user_id}/personalize", response_model=dict)
async def personalize_user_experience(user_id: int, db: AsyncSession = Depends(get_db)):
    return await personalization_service.personalize_user_experience(user_id, db)
