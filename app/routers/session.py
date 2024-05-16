from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models import Session as SessionModel, User as UserModel
from app.schemas import Session, SessionCreate, SessionUpdate
from app.dependencies import get_db

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=Session)
async def create_session(session: SessionCreate, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(UserModel).filter(UserModel.id == session.user_id)
        )
        db_user = result.scalars().first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        db_session = SessionModel(
            user_id=session.user_id,
            session_date=session.session_date,
            session_type=session.session_type,
            session_status=session.session_status,
            session_duration=session.session_duration,
            session_description=session.session_description,
        )
        db.add(db_session)
        await db.commit()
        await db.refresh(db_session)
        return db_session


@router.get("/", response_model=List[Session])
async def read_sessions(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(SessionModel).offset(skip).limit(limit))
        sessions = result.scalars().all()
        return sessions


@router.get("/{session_id}", response_model=Session)
async def read_session(session_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(SessionModel).filter(SessionModel.id == session_id)
        )
        session = result.scalars().first()
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session


@router.put("/{session_id}", response_model=Session)
async def update_session(
    session_id: int, session: SessionUpdate, db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(
            select(SessionModel).filter(SessionModel.id == session_id)
        )
        db_session = result.scalars().first()
        if db_session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.session_date:
            db_session.session_date = session.session_date
        if session.session_type:
            db_session.session_type = session.session_type
        if session.session_status:
            db_session.session_status = session.session_status
        if session.session_duration:
            db_session.session_duration = session.session_duration
        if session.session_description:
            db_session.session_description = session.session_description

        await db.commit()
        await db.refresh(db_session)
        return db_session


@router.delete("/{session_id}", response_model=Session)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    async with db.begin():
        result = await db.execute(
            select(SessionModel).filter(SessionModel.id == session_id)
        )
        db_session = result.scalars().first()
        if db_session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        await db.delete(db_session)
        await db.commit()
        return db_session
