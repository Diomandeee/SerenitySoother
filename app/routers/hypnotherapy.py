from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies import get_db
from app.services import hypnotherapy_service
from typing import List
from app.schemas import HypnotherapyScript, HypnotherapyScriptCreate, HypnotherapySession, HypnotherapySessionCreate

router = APIRouter(
    prefix="/hypnotherapy",
    tags=["hypnotherapy"],
    responses={404: {"description": "Not found"}},
)

@router.post("/scripts/", response_model=HypnotherapyScript)
async def create_script(script: HypnotherapyScriptCreate, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.create_hypnotherapy_script(
        user_id=script.user_id,
        title=script.title,
        content=script.content,
        db=db
    )

@router.get("/scripts/{user_id}", response_model=List[HypnotherapyScript])
async def get_scripts(user_id: int, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.get_hypnotherapy_scripts(user_id, db)

@router.post("/sessions/", response_model=HypnotherapySession)
async def create_session(session: HypnotherapySessionCreate, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.create_hypnotherapy_session(
        user_id=session.user_id,
        script_id=session.script_id,
        session_notes=session.session_notes,
        db=db
    )

@router.get("/sessions/{user_id}", response_model=List[HypnotherapySession])
async def get_sessions(user_id: int, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.get_hypnotherapy_sessions(user_id, db)

@router.get("/session/{session_id}", response_model=HypnotherapySession)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.get_hypnotherapy_session(session_id, db)

@router.put("/session/{session_id}", response_model=HypnotherapySession)
async def update_session(session_id: int, session_notes: str, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.update_hypnotherapy_session(session_id, session_notes, db)

@router.delete("/session/{session_id}")
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.delete_hypnotherapy_session(session_id, db)

@router.get("/session/{session_id}/analyze")
async def analyze_session(session_id: int, db: AsyncSession = Depends(get_db)):
    return await hypnotherapy_service.analyze_hypnotherapy_session(session_id, db)