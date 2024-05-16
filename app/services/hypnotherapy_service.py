from sqlalchemy.ext.asyncio import AsyncSession
from app.models import HypnotherapyScript, HypnotherapySession
from fastapi import HTTPException
from sqlalchemy.future import select
from typing import List
import logging

logger = logging.getLogger(__name__)

async def create_hypnotherapy_script(user_id: int, title: str, content: str, db: AsyncSession):
    try:
        new_script = HypnotherapyScript(user_id=user_id, title=title, content=content)
        db.add(new_script)
        await db.commit()
        await db.refresh(new_script)
        return new_script
    except Exception as e:
        logger.error(f"Error creating hypnotherapy script for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the hypnotherapy script.")

async def get_hypnotherapy_scripts(user_id: int, db: AsyncSession) -> List[HypnotherapyScript]:
    try:
        result = await db.execute(select(HypnotherapyScript).filter(HypnotherapyScript.user_id == user_id))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting hypnotherapy scripts for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching hypnotherapy scripts.")

async def create_hypnotherapy_session(user_id: int, script_id: int, session_notes: str, db: AsyncSession):
    try:
        new_session = HypnotherapySession(user_id=user_id, script_id=script_id, session_notes=session_notes)
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session
    except Exception as e:
        logger.error(f"Error creating hypnotherapy session for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the hypnotherapy session.")

async def get_hypnotherapy_sessions(user_id: int, db: AsyncSession) -> List[HypnotherapySession]:
    try:
        result = await db.execute(select(HypnotherapySession).filter(HypnotherapySession.user_id == user_id))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error getting hypnotherapy sessions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching hypnotherapy sessions.")

async def get_hypnotherapy_session(session_id: int, db: AsyncSession) -> HypnotherapySession:
    try:
        result = await db.execute(select(HypnotherapySession).filter(HypnotherapySession.id == session_id))
        session = result.scalars().first()
        if not session:
            raise HTTPException(status_code=404, detail="Hypnotherapy session not found")
        return session
    except Exception as e:
        logger.error(f"Error getting hypnotherapy session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching hypnotherapy session.")
    
async def update_hypnotherapy_session(session_id: int, session_notes: str, db: AsyncSession) -> HypnotherapySession:
    try:
        result = await db.execute(select(HypnotherapySession).filter(HypnotherapySession.id == session_id))
        session = result.scalars().first()
        if not session:
            raise HTTPException(status_code=404, detail="Hypnotherapy session not found")
        session.session_notes = session_notes
        await db.commit()
        await db.refresh(session)
        return session
    except Exception as e:
        logger.error(f"Error updating hypnotherapy session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while updating hypnotherapy session.")
    
async def delete_hypnotherapy_session(session_id: int, db: AsyncSession):   
    try:
        result = await db.execute(select(HypnotherapySession).filter(HypnotherapySession.id == session_id))
        session = result.scalars().first()
        if not session:
            raise HTTPException(status_code=404, detail="Hypnotherapy session not found")
        db.delete(session)
        await db.commit()
    except Exception as e:
        logger.error(f"Error deleting hypnotherapy session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting hypnotherapy session.")
    

async def analyze_hypnotherapy_session(session_id: int, db: AsyncSession):
    try:
        result = await db.execute(select(HypnotherapySession).filter(HypnotherapySession.id == session_id))
        session = result.scalars().first()
        if not session:
            raise HTTPException(status_code=404, detail="Hypnotherapy session not found")
        # Perform analysis on the session
        return {"analysis": "Session analysis not implemented yet"}
    except Exception as e:
        logger.error(f"Error analyzing hypnotherapy session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing hypnotherapy session.")