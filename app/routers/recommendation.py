from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies import get_db
from app.services import recommendation_service

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)


@router.get("/{user_id}", response_model=dict)
async def get_recommendations(user_id: int, db: AsyncSession = Depends(get_db)):
    try:
        return await recommendation_service.recommend_for_user(user_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
