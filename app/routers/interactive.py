from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.dependencies import get_db
from app.schemas import TradingCard
from app.models import Scene, Element
from app.services.scene_element_service import handle_user_action, swap_scene_element, create_scene_element_matrix, log_scene_element_matrix

router = APIRouter()

@router.post("/combine/", response_model=TradingCard)
async def combine_scene_element_endpoint(user_id: int, scene_id: int, element_id: int, db: AsyncSession = Depends(get_db)):
    try:
        trading_card = await handle_user_action(user_id, scene_id, element_id, db)
        return trading_card
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/swap/", response_model=TradingCard)
async def swap_scene_element_endpoint(user_id: int, scene_id: int, element_id: int, db: AsyncSession = Depends(get_db)):
    try:
        trading_card = await swap_scene_element(user_id, scene_id, element_id, db)
        return trading_card
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/matrix/", response_model=dict)
async def get_scene_element_matrix(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(Scene))
        scenes = result.scalars().all()
        result = await db.execute(select(Element))
        elements = result.scalars().all()
        matrix = create_scene_element_matrix(scenes, elements)
        log_scene_element_matrix(matrix, scenes, elements)
        return {"matrix": matrix}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
