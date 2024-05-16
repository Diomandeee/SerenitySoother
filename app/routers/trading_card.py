from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.schemas import TradingCard, TradingCardCreate, TradingCardUpdate, User
from app.dependencies import get_db
from app.services import trading_card_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trading_cards",
    tags=["trading_cards"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=TradingCard)
async def create_trading_card(
    trading_card: TradingCardCreate, db: AsyncSession = Depends(get_db)
):
    return await trading_card_service.create_trading_card(
        user_id=trading_card.user_id,
        card_type=trading_card.card_type,
        card_design=trading_card.card_design,
        realm_access_url=trading_card.realm_access_url,
        box_size=trading_card.box_size,
        border=trading_card.border,
        fill_color=trading_card.fill_color,
        back_color=trading_card.back_color,
        logo_path=trading_card.logo_path,
        background_image_path=trading_card.background_image_path,
        db=db,
    )


@router.get("/", response_model=List[TradingCard])
async def read_trading_cards(
    skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    return await trading_card_service.get_all_trading_cards(
        db=db, skip=skip, limit=limit
    )


@router.get("/{trading_card_id}", response_model=TradingCard)
async def read_trading_card(trading_card_id: int, db: AsyncSession = Depends(get_db)):
    return await trading_card_service.get_trading_card_by_id(
        trading_card_id=trading_card_id, db=db
    )


@router.put("/{trading_card_id}", response_model=TradingCard)
async def update_trading_card(
    trading_card_id: int,
    trading_card: TradingCardUpdate,
    db: AsyncSession = Depends(get_db),
):
    return await trading_card_service.update_trading_card(
        trading_card_id=trading_card_id,
        card_type=trading_card.card_type,
        card_design=trading_card.card_design,
        realm_access_url=trading_card.realm_access_url,
        qr_code_url=trading_card.qr_code_url,
        db=db,
    )


@router.delete("/{trading_card_id}", response_model=TradingCard)
async def delete_trading_card(trading_card_id: int, db: AsyncSession = Depends(get_db)):
    return await trading_card_service.delete_trading_card(
        trading_card_id=trading_card_id, db=db
    )


@router.get("/recommend/{user_id}", response_model=List[TradingCard])
async def recommend_trading_cards(user_id: int, db: AsyncSession = Depends(get_db)):
    return await trading_card_service.recommend_trading_cards(user_id=user_id, db=db)


@router.get("/popular", response_model=List[TradingCard])
async def get_popular_trading_cards(db: AsyncSession = Depends(get_db)):
    return await trading_card_service.popular_trading_cards(db=db)


@router.get("/recent", response_model=List[TradingCard])
async def get_recent_trading_cards(db: AsyncSession = Depends(get_db)):
    return await trading_card_service.recent_trading_cards(db=db)


@router.get("/designs", response_model=List[str])
async def get_trading_card_designs(db: AsyncSession = Depends(get_db)):
    return await trading_card_service.trading_card_designs(db=db)


@router.get("/types", response_model=List[str])
async def get_trading_card_types(db: AsyncSession = Depends(get_db)):
    return await trading_card_service.trading_card_types(db=db)


@router.get("/by_qr_code", response_model=TradingCard)
async def get_trading_card_by_qr_code(
    qr_code_url: str, db: AsyncSession = Depends(get_db)
):
    return await trading_card_service.get_trading_card_by_qr_code(
        qr_code_url=qr_code_url, db=db
    )


@router.get("/by_type", response_model=List[TradingCard])
async def get_trading_cards_by_type(card_type: str, db: AsyncSession = Depends(get_db)):
    return await trading_card_service.get_trading_cards_by_type(
        card_type=card_type, db=db
    )


@router.get("/by_design", response_model=List[TradingCard])
async def get_trading_cards_by_design(
    card_design: str, db: AsyncSession = Depends(get_db)
):
    return await trading_card_service.get_trading_cards_by_design(
        card_design=card_design, db=db
    )


@router.get("/by_user", response_model=List[TradingCard])
async def get_trading_cards_by_user(user_id: int, db: AsyncSession = Depends(get_db)):
    return await trading_card_service.get_trading_cards_by_user(user_id=user_id, db=db)


@router.get("/recommended/{user_id}", response_model=List[TradingCard])
async def get_user_recommended_trading_cards(
    user_id: int, db: AsyncSession = Depends(get_db)
):
    return await trading_card_service.get_user_recommended_trading_cards(
        user_id=user_id, db=db
    )


@router.get("/similar_users/{user_id}", response_model=List[User])
async def get_similar_users(user_id: int, db: AsyncSession = Depends(get_db)):
    return await trading_card_service.get_similar_users(user_id=user_id, db=db)
