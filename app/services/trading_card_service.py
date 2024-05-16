from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, TradingCard
from app.services import qr_code_service
from sqlalchemy.future import select
from fastapi import HTTPException
from collections import Counter
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_trading_card(
    user_id: int,
    card_type: str,
    card_design: str,
    realm_access_url: str,
    box_size: int,
    border: int,
    fill_color: str,
    back_color: str,
    logo_path: str,
    background_image_path: str,
    db: AsyncSession,
):
    try:
        async with db.begin():
            result = await db.execute(select(User).filter(User.id == user_id))
            user = result.scalars().first()
            if not user:
                logger.error(f"User with ID {user_id} not found")
                raise HTTPException(status_code=400, detail="User not found")

            qr_code_url = qr_code_service.generate_qr_code(
                data=realm_access_url,
                box_size=box_size,
                border=border,
                fill_color=fill_color,
                back_color=back_color,
                logo_path=logo_path,
                background_image_path=background_image_path,
            )

            db_trading_card = TradingCard(
                user_id=user_id,
                card_type=card_type,
                card_design=card_design,
                realm_access_url=realm_access_url,
                qr_code_url=qr_code_url,
            )
            db.add(db_trading_card)
        await db.commit()
        await db.refresh(db_trading_card)
        logger.info(f"Trading card created with ID {db_trading_card.id}")
        return db_trading_card
    except Exception as e:
        logger.error(f"Error creating trading card: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating trading card: {str(e)}"
        )


async def get_all_trading_cards(
    db: AsyncSession, skip: int = 0, limit: int = 10
) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(select(TradingCard).offset(skip).limit(limit))
        trading_cards = result.scalars().all()
        return trading_cards


async def get_trading_card_by_id(trading_card_id: int, db: AsyncSession) -> TradingCard:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.id == trading_card_id)
        )
        trading_card = result.scalars().first()
        if trading_card is None:
            raise HTTPException(status_code=404, detail="Trading card not found")
        return trading_card


async def update_trading_card(
    trading_card_id: int,
    card_type: str,
    card_design: str,
    realm_access_url: str,
    qr_code_url: str,
    db: AsyncSession,
) -> TradingCard:
    try:
        # Using a separate transaction context manager for query and update
        async with db.begin():
            result = await db.execute(
                select(TradingCard).filter(TradingCard.id == trading_card_id)
            )
            db_trading_card = result.scalars().first()
            if db_trading_card is None:
                raise HTTPException(status_code=404, detail="Trading card not found")

        # Update the trading card details
        async with db.begin():
            db_trading_card.card_type = card_type
            db_trading_card.card_design = card_design
            db_trading_card.realm_access_url = realm_access_url
            db_trading_card.qr_code_url = qr_code_url

            db.add(db_trading_card)
            await db.commit()
            await db.refresh(db_trading_card)
            return db_trading_card
    except Exception as e:
        logger.error(f"Error updating trading card: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating trading card: {str(e)}"
        )


async def delete_trading_card(trading_card_id: int, db: AsyncSession) -> TradingCard:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.id == trading_card_id)
        )
        db_trading_card = result.scalars().first()
        if db_trading_card is None:
            raise HTTPException(status_code=404, detail="Trading card not found")

        await db.delete(db_trading_card)
        await db.commit()
        return db_trading_card


async def recommend_trading_cards(user_id: int, db: AsyncSession) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.user_id == user_id)
        )
        user_trading_cards = result.scalars().all()

        user_card_types = [card.card_type for card in user_trading_cards]
        result = await db.execute(select(TradingCard))
        all_trading_cards = result.scalars().all()

        recommended_cards = [
            card for card in all_trading_cards if card.card_type not in user_card_types
        ]

        # Further filter recommendations by collaborative filtering
        user_similarity = await calculate_user_similarity(
            user_id, user_trading_cards, db
        )
        similar_users = sorted(
            user_similarity.items(), key=lambda x: x[1], reverse=True
        )[:5]

        for similar_user_id, _ in similar_users:
            similar_user_cards = await get_user_trading_cards(similar_user_id, db)
            for card in similar_user_cards:
                if card.card_type not in user_card_types:
                    recommended_cards.append(card)

        return list(set(recommended_cards))[:3]  # Return top 3 recommendations


async def get_user_trading_cards(user_id: int, db: AsyncSession) -> List[TradingCard]:
    result = await db.execute(
        select(TradingCard).filter(TradingCard.user_id == user_id)
    )
    return result.scalars().all()


async def calculate_user_similarity(
    user_id: int, user_trading_cards: List[TradingCard], db: AsyncSession
):
    user_card_types = [card.card_type for card in user_trading_cards]
    result = await db.execute(select(User))
    all_users = result.scalars().all()

    user_similarity = {}
    for other_user in all_users:
        if other_user.id == user_id:
            continue
        other_user_cards = await get_user_trading_cards(other_user.id, db)
        other_user_card_types = [card.card_type for card in other_user_cards]
        common_types = set(user_card_types) & set(other_user_card_types)
        if not common_types:
            continue
        user_similarity[other_user.id] = len(common_types) / len(
            set(user_card_types) | set(other_user_card_types)
        )

    return user_similarity


async def popular_trading_cards(db: AsyncSession) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(select(TradingCard))
        trading_cards = result.scalars().all()

        card_types = [card.card_type for card in trading_cards]
        popular_card_type = Counter(card_types).most_common(1)[0][0]

        return [card for card in trading_cards if card.card_type == popular_card_type]


async def recent_trading_cards(db: AsyncSession) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).order_by(TradingCard.created_at.desc())
        )
        trading_cards = result.scalars().all()
        return trading_cards


async def trading_card_designs(db: AsyncSession) -> List[str]:
    async with db.begin():
        result = await db.execute(select(TradingCard.card_design).distinct())
        designs = result.scalars().all()
        return designs


async def trading_card_types(db: AsyncSession) -> List[str]:
    async with db.begin():
        result = await db.execute(select(TradingCard.card_type).distinct())
        types = result.scalars().all()
        return types


async def get_trading_card_by_qr_code(
    qr_code_url: str, db: AsyncSession
) -> TradingCard:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.qr_code_url == qr_code_url)
        )
        trading_card = result.scalars().first()
        if trading_card is None:
            raise HTTPException(status_code=404, detail="Trading card not found")
        return trading_card


async def get_trading_cards_by_type(
    card_type: str, db: AsyncSession
) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.card_type == card_type)
        )
        trading_cards = result.scalars().all()
        return trading_cards


async def get_trading_cards_by_design(
    card_design: str, db: AsyncSession
) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.card_design == card_design)
        )
        trading_cards = result.scalars().all()
        return trading_cards


async def get_trading_cards_by_user(
    user_id: int, db: AsyncSession
) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.user_id == user_id)
        )
        trading_cards = result.scalars().all()
        return trading_cards


async def get_user_recommended_trading_cards(
    user_id: int, db: AsyncSession
) -> List[TradingCard]:
    async with db.begin():
        result = await db.execute(
            select(TradingCard).filter(TradingCard.user_id == user_id)
        )
        user_trading_cards = result.scalars().all()

        user_card_types = [card.card_type for card in user_trading_cards]
        result = await db.execute(select(TradingCard))
        all_trading_cards = result.scalars().all()

        recommended_cards = [
            card for card in all_trading_cards if card.card_type not in user_card_types
        ]

        # Further filter recommendations by collaborative filtering
        user_similarity = await calculate_user_similarity(
            user_id, user_trading_cards, db
        )
        similar_users = sorted(
            user_similarity.items(), key=lambda x: x[1], reverse=True
        )[:5]

        for similar_user_id, _ in similar_users:
            similar_user_cards = await get_user_trading_cards(similar_user_id, db)
            for card in similar_user_cards:
                if card.card_type not in user_card_types:
                    recommended_cards.append(card)

        return list(set(recommended_cards))[:3]  # Return top 3 recommendations


async def get_similar_users(user_id: int, db: AsyncSession) -> List[User]:
    async with db.begin():
        result = await db.execute(select(User))
        all_users = result.scalars().all()

        user_similarity = await calculate_user_similarity(user_id, [], db)
        similar_users = sorted(
            user_similarity.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return [
            user
            for user in all_users
            if user.id in [similar_user_id for similar_user_id, _ in similar_users]
        ]
