from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, Script, Scene, TradingCard, Session
from app.services.utils import (
    get_user,
    get_user_sessions,
    get_user_scripts,
    get_user_trading_cards,
    calculate_similarity,
    calculate_user_similarity,
)
from collections import Counter
from typing import List
from sqlalchemy.future import select


async def recommend_for_user(user_id: int, db: AsyncSession):
    user = await get_user(user_id, db)
    sessions = await get_user_sessions(user_id, db)
    scripts = await get_user_scripts([session.id for session in sessions], db)
    trading_cards = await get_user_trading_cards(user_id, db)

    recommended_scripts = await recommend_scripts(user, sessions, scripts, db)
    recommended_scenes = await recommend_scenes(user, sessions, scripts, db)
    recommended_trading_cards = await recommend_trading_cards(user, trading_cards, db)

    return {
        "message": f"Personalized recommendations for {user.username}",
        "recommended_scripts": recommended_scripts,
        "recommended_scenes": recommended_scenes,
        "recommended_trading_cards": recommended_trading_cards,
    }


async def recommend_scripts(
    user, sessions: List[Session], scripts: List[Script], db: AsyncSession
):
    user_sessions = [session.id for session in sessions]
    user_scripts = [script for script in scripts if script.session_id in user_sessions]
    user_ratings = {
        script.id: script.script_rating
        for script in user_scripts
        if script.script_rating
    }

    result = await db.execute(select(User))
    all_users = result.scalars().all()

    user_similarity = {}
    for other_user in all_users:
        if other_user.id == user.id:
            continue
        other_sessions = await get_user_sessions(other_user.id, db)
        other_scripts = await get_user_scripts(
            [session.id for session in other_sessions], db
        )
        other_ratings = {
            script.id: script.script_rating
            for script in other_scripts
            if script.script_rating
        }
        similarity = calculate_similarity(user_ratings, other_ratings)
        user_similarity[other_user.id] = similarity

    similar_users = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]

    recommended_scripts = set()
    for similar_user_id, _ in similar_users:
        similar_user_sessions = await get_user_sessions(similar_user_id, db)
        similar_user_scripts = await get_user_scripts(
            [session.id for session in similar_user_sessions], db
        )
        for script in similar_user_scripts:
            if script.id not in user_ratings:
                recommended_scripts.add(script)

    return list(recommended_scripts)[:3]


async def recommend_scenes(
    user: User, sessions: List[Session], scripts: List[Script], db: AsyncSession
):
    if not scripts:
        return []

    script_types = [script.script_type for script in scripts]
    most_common_type = Counter(script_types).most_common(1)[0][0]

    result = await db.execute(
        select(Scene).filter(
            Scene.script_id.in_(
                [
                    script.id
                    for script in scripts
                    if script.script_type == most_common_type
                ]
            )
        )
    )
    scenes = result.scalars().all()

    return [scene for scene in scenes[:3]]


async def recommend_trading_cards(
    user: User, trading_cards: List[TradingCard], db: AsyncSession
):
    if not trading_cards:
        return []

    user_card_types = [card.card_type for card in trading_cards]

    result = await db.execute(select(TradingCard))
    all_cards = result.scalars().all()

    recommended_cards = [
        card for card in all_cards if card.card_type not in user_card_types
    ]

    user_similarity = await calculate_user_similarity(user.id, trading_cards, db)
    similar_users = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]

    for similar_user_id, _ in similar_users:
        similar_user_cards = await get_user_trading_cards(similar_user_id, db)
        for card in similar_user_cards:
            if card.card_type not in user_card_types:
                recommended_cards.append(card)

    return list(set(recommended_cards))[:3]
