from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User, Script, Scene, TradingCard, Session
from sqlalchemy.future import select
from app.helper import log_handler
from fastapi import HTTPException
from collections import Counter
from app.services.utils import (
    get_user,
    get_user_sessions,
    get_user_scripts,
    get_user_trading_cards,
    get_user_goals,
    get_user_emotions,
    calculate_similarity,
    calculate_user_similarity,
    recommend_goals,
    recommend_based_on_mood,
)
from typing import List


async def personalize_user_experience(user_id: int, db: AsyncSession):
    try:
        user = await get_user(user_id, db)
        sessions = await get_user_sessions(user_id, db)
        scripts = await get_user_scripts([session.id for session in sessions], db)
        trading_cards = await get_user_trading_cards(user_id, db)
        goals = await get_user_goals(user_id, db)
        emotions = await get_user_emotions(user_id, db)

        recommended_scripts = await recommend_scripts(user, sessions, scripts, db)
        recommended_scenes = await recommend_scenes(user, sessions, scripts, db)
        recommended_trading_cards = await recommend_trading_cards(
            user, trading_cards, db
        )
        recommended_goals = recommend_goals(goals)
        mood_based_recommendations = await recommend_based_on_mood(emotions, db)

        return {
            "message": f"Personalized experience for {user.username}",
            "recommended_scripts": recommended_scripts,
            "recommended_scenes": recommended_scenes,
            "recommended_trading_cards": recommended_trading_cards,
            "recommended_goals": recommended_goals,
            "mood_based_recommendations": mood_based_recommendations,
        }
    except Exception as e:
        log_handler(f"Error personalizing user experience for user_id {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while personalizing the user experience.",
        )


async def recommend_scripts(
    user, sessions: List[Session], scripts: List[Script], db: AsyncSession
):
    try:
        user_sessions = [session.id for session in sessions]
        user_scripts = [
            script for script in scripts if script.session_id in user_sessions
        ]
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

        similar_users = sorted(
            user_similarity.items(), key=lambda x: x[1], reverse=True
        )[:5]

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
    except Exception as e:
        log_handler(f"Error recommending scripts for user {user.username}: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while recommending scripts."
        )


async def recommend_scenes(
    user: User, sessions: List[Session], scripts: List[Script], db: AsyncSession
):
    try:
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
    except Exception as e:
        log_handler(f"Error recommending scenes for user {user.username}: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while recommending scenes."
        )


async def recommend_trading_cards(
    user: User, trading_cards: List[TradingCard], db: AsyncSession
):
    try:
        if not trading_cards:
            return []

        user_card_types = [card.card_type for card in trading_cards]

        result = await db.execute(select(TradingCard))
        all_cards = result.scalars().all()

        recommended_cards = [
            card for card in all_cards if card.card_type not in user_card_types
        ]

        user_similarity = await calculate_user_similarity(user.id, trading_cards, db)
        similar_users = sorted(
            user_similarity.items(), key=lambda x: x[1], reverse=True
        )[:5]

        for similar_user_id, _ in similar_users:
            similar_user_cards = await get_user_trading_cards(similar_user_id, db)
            for card in similar_user_cards:
                if card.card_type not in user_card_types:
                    recommended_cards.append(card)

        return list(set(recommended_cards))[:3]
    except Exception as e:
        log_handler(f"Error recommending trading cards for user {user.username}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while recommending trading cards.",
        )
