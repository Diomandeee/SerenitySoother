from app.services.trading_card_service import create_trading_card
from sqlalchemy.ext.asyncio import AsyncSession
from colorsys import rgb_to_hsv, rgb_to_yiq
from app.schemas import TradingCardCreate
from sqlalchemy.future import select
from app.helper import log_handler
from typing import List, Tuple
from app.models import (
    Scene,
    Element,
    TradingCard,
    User,
    scene_element_association,
    Session,
    Emotion,
)
import numpy as np


# Compatibility rules dictionary
COMPATIBILITY_RULES = {
    "Nature": ["Plants", "Landscapes", "Water features", "Celestial bodies"],
    "Fantasy": [
        "Creatures",
        "Abstract shapes",
        "Celestial bodies",
        "Musical instruments",
    ],
    "Abstract": ["Abstract shapes", "Fractals"],
    "Urban": ["Objects", "Architecture", "Musical instruments"],
    "Futuristic": [
        "Objects",
        "Abstract shapes",
        "Celestial bodies",
        "Musical instruments",
        "Fractals",
    ],
    "Historical": ["Objects", "Architecture", "Landscapes", "Musical instruments"],
    "Dreamscapes": [
        "Creatures",
        "Abstract shapes",
        "Celestial bodies",
        "Musical instruments",
    ],
    "Surrealism": ["Abstract shapes", "Fractals", "Celestial bodies"],
    "Steampunk": ["Objects", "Architecture", "Musical instruments"],
    "Cyberpunk": [
        "Objects",
        "Architecture",
        "Abstract shapes",
        "Celestial bodies",
        "Musical instruments",
    ],
}


def is_compatible(scene: Scene, element: Element) -> bool:
    """
    Define the compatibility rules between a scene and an element.
    """
    return element.element_type in COMPATIBILITY_RULES.get(scene.scene_type, [])


def rgb_to_lab(rgb):
    """Convert RGB to LAB color space."""

    def pivot_rgb(value):
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = map(pivot_rgb, rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x, y, z = [
        value / ref for value, ref in zip((x, y, z), (0.95047, 1.00000, 1.08883))
    ]

    def pivot_xyz(value):
        return value ** (1 / 3) if value > 0.008856 else (7.787 * value) + (16 / 116)

    l = (116 * pivot_xyz(y)) - 16
    a = 500 * (pivot_xyz(x) - pivot_xyz(y))
    b = 200 * (pivot_xyz(y) - pivot_xyz(z))

    return l, a, b


def delta_e(lab1, lab2):
    """Calculate the Delta E (CIE76) distance between two LAB colors."""
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(lab1, lab2)))


def color_harmony(
    scene_color: Tuple[int, int, int], element_color: Tuple[int, int, int]
) -> float:
    """
    Calculate the color harmony between scene and element using advanced color theory concepts.
    """
    # Convert RGB to HSV
    scene_hsv = rgb_to_hsv(
        scene_color[0] / 255, scene_color[1] / 255, scene_color[2] / 255
    )
    element_hsv = rgb_to_hsv(
        element_color[0] / 255, element_color[1] / 255, element_color[2] / 255
    )

    # Calculate hue difference for complementary harmony
    hue_diff = abs(scene_hsv[0] - element_hsv[0])
    complementary_harmony = max(0, 1 - hue_diff)

    # Calculate color distance in LAB color space
    scene_lab = rgb_to_lab([x / 255.0 for x in scene_color])
    element_lab = rgb_to_lab([x / 255.0 for x in element_color])
    lab_distance = delta_e(scene_lab, element_lab)

    # Calculate contrast ratio (YIQ color space)
    scene_yiq = rgb_to_yiq(
        scene_color[0] / 255, scene_color[1] / 255, scene_color[2] / 255
    )
    element_yiq = rgb_to_yiq(
        element_color[0] / 255, element_color[1] / 255, element_color[2] / 255
    )
    contrast_ratio = abs(scene_yiq[0] - element_yiq[0])

    # Calculate final harmony score
    harmony_score = (
        (complementary_harmony * 0.4)
        + ((100 - lab_distance) * 0.4)
        + (contrast_ratio * 0.2)
    )
    return harmony_score


def generate_trading_card(scene: Scene, element: Element, user: User) -> TradingCard:
    """
    Generate a trading card based on the combined scene and element.
    """
    card_design = f"{scene.scene_type}-{element.element_type}"
    realm_access_url = f"http://example.com/realm/{scene.id}/{element.id}"
    qr_code_url = f"http://example.com/qrcode/{scene.id}/{element.id}"

    additional_attributes = {
        "box_size": 10,
        "border": 4,
        "fill_color": "black",
        "back_color": "white",
        "logo_path": None,
        "background_image_path": None,
    }

    return TradingCard(
        user_id=user.id,
        card_type="Combination",
        card_design=card_design,
        realm_access_url=realm_access_url,
        qr_code_url=qr_code_url,
        **additional_attributes,
    )


def create_scene_element_matrix(
    scenes: List[Scene], elements: List[Element]
) -> np.ndarray:
    """
    Create an adjacency matrix representing the compatibility between scenes and elements.
    """
    matrix = np.zeros((len(scenes), len(elements)), dtype=int)
    for i, scene in enumerate(scenes):
        for j, element in enumerate(elements):
            if is_compatible(scene, element):
                matrix[i, j] = 1
    return matrix


async def get_user_engagement_score(user_id: int, db: AsyncSession) -> int:
    """
    Calculate user engagement score based on various factors.
    """
    try:
        # Fetch user sessions
        result = await db.execute(select(Session).filter(Session.user_id == user_id))
        sessions = result.scalars().all()

        # Fetch user emotions
        result = await db.execute(select(Emotion).filter(Emotion.user_id == user_id))
        emotions = result.scalars().all()

        # Calculate engagement metrics
        num_sessions = len(sessions)
        avg_session_duration = (
            np.mean(
                [
                    session.session_duration
                    for session in sessions
                    if session.session_duration
                ]
            )
            if sessions
            else 0
        )
        num_emotions_logged = len(emotions)

        # Define weights for different metrics
        weights = {
            "num_sessions": 0.4,
            "avg_session_duration": 0.3,
            "num_emotions_logged": 0.3,
        }

        # Calculate engagement score
        engagement_score = (
            weights["num_sessions"] * num_sessions
            + weights["avg_session_duration"] * avg_session_duration
            + weights["num_emotions_logged"] * num_emotions_logged
        )

        return int(engagement_score)
    except Exception as e:
        log_handler(f"Error in get_user_engagement_score: {e}")
        return 0


def calculate_connection_score(
    scene: Scene, element: Element, user_engagement_score: int
) -> int:
    """
    Calculate the compatibility score between a scene and an element.
    """
    try:
        # Base score based on thematic compatibility
        base_score = 50 if is_compatible(scene, element) else 0

        # Add user engagement score
        engagement_weight = 0.5
        final_score = base_score + (engagement_weight * user_engagement_score)

        return int(final_score)
    except Exception as e:
        log_handler(f"Error in calculate_connection_score: {e}")
        return 0


async def combine_scene_element(
    scene_id: int, element_id: int, db: AsyncSession
) -> TradingCard:
    """
    Combine a scene and an element to create a trading card.
    """
    try:
        # Check if the scene and element combination is valid
        result = await db.execute(
            select(scene_element_association).where(
                scene_element_association.c.scene_id == scene_id,
                scene_element_association.c.element_id == element_id,
            )
        )
        valid_combination = result.scalar()

        if not valid_combination:
            raise ValueError("Invalid scene and element combination")

        # Fetch the scene, element, and user from the database
        scene = await db.get(Scene, scene_id)
        element = await db.get(Element, element_id)

        if not scene or not element:
            raise ValueError("Invalid scene or element ID")

        user = await db.get(User, scene.script.session.user_id)
        if not user:
            raise ValueError("Invalid user ID")

        # Calculate user engagement score
        user_engagement_score = await get_user_engagement_score(user.id, db)

        # Calculate connection score
        scene_color = (255, 255, 255)  # Placeholder for scene's color
        element_color = (255, 255, 255)  # Placeholder for element's color
        color_harmony_score = color_harmony(scene_color, element_color)
        connection_score = calculate_connection_score(
            scene, element, user_engagement_score
        )

        # Process the combination and generate a trading card
        trading_card_data = TradingCardCreate(
            user_id=user.id,
            card_type="Combination",
            card_design=f"{scene.scene_type}-{element.element_type}",
            realm_access_url=f"http://example.com/realm/{scene.id}/{element.id}",
            qr_code_url=f"http://example.com/qrcode/{scene.id}/{element.id}",
            box_size=10,
            border=4,
            fill_color="black",
            back_color="white",
            logo_path=None,
            background_image_path=None,
        )

        # Use the create_trading_card function
        trading_card = await create_trading_card(trading_card_data, db)

        log_handler(
            f"Trading card created successfully for user {user.id} with connection score {connection_score}"
        )

        return trading_card
    except Exception as e:
        log_handler(f"Error in combine_scene_element: {e}")
        await db.rollback()
        raise e


async def handle_user_action(
    user_id: int, scene_id: int, element_id: int, db: AsyncSession
) -> TradingCard:
    """
    Handle the user's action to combine a scene and an element.
    """
    try:
        # Check if the user has access to the scene
        scene = await db.get(Scene, scene_id)
        if not scene or scene.script.session.user_id != user_id:
            raise ValueError("User does not have access to the scene")

        # Check if the element is valid
        element = await db.get(Element, element_id)
        if not element:
            raise ValueError("Invalid element")

        # Combine the scene and element
        trading_card = await combine_scene_element(scene_id, element_id, db)
        return trading_card
    except ValueError as e:
        log_handler(f"ValueError in handle_user_action: {e}")
        raise
    except Exception as e:
        log_handler(f"Exception in handle_user_action: {e}")
        raise


async def swap_scene_element(
    user_id: int, scene_id: int, element_id: int, db: AsyncSession
) -> TradingCard:
    """
    Swap a scene and an element interactively, ensuring compatibility and generating a trading card.
    """
    try:
        # Check if the scene and element combination is valid
        result = await db.execute(
            select(scene_element_association).where(
                scene_element_association.c.scene_id == scene_id,
                scene_element_association.c.element_id == element_id,
            )
        )
        valid_combination = result.scalar()

        if not valid_combination:
            raise ValueError("Invalid scene and element combination")

        # Fetch the scene, element, and user from the database
        scene = await db.get(Scene, scene_id)
        element = await db.get(Element, element_id)
        user = await db.get(User, user_id)

        if not scene or not element or not user:
            raise ValueError("Invalid scene, element, or user ID")

        # Calculate user engagement score
        user_engagement_score = await get_user_engagement_score(user.id, db)

        # Calculate connection score
        scene_color = (255, 255, 255)  # Placeholder for scene's color
        element_color = (255, 255, 255)  # Placeholder for element's color
        color_harmony_score = color_harmony(scene_color, element_color)
        connection_score = calculate_connection_score(
            scene, element, user_engagement_score
        )

        # Generate a trading card based on the scene and element
        trading_card = generate_trading_card(scene, element, user)

        # Save the trading card to the database
        db.add(trading_card)
        await db.commit()
        log_handler(
            f"Trading card created successfully for user {user_id} with connection score {connection_score}"
        )

        return trading_card
    except Exception as e:
        log_handler(f"Error in swap_scene_element: {e}")
        await db.rollback()
        raise e


async def log_scene_element_matrix(
    matrix: np.ndarray, scenes: List[Scene], elements: List[Element], db: AsyncSession
) -> None:
    """
    Log the scene-element compatibility matrix to the database.
    """
    try:
        for i, scene in enumerate(scenes):
            for j, element in enumerate(elements):
                association = scene_element_association.insert().values(
                    scene_id=scene.id, element_id=element.id, is_compatible=matrix[i, j]
                )
                await db.execute(association)
        await db.commit()
    except Exception as e:
        log_handler(f"Error in log_scene_element_matrix: {e}")
        await db.rollback()
        raise e
