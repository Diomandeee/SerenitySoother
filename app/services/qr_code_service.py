import qrcode
from qrcode.image.pil import PilImage
from io import BytesIO
from fastapi import HTTPException
import base64
from typing import Optional
from PIL import Image, ImageDraw
import os
import logging

QR_CODES_DIR = "images/qr_codes"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_qr_code(
    data: str,
    box_size: int = 10,
    border: int = 4,
    fill_color: str = "black",
    back_color: str = "white",
    logo_path: Optional[str] = None,
    background_image_path: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    try:
        if not os.path.exists(QR_CODES_DIR):
            os.makedirs(QR_CODES_DIR)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")

        if background_image_path:
            img = blend_with_background(img, background_image_path)

        if logo_path:
            img = add_logo(img, logo_path)

        if not filename:
            filename = f"{base64.urlsafe_b64encode(os.urandom(6)).decode()}.png"

        file_path = os.path.join(QR_CODES_DIR, filename)
        img.save(file_path)

        return file_path
    except Exception as e:
        logger.error(f"QR code generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"QR code generation failed: {str(e)}"
        )


def add_logo(qr_img: PilImage, logo_path: str) -> PilImage:
    try:
        logo = Image.open(logo_path)

        # Ensure logo has an alpha channel
        if logo.mode != "RGBA":
            logo = logo.convert("RGBA")

        qr_width, qr_height = qr_img.size
        logo_size = int(qr_width * 0.2)
        logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)

        # Create a transparent mask for the logo
        mask = Image.new("L", logo.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, logo_size, logo_size), fill=255)

        logo_position = ((qr_width - logo_size) // 2, (qr_height - logo_size) // 2)
        qr_img.paste(logo, logo_position, mask=logo)
        return qr_img
    except Exception as e:
        logger.error(f"Failed to add logo to QR code: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add logo to QR code: {str(e)}"
        )


def blend_with_background(qr_img: PilImage, background_image_path: str) -> PilImage:
    try:
        background = Image.open(background_image_path)
        background = background.convert("RGBA")
        qr_img = qr_img.convert("RGBA")

        bg_width, bg_height = background.size
        qr_width, qr_height = qr_img.size

        # Ensure the background is large enough to hold the QR code
        if bg_width < qr_width or bg_height < qr_height:
            raise HTTPException(status_code=400, detail="Background image is too small")

        # Center the QR code on the background
        position = ((bg_width - qr_width) // 2, (bg_height - qr_height) // 2)
        combined = Image.alpha_composite(
            background, qr_img.resize((bg_width, bg_height), Image.Resampling.LANCZOS)
        )
        return combined
    except Exception as e:
        logger.error(f"Failed to blend QR code with background: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to blend QR code with background: {str(e)}"
        )


def save_qr_code_to_file(
    data: str,
    file_path: str,
    box_size: int = 10,
    border: int = 4,
    fill_color: str = "black",
    back_color: str = "white",
    logo_path: Optional[str] = None,
    background_image_path: Optional[str] = None,
):
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")

        if background_image_path:
            img = blend_with_background(img, background_image_path)

        if logo_path:
            img = add_logo(img, logo_path)

        img.save(file_path)
    except Exception as e:
        logger.error(f"Failed to save QR code to file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save QR code to file: {str(e)}"
        )
