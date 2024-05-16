from fastapi import APIRouter, HTTPException
from app.services.qr_code_service import generate_qr_code
from pydantic import BaseModel
from typing import Optional

router = APIRouter(
    prefix="/generate_qr_code",
    tags=["qr_code"],
    responses={404: {"description": "Not found"}},
)


class QRCodeRequest(BaseModel):
    data: str
    box_size: int = 10
    border: int = 4
    fill_color: str = "black"
    back_color: str = "white"
    logo_path: Optional[str] = None
    background_image_path: Optional[str] = None
    filename: Optional[str] = None


@router.post("/", response_model=dict)
async def create_qr_code(request: QRCodeRequest):
    try:
        qr_code_url = generate_qr_code(
            data=request.data,
            box_size=request.box_size,
            border=request.border,
            fill_color=request.fill_color,
            back_color=request.back_color,
            logo_path=request.logo_path,
            background_image_path=request.background_image_path,
            filename=request.filename,
        )
        return {"qr_code_url": qr_code_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


