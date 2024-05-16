from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.models import Base
from app.routers import (
    user,
    session,
    script,
    scene,
    element,
    emotion,
    goal,
    progress,
    notification,
    memory,
    setting,
    trading_card,
    thought,
    qr_code,
    recommendation,
    gamification,
    hypnotherapy,
)
from contextlib import asynccontextmanager

DATABASE_URL = "sqlite+aiosqlite:///./serenity_soother.db"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to Serenity Soother"}


app.include_router(user.router, prefix="/users", tags=["users"])
app.include_router(session.router, prefix="/sessions", tags=["sessions"])
app.include_router(script.router, prefix="/scripts", tags=["scripts"])
app.include_router(scene.router, prefix="/scenes", tags=["scenes"])
app.include_router(element.router, prefix="/elements", tags=["elements"])
app.include_router(emotion.router, prefix="/emotions", tags=["emotions"])
app.include_router(goal.router, prefix="/goals", tags=["goals"])
app.include_router(progress.router, prefix="/progress", tags=["progress"])
app.include_router(notification.router, prefix="/notifications", tags=["notifications"])
app.include_router(memory.router, prefix="/memories", tags=["memories"])
app.include_router(setting.router, prefix="/settings", tags=["settings"])
app.include_router(trading_card.router, prefix="/trading_cards", tags=["trading_cards"])
app.include_router(thought.router, prefix="/thoughts", tags=["thoughts"])
app.include_router(qr_code.router, prefix="/qr_codes", tags=["qr_codes"])
app.include_router(thought.router, prefix="/thoughts", tags=["thoughts"])
app.include_router(recommendation.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(gamification.router, prefix="/gamification", tags=["gamification"])
app.include_router(hypnotherapy.router, prefix="/hypnotherapy", tags=["hypnotherapy"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
