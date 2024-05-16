import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "your_secret_key")
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL", "sqlite+aiosqlite:///./serenity_soother.db"
    )
    DATABASE_TEST_URL: str = os.environ.get(
        "DATABASE_TEST_URL", "sqlite+aiosqlite:///./test_serenity_soother.db"
    )
    ENV: str = os.environ.get("ENV", "development")

    class Config:
        env_file = ".env"


settings = Settings()
