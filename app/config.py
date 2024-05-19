from pydantic import BaseSettings, Field
from typing import Optional
import secrets


class Settings(BaseSettings):
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    DATABASE_URL: str = "sqlite+aiosqlite:///./db/serenity_soother.db"
    DATABASE_TEST_URL: str = "sqlite+aiosqlite:///./db/test_serenity_soother.db"
    ENV: str = "development"
    SECRET: Optional[str] = None
    ACCESS_KEY: Optional[str] = None
    ACCOUNT: Optional[str] = None
    SERVICE_ACCOUNT: Optional[str] = None
    URL: Optional[str] = None
    BUCKET_NAME: Optional[str] = None
    SHEET_NAME: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"

    @property
    def CREDENTIALS(self):
        return {
            "credentials": {
                "secret": self.SECRET,
                "access_key": self.ACCESS_KEY,
                "account": self.ACCOUNT,
                "service_account": self.SERVICE_ACCOUNT,
                "url": self.URL,
                "bucket_name": self.BUCKET_NAME,
                "sheet_name": self.SHEET_NAME,
            }
        }

    @property
    def OPENAI_API(self):
        return self.OPENAI_API_KEY

    class Config:
        env_file = ".env"


settings = Settings()
