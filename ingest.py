import os
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import init_db, engine
from app.utils import parse_directory
from app.database.insertion import insert_data

async def main():
    await init_db()

    base_path = 'data'
    parsed_data = parse_directory(base_path)

    async with AsyncSession(engine) as session:
        await insert_data(session, parsed_data)

if __name__ == "__main__":
    asyncio.run(main())

