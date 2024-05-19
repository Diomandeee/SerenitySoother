import os
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import init_db, engine
from app.services.nft_service import DataUploader
from app.database.insertion import insert_data


async def main():
    await init_db()

    base_path = "/Users/mohameddiomande/Desktop/1792x1024/data"
    data_uploader = DataUploader(directory=base_path)
    parsed_data = data_uploader.upload_all_media_in_parallel(
        "/Users/mohameddiomande/Desktop/1792x1024/data"
    )

    print(parsed_data)

    # async with AsyncSession(engine) as session:
    #     await insert_data(session, parsed_data)


if __name__ == "__main__":
    asyncio.run(main())
