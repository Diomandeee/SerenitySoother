from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Script, Scene, Element, Session, User, Section

async def insert_data(db: AsyncSession, data: dict):
    try:
        async with db.begin():
            for data_id, content in data.items():
                session = Session(
                    user_id=1,  # assuming user ID 1 for this example
                    session_type="",
                    session_status="Completed",
                    session_description=f"Session for script {data_id}"
                )
                db.add(session)
                await db.flush()

                script = Script(
                    session_id=session.id,
                    script_type="",
                    script_content="",
                    script_description=f"Script for {data_id}"
                )
                db.add(script)
                await db.flush()

                # Create a list to hold section contents
                sections_content = []

                for index, section in content['sections']:
                    section_data = section.split('\n', 1)
                    part_title = section_data[0].strip()
                    content_text = section_data[1].strip() if len(section_data) > 1 else ''
                    section_entry = Section(
                        script_id=script.id,
                        part_title=part_title,
                        content=content_text
                    )
                    sections_content.append(content_text)
                    db.add(section_entry)
                    await db.flush()

                # Join section contents to form the script content
                script.script_content = '\n'.join(sections_content)
                db.add(script)
                await db.flush()

                for index, text in content['scene_descriptions']:
                    scene = Scene(
                        script_id=script.id,
                        scene_type="",
                        scene_description=text
                    )
                    db.add(scene)
                    await db.flush()

                    for img_index, img_path in content['images']:
                        if img_index == index:
                            scene.scene_image = img_path
                            db.add(scene)
                            await db.flush()

                    for audio_index, audio_path in content['audios']:
                        if audio_index == index:
                            scene.scene_audio = audio_path
                            db.add(scene)
                            await db.flush()
                    
    except Exception as e:
        await db.rollback()
        raise e
