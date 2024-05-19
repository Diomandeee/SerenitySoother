from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Union, Optional
from app.chain_tree.state import chain_manager, StateMachine
from app.chain_tree.schemas import (
    Content as PydanticContent,
    Author as PydanticAuthor,
    ChainCoordinate as PydanticChainCoordinate,
    Chain as PydanticChain,
)
from app.dependencies import get_db

router = APIRouter(
    prefix="/trees",
    tags=["trees"],
    responses={404: {"description": "Not found"}},
)


@router.post("/conversations/start")
async def start_conversation(initial_message: str):
    try:
        return await chain_manager.start_conversation(initial_message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/create")
async def create_conversation():
    try:
        return await chain_manager.create_conversation()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/add")
async def add_conversation(conversation: StateMachine):
    try:
        await chain_manager.add_conversation(conversation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        return await chain_manager.get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/conversations/{conversation_id}/rewind")
async def rewind_conversation(conversation_id: str, steps: int = 1):
    try:
        await chain_manager.rewind_conversation(conversation_id, steps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}/print")
async def print_conversation(conversation_id: str):
    try:
        await chain_manager.print_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/end")
async def end_conversation(conversation_id: str):
    try:
        await chain_manager.end_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/restart")
async def restart_conversation(conversation_id: str):
    try:
        await chain_manager.restart_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    try:
        return await chain_manager.get_conversation_history(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/conversations/{conversation_id}/messages/add")
async def add_message(
    conversation_id: str,
    message_id: str,
    content: PydanticContent,
    author: PydanticAuthor,
    coordinate: Optional[PydanticChainCoordinate] = None,
    embedding: List[float] = None,
    parent: str = None,
    save: bool = False,
    db: AsyncSession = Depends(get_db),
):
    try:
        await chain_manager.add_message(
            conversation_id,
            message_id,
            content,
            author,
            coordinate,
            embedding,
            parent,
            save,
            db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/conversations/{conversation_id}/messages/update")
async def update_message(
    conversation_id: str, message_id: str, new_message: PydanticChain
):
    try:
        await chain_manager.update_message(conversation_id, message_id, new_message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/conversations/{conversation_id}/messages/{message_id}")
async def delete_message(conversation_id: str, message_id: str):
    try:
        return await chain_manager.delete_message(conversation_id, message_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/conversations/{conversation_id}/messages/{message_id}")
async def get_message(conversation_id: str, message_id: str):
    try:
        return await chain_manager.get_message(conversation_id, message_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/conversations/{conversation_id}/messages/{message_id}/move")
async def move_message(conversation_id: str, message_id: str, new_parent_id: str):
    try:
        await chain_manager.move_message(conversation_id, message_id, new_parent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/merge")
async def merge_conversations(
    conversation_id_1: str, conversation_id_2: str, db: AsyncSession = Depends(get_db)
):
    try:
        await chain_manager.merge_conversations(
            conversation_id_1, conversation_id_2, db
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations")
async def get_conversations():
    try:
        return await chain_manager.get_conversations()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/ids")
async def get_conversation_ids():
    try:
        return await chain_manager.get_conversation_ids()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/titles")
async def get_conversation_titles():
    try:
        return await chain_manager.get_conversation_titles()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/titles_ids")
async def get_conversation_titles_and_ids():
    try:
        return await chain_manager.get_conversation_titles_and_ids()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        await chain_manager.delete_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/conversations")
async def delete_all_conversations():
    try:
        await chain_manager.delete_all_conversations()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/cleanup")
async def cleanup_inactive_conversations(inactivity_threshold_in_hours: int = 1):
    try:
        await chain_manager.cleanup_inactive_conversations(
            inactivity_threshold_in_hours
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/export")
async def export_conversations_to_json():
    try:
        return await chain_manager.export_conversations_to_json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/import")
async def import_conversations_from_json(json_data: str):
    try:
        await chain_manager.import_conversations_from_json(json_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/messages/user_input")
async def handle_user_input(
    conversation_ids: Union[str, List[str]],
    user_input: Union[str, List[str]],
    parent_ids: Union[str, List[str], None] = None,
):
    try:
        await chain_manager.handle_user_input(conversation_ids, user_input, parent_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/messages/agent_response")
async def handle_agent_response(
    conversation_ids: Union[str, List[str]],
    agent_response: Union[str, List[str]],
    parent_ids: Union[str, List[str], None] = None,
):
    try:
        await chain_manager.handle_agent_response(
            conversation_ids, agent_response, parent_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/messages/system_message")
async def handle_system_message(
    conversation_ids: Union[str, List[str]],
    system_message: Union[str, List[str]],
    parent_ids: Union[str, List[str], None] = None,
):
    try:
        await chain_manager.handle_system_message(
            conversation_ids, system_message, parent_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    try:
        return await chain_manager.get_messages(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/conversations/{conversation_id}/load")
async def load_conversation(conversation_id: str, title: str = "Untitled"):
    try:
        await chain_manager.load_conversation(conversation_id, title)
        return "Conversation loaded successfully."
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversations/{conversation_id}/save")
async def save_conversation(conversation_id: str):
    try:
        await chain_manager.save_conversation(conversation_id)
        return "Conversation saved successfully."
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
