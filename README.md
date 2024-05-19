app/
├── **init**.py
├── config.py
├── dependencies.py
├── enums.py
├── main.py
├── models.py
├── chain_tree/
│ ├── **init**.py
│ ├── models.py # SQLAlchemy models for chain_tree
│ ├── state_machine.py # The StateMachine and ChainManager classes
│ ├── utils.py # Utility functions like filter_none_values, etc.
│ ├── schemas/
│ │ ├── **init**.py
│ │ ├── author.py # Pydantic schemas for chain_tree
│ │ ├── content.py
│ │ ├── finish_details.py
│ │ ├── attachment.py
│ │ ├── metadata.py
│ │ ├── chain_coordinate.py
│ │ ├── chain_message.py
│ │ ├── chain_map.py
│ │ ├── chain_tree.py
├── routers/
│ ├── **init**.py
│ ├── chain_tree_router.py # FastAPI router for ChainManager and StateMachine endpoints
│ ├── element.py
│ ├── emotion.py
│ ├── gamification.py
│ ├── goal.py
│ ├── hypnotherapy.py
│ ├── interactive.py
│ ├── memory.py
│ ├── notification.py
│ ├── progress.py
│ ├── qr_code.py
│ ├── recommendation.py
│ ├── scene.py
│ ├── script.py
│ ├── session.py
│ ├── setting.py
│ ├── thought.py
│ ├── trading_card.py
│ ├── user.py
├── schemas.py # General Pydantic schemas
├── services/
│ ├── **init**.py
│ ├── gamification_service.py
│ ├── hypnotherapy_service.py
│ ├── personalization_service.py
│ ├── qr_code_service.py
│ ├── recommendation_service.py
│ ├── scene_element_service.py
│ ├── trading_card_service.py
│ ├── task_scheduler.py
│ ├── utils.py
├── database/
│ ├── **init**.py
│ ├── insertion.py
├── test/
│ ├── **init**.py
│ ├── test_hypnotherapy.py
│ ├── test_user.py
│ ├── test_session.py
│ ├── test_conversation.py
│ ├── test_trading_card.py
│ ├── test_notification.py
└── data/
└── scripts/
