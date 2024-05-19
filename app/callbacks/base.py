from typing import Any, Dict, List, Optional, Union
from app.chain_tree.base import EntityAction, EntityFinish, LLMResult, BaseMessage
from datetime import datetime
from uuid import UUID


class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""


class ChainManagerMixin:
    """Mixin for chain callbacks."""

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""

    def on_agent_action(
        self,
        action: EntityAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""

    def on_agent_finish(
        self,
        finish: EntityFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""


class ToolManagerMixin:
    """Mixin for tool callbacks."""

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""


class CallbackManagerMixin:
    """Mixin for callback manager."""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""


class RunManagerMixin:
    """Mixin for run manager."""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on arbitrary text."""


class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """Base callback handler that can be used to handle callbacks from langchain."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False


class AsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""

    async def on_agent_action(
        self,
        action: EntityAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""

    async def on_agent_finish(
        self,
        finish: EntityFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""


class BaseCallbackManager(CallbackManagerMixin):
    """Base callback manager that can be used to handle callbacks from LangChain."""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: Optional[List[BaseCallbackHandler]] = None,
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        """Initialize callback manager."""
        self.handlers: List[BaseCallbackHandler] = handlers
        self.inheritable_handlers: List[BaseCallbackHandler] = (
            inheritable_handlers or []
        )
        self.parent_run_id: Optional[UUID] = parent_run_id

    @property
    def is_async(self) -> bool:
        """Whether the callback manager is async."""
        return False

    def add_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)
        if inherit:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)
        self.inheritable_handlers.remove(handler)

    def set_handlers(
        self, handlers: List[BaseCallbackHandler], inherit: bool = True
    ) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        """Set handler as the only handler on the callback manager."""
        self.set_handlers([handler], inherit=inherit)

    def configure(
        self,
        handlers: Optional[List[BaseCallbackHandler]] = None,
        inheritable_handlers: Optional[List[BaseCallbackHandler]] = None,
        verbose: bool = False,
    ) -> "BaseCallbackManager":
        """Configure the callback manager."""
        pass

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str]
    ) -> "BaseCallbackManager":
        """Run when LLM starts running."""
        pass


class BaseCallback:
    def on_processing_start(self, files):
        pass

    def on_file_preprocess(self, filename):
        pass

    def on_file_postprocess(self, filename, prompt_data):
        pass

    def on_prompt_preprocess(self, prompt_data):
        pass

    def on_prompt_postprocess(self, prompt_data):
        pass

    def transform(self, prompt_data):
        return prompt_data

    def filter(self, prompt_data):
        return True

    def on_processing_end(self, prompts):
        pass


class TimestampCallback(BaseCallback):
    def transform(self, prompt_data):
        prompt_data["timestamp"] = datetime.now().isoformat()
        return prompt_data


class AuthorValidationCallback(BaseCallback):
    def on_prompt_start(self, prompt_data):
        if "author" not in prompt_data:
            raise ValueError("Missing 'author' field in prompt data")


class LoggingCallback(BaseCallback):
    def on_file_preprocess(self, filename):
        print(f"Starting processing of file: {filename}")

    def on_file_postprocess(self, filename, prompt_data):
        print(f"Finished processing of file: {filename}")


class TransformationCallback(BaseCallback):
    def transform(self, prompt_data):
        prompt_data["content"] = prompt_data["content"].upper()
        return prompt_data

    def filter(self, prompt_data):
        return len(prompt_data["content"]) > 5


class SummaryCallback(BaseCallback):
    def on_processing_end(self, prompts):
        print(f"Processed {len(prompts)} prompts")


class ProgressCallback(BaseCallback):
    def on_processing_start(self, files):
        self.total_files = len(files)
        self.current_file = 0

    def on_file_postprocess(self, filename, prompt_data):
        self.current_file += 1
        print(f"Processed {self.current_file} of {self.total_files} files.")


class DataEnhancementCallback(BaseCallback):
    def transform(self, prompt_data):
        # Example: Add word count
        prompt_data["word_count"] = len(prompt_data["content"].split())
        return prompt_data


class MonitoringCallback(BaseCallback):
    def on_file_preprocess(self, filename):
        self.start_time = datetime.now()

    def on_file_postprocess(self, filename, prompt_data):
        time_taken = datetime.now() - self.start_time
        print(f"Time taken for {filename}: {time_taken.total_seconds()} seconds.")


class CallbackHandler:
    def __init__(self, *callbacks, config=None):
        self.callbacks = list(callbacks)
        self.enabled_callbacks = {
            callback.__class__.__name__: (0, callback) for callback in callbacks
        }  # Default priority 0
        self.config = config or {}

    def enable(self, name, priority=0, config=None):
        """Enable a specific callback by name and provide additional configuration."""
        callback_class = globals()[name]
        callback_instance = callback_class(config=config or self.config.get(name))
        self.enabled_callbacks[name] = (priority, callback_instance)

    def disable(self, name):
        """Disable a specific callback by name."""
        if name in self.enabled_callbacks:
            del self.enabled_callbacks[name]

    def enable_if(self, name, condition):
        """Enable a specific callback if the condition is True."""
        if condition:
            self.enable(name)

    def disable_if(self, name, condition):
        """Disable a specific callback if the condition is True."""
        if condition:
            self.disable(name)

    def set_priority(self, name, priority):
        """Set priority for a specific callback."""
        callback = self.enabled_callbacks.get(name)
        if callback:
            self.enabled_callbacks[name] = (priority, callback[1])

    def get_enabled_callbacks(self):
        """Get all enabled callbacks sorted by priority."""
        return [
            callback
            for _, callback in sorted(
                self.enabled_callbacks.values(), key=lambda x: x[0]
            )
        ]

    def invoke(self, method_name, *args, **kwargs):
        """Invoke a specific method across all enabled callbacks."""
        for callback in self.get_enabled_callbacks():
            try:
                method = getattr(callback, method_name, None)
                if method:
                    method(*args, **kwargs)
            except Exception as e:
                print(f"Error in callback {callback.__class__.__name__}: {str(e)}")

    # Example of mapping methods for easy access
    def on_processing_start(self, files):
        self.invoke("on_processing_start", files)

    def on_file_preprocess(self, filename):
        self.invoke("on_file_preprocess", filename)

    def on_file_postprocess(self, filename, prompt_data):
        self.invoke("on_file_postprocess", filename, prompt_data)

    def transform(self, prompt_data):
        for callback in self.get_enabled_callbacks():
            prompt_data = callback.transform(prompt_data)
        return prompt_data

    def filter(self, prompt_data):
        for callback in self.get_enabled_callbacks():
            if not callback.filter(prompt_data):
                return False
        return True

    def on_processing_end(self, prompts):
        self.invoke("on_processing_end", prompts)
