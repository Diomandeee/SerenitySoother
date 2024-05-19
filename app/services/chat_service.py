from app.services.generation_service import BaseLanguageModel, get_buffer_string
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from app.services.embedding_service import OpenAIEmbedding
from app.services.cloud_service import CloudManager
from app.helper import log_handler, backoff_handler
from app.callbacks.base import BaseCallbackManager
from pydantic import Extra, Field, root_validator
from app.callbacks.manager import (
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from app.services.prompt_service import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_3,
)
from tenacity import (
    retry,
    before_sleep_log,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)

from app.chain_tree.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
    ChainResult,
    ChainGeneration,
    PromptValue,
    UserChain,
    LLMResult,
)

from abc import ABC, abstractmethod
from app.chain_tree.schemas import *
import warnings
import inspect
import logging
import time
import os


logger = logging.getLogger(__name__)


def _get_verbosity() -> bool:
    return True


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseChatModel(BaseLanguageModel, ABC):
    """Whether to print out response text."""

    verbose: bool = Field(default_factory=_get_verbosity)

    callbacks: Callbacks = Field(default=None, exclude=True)
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @abstractmethod
    def _generate(
        self,
        messages: List[Chain],
        prompt_num: Optional[int] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_func: Optional[callable] = None,
    ) -> ChainResult:
        """Generate a response for the given messages.

        Args:
            messages: The messages to generate a response for.
            stop: A list of strings to stop generation at.
            run_manager: The run manager to use for this generation.

        Returns:
            The generated response.

        """
        pass

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {}

    def generate(
        self,
        messages: List[List[Chain]],
        prompt_num: Optional[int] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        generate_func: Optional[callable] = None,
    ) -> LLMResult:
        """Top Level call"""

        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        message_strings = [get_buffer_string(m) for m in messages]
        run_manager = callback_manager.on_llm_start(
            {"name": self.__class__.__name__}, message_strings
        )

        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        try:
            results = [
                (
                    self._generate(
                        m,
                        prompt_num,
                        stop=stop,
                        run_manager=run_manager,
                        generate_func=generate_func,
                    )
                    if new_arg_supported
                    else self._generate(
                        m,
                        prompt_num,
                        stop=stop,
                        run_manager=run_manager,
                        generate_func=generate_func,
                    )
                )
                for m in messages
            ]
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_llm_error(e)
            raise e
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        run_manager.on_llm_end(output)
        return output

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt_messages = [p.to_chain() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks)

    def call_as_llm(self, message: str, stop: Optional[List[str]] = None) -> str:

        generation = self.generate(
            [
                [UserChain(content=Content(text=message))],
            ],
            stop=stop,
        ).generations[0][0]
        if isinstance(generation, ChainGeneration):
            return generation.message.content.text
        else:
            raise ValueError("Unexpected generation type")

    def __call__(
        self,
        messages: List[Chain],
        prompt_num: Optional[int] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        generate_func: Optional[callable] = None,
    ) -> Chain:
        generation = self.generate(
            [messages],
            prompt_num,
            stop=stop,
            callbacks=callbacks,
            generate_func=generate_func,
        ).generations[0][0]
        if isinstance(generation, ChainGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")


class AI(BaseChatModel):
    """Chat wrapper for openai API.

    Args:
        model_name (str): Model name to use.
        temperature (float): What sampling temperature to use.
        model_kwargs (Dict[str, Any]): Holds any model parameters valid for `create` call not explicitly specified.
        openai_api_key (Optional[str]): openai API key, if not available as an environment variable.
        openai_organization (Optional[str]): openai organization, if not available as an environment variable.
        max_retries (int): Maximum number of retries to make when generating.
        streaming (bool): Whether to stream the results or not.
        n (int): Number of chat completions to generate for each prompt.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        frequency_penalty (Optional[float]): Frequency penalty to use.
        presence_penalty (Optional[float]): Presence penalty to use.

    """

    client: Any = Field(default_factory=None, init=False)
    embedder: Any = Field(default_factory=None, init=False)
    audio: Any = Field(default_factory=None, init=False)
    transcripter: Any = Field(default_factory=None, init=False)
    imager: Any = Field(default_factory=None, init=False)
    prompt_manager: Any = Field(default_factory=None, init=False)
    model_name: str = "gpt-3.5-turbo-0125"
    """Model name to use."""
    temperature: float = 1
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    audio_func: Optional[bool] = False

    provider: Optional[str] = None

    credentials: Any = None

    """Timeout for requests to openai completion API. Default is 600 seconds."""
    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    streaming: bool = True

    api_key: Optional[str] = None

    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = 4096
    """Maximum number of tokens to generate."""
    frequency_penalty: Optional[float] = 1

    presence_penalty: Optional[float] = 1

    storage: Optional[str] = "store"

    subdirectory: Optional[str] = None

    size = ["1024x1792", "1792x1024"]

    target_tokens = 16385
    create: bool = False
    play: bool = False
    internal: bool = False
    show: bool = False
    process_media: bool = False
    embedding_cache = {}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                log_handler(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        disallowed_model_kwargs = all_required_field_names | {"model"}
        invalid_model_kwargs = disallowed_model_kwargs.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        provider = values.get("provider", "openai")

        credentials = values.get("credentials", None)["credentials"]
        try:
            from openai import OpenAI
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        try:
            client_openai = OpenAI(api_key=api_key)

            if provider == "openai":
                client = client_openai

            elif provider == "ollama":
                client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

            else:
                raise ValueError(
                    "Provider must be 'openai', 'openai', 'llama', or 'ollama'."
                )

            values["client"] = client.chat.completions
            values["audio"] = (
                client_openai.audio.speech if client_openai.audio else None
            )
            values["transcripter"] = (
                client_openai.audio.transcriptions if client_openai.audio else None
            )
            values["imager"] = client_openai.images if client_openai.images else None
            values["embedder"] = OpenAIEmbedding(api_key=api_key)
            values["prompt_manager"] = CloudManager(credentials=credentials)

        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the Groq package. Try upgrading it "
                "with `pip install --upgrade Groq`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        if "API_KEYS" in values:
            values["api_keys"] = values["API_KEYS"]
        return values

    def update_client(self, client: Any) -> None:
        self.client = client.chat.completions

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Groq API."""

        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _get_run_manager_kwargs(
        self,
        prompt_num: Optional[int] = None,
        manager: Optional[object] = None,
        generate_func: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        base_path_types = [
            "audio",
            "image",
            "embed",
            "text",
            "prompt",
            "caption",
        ]
        base_paths = {
            f"base_path_{type}": f"{self.storage}/{self.subdirectory}/{type}/{prompt_num}"
            for type in base_path_types
        }

        return {
            **base_paths,
            "generate": self.audio_func,
            "create": self.create,
            "play": self.play,
            "internal": self.internal,
            "show": self.show,
            "audio_func": self.audio_func,
            "process_media": self.process_media,
            "manager": manager,
            "prompt_num": prompt_num,
            "embedder": self.embedder,
            "base_path_dir": self.storage,
            "prompt_manager": self.prompt_manager,
            "token_count_func": self.get_num_tokens,
            "generate_audio_func": self.generate_audio,
            "generate_embeddings_func": self.generate_embeddings,
            "generate_imagine_func_dalle": self.generate_image_dalle,
            "generate_transcript_func:": self.generate_transcript,
            "generate_func": generate_func,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        """Create a retry decorator for the completion call."""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
        )

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChainResult:
        generations = []
        for res in response["content"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChainGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChainResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[Chain], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _handle_non_streaming_generation(
        self, message_dicts: dict, params: dict
    ) -> ChainResult:
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _process_stream_response(
        self,
        stream_resp: dict,
        inner_completion: str,
        role: str,
        run_manager: Optional[CallbackManagerForLLMRun],
        prompt_num: Optional[int] = None,
        manager: Optional[object] = None,
        generate_func: Optional[Callable] = None,
    ) -> tuple[str, str]:
        # Safely extract the first choice and its delta
        first_choice = stream_resp.choices[0] if stream_resp.choices else None
        delta = first_choice.delta if first_choice else None

        if delta:
            role = delta.role if delta.role is not None else role
            token = delta.content if delta.content is not None else ""
            inner_completion += token

            if (
                run_manager and token.strip()
            ):  # Check if the token is not just whitespace
                kwargs = self._get_run_manager_kwargs(
                    prompt_num, manager, generate_func
                )
                run_manager.on_llm_new_token(token, **kwargs)

        return inner_completion, role

    def _handle_streaming_generation(
        self,
        message_dicts: dict,
        params: dict,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        prompt_num: Optional[int] = None,
        manager: Optional[object] = None,
        generate_func: Optional[Callable] = None,
    ) -> ChainResult:
        inner_completion = ""
        role = "assistant"
        params["stream"] = True

        try:
            for stream_resp in self.completion_with_retry(
                messages=message_dicts, **params
            ):
                # Process each streamed response
                inner_completion, role = self._process_stream_response(
                    stream_resp,
                    inner_completion,
                    role,
                    run_manager,
                    prompt_num,
                    manager,
                    generate_func,
                )
        finally:
            # After streaming is complete, call on_stream_end to handle any remaining audio
            if run_manager:
                kwargs = self._get_run_manager_kwargs(prompt_num, manager)
                run_manager.on_stream_end(**kwargs)
        run_manager.on_cycle_end(**kwargs)
        message = _convert_dict_to_message(
            {
                "content": inner_completion,
                "role": role,
                "id": prompt_num,
            }
        )
        return ChainResult(generations=[ChainGeneration(message=message)])

    def _generate(
        self,
        messages: List[Chain],
        prompt_num: Optional[int] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        manager: Optional[object] = None,
        generate_func: Optional[Callable] = None,
    ) -> ChainResult:
        # Prepare message dictionaries and parameters
        message_dicts, params = self._create_message_dicts(messages, stop)

        if self.streaming:
            return self._handle_streaming_generation(
                message_dicts=message_dicts,
                params=params,
                run_manager=run_manager,
                prompt_num=prompt_num,
                manager=manager,
                generate_func=generate_func,
            )
        else:
            return self._handle_non_streaming_generation(
                message_dicts=message_dicts, params=params
            )

    def _call_audio_generation_api(self, prompt: str, lang: str = "fr-FR") -> dict:
        response = self.audio.create(
            model="tts-1",
            voice="alloy",
            input=prompt,
        )
        return response

    def _call_transcript_generation_api(self, temp_file: Union[str, object]) -> dict:
        from pathlib import Path

        temp_file = Path(temp_file)

        text = self.transcripter.create(
            model="whisper-1", file=temp_file, response_format="text"
        )
        return text

    def _call_dall_image_generation_api(self, prompt: str) -> dict:
        response = self.imager.generate(
            model="dall-e-3",
            prompt=prompt,
            size=self.size[1],
            quality="hd",
            n=1,
        )
        return response

    def _call_revised_generation_api(self, prompt: str) -> dict:

        response = self.client.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_2},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_serenity_generation_api(self, prompt: str) -> dict:

        response = self.client.create(
            model="ft:gpt-3.5-turbo-0125:personal:serenity-soother:9KtNHWL2",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_generation_api(
        self,
        prompt: str,
        step_content: str = "",
        next_step: str = None,
        post_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        original_prompt: bool = False,
        model_name: Optional[str] = None,
    ) -> dict:
        """
        Generalized method to call the generation API with different steps.

        Args:
        prompt (str): The user's prompt.
        step_content (str): Specific content related to the step.
        next_step (str, optional): Identifier for the next step to stop the response.
        post_prompt (str, optional): Additional prompt content.
        max_tokens (int, optional): The maximum number of tokens to generate.

        Returns:
        dict: The response from the API.
        """
        if original_prompt is not True:
            user_content_prompt = prompt + "\n" + step_content
            if post_prompt:
                user_content_prompt += "\n" + post_prompt

            stop = next_step if next_step is not None else None

        else:
            pre_prompt = "Step 0: Imagine That:\n"
            post_prompt = "Imagine That:"
            user_content_prompt = pre_prompt + prompt + "\n" + post_prompt
            stop = next_step if next_step is not None else None

        if model_name is None:
            model_name = self.model_name
        else:
            model_name = model_name

        response = self.client.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_3},
                {"role": "user", "content": user_content_prompt},
            ],
            stop=stop,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        return response.choices[0].message.content

    def _call_imagine_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(
            prompt, "Step 0: Imagine That:\n", "Brainstorming:"
        )

    def _call_brainstorm_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(
            prompt,
            "Step 1: Brainstorming:\n",
            "Step 2:" or "Thought Provoking Questions:",
        )

    def _call_questions_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(
            prompt, "Step 2: Thought Provoking Questions:\n", "Create:"
        )

    def _call_create_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(
            prompt, "Step 3: Create Prompts:\n", "Synergetic:"
        )

    def _call_synergetic_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(
            prompt, "Step 4: Synergetic Prompt:\n", "Category:"
        )

    def _call_category_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(prompt, "Step 5: Category:\n", "Revised:")

    def _call_spf_generation_api(self, prompt: str) -> dict:
        return self._call_generation_api(prompt, original_prompt=True)

    def _retry_api_call(self, api_func, *args, **kwargs) -> dict:
        for attempt in range(self.max_retries):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                log_handler(f"API call failed on attempt {attempt + 1}: {e}")
                time.sleep(backoff_handler(attempt + 1))
        raise Exception("Max retries reached. Aborting...")

    def _validate_image_response(self, image_response: dict) -> None:
        # Validate the response from the image generation API
        data_item = image_response.data[0]
        revised_prompt = getattr(data_item, "revised_prompt", None)
        image_url = getattr(data_item, "url", None)

        return revised_prompt, image_url

    def generate_image_dalle(self, prompt: str) -> Any:
        try:
            image_response = self._retry_api_call(
                self._call_dall_image_generation_api, prompt
            )
            image_response = self._validate_image_response(image_response)
            log_handler("Image generated successfully")
            return image_response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_audio(self, prompt: str, lang: str = "fr-FR") -> Any:
        try:
            audio_response = self._retry_api_call(
                self._call_audio_generation_api, prompt, lang
            )
            log_handler("Audio generated successfully")
            return audio_response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_transcript(self, file: str) -> Any:
        try:
            transcript_response = self._retry_api_call(
                self._call_transcript_generation_api, file
            )
            log_handler("Transcript generated successfully")
            return transcript_response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_embeddings(self, prompts: List[str]) -> Any:
        """
        Generate embeddings for a list of prompts. Assume the embedder returns a list of arrays.
        """
        try:
            embeddings = self.embedder(
                prompts
            )  # Assuming this returns a list of numpy arrays
            logging.info("Embeddings generated successfully for batch")
            return embeddings
        except Exception as e:
            logging.error(f"Failed to generate embeddings for batch: {e}")
            raise

    def generate_imagine(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(self._call_imagine_generation_api, prompt)
            log_handler("Imagine step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_brainstorm(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(
                self._call_brainstorm_generation_api, prompt
            )
            log_handler("Brainstorm step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_questions(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(self._call_questions_generation_api, prompt)
            log_handler("Questions step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_create(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(self._call_create_generation_api, prompt)
            log_handler("Create step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_synergetic(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(
                self._call_synergetic_generation_api, prompt
            )
            log_handler("Synergetic step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_category(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(self._call_category_generation_api, prompt)
            log_handler("Category step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_spf(self, prompt: str) -> Any:
        try:
            response = self._retry_api_call(self._call_spf_generation_api, prompt)
            log_handler("SPF step generated successfully")
            return response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_revised(self, prompt: str) -> Any:
        try:
            revised_response = self._retry_api_call(
                self._call_revised_generation_api, prompt
            )
            log_handler("Revised prompt generated successfully")
            return revised_response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_serenity(self, prompt: str) -> Any:
        try:
            revised_response = self._retry_api_call(
                self._call_serenity_generation_api, prompt
            )
            log_handler("Revised prompt generated successfully")
            return revised_response
        except Exception as e:
            log_handler(f"Max retries reached or API call failed: {e}")
            raise

    def generate_embeddings_cache(self, prompts: List[str]) -> List[List[float]]:
        # Check cache first
        new_prompts = [p for p in prompts if p not in self.embedding_cache]
        if new_prompts:
            new_embeddings = self.embedder(new_prompts)
            for prompt, emb in zip(new_prompts, new_embeddings):
                self.embedding_cache[prompt] = emb
        return [self.embedding_cache[p] for p in prompts]

    def get_num_tokens(
        self,
        text: str,
        last_user_msg: Optional[str] = None,
        similarity_weight: float = 0.5,
        use_similarity: bool = False,
        return_count: bool = False,
        semantic_similarity_cosine_func: callable = None,
    ) -> int:
        """Calculate number of tokens in a message, adjusted by semantic similarity if provided."""

        try:
            import tiktoken
        except ImportError:
            logging.error("Could not import tiktoken. Please install it via pip.")
            raise ImportError(
                "Could not import tiktoken. Install it via pip install tiktoken."
            )

        # Tokenize text to calculate base token count
        try:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
            tokenized_text = enc.encode(text)
            base_token_count = len(tokenized_text)
            if return_count:
                return base_token_count
        except Exception as e:
            logging.error(f"Tokenization failed due to: {e}")
            raise RuntimeError(f"Error during tokenization: {e}")

        # If a last_user_msg exists, adjust the token count based on semantic similarity
        if last_user_msg:
            try:
                if use_similarity:
                    similarity = semantic_similarity_cosine_func(
                        text,
                        last_user_msg,
                    )

                    # print the similarity score
                    log_handler(
                        f"Similarity score: {similarity}",
                        level="info",
                        verbose=self.verbose,
                    )

                    # Adjust token count based on similarity
                    adjusted_token_count = int(
                        base_token_count * (1 - similarity_weight * similarity)
                    )
                    return adjusted_token_count
            except Exception as e:
                logging.error(f"Error adjusting token count: {e}")
                return base_token_count

        return base_token_count

    def _truncate_conversation_history(
        self,
        conversation_history: List[Any],
        include_all_context: bool = False,
        verbose: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Truncates the conversation history to fit within a specified token limit,
        while preserving important context.

        Args:
            conversation_history (List[Any]): The full conversation history.
            preserve_context (bool): Flag to preserve important contextual messages.
            min_context_tokens (int): Minimum number of tokens to preserve for context.
            dynamic_threshold_window (int): Number of recent messages to consider for dynamic adjustments.
            include_all_context (bool): Flag to include the entire context, bypassing truncation.
            verbose (bool): Flag for verbose logging.

        Returns:
            List[Dict[str, str]]: The truncated conversation history.
        """

        if not conversation_history:
            return []

        if include_all_context:
            log_handler(
                f"Conversation history: {conversation_history}", verbose=verbose
            )
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation_history
            ]

        # Adjust token allocation multipliers
        assistant_token_multiplier = 2.0
        user_token_multiplier = 1.0  # Standard allocation for user messages

        # Existing logic to process and tokenize the conversation history
        conversation_history_dicts = []
        for msg in conversation_history:
            try:
                role = msg.__class__.__name__.replace("Chain", "").lower()
                content = msg.content.text if msg.content else ""
                if not content:
                    continue
                conversation_history_dicts.append({"role": role, "content": content})
            except AttributeError as e:
                logging.error(f"Error processing message: {e}")

        # Calculate tokens and apply multipliers
        adjusted_tokens = []
        for msg in conversation_history_dicts:
            base_tokens = self.get_num_tokens(
                msg["content"], conversation_history_dicts[-1]["content"]
            )
            if msg["role"] == "assistant":
                tokens = base_tokens * assistant_token_multiplier
            else:
                tokens = base_tokens * user_token_multiplier
            adjusted_tokens.append({**msg, "tokens": tokens})

        # Initialize variables for tracking tokens and managing truncated history
        truncated_history = []
        tokens_so_far = 0

        # Truncate conversation history based on adjusted tokens
        for message in reversed(adjusted_tokens):
            if tokens_so_far + message["tokens"] <= self.target_tokens:
                truncated_history.insert(
                    0, {"role": message["role"], "content": message["content"]}
                )
                tokens_so_far += message["tokens"]

        # Log the truncated history and token usage
        if verbose:
            log_handler(
                f"Truncated conversation history: {truncated_history}", verbose=verbose
            )
            log_handler(f"Tokens used: {tokens_so_far}", verbose=verbose)

        return truncated_history

    def _add_prompt_to_history(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]],
        role: str = "user",
    ) -> List[Dict[str, str]]:
        """
        Add a prompt to the conversation history.

        Args:
            prompt (str): The prompt to add to the conversation history.
            conversation_history (List[Dict[str, str]]): The conversation history.
            role (str): The role of the prompt.

        Returns:
            List[Dict[str, str]]: The updated conversation history.
        """
        conversation_history.append({"role": role, "content": prompt})
        return conversation_history

    def truncate_conversation_history(
        self,
        conversation_history: List[Dict[str, str]] = [],
        include_all_context: bool = False,
        verbose: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Truncate the conversation history to fit within the token limit.

        Args:
            use_basic_truncation (bool): Flag to use basic truncation.
            prompt (str): The prompt to add to the conversation history.
            conversation_history (List[Dict[str, str]]): The conversation history.
            include_all_context (bool): Flag to include the entire context, bypassing truncation.
            verbose (bool): Flag for verbose logging.

        Returns:
            List[Dict[str, str]]: The truncated conversation history.
        """

        return self._truncate_conversation_history(
            conversation_history, include_all_context, verbose
        )

    def _process_conversation_history(
        self,
        conversation_history: List[Dict[str, str]],
        prompt: Optional[str] = None,
        use_basic_truncation: bool = True,
        include_all_context: bool = False,
        role: str = "user",
        verbose: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Process the conversation history and add the prompt to it.

        Args:
            conversation_history (List[Dict[str, str]]): The conversation history.
            prompt (str): The prompt to add to the conversation history.
            role (str): The role of the prompt.
            include_all_context (bool): Flag to include the entire context, bypassing truncation.

        Returns:
            List[Dict[str, str]]: The updated conversation history.
        """
        if include_all_context:
            return self._add_prompt_to_history(prompt, conversation_history, role)

        truncated_history = self.truncate_conversation_history(
            conversation_history=conversation_history,
            include_all_context=include_all_context,
            verbose=verbose,
        )
        if prompt:
            return self._add_prompt_to_history(prompt, truncated_history, role)

        else:
            return truncated_history
