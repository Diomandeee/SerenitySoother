from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, root_validator, Field
from chain_tree.models.chain import Chain
import numpy as np
import tiktoken
import torch
import os

tokenizer_tiktoken = tiktoken.get_encoding(
    "cl100k_base"
)  # The encoding scheme to use for tokenization

# Constants
CHUNK_SIZE = 200  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = int(
    os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", 128)
)  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""


class Generation(BaseModel):
    """Output of a single generation."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""


class ChainGeneration(Generation):
    """Output of a single generation."""

    text = ""
    message: Chain

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["text"] = values["message"].content
        return values


class ChainResult(BaseModel):
    """Class that contains all relevant information for a Chat Result."""

    generations: List[ChainGeneration]
    """List of the things generated."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_chain(self) -> List[Chain]:
        """Return prompt as messages."""


class BaseLanguageModel(BaseModel, ABC):
    @abstractmethod
    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: object = None,
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token present in the text."""
        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        return len(self.get_token_ids(text))

    def truncate_prompt(self, prompt: str, max_length: int = 2000) -> str:
        """
        Truncate the prompt to the maximum allowed length by removing sentences from the beginning.

        Args:
            prompt (str): The original prompt text.
            max_length (int): The maximum length allowed for the prompt.

        Returns:
            str: The truncated prompt.
        """
        # Split the prompt into sentences
        sentences = prompt.split(".")
        # Initialize the truncated prompt
        truncated_prompt = ""
        # Iterate through the sentences
        for sentence in sentences:
            # If the length of the truncated prompt plus the length of the sentence is less than the maximum length
            if len(truncated_prompt) + len(sentence) < max_length:
                # Add the sentence to the truncated prompt
                truncated_prompt += sentence + "."
            # Otherwise
            else:
                # Break out of the loop
                break
        # Return the truncated prompt
        return truncated_prompt

    def _calculate_content_variance(self, messages: List[BaseMessage]) -> float:
        """Calculate the variance of the content length of the messages."""
        content_lengths = [len(m.content) for m in messages]
        return np.var(content_lengths)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the message."""
        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])

    @classmethod
    def all_required_field_names(cls) -> Set:
        all_required_field_names = set()
        for field in cls.__fields__.values():
            all_required_field_names.add(field.name)
            if field.has_alias:
                all_required_field_names.add(field.alias)
        return all_required_field_names


def get_buffer_string(
    messages: List[Dict[str, Any]], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if "role" in m and "content" in m:
            if m["role"] == "user":
                role = human_prefix
            elif m["role"] == "assistant":
                role = ai_prefix
            elif m["role"] == "system":
                role = "System"
            else:
                raise ValueError(f"Got unsupported message type: {m}")

            string_messages.append(f"{role}: {m['content']}")
        else:
            raise ValueError(f"Invalid message format: {m}")

    return "\n".join(string_messages)


def _get_token_ids_default_method(
    text: str,
    verbose: bool = True,
    model_name: str = "distilbert-base-uncased",
    embeddings: bool = False,
    to_list: bool = True,
    max_token_count: Optional[int] = None,
    chunk_token_size: Optional[int] = None,
) -> Union[Dict, List[int]]:
    try:
        from transformers import AutoModel, GPT2TokenizerFast
    except ImportError:
        raise ValueError(
            "Could not import transformers python package. "
            "This is needed to tokenize the text. "
            "Please install it with `pip install transformers`."
        )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = AutoModel.from_pretrained(model_name) if embeddings else None

    if verbose and embeddings:
        split_texts = text.split("\n\n") if "\n\n" in text else [text]

        aggregated_tokens = []
        aggregated_token_ids = []
        aggregated_token_count = 0
        aggregated_embeddings = []

        for split_text in split_texts:
            inputs = tokenizer(split_text, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            token_ids = inputs.input_ids[0].tolist()
            token_count = inputs.input_ids[0].shape[0]

            # Truncation logic here
            if max_token_count is not None:
                if aggregated_token_count + token_count > max_token_count:
                    truncate_by = aggregated_token_count + token_count - max_token_count
                    tokens = tokens[:-truncate_by]
                    token_ids = token_ids[:-truncate_by]
                    token_count -= truncate_by

            if embeddings:
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                aggregated_embeddings.append(last_hidden_states)

            aggregated_tokens.extend(tokens)
            aggregated_token_ids.extend(token_ids)
            aggregated_token_count += token_count

        # Truncate results if necessary
        if max_token_count is not None and aggregated_token_count > max_token_count:
            aggregated_tokens = aggregated_tokens[:max_token_count]
            aggregated_token_ids = aggregated_token_ids[:max_token_count]
            aggregated_token_count = max_token_count

        result = {
            "tokens": aggregated_tokens,
            "token_ids": aggregated_token_ids,
            "token_count": aggregated_token_count,
        }

        if embeddings:
            aggregated_embeddings = torch.cat(aggregated_embeddings, dim=0)
            result["embedding"] = (
                aggregated_embeddings.tolist() if to_list else aggregated_embeddings
            )

        return result

    else:
        # Perform advanced tokenization
        tokens = tokenizer_tiktoken.encode(text, disallowed_special=())
        chunks = []
        chunk_size = chunk_token_size or CHUNK_SIZE
        num_chunks = 0

        while tokens and num_chunks < MAX_NUM_CHUNKS:
            chunk = tokens[:chunk_size]
            chunk_text = tokenizer_tiktoken.decode(chunk)

            last_punctuation = max(
                chunk_text.rfind("."),
                chunk_text.rfind("?"),
                chunk_text.rfind("!"),
                chunk_text.rfind("\n"),
            )

            if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
                chunk_text = chunk_text[: last_punctuation + 1]

            chunk_text_to_append = chunk_text.replace("\n", " ").strip()

            if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
                chunks.append(chunk_text_to_append)

            tokens = tokens[
                len(tokenizer_tiktoken.encode(chunk_text, disallowed_special=())) :
            ]
            num_chunks += 1

        if tokens:
            remaining_text = (
                tokenizer_tiktoken.decode(tokens).replace("\n", " ").strip()
            )
            if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
                chunks.append(remaining_text)

        return chunks


def reverse_tokenize(token_ids: Union[List[int], List[List[int]]]) -> str:
    """
    Converts a list of token IDs back to the original text.

    Parameters:
        - token_ids (Union[List[int], List[List[int]]]): A list of token IDs or a list of lists of token IDs.

    Returns:
        - str: The original text.
    """

    # Check if token_ids is a list of lists
    if isinstance(token_ids[0], list):
        # Decode each sublist and join them with double newlines
        return "\n\n".join(
            [
                tokenizer_tiktoken.decode(ids, skip_special_tokens=True)
                for ids in token_ids
            ]
        )
    else:
        # Decode the list of token IDs
        return tokenizer_tiktoken.decode(token_ids, skip_special_tokens=True)


def get_token_ids(
    text: str,
    verbose: bool = True,  # Changed to True as per your requirement
    model_name: str = "distilbert-base-uncased",
    embeddings: bool = False,
    to_list: bool = True,
    max_token_count: Optional[int] = None,
    chunk_token_size: Optional[int] = None,
) -> Union[Dict, List[int]]:
    """Get the token present in the text."""
    return _get_token_ids_default_method(
        text,
        verbose,
        model_name,
        embeddings,
        to_list,
        max_token_count,
        chunk_token_size,
    )
