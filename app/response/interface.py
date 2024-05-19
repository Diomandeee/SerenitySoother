from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, List


class ChainBuilder(ABC):
    @abstractmethod
    def build_system_chain(self, content, coordinate):
        pass

    @abstractmethod
    def build_assistant_chain(self, content, coordinate):
        pass

    @abstractmethod
    def build_user_chain(self, content, coordinate):
        pass

    def get_result(self):
        return self.chain_tree


class Technique(ABC):
    def __init__(
        self,
        builder: ChainBuilder,
        custom_challenges: Optional[List[str]] = None,
        custom_prompts: Optional[List[str]] = None,
    ):
        self.builder = builder
        self.custom_challenges = custom_challenges or []
        self.custom_prompts = custom_prompts or []

    def get_external_data_instructions(self):
        pass

    @abstractmethod
    def _fallback(self):
        pass

    @abstractmethod
    def _compute_novelty_factor(self):
        pass

    @abstractmethod
    def _generate_prompt(
        self, prompt: str, selected_option: str, selected_dynamic_prompt: str
    ) -> str:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class IChainTree(ABC):
    @abstractmethod
    def add_chain(
        self,
        chain_type: Type[Any],
        id: str,
        content: Any,
        coordinate: Any,
        metadata: Optional[Dict[str, Any]],
    ):
        pass

    @abstractmethod
    def get_chains(self):
        pass

    @abstractmethod
    def get_chain(self, id: str):
        pass

    @abstractmethod
    def get_last_chain(self):
        pass

    @abstractmethod
    def get_chains_by_type(self, chain_type: str):
        pass

    @abstractmethod
    def get_chains_by_coordinate(self, coordinate: Any):
        pass

    @abstractmethod
    def remove_chain(self, id: str):
        pass

    @abstractmethod
    def update_chain(
        self,
        id: str,
        new_content: Optional[Any] = None,
        new_coordinate: Optional[Any] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
    ):
        pass

    def add_link(self, link: dict):
        pass


class IChainFactory(ABC):
    @abstractmethod
    def create_chain(
        self,
        chain_type: str,
        id: str,
        content: Any,
        coordinate: Any,
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        pass

    @abstractmethod
    def generate_id(self) -> str:
        pass


class IChainTreeFactory(ABC):
    @abstractmethod
    def create_chain_tree(self) -> IChainTree:
        pass
