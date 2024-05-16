from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, List
from chain_tree.models import Content, Chain, ChainCoordinate


class ChainBuilder(ABC):
    @abstractmethod
    def build_system_chain(self, content: Content, coordinate: ChainCoordinate):
        pass

    @abstractmethod
    def build_assistant_chain(self, content: Content, coordinate: ChainCoordinate):
        pass

    @abstractmethod
    def build_user_chain(self, content: Content, coordinate: ChainCoordinate):
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
        chain_type: Type[Chain],
        id: str,
        content: Content,
        coordinate: ChainCoordinate,
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
    def get_chains_by_coordinate(self, coordinate: ChainCoordinate):
        pass

    @abstractmethod
    def remove_chain(self, id: str):
        pass

    @abstractmethod
    def update_chain(
        self,
        id: str,
        new_content: Optional[Content] = None,
        new_coordinate: Optional[ChainCoordinate] = None,
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
        content: Content,
        coordinate: ChainCoordinate,
        metadata: Optional[Dict[str, Any]],
    ) -> Chain:
        pass

    @abstractmethod
    def generate_id(self) -> str:
        pass


class IChainTreeFactory(ABC):
    @abstractmethod
    def create_chain_tree(self) -> IChainTree:
        pass
