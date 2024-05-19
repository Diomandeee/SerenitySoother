from typing import List
from app.chain_tree.base import SynthesisTechnique
from app.response.sernity import SerenityScribe
import random


class SynthesisTechniqueCohort:
    """
    Manages a collection of synthesis techniques and provides methods to
    access and manipulate them.
    """

    def __init__(self):
        """Initializes the list of available synthesis techniques."""
        self.synthesis_techniques: List[SynthesisTechnique] = [SerenityScribe()]

    def register_synthesis_technique(self, new_technique: SynthesisTechnique):
        """
        Registers a new synthesis technique to the cohort.

        Args:
            new_technique (SynthesisTechnique): The new synthesis technique to be added.

        Raises:
            ValueError: If a technique with the same name already exists.
        """
        if any(
            technique.technique_name == new_technique.technique_name
            for technique in self.synthesis_techniques
        ):
            raise ValueError(
                f"A technique with the name '{new_technique.technique_name}' already exists."
            )
        self.synthesis_techniques.append(new_technique)

    def get_random_synthesis_technique_name(self) -> str:
        """
        Returns a random synthesis technique name.

        Returns:
            str: Name of a randomly chosen synthesis technique.
        """
        return random.choice(self.get_synthesis_technique_names())

    def get_synthesis_technique(self, name: str) -> SynthesisTechnique:
        """
        Fetches the synthesis technique by name.

        Args:
            name (str): Name of the desired synthesis technique.

        Returns:
            SynthesisTechnique: The synthesis technique object with the provided name.

        Raises:
            ValueError: If the given synthesis technique name does not exist.
        """
        for synthesis_technique in self.synthesis_techniques:
            if (
                synthesis_technique.technique_name == name
                or synthesis_technique.name == name
            ):
                return synthesis_technique
        # get random technique if not found
        return random.choice(self.synthesis_techniques)

    def get_synthesis_technique_names(self) -> List[str]:
        """
        Retrieves all synthesis technique names.

        Returns:
            List[str]: List of names of all synthesis techniques.
        """
        return [
            synthesis_technique.technique_name
            for synthesis_technique in self.synthesis_techniques
        ]

    def get_synthesis_technique_epithets(self) -> List[str]:
        """
        Retrieves epithets of all synthesis techniques.

        Returns:
            List[str]: List of epithets for all synthesis techniques.
        """
        return [
            synthesis_technique.epithet
            for synthesis_technique in self.synthesis_techniques
        ]

    def get_synthesis_technique_imperatives(self) -> List[str]:
        """
        Retrieves imperatives of all synthesis techniques.

        Returns:
            List[str]: List of imperatives for all synthesis techniques.
        """
        return [
            synthesis_technique.imperative
            for synthesis_technique in self.synthesis_techniques
        ]

    def get_synthesis_technique_prompts(self) -> List[str]:
        """
        Retrieves prompts of all synthesis techniques.

        Returns:
            List[str]: List of prompts for all synthesis techniques.
        """
        return [
            synthesis_technique.prompts
            for synthesis_technique in self.synthesis_techniques
        ]

    def create_synthesis_technique(self, name: str) -> SynthesisTechnique:
        """
        Creates (or fetches) a synthesis technique by its name.

        Args:
            name (str): Name of the desired synthesis technique.

        Returns:
            SynthesisTechnique: The synthesis technique object with the provided name.
        """
        return self.get_synthesis_technique(name)
