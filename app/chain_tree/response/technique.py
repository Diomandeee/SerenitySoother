from typing import Optional, List, Iterator, Dict, Any
from chain_tree.response.cohort import SynthesisTechniqueCohort
from chain_tree.interface import ChainBuilder
from chain_tree.interface import Technique
from chain_tree.models import Content, ChainCoordinate
import random
import re


class SynthesisTechniqueDirector(Technique):
    def __init__(
        self,
        builder: ChainBuilder,
        technique_manager: SynthesisTechniqueCohort,
        technique_name: str,
        max_depth: int = 5,
        recursion_depth: int = 3,
        novelty_threshold: float = 0.25,
    ):
        if not isinstance(builder, ChainBuilder):
            raise ValueError("The builder must be an instance of ChainBuilder.")

        self.builder = builder
        self.technique_manager = technique_manager
        self.max_depth = max_depth
        self.novelty_threshold = novelty_threshold
        self.recursion_depth = recursion_depth
        self.conversation_history = ""
        self.last_prompt = None
        self.last_option = None
        self.last_dynamic_prompt = None

        self.set_technique(technique_name)

    def save_conversation_history(self, filename: str) -> None:
        """Saves the conversation history to the given file."""
        with open(filename, "w") as f:
            f.write(self.conversation_history)

    def reset(self) -> None:
        """Resets the state of the director."""
        self.conversation_history = ""
        self.set_technique(self.technique_manager.get_synthesis_technique_names()[0])

    def get_synthesis_technique_info(self) -> Dict[str, Any]:
        """Returns the information about the current synthesis technique."""
        return {
            "name": self.technique.technique_name,
            "imperative": self.technique.imperative,
            "prompts": self.technique.prompts,
        }

    def set_technique(self, technique_name: str) -> None:
        """Sets the current synthesis technique to the one specified by the given name."""
        self.technique = self.technique_manager.create_synthesis_technique(
            technique_name
        )
        self.prompts_cycle = self._create_cycle(list(self.technique.prompts.keys()))

    def _create_cycle(self, items: List[str]) -> Iterator[str]:
        """
        Create a non-repeating, cyclic iterator over the given items.

        Args:
            items (List[str]): List of items to cycle over.

        Returns:
            Iterator[str]: Cyclic iterator over the shuffled items.
        """
        while True:
            random.shuffle(items)
            for item in items:
                yield item

    def _fallback(self):
        """
        Handle situations when the novelty factor of the generated message
        is below the desired threshold.

        To ensure variety in the conversation, this method switches to the next
        synthesis technique in the list. If the current technique is the last one,
        it wraps around to the beginning of the list. This method then resets
        the technique's state and the prompt history, ensuring a fresh start
        for message generation with the new technique.

        The flow is as follows:
        1. Find the index of the current technique in the list.
        2. Calculate the index for the next technique, wrapping around
        if the end of the list is reached.
        3. Fetch the name of the new technique based on its index.
        4. Set this new technique as the current technique.
        5. Reset the state related to prompts to ensure no repetition
        with the new technique.
        """
        current_technique_index = (
            self.technique_manager.get_synthesis_technique_names().index(
                self.technique.technique_name
            )
        )
        new_technique_index = (current_technique_index + 1) % len(
            self.technique_manager.get_synthesis_technique_names()
        )
        new_technique_name = self.technique_manager.get_synthesis_technique_names()[
            new_technique_index
        ]

        self.set_technique(new_technique_name)

        # Reset the state of the technique and prompt history for the new technique
        self.last_prompt = None
        self.last_option = None
        self.last_dynamic_prompt = None

    def _compute_novelty_factor(self, message: str, conversation_history: str) -> float:
        """
        Compute the novelty factor of a given message with respect to the entire conversation history.

        The novelty factor is calculated based on:
        1. Sorensen-Dice coefficient between the message and the conversation history.
        2. A decay factor that reduces the importance of earlier parts of the conversation.
        3. An adjustment based on the length of the current message.

        The resulting novelty factor will always be in the range [0, 1], where 0 indicates no novelty and 1 indicates maximum novelty.

        Parameters:
        - message (str): The message for which the novelty factor needs to be computed.
        - conversation_history (str): The entire conversation history up to this point.

        Returns:
        - float: The computed novelty factor, between 0 and 1.
        """
        message_words = set(message.lower().split())
        history_words = set(conversation_history.lower().split())
        common_words = message_words.intersection(history_words)

        if len(message_words) == 0 and len(history_words) == 0:
            return 0.0

        sorensen_dice_coefficient = (2 * len(common_words)) / (
            len(message_words) + len(history_words)
        )
        novelty_factor = 1 - sorensen_dice_coefficient

        # Apply a decay factor based on the conversation length to prioritize recent messages
        conversation_length = len(conversation_history.split())
        decay_factor = max(1 - (conversation_length / self.max_depth), 0.2)
        novelty_factor *= decay_factor

        # Adjust novelty factor based on the length of the message
        message_length_factor = max(len(message_words) / self.max_depth, 0.2)
        novelty_factor *= message_length_factor

        # Ensure novelty factor is within the range of 0 to 1
        novelty_factor = max(min(novelty_factor, 1.0), 0.0)

        return novelty_factor

    def _generate_prompt(
        self, prompt: str, selected_option: str, selected_dynamic_prompt: str
    ) -> str:
        """
        Generate an extended prompt by incorporating the given branching option and dynamic prompt into the technique's template.

        The method follows these steps:
        1. Retrieve the template associated with the prompt.
        2. Substitute placeholders in the template with the selected option and dynamic prompt.
        3. Format and capitalize the constructed prompt.
        4. Highlight the selected branching option with asterisks for emphasis.
        5. Return the constructed extended prompt with added exclamation for emphasis.

        Parameters:
        - prompt (str): The base prompt name.
        - selected_option (str): The selected branching option.
        - selected_dynamic_prompt (str): The selected dynamic prompt.

        Returns:
        - str: The constructed extended prompt with the selected option and dynamic prompt integrated and formatted.
        """

        prompt_data = self.technique.prompts[prompt]
        prompt_template = prompt_data.get("template")
        option_placeholder = "{option}"
        dynamic_prompt_placeholder = "{dynamic_prompt}"

        if prompt_template:
            # Replace placeholders in the template with the selected option and dynamic prompt
            prompt_text = prompt_template.replace(option_placeholder, selected_option)
            prompt_text = prompt_text.replace(
                dynamic_prompt_placeholder, selected_dynamic_prompt
            )

            # Capitalize the prompt text
            prompt_text = prompt_text.capitalize()
        else:
            # If a template is not provided, fallback to a default prompt construction
            prompt_text = f"{selected_option} {selected_dynamic_prompt}"

        # Combine the prompt text with the prompt name
        extended_prompt = f"{prompt} - {prompt_text}"

        # Remove extra whitespace and capitalize the prompt name and dynamic prompt
        extended_prompt = re.sub(r"\s+", " ", extended_prompt).strip()
        prompt_name, _, dynamic_prompt = extended_prompt.partition(" - ")
        dynamic_prompt = dynamic_prompt.capitalize()
        extended_prompt = f"{prompt_name} - {dynamic_prompt}"

        # Add an exclamation mark at the end of the prompt for emphasis
        extended_prompt += "!"

        # Highlight the selected option using asterisks
        highlighted_prompt = re.sub(
            re.escape(selected_option),
            lambda match: f"*{match.group(0)}*",
            extended_prompt,
            flags=re.IGNORECASE,
        )

        return highlighted_prompt

    def _build_synthesis(self, coordinate: Optional[ChainCoordinate] = None):
        """
        Constructs a synthesis chain based on the current technique,
        considering parameters like branching options, dynamic prompts,
        novelty threshold, and conversation history.

        The process involves the following steps:
        1. Choose a non-repeating prompt based on the current synthesis technique.
        2. For the chosen prompt, extract its associated branching options
        and dynamic prompts.
        3. Select a non-repeated branching option and dynamic prompt.
        4. Generate an extended prompt using the selected values.
        5. Build the system chain with the combined content derived from
        the technique's epithet and the extended prompt.
        6. Check the novelty factor of the generated message against the conversation history.
        If the novelty factor is below the threshold, use a fallback mechanism.

        Args:
            coordinate (Optional[Coordinate]): The starting spatial coordinate
            for the synthesis. If not provided, it defaults to (0, 0, 0, 4).

        Returns:
            str: The epithet of the technique used for this synthesis.
        """
        if coordinate is None:
            coordinate = ChainCoordinate(x=0, y=0, z=0, t=4)

        # Get the list of prompts
        prompts = list(self.technique.prompts.keys())

        # Shuffle the prompts to ensure randomization
        random.shuffle(prompts)

        selected_prompt = None
        selected_option = None
        selected_dynamic_prompt = None

        # Find the first non-repeating prompt
        for prompt in prompts:
            if prompt != self.last_prompt:
                selected_prompt = prompt
                break

        self.last_prompt = selected_prompt

        if not self.technique.prompts[selected_prompt]:
            return

        # Get the branching options and dynamic prompts for the selected prompt
        branching_options = self.technique.prompts[selected_prompt]["branching_options"]
        dynamic_prompts = self.technique.prompts[selected_prompt]["dynamic_prompts"]

        # Shuffle the branching options and dynamic prompts to ensure randomization
        random.shuffle(branching_options)
        random.shuffle(dynamic_prompts)

        # Find the first non-repeating branching option
        for option in branching_options:
            if option != self.last_option:
                selected_option = option
                break

        self.last_option = selected_option

        # Find the first non-repeating dynamic prompt
        for dynamic_prompt in dynamic_prompts:
            if dynamic_prompt != self.last_dynamic_prompt:
                selected_dynamic_prompt = dynamic_prompt
                break

        self.last_dynamic_prompt = selected_dynamic_prompt

        extended_prompt = self._generate_prompt(
            selected_prompt, selected_option, selected_dynamic_prompt
        )

        pre_amble = f"Epithet of {self.technique.name}: {self.technique.epithet}\n\n{self.technique.description} "

        if self.technique.system_prompt is None:
            combined_content = pre_amble + extended_prompt

        else:
            combined_content = (
                f"Epithet of {self.technique.name}:\n\n{self.technique.system_prompt} "
            )

        content = Content(text=combined_content)

        self.builder.build_system_chain(content=content, coordinate=coordinate)

        new_message = extended_prompt

        if (
            self._compute_novelty_factor(new_message, self.conversation_history)
            < self.novelty_threshold
        ):
            return self._fallback()

        self.conversation_history += new_message

        return self.technique.epithet

    def _build_synthesis_recursive(
        self, coordinate: Optional[ChainCoordinate] = None, depth: int = 0
    ):
        """
        Recursively constructs a synthesis chain based on the current technique,
        considering parameters like branching options, dynamic prompts,
        novelty threshold, and conversation history.

        The process involves the following steps:
        1. Build a synthesis chain with the current technique.
        2. If the depth is less than the maximum depth, recursively build a synthesis chain
        with a new technique.

        Args:
            coordinate (Optional[Coordinate]): The starting spatial coordinate
            for the synthesis. If not provided, it defaults to (0, 0, 0, 4).
            depth (int): The current depth of the recursion. Defaults to 0.

        Returns:
            str: The epithet of the technique used for this synthesis.
        """
        epithet = self._build_synthesis(coordinate)

        if depth < self.recursion_depth:
            return self._build_synthesis_recursive(coordinate, depth + 1)

        return epithet

    def build_synthesis(
        self, coordinate: Optional[ChainCoordinate] = None, recursive: bool = True
    ):
        """
        Constructs a synthesis chain based on the current technique,
        considering parameters like branching options, dynamic prompts,
        novelty threshold, and conversation history.

        The process involves the following steps:
        1. Choose a non-repeating prompt based on the current synthesis technique.
        2. For the chosen prompt, extract its associated branching options
        and dynamic prompts.
        3. Select a non-repeated branching option and dynamic prompt.
        4. Generate an extended prompt using the selected values.
        5. Build the system chain with the combined content derived from
        the technique's epithet and the extended prompt.
        6. Check the novelty factor of the generated message against the conversation history.
        If the novelty factor is below the threshold, use a fallback mechanism.

        Args:
            coordinate (Optional[Coordinate]): The starting spatial coordinate
            for the synthesis. If not provided, it defaults to (0, 0, 0, 4).
            recursive (bool): Whether to recursively build a synthesis chain
            with a new technique. Defaults to False.

        Returns:
            str: The epithet of the technique used for this synthesis.
        """
        if recursive:
            return self._build_synthesis_recursive(coordinate)

        syn = self._build_synthesis(coordinate)

        return syn
