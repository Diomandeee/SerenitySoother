from typing import Optional, List
from chain_tree.response.technique import SynthesisTechniqueDirector
from chain_tree.models import Content, ChainCoordinate
from chain_tree.utils import log_handler


class ChainDirector:
    """
    Manages and constructs chains.
    Attributes:
        technique_director (SynthesisTechniqueDirector): A director for synthesis techniques.
        builder (Type): The builder from the technique director.
        custom_challenges (List[str]): Custom challenges, if any.
        custom_prompts (List[str]): Custom prompts, if any.
        chain_steps (List[ChainStep]): List of chain steps to execute.

    Methods:
        validate_input(prompt: str, response: Optional[str] = None) -> bool: Validates the input.
        _construct(prompt: str, response: Optional[str] = None): Constructs the chain by executing the steps.
    """

    def __init__(
        self,
        technique_director: SynthesisTechniqueDirector,
        custom_challenges: Optional[List[str]] = None,
        custom_prompts: Optional[List[str]] = None,
    ):
        if not isinstance(technique_director, SynthesisTechniqueDirector):
            raise ValueError("The builder must be an instance of ChainBuilder.")

        self.technique_director = technique_director
        self.builder = self.technique_director.builder
        self.custom_challenges = custom_challenges or []
        self.custom_prompts = custom_prompts or []

    def validate_input(self, prompt: str, response: Optional[str] = None) -> bool:
        """
        Validates the input prompt and optional response.

        Ensures that the provided prompt and response are both strings. If they are not,
        a ValueError is raised.

        Args:
            prompt (str): The input prompt.
            response (Optional[str]): The optional input response.

        Returns:
            bool: True if the input is valid, False otherwise.

        Raises:
            ValueError: If either the prompt or response is not a string.
        """
        if not isinstance(prompt, str) or (
            response is not None and not isinstance(response, str)
        ):
            raise ValueError("Prompt and response must be strings.")
        return True

    def _build_assistant_chain(
        self, answer: str, coordinate: Optional[ChainCoordinate] = None
    ):
        """
        Constructs a chain for the assistant's response in a given spatial coordinate.

        Args:
            answer (str): The assistant's response.
            coordinate (Optional[Coordinate]): The spatial coordinate where the chain
                should be built. Defaults to Coordinate(x=0, y=0, z=0, t=1).

        Side Effects:
            Updates the state of the builder by constructing an assistant chain.
        """
        if coordinate is None:
            coordinate = ChainCoordinate(x=0, y=0, z=0, t=1)

        content = Content(
            text=answer, parts=[answer], part_lengths=len(answer.split("\n\n"))
        )
        self.builder.build_assistant_chain(content=content, coordinate=coordinate)

    def _build_user_chain(
        self, question: str, coordinate: Optional[ChainCoordinate] = None
    ):
        """
        Constructs a chain for the user's question in a given spatial coordinate.

        Args:
            question (str): The user's question.
            coordinate (Optional[Coordinate]): The spatial coordinate where the chain
                should be built. Defaults to Coordinate(x=0, y=0, z=0, t=3).

        Side Effects:
            Updates the state of the builder by constructing a user chain.
        """
        if coordinate is None:
            coordinate = ChainCoordinate(x=0, y=0, z=0, t=3)

        content = Content(
            text=question, parts=[question], part_lengths=len(question.split("\n\n"))
        )
        self.builder.build_user_chain(content=content, coordinate=coordinate)

    def _build_system_chain(
        self, content: str, coordinate: Optional[ChainCoordinate] = None
    ):
        """
        Constructs a chain for the system with the given content.

        Args:
            content (str): The content to be embedded within the system chain.

        Side Effects:
            Updates the state of the builder by constructing a system chain.
        """
        system_content = Content(
            text=content, parts=[content], part_lengths=len(content.split("\n\n"))
        )

        if coordinate is None:
            coordinate = ChainCoordinate(x=0, y=0, z=0, t=0)
        self.builder.build_system_chain(content=system_content, coordinate=coordinate)

    def construct_predefined_chain(
        self, prompt: str, post_prompt: Optional[str] = None
    ):
        """
        Constructs a predefined chain with system, user, and assistant roles.

        Args:
            prompt (str): The user's initial prompt.
            post_prompt (str): Additional content to append to the user's prompt.
        """
        if post_prompt is None:
            post_prompt = "Step 4: Synergetic Prompt:\nFinally, create"

        else:
            post_prompt = post_prompt

        # System's message
        system_content = (
            "Irrespective of the user prompt, the system will always "
            "interpret the prompt as a Synergetic Prompt. Do not use "
            "list formatting or bullet points. You will always enhance "
            "respond with challenge accepted."
        )
        self._build_system_chain(system_content)

        # User's first message
        user_content_step_0 = "Step 0: Imagine That:\n"
        self._build_user_chain(user_content_step_0)

        # User's second message (with prompt and post_prompt)
        user_content_prompt = prompt + "\n" + post_prompt
        self._build_user_chain(user_content_prompt)

        # Assistant's response
        assistant_content = "Create a..."
        self._build_assistant_chain(assistant_content)


class ReplyChainDirector(ChainDirector):
    """
    Specialized director for constructing reply chains.

    This director builds reply chains and provides functionality for detecting
    similar prompts to those previously processed.
    """

    def __init__(
        self,
        technique_director: SynthesisTechniqueDirector,
        custom_challenges: Optional[List[str]] = None,
        custom_prompts: Optional[List[str]] = None,
    ):
        """
        Initializes the ReplyChainDirector.

        Args:
            technique_director (SynthesisTechniqueDirector): A director for the synthesis technique.
            custom_challenges (Optional[List[str]]): A list of custom challenges. Default is None.
            custom_prompts (Optional[List[str]]): A list of custom prompts. Default is None.
        """
        super().__init__(technique_director, custom_challenges, custom_prompts)
        self.previous_prompts = []
        self.previous_responses = []
        self.synthesis_called = False
        self.system_prompt_called = False

    def handle_similar_prompt(self, prompt: str):
        """
        Handles the scenario when a similar prompt is detected.

        Args:
            prompt (str): The detected similar prompt.

        Side Effects:
            Logs and prints the information about the detected similar prompt.
        """
        print(f"Detected a similar prompt: {prompt}")
        log_handler(f"Similar prompt detected: {prompt}")

    def _construct_chain_parts(self, prompt: str, response: Optional[str]):
        """
        Constructs individual parts of the chain.

        Args:
            prompt (str): The user's question.
            response (Optional[str]): The assistant's response.
        """

        if not response:
            self._build_user_chain(prompt)

        if response:
            self._build_assistant_chain(response)
            self._build_user_chain(prompt)

    def build_synthesis(self) -> Optional[str]:
        """
        Triggers the synthesis process.

        Calls the synthesis only if it hasn't been called before for the current instance.

        Returns:
            Optional[str]: The name of the synthesis if it was built, otherwise None.
        """
        if not self.synthesis_called:
            name = self.technique_director._build_synthesis()
            self.synthesis_called = True
            return name
        return None

    def _cache_prompt_and_response(self, prompt: str, response: Optional[str]):
        """
        Caches the prompt and response for future similarity checks.

        Args:
            prompt (str): The user's question.
            response (Optional[str]): The assistant's response.
        """
        self.previous_prompts.append(prompt)
        self.previous_responses.append(response)

    def construct(
        self,
        prompt: str,
        response: Optional[str] = None,
        check_similarity: bool = False,
        use_predefined_chain: bool = False,
        post_prompt: Optional[str] = None,
        synthesis: bool = True,
    ):
        """
        Constructs the reply chain for the given prompt and response.

        Args:
            prompt (str): The user's question or statement.
            response (Optional[str]): The assistant's response. Default is None.
            check_similarity (bool): Flag indicating whether to check for prompt similarity.
                                     Default is False.

        Side Effects:
            Constructs user and assistant chains. Checks for prompt similarity and handles
            similar prompts if detected. Caches prompts and responses for future checks.
        """

        if synthesis:
            self.build_synthesis()

        if use_predefined_chain:
            self.construct_predefined_chain(prompt, post_prompt)

        else:
            self._construct_chain_parts(prompt, response)

        # Cache for future similarity checks
        self._cache_prompt_and_response(prompt, response)
