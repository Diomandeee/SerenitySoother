from typing import List, Dict, Optional, Union
from app.response.cohort import SynthesisTechniqueCohort
from app.response.builder import ReplyChainBuilder
from app.response.technique import SynthesisTechniqueDirector
from app.response.director import ReplyChainDirector
from app.helper import log_handler
import math

TOTAL_MAX_TOKEN_COUNT = 16385
MAX_TOKEN_COUNT_PER_TEXT = 4096 * 2


class ReplyChainSystem:
    """
    System to manage and construct reply chains using specified synthesis techniques.

    Attributes:
        reply_chain_builder (ReplyChainBuilder): Builder for the reply chains.
        technique_manager (SynthesisTechniqueCohort): Manager for synthesis techniques.
        name (str): Name of the random synthesis technique.
        tech_director (SynthesisTechniqueDirector): Director for synthesis techniques.
        director (ReplyChainDirector): Director for reply chains.
    """

    def __init__(
        self,
        verbose: bool = False,
        name: Optional[str] = None,
        register_synthesis_technique: Optional[object] = None,
    ):
        """
        Initializes the ReplyChainSystem with necessary components.
        """
        self.reply_chain_builder = ReplyChainBuilder()

        self.technique_manager = SynthesisTechniqueCohort()
        if register_synthesis_technique:
            self.technique_manager.register_synthesis_technique(
                register_synthesis_technique
            )
            self.name = register_synthesis_technique.technique_name

        else:
            if name:
                self.name = name
            else:
                self.name = self.technique_manager.get_random_synthesis_technique_name()

        self.tech_director = SynthesisTechniqueDirector(
            technique_name=self.name,
            builder=self.reply_chain_builder,
            technique_manager=self.technique_manager,
        )

        self.director = ReplyChainDirector(
            technique_director=self.tech_director,
        )

        self.verbose = verbose
        self.counter = 0

    def _validate_parameters(
        self,
        max_history_length: Optional[int],
        custom_conversation_data: Optional[List[Dict[str, str]]],
    ) -> None:
        """
        Validates the parameters.
        """
        if max_history_length is not None and not isinstance(max_history_length, int):
            raise ValueError("max_history_length must be an integer.")

        if custom_conversation_data is not None and not isinstance(
            custom_conversation_data, list
        ):
            raise ValueError("custom_conversation_data must be a list of dictionaries.")

    def clear_chains(self):
        """
        Clears the chains from the chain chain_tree.tree.

        Returns:
            None
        """
        self.chain_tree.clear_chains()
        self.director.synthesis_called = False

    def switch_synthesis_technique(self, name: Optional[str] = None):
        """
        Switches the synthesis technique to the specified name.

        Returns:
            None
        """
        self.chain_tree.clear_chains()
        self.director.synthesis_called = False

        if name:
            self.name = name

        self.tech_director = SynthesisTechniqueDirector(
            technique_name=self.name,
            builder=self.reply_chain_builder,
            technique_manager=self.technique_manager,
        )

        self.director = ReplyChainDirector(
            technique_director=self.tech_director,
        )

    def remove_last_chain(self):
        """
        Removes the last chain from the chain chain_tree.tree.

        Returns:
            None
        """
        self.chain_tree.remove_last_chain()

    def remove_first_chains(self, n: int):
        """
        Removes the first n chains from the chain chain_tree.tree.

        Args:
            n (int): Number of chains to remove.

        Returns:
            None
        """
        self.chain_tree.remove_first_chains(n)

    def remove_last_chains(self, n: int):
        """
        Removes the last n chains from the chain chain_tree.tree.

        Args:
            n (int): Number of chains to remove.

        Returns:
            None
        """
        self.chain_tree.remove_last_chains(n)

    def get_chains(self):
        """
        Retrieves chains from the chain chain_tree.tree.

        Returns:
            list: List of chains from the chain chain_tree.tree.
        """
        return self.chain_tree.get_chains()

    def _validate_conversation_data(self, data: List[Dict[str, Union[str, None]]]):
        """
        Validates the structure and content of the conversation data.

        Args:
            data (List[Dict[str, Union[str, None]]]): List of conversation items.

        Raises:
            ValueError: If the structure or content of the data is invalid.
            ValueError: If the number of tokens in a conversation item exceeds the limit.
            ValueError: If the cumulative tokens of all items exceed the total limit.
        """
        if not isinstance(data, list):
            raise ValueError("Conversation data must be a list of dictionaries.")

        if not data:
            raise ValueError("Conversation data must not be empty.")

        for item in data:
            # Validate the structure of each item
            if not isinstance(item, dict):
                raise ValueError("Each conversation item must be a dictionary.")

            if set(item.keys()) != {"prompt", "response"}:
                raise ValueError(
                    "Each conversation item must only have 'prompt' and 'response' keys."
                )

            if "prompt" not in item or not isinstance(item["prompt"], str):
                raise ValueError("The 'prompt' key must exist and be a string.")

            if "response" in item and not (
                isinstance(item["response"], str) or item["response"] is None
            ):
                raise ValueError("The 'response' key must be either a string or None.")

    def process_conversations(self, data: List[Dict[str, str]]) -> None:
        """
        Processes a list of conversation data to construct reply chains.

        Args:
        - data (List[Dict[str, str]]): List of dictionaries containing 'prompt' and 'response' as keys.

        Returns:
        - None
        """

        # Validate the conversation data to ensure it meets expectations
        try:
            self._validate_conversation_data(data)
        except ValueError as e:
            log_handler(
                f"Validation failed: {e}",
                step="process_conversations",
                verbose=self.verbose,
            )
            return

        # Loop through each conversation data item to construct reply chains
        for item in data:
            try:
                log_handler(
                    "Constructing reply chain...",
                    step="process_conversations",
                    verbose=self.verbose,
                )
                self.director.construct(item["prompt"], item["response"])
                self.counter += 1
                log_handler(
                    f"Successfully constructed reply chain for item: {item}",
                    step="process_conversations",
                    verbose=self.verbose,
                )
            except Exception as e:
                log_handler(
                    f"Failed to construct reply chain for item: {item}. Error: {e}",
                    step="process_conversations",
                )
                continue

        # Get the resulting chain tree
        self.chain_tree = self.reply_chain_builder.get_result()

        # Log completion message
        log_handler(
            f"Successfully constructed {self.counter} reply chains.",
            step="process_conversations",
            verbose=self.verbose,
        )

    def add_nodes_from_chains(
        self, chains: Optional[List[Union[str, dict]]] = None
    ) -> None:
        """
        Add nodes from chains to the chain chain_tree.tree. Each node in the tree represents a conversation segment
        from a given chain.

        Args:
        - chains (Optional[List[Union[str, dict]]]): List of conversation chains. If not provided, will use the default chains from the instance.

        Returns:
        - None
        """

        # Check if 'chains' is provided or default to instance chains.
        if chains is None:
            log_handler(
                "No chains provided. Fetching default chains.",
                step="add_nodes_from_chains",
                verbose=self.verbose,
            )
            chains = self.get_chains()

        # Validate that chains are provided and are in expected format
        if not chains or not isinstance(chains, list):
            log_handler(
                "Chains are either empty or not in the expected format. Exiting.",
                step="add_nodes_from_chains",
                verbose=self.verbose,
            )
            return

        # Iterating through each chain and adding them to the tree
        for chain in chains:
            try:
                # If chain is a dictionary, try to get the 'content' and 'text'. Otherwise, assume it's a string.
                content_text = (
                    chain.get("content", {}).get("text", "")
                    if isinstance(chain, dict)
                    else chain
                )

                # Ensure the chain has content and the content has text
                if content_text:
                    # Generate a new unique node ID based on the current number of nodes
                    node_id = len(self.chain_tree.get_nodes()) + 1

                    # Add the node to the chain tree
                    self.chain_tree.add_node(content_text, node_id)
                else:
                    log_handler(
                        f"Skipped chain due to missing content or text.",
                        step="add_nodes_from_chains",
                        verbose=self.verbose,
                    )
            except Exception as e:
                log_handler(
                    f"An exception occurred while processing a chain: {str(e)}",
                    step="add_nodes_from_chains",
                    verbose=self.verbose,
                )

    def _truncate_conversation(
        self,
        chains: Union[List[str], List[List[str]]],
        max_history_length: Optional[int] = None,
        prioritize_recent: bool = True,
        use_tokenizer: bool = False,
        get_token_ids: callable = None,
        reverse_tokenize: callable = None,
    ) -> List[str]:
        # Initialize tokens_to_remove to 0
        tokens_to_remove = 0

        if isinstance(chains[0], list):
            log_handler(
                "Chains provided as a list of lists. Flattening...",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            combined_chains = [chain for chains in chains for chain in chains]
        else:
            log_handler(
                "Chains provided as a single list.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            combined_chains = chains

        if not combined_chains:
            log_handler(
                "Provided chains parameter is empty. Raising a ValueError.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            raise ValueError("Conversation history must not be empty.")

        if max_history_length is not None:
            if prioritize_recent:
                log_handler(
                    "Prioritizing recent chains based on number of chains.",
                    step="_truncate_conversation",
                    verbose=self.verbose,
                )
                combined_chains = combined_chains[-max_history_length:]
            else:
                log_handler(
                    "Prioritizing older chains based on number of chains.",
                    step="_truncate_conversation",
                    verbose=self.verbose,
                )
                combined_chains = combined_chains[:max_history_length]

        token_count = 0
        truncated_chains = []
        effective_max_history_length = (
            min(TOTAL_MAX_TOKEN_COUNT, max_history_length)
            if max_history_length
            else TOTAL_MAX_TOKEN_COUNT
        )

        if prioritize_recent:
            log_handler(
                "Prioritizing recent chains.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            for chain in reversed(combined_chains):
                chain_tokens = 1
                new_token_count = token_count + len(chain_tokens)
                if new_token_count > effective_max_history_length:
                    break
                token_count = new_token_count
                truncated_chains.insert(0, chain)

            log_handler(
                f"Truncated chains have {token_count} tokens.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
        else:
            log_handler(
                "Prioritizing older chains.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            for chain in combined_chains:
                if use_tokenizer:
                    chain_tokens = get_token_ids(
                        chain.content.text,
                        verbose=True,
                        max_token_count=MAX_TOKEN_COUNT_PER_TEXT,
                    )
                else:
                    chain_tokens = [1]
                new_token_count = token_count + len(chain_tokens)
                if new_token_count > effective_max_history_length:
                    break
                token_count = new_token_count
                truncated_chains.append(chain)

            log_handler(
                f"Truncated chains have {token_count} tokens.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
        if token_count > TOTAL_MAX_TOKEN_COUNT:
            log_handler(
                "Exceeded TOTAL_MAX_TOKEN_COUNT. Applying additional truncation.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            median_index = len(truncated_chains) // 2
            truncated_chains.pop(median_index)
            new_token_count = sum(
                len(get_token_ids(chain.content.text, verbose=True))
                for chain in truncated_chains
            )

            log_handler(
                f"New token count after median truncation: {new_token_count}",
                step="_truncate_conversation",
                verbose=self.verbose,
            )

            tokens_to_remove = new_token_count - TOTAL_MAX_TOKEN_COUNT
            tokens_to_remove_per_chain = math.ceil(
                tokens_to_remove / len(truncated_chains)
            ).astype(int)

            log_handler(
                f"Removing {tokens_to_remove} tokens from {len(truncated_chains)} chains.",
                step="_truncate_conversation",
                verbose=self.verbose,
            )
            for chain in truncated_chains:
                chain_tokens = get_token_ids(chain.content.text, verbose=True)
                if len(chain_tokens) > tokens_to_remove_per_chain:
                    truncated_chain_tokens = chain_tokens[:-tokens_to_remove_per_chain]
                    chain.content.text = reverse_tokenize(truncated_chain_tokens)
                else:
                    truncated_chains.remove(chain)
                    tokens_to_remove -= len(chain_tokens)

        log_handler(
            f"Final truncated chains have {TOTAL_MAX_TOKEN_COUNT - tokens_to_remove} tokens.",
            step="_truncate_conversation",
            verbose=self.verbose,
        )
        return truncated_chains

    def construct_reply_chain(
        self,
        prompt: str,
        response: Optional[str] = None,
        check_similarity: bool = False,
        use_predefined_chain: bool = False,
        post_prompt: Optional[str] = None,
    ):
        """
        Constructs a reply chain using the given prompt and response.

        Args:
            prompt (str): User's prompt.
            response (Optional[str]): System's response. Default is None.

        Raises:
            ValueError: If prompt or response are not of type string.
        """
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        if response is not None and not isinstance(response, str):
            print(type(response))

        self.director.construct(
            prompt, response, check_similarity, use_predefined_chain, post_prompt
        )
        self.chain_tree = self.reply_chain_builder.get_result()

    def _process_custom_conversation_data(
        self,
        custom_conversation_data: List[Dict[str, str]],
        use_process_conversations: bool,
        check_similarity: bool = False,
        use_predefined_chain: bool = False,
        post_prompt: Optional[str] = None,
    ) -> None:
        """
        Processes custom conversation data based on the given flag use_process_conversations.
        """
        if use_process_conversations:
            log_handler(
                "Using process_conversations method for custom data",
                step="custom_data_process",
                verbose=self.verbose,
            )
            self.process_conversations(custom_conversation_data)
        else:
            log_handler(
                "Using individual construct_reply_chain calls for custom data",
                step="custom_data_process",
                verbose=self.verbose,
            )
            for conversation_item in custom_conversation_data:
                if "prompt" not in conversation_item:
                    raise ValueError(
                        "Each dictionary in custom_conversation_data should have a 'prompt' key."
                    )
                prompt = conversation_item["prompt"]
                response = conversation_item.get("response")
                self.construct_reply_chain(
                    prompt,
                    response,
                    check_similarity,
                    use_predefined_chain,
                    post_prompt,
                )

    def _process_single_conversation(
        self,
        prompt: Optional[str],
        response: Optional[str],
        use_process_conversations: bool,
        check_similarity: bool = False,
        use_predefined_chain: bool = False,
        post_prompt: Optional[str] = None,
    ) -> None:
        """
        Processes a single conversation based on the given flag use_process_conversations.
        """
        if use_process_conversations:
            log_handler(
                "Using process_conversations method for single conversation",
                step="single_conversation_process",
                verbose=self.verbose,
            )
            conversation_data = [{"prompt": prompt, "response": response}]
            self.process_conversations(conversation_data)
        else:
            log_handler(
                "Using construct_reply_chain for single conversation",
                step="single_conversation_process",
                verbose=self.verbose,
            )
            self.construct_reply_chain(
                prompt, response, check_similarity, use_predefined_chain, post_prompt
            )

    def prepare_conversation_history(
        self,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        use_process_conversations: bool = False,
        custom_conversation_data: Optional[List[Dict[str, str]]] = None,
        max_history_length: Optional[int] = None,
        prioritize_recent: bool = False,
        check_similarity: bool = False,
        use_predefined_chain: bool = False,
        post_prompt: Optional[str] = None,
    ) -> str:
        log_handler("Starting the preparation of conversation history", step="start")

        # Validate the parameters
        log_handler("Validating parameters", step="validation", verbose=self.verbose)
        self._validate_parameters(max_history_length, custom_conversation_data)

        # Process custom conversation data if provided
        if custom_conversation_data:
            log_handler(
                "Custom conversation data provided",
                step="custom_data",
                verbose=self.verbose,
            )

            # add the prompt and response to the custom conversation data
            if prompt:
                custom_conversation_data.append(
                    {"prompt": prompt, "response": response}
                )
            self._process_custom_conversation_data(
                custom_conversation_data, use_process_conversations
            )
        else:
            log_handler(
                "Single conversation provided",
                step="single_conversation",
                verbose=self.verbose,
            )
            self._process_single_conversation(
                prompt,
                response,
                use_process_conversations,
                check_similarity,
                use_predefined_chain,
                post_prompt,
            )
        # Truncate the conversation history if needed
        log_handler(
            "Truncating conversation history", step="truncate", verbose=self.verbose
        )
        truncated_history = self._truncate_conversation(
            self.get_chains(),
            max_history_length=max_history_length,
            prioritize_recent=prioritize_recent,
        )

        log_handler(
            "Completed the preparation of conversation history",
            step="complete",
            verbose=self.verbose,
        )

        return truncated_history
