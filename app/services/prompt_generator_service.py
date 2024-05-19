import asyncio
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.callbacks.streaming import StreamingHandler
from app.response.system import ReplyChainSystem
from app.services.chat_service import AI
from app.helper import log_handler
from functools import lru_cache, wraps
from tqdm import tqdm
import random
import time
import uuid


class PromptGenerator:
    def __init__(
        self,
        callback: Optional[StreamingHandler] = None,
        segment_delimiter: Optional[str] = ".",
        name: Optional[str] = None,
        technique: Optional[object] = None,
        upload: Optional[bool] = None,
        path: Optional[str] = None,
        with_responses: Optional[bool] = False,
        storage: Optional[str] = None,
        verbose: Optional[bool] = False,
        create: Optional[bool] = False,
        stop: Optional[str] = None,
        provider: Optional[str] = "openai",
        convert: Optional[bool] = False,
        subdirectory: Optional[str] = None,
        chat: Optional[AI] = None,
    ):
        self.stop = stop
        self.name = name
        self.path = path
        self.build = create
        self.upload = upload
        self.verbose = verbose
        self.convert = convert
        self.storage = storage
        self.provider = provider
        self.subdirectory = subdirectory
        self.with_responses = with_responses

        if subdirectory is None:
            subdirectory = str(uuid.uuid4())

        self.subdirectory = subdirectory

        self.reply_chain_system = ReplyChainSystem(
            name=name,
            register_synthesis_technique=technique,
            verbose=verbose,
        )
        self.callback = (
            callback
            if callback
            else StreamingHandler(
                segment_delimiter=segment_delimiter,
            )
        )
        self.chat = chat

    @lru_cache(maxsize=None)
    def cached_generate_prompt_task(
        self,
        prompt: str,
        response: str,
        use_process_conversations: bool = False,
        custom_conversation_data: dict = None,
        generate_prompt: Optional[callable] = None,
    ) -> str:
        """Generate a prompt with the cached conversation data."""
        if use_process_conversations:
            return asyncio.run(
                generate_prompt(
                    prompt, response, custom_conversation_data=custom_conversation_data
                )
            )
        else:
            return asyncio.run(
                generate_prompt(prompt, response, custom_conversation_data=None)
            )

    def log_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            log_handler(
                f"Function {f.__name__} executed with arguments {args} and keyword arguments {kwargs}"
            )
            return result

        return wrapper

    @log_decorator
    def generate_parallel(
        self,
        num_prompts: int,
        max_workers: int,
        batch_size: int,
        example_pairs: Optional[List[Tuple[str, str]]] = None,
        use_process_conversations: bool = False,
        custom_conversation_data: Optional[List[dict]] = None,
        generate_prompt: Optional[callable] = None,
    ) -> List[str]:
        # Store adaptive timeout
        task_times = {}

        # Intelligent error handling
        adaptive_retry = {}

        # Store worker performance
        worker_performance = {}

        results = []
        generated_count = 0

        num_prompts = min(num_prompts, len(example_pairs))

        # Load balancing
        sorted_pairs = sorted(example_pairs, key=lambda x: task_times.get(x, 0))
        heavy_tasks = sorted_pairs[: len(sorted_pairs) // 2]
        light_tasks = sorted_pairs[len(sorted_pairs) // 2 :]
        random.shuffle(light_tasks)
        balanced_pairs = heavy_tasks + light_tasks

        # Only take the desired number of prompts
        balanced_pairs = balanced_pairs[:num_prompts]

        # Function to yield batches of tasks
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for batch in chunks(
                balanced_pairs, batch_size
            ):  # Use total_batches to process in chunks
                for prompt, response in batch:
                    timeout = min(5, task_times.get((prompt, response), 5))
                    worker = max(
                        worker_performance, key=worker_performance.get, default=None
                    )

                    future = executor.submit(
                        self.cached_generate_prompt_task,
                        prompt,
                        response,
                        use_process_conversations,
                        custom_conversation_data,
                        generate_prompt,
                    )
                    futures[future] = (prompt, response, time.time(), worker)

                for future in tqdm(
                    as_completed(futures),
                    total=len(balanced_pairs),
                    desc="Generating prompts",
                ):
                    (prompt, response, start_time, worker) = futures[future]
                    end_time = time.time()
                    task_duration = end_time - start_time

                    # Update average task time
                    if (prompt, response) in task_times:
                        task_times[(prompt, response)] = (
                            task_times[(prompt, response)] + task_duration
                        ) / 2
                    else:
                        task_times[(prompt, response)] = task_duration

                    # Update worker performance
                    if worker in worker_performance:
                        worker_performance[worker] = (
                            worker_performance[worker] + task_duration
                        ) / 2
                    else:
                        worker_performance[worker] = task_duration

                    try:
                        result = future.result(timeout=timeout)
                        results.append(result)
                        generated_count += 1
                    except Exception as e:
                        retries = adaptive_retry.get((prompt, response), 0)
                        if retries < 5:
                            adaptive_retry[(prompt, response)] = retries + 1
                        else:
                            print(
                                f"Failed generating prompt for pair: {(prompt, response)} after {retries} retries."
                            )

        return results
