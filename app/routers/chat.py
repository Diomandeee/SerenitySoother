from app.callbacks.streaming import StreamingHandler
from concurrent.futures import ThreadPoolExecutor
from app.response.system import ReplyChainSystem
from app.services.chat_service import AI, Chain
from app.chain_tree.state import chain_manager
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Tuple
from app.config import settings
from pydantic import BaseModel
import uuid


class GeneratePromptRequest(BaseModel):
    prompt: Optional[str] = None
    response: Optional[str] = None


class RunInteractiveRequest(BaseModel):
    mode: str = "run"
    answer_split: bool = False
    max_workers: int = 1
    batch_size: int = 4
    example_pairs: Optional[List[Tuple[str, str]]] = None
    with_responses: Optional[bool] = True


class GenerateEmbeddingsRequest(BaseModel):
    prompts: List[str]


class GenerateAudioRequest(BaseModel):
    prompt: str
    lang: str = "fr-FR"


class GenerateTranscriptRequest(BaseModel):
    file: str


class GenerateImageRequest(BaseModel):
    prompt: str


class GenerateMessageRequest(BaseModel):
    prompt: str


class UploadFileRequest(BaseModel):
    file_path: str
    bucket_subdir: str


class UploadBatchRequest(BaseModel):
    mode: str
    batch_files: List[str]
    conversation_id: str
    path: str
    verbose: bool = True


class UploadMediaRequest(BaseModel):
    conversation_id: str
    path: Optional[str] = None


class DownloadFileRequest(BaseModel):
    source_blob_name: str
    destination_file_name: str


router = APIRouter()

# Initialize the chat model and other necessary components
callback = StreamingHandler(segment_delimiter=".")
reply_chain_system = ReplyChainSystem()

chat_service = AI(
    storage="store",
    callbacks=[callback],
    model_name="gpt-3.5-turbo",
    credentials=settings.CREDENTIALS,
    max_tokens=4096,
    create=False,
    api_key=settings.OPENAI_API_KEY,
    play=False,
    process_media=False,
    internal=False,
    show=False,
    audio_func=False,
    provider="openai",
    subdirectory=str(uuid.uuid4()),
    target_tokens=16385,
)


def generate_prompt_parts(
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    conversation_id: Optional[str] = None,
    use_basic_truncation: Optional[bool] = False,
    **kwargs,
) -> Chain:
    """Generate prompt parts for the conversation."""
    conversation_history = reply_chain_system.prepare_conversation_history(
        prompt, response, **kwargs
    )

    truncated_history = chat_service._process_conversation_history(
        conversation_history,
        prompt=prompt,
        use_basic_truncation=use_basic_truncation,
        verbose=True,
    )

    return chat_service(truncated_history, conversation_id, None)


async def generate_prompt(
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    **kwargs,
) -> str:

    text = await chain_manager._generic_creation(
        prompt=prompt,
        response=response,
        generation_function=generate_prompt_parts,
        creation_function=_create_prompt_and_embedding,
        **kwargs,
    )
    return text


@router.post("/create_prompt")
async def _create_prompt_and_embedding(
    text: str,
    conversation_id: str,
    prompt: Optional[str] = None,
    upload: Optional[bool] = None,
    **kwargs,
) -> None:
    """Helper function to create prompt object and generate embedding."""
    embedding = chat_service.generate_embeddings([text])
    await chat_service.prompt_manager.create_prompt(
        prompt=prompt,
        prompt_parts=text.split("\n\n"),
        id=conversation_id,
        embedding=embedding,
        upload=upload,
        **kwargs,
    )


@router.post("/generate_prompt")
async def generate_prompt_endpoint(request: GeneratePromptRequest):
    try:
        text = await chain_manager._generic_creation(
            prompt=request.prompt,
            response=request.response,
            generation_function=generate_prompt_parts,
            creation_function=_create_prompt_and_embedding,
            upload=True,
        )
        return {"result": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/run_thread")
async def run_thread_endpoint(request: GenerateMessageRequest):
    try:
        text = await chain_manager._generic_prompt_creation(
            prompt=request.prompt,
            generation_function=generate_prompt_parts,
            creation_function=_create_prompt_and_embedding,
        )
        return {"result": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/run_chat")
async def run_chat_endpoint():
    try:
        result = await chain_manager.run_chat(
            generate_prompt=generate_prompt_parts,
            chat=chat_service,
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_prompt_parts")
async def generate_prompt_parts_endpoint(
    request: GeneratePromptRequest,
    conversation_id: Optional[str] = None,
    use_basic_truncation: Optional[bool] = False,
):
    try:
        result = generate_prompt_parts(
            prompt=request.prompt,
            response=request.response,
            conversation_id=conversation_id,
            use_basic_truncation=use_basic_truncation,
        )
        return {"result": result.content.raw}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _create_future_tasks(
    executor: ThreadPoolExecutor,
    prompt: str,
    answer: str = None,
    parent_id: str = None,
    answer_split: bool = False,
    mode: str = "run",
    **kwargs,
):
    futures = []
    if answer_split:
        responses = answer.split("\n\n")
    else:
        responses = [answer]

    for index, response in enumerate(responses):
        task_parent_id = f"{parent_id}_{index}" if answer_split else parent_id

        if mode == "run":
            future = executor.submit(run_thread_endpoint, prompt=prompt, **kwargs)
        elif mode == "run_chat":
            future = executor.submit(run_chat_endpoint, **kwargs)

        elif mode == "generate_prompt":
            future = executor.submit(
                generate_prompt_endpoint, prompt, response, **kwargs
            )

        elif mode == "generate_prompt_parts":
            future = executor.submit(
                generate_prompt_parts_endpoint,
                prompt=prompt,
                response=response,
                conversation_id=task_parent_id,
                **kwargs,
            )

        elif mode == "create_prompt":
            future = executor.submit(
                _create_prompt_and_embedding,
                text=prompt,
                conversation_id=task_parent_id,
                prompt=prompt,
                upload=True,
                **kwargs,
            )

        elif mode == "generate_audio":
            future = executor.submit(generate_audio_endpoint, prompt, **kwargs)

        elif mode == "generate_transcript":
            future = executor.submit(generate_transcript_endpoint, prompt, **kwargs)

        elif mode == "generate_image":
            future = executor.submit(generate_image_endpoint, prompt, **kwargs)

        elif mode == "generate_message":
            future = executor.submit(generate_message_endpoint, prompt, **kwargs)

        elif mode == "generate_imagine":
            future = executor.submit(generate_imagine, prompt, **kwargs)

        elif mode == "generate_brainstorm":
            future = executor.submit(generate_brainstorm, prompt, **kwargs)

        elif mode == "generate_questions":
            future = executor.submit(generate_questions, prompt, **kwargs)

        elif mode == "generate_create":
            future = executor.submit(generate_create, prompt, **kwargs)

        elif mode == "generate_synergetic":
            future = executor.submit(generate_synergetic, prompt, **kwargs)

        elif mode == "generate_category":
            future = executor.submit(generate_category, prompt, **kwargs)

        elif mode == "generate_revised":
            future = executor.submit(generate_revised, prompt, **kwargs)

        elif mode == "generate_spf":
            future = executor.submit(generate_spf, prompt, **kwargs)
        else:
            future = executor.submit(
                generate_prompt_endpoint, prompt, response, **kwargs
            )
        futures.append(future)
    return futures


def _get_unique_prompt(example_pairs: List, generated_prompts: set):
    parent_id = str(uuid.uuid4())

    prompt, answer = example_pairs.pop()

    while (prompt, answer, parent_id) in generated_prompts:
        prompt, answer, parent_id = example_pairs.pop()
    generated_prompts.add((prompt, answer, parent_id))
    return prompt, answer, parent_id


@router.post("/run_interactive")
async def run_interactive_endpoint(request: RunInteractiveRequest):
    try:
        num_prompts = len(request.example_pairs)

        generated_prompts = set()
        total_batches = (num_prompts + request.batch_size - 1) // request.batch_size
        valid_count = 0

        for batch_num in range(total_batches):
            with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
                futures = []

                for _ in range(min(request.batch_size, num_prompts - valid_count)):
                    prompt, answer, parent_id = _get_unique_prompt(
                        request.example_pairs, generated_prompts
                    )
                    if request.with_responses:
                        futures.extend(
                            await _create_future_tasks(
                                executor,
                                prompt,
                                answer,
                                parent_id,
                                request.answer_split,
                                request.mode,
                            )
                        )
                    else:
                        futures.extend(
                            await _create_future_tasks(
                                executor,
                                prompt,
                                None,
                                parent_id,
                                request.answer_split,
                                request.mode,
                            )
                        )

                for future in futures:
                    await future.result()

            valid_count += len(futures)

        return {"status": f"Generated {valid_count} prompts."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_embeddings")
def generate_embeddings_endpoint(request: GenerateEmbeddingsRequest):
    try:
        embeddings = chat_service.generate_embeddings(request.prompts)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_audio")
async def generate_audio_endpoint(request: GenerateAudioRequest):
    try:
        audio_response = await chat_service.generate_audio(request.prompt, request.lang)
        return {"audio_response": audio_response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_transcript")
async def generate_transcript_endpoint(request: GenerateTranscriptRequest):
    try:
        transcript_response = await chat_service.generate_transcript(request.file)
        return {"transcript_response": transcript_response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_image")
async def generate_image_endpoint(request: GenerateImageRequest):
    try:
        image_response = chat_service.generate_image_dalle(request.prompt)
        return {"image_response": image_response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_message")
async def generate_message_endpoint(request: GenerateMessageRequest):
    try:
        response = chat_service.call_as_llm(request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload_file")
async def upload_file_endpoint(request: UploadFileRequest):
    try:
        response = chat_service.prompt_manager.upload_file(
            request.file_path, request.bucket_subdir
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload_batch")
async def upload_batch_endpoint(request: UploadBatchRequest):
    try:
        chat_service.prompt_manager.upload_batch(
            request.mode,
            request.batch_files,
            request.conversation_id,
            request.path,
            request.verbose,
        )
        return {"response": "Batch upload completed."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload_media")
async def upload_media_endpoint(request: UploadMediaRequest):
    try:
        chat_service.prompt_manager.upload_media_in_parallel(
            request.conversation_id, request.path
        )
        return {"response": "Media upload completed."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/download_file")
async def download_file_endpoint(request: DownloadFileRequest):
    try:
        chat_service.prompt_manager.download_file(
            request.source_blob_name, request.destination_file_name
        )
        return {"response": "File downloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/list_files")
async def list_files_endpoint():
    try:
        files = chat_service.prompt_manager.list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/list_buckets")
async def list_buckets_endpoint():
    try:
        buckets = chat_service.prompt_manager.list_buckets()
        return {"buckets": buckets}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_imagine")
async def generate_imagine(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_imagine, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_brainstorm")
async def generate_brainstorm(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_brainstorm, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_questions")
async def generate_questions(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_questions, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_create")
async def generate_create(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_create, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_synergetic")
async def generate_synergetic(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_synergetic, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_category")
async def generate_category(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_category, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_revised")
async def generate_revised(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_revised, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_spf")
async def generate_spf(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_spf, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_serenity")
async def generate_serenity(prompt: str):
    try:
        response = chain_manager._generic_prompt_creation(
            prompt, chat_service.generate_serenity, _create_prompt_and_embedding
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# def generate_batch_examples(
#     data: pd.DataFrame,
#     filename: str = "batch_requests",
#     model: str = "gpt-3.5-turbo-0125",
#     max_tokens: int = 1000,
#     api_endpoint: str = "/v1/chat/completions"
# ) -> None:
#     """
#     Generate a batch file for API requests from a DataFrame.

#     Parameters:
#         data (pd.DataFrame): DataFrame containing the columns for system, user messages.
#         filename (str): Name of the file to save the batch requests.
#         model (str): Model identifier for the API request.
#         max_tokens (int): Maximum number of tokens for the response.
#         api_endpoint (str): API endpoint for the request.
#     """
#     with open(f"{filename}.jsonl", "w") as file:
#         for index, row in data.iterrows():
#             request_data = {
#                 "custom_id": f"request-{index}",
#                 "method": "POST",
#                 "url": api_endpoint,
#                 "body": {
#                     "model": model,
#                     "messages": [],
#                     "max_tokens": max_tokens
#                 }
#             }

#             # Append system message if available
#             if 'system_message' in row:
#                 request_data['body']['messages'].append({"role": "system", "content": row['system_message']})

#             # Append user and assistant messages
#             if 'user_message' in row and 'assistant_message' in row:
#                 request_data['body']['messages'].append({"role": "user", "content": row['user_message']})
#                 request_data['body']['messages'].append({"role": "assistant", "content": row['assistant_message']})

#             file.write(json.dumps(request_data) + "\n")

#     print(f"Batch requests saved to {filename}.jsonl")

# @router.post("/generate_batch_examples")
# async def generate_batch_examples_endpoint(
#     data: pd.DataFrame,
#     filename: Optional[str] = "batch_requests",
#     model: Optional[str] = "gpt-3.5-turbo-0125",
#     max_tokens: Optional[int] = 1000,
#     api_endpoint: Optional[str] = "/v1/chat/completions"
# ):
#     try:
#         generate_batch_examples(data, filename, model, max_tokens, api_endpoint)
#         return {"status": "Batch requests generated successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# # # Example of calling the function
# # data = pd.DataFrame({
# #     'system_message': ["You are a helpful assistant.", "You are an unhelpful assistant."],
# #     'user_message': ["Hello world!", "Hello world!"],
# #     'assistant_message': ["How can I assist you today?", "I'm not sure how to help."]
# # })
# # generate_batch_examples(data)
