from typing import Optional, Any
from google.api_core.exceptions import BadRequest, GoogleAPIError
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from datetime import datetime
from PIL import Image
import requests
import logging
import base64
import tqdm
import json
import glob
import uuid
import os
import re
import io


class PromptManager:
    def __init__(
        self,
        directory: str = "chain_database/prompts",
        filename_pattern: str = "synergy_chat_{}.json",
        credentials: dict = None,
    ):
        """
        Initialize the PromptManager.

        Args:
        - directory (str): The directory where prompts will be stored.
        - filename_pattern (str): The pattern for filenames. '{}' is replaced by the prompt number.
        """
        self.directory = directory
        self.filename_pattern = filename_pattern
        self.credentials = credentials

        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def create_directory(self, name: str) -> None:
        if not os.path.exists(name):
            os.makedirs(name)

    def _get_prompt_file_path(self, prompt_num: int) -> str:
        """Get the file path for a given prompt number."""
        filename = self.filename_pattern.format(prompt_num)
        return os.path.join(self.directory, filename)

    def load_prompt(self, prompt_num: Optional[int] = None) -> dict:
        """Load a prompt object."""
        if not prompt_num:
            prompt_num = self.get_last_prompt_num()

        prompt_path = self._get_prompt_file_path(prompt_num)
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                return json.load(f)
        else:
            self.logger.warning(f"Prompt {prompt_num} file does not exist.")
            return {"status": "NOT_FOUND"}

    def delete_all_prompts(self):
        """Remove all prompts."""
        for f in glob.glob(
            os.path.join(self.directory, self.filename_pattern.format("*"))
        ):
            os.remove(f)
        self.logger.info("All prompts deleted.")

    def get_last_prompt_num(self) -> int:
        """Retrieve the number of the latest prompt."""
        prompt_files = glob.glob(
            os.path.join(self.directory, self.filename_pattern.format("*"))
        )
        if prompt_files:
            prompt_files.sort(
                key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group())
            )
            return int(re.search(r"\d+", os.path.basename(prompt_files[-1])).group())
        return 0

    async def create_prompt_object(
        self,
        prompt_parts: list,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        id: Optional[str] = None,
        embedding: Optional[object] = None,
        image_format: str = "PNG",
    ) -> dict:
        """Construct a new prompt object and fetch image from URL to store in-memory."""

        # Validate input parameters
        if not prompt_parts:
            self.logger.error("No content provided for the prompt.")
            return {"status": "FAILURE"}

        prompt_num = self.get_last_prompt_num() + 1

        # Initialize the prompt object
        prompt_object = {
            "id": id,
            "create_time": str(datetime.now()),
            "prompt_num": prompt_num,
            "prompt": prompt,
            "response": prompt_parts,
            "revised_prompt": revised_prompt,
            "embedding": embedding,
            "image": None,
        }

        if prompt:
            prompt_object["prompt"] = prompt

        if revised_prompt:
            prompt_object["revised_prompt"] = revised_prompt

        # If an image URL is provided, fetch the image and store it in memory
        if image_url:
            try:
                response = requests.get(image_url)
                response.raise_for_status()

                # Load image into PIL
                image = Image.open(io.BytesIO(response.content))

                # Save the image in the specified format in-memory
                image_bytes_io = io.BytesIO()
                image.save(image_bytes_io, format=image_format)
                image_bytes_io.seek(0)

                # Convert image data to base64-encoded string
                base64_encoded_image = base64.b64encode(
                    image_bytes_io.getvalue()
                ).decode("utf-8")

                # Store the base64-encoded image data
                prompt_object["image"] = base64_encoded_image

            except requests.RequestException as e:
                self.logger.error(f"Request failed for image URL {image_url}: {e}")
                prompt_object["image"] = "Request failed"

            except IOError as e:
                self.logger.error(
                    f"IOError while processing image from URL {image_url}: {e}"
                )
                prompt_object["image"] = "IOError"

        return prompt_object


class CloudManager(PromptManager):
    def __init__(
        self,
        credentials: dict,
        directory: str = "prompts",
        filename_pattern: str = "synergy_chat_{}.json",
    ):
        """
        Initialize the CloudManager.

        Args:
        - directory (str): The directory where prompts will be stored.
        - filename_pattern (str): The pattern for filenames. '{}' is replaced by the prompt number.
        - bucket_name (str): The name of the bucket to use for cloud storage.
        - project: The project ID. If not passed, falls back to the default inferred from the environment.
        - credentials: The credentials to use when creating the client. If None, falls back to the default inferred from the environment.
        - http: The object to be used for HTTP requests. If None, an http client will be created.
        - client_info: The client info used to send a user-agent string along with API requests.
        - client_options: The options used to create a client.
        - use_auth_w_custom_endpoint: Boolean indicating whether to use the auth credentials when a custom API endpoint is set.
        """
        super().__init__(
            directory=directory,
            filename_pattern=filename_pattern,
        )
        self.client = storage.Client.from_service_account_json(
            json_credentials_path=credentials["service_account"]
        )

        bucket_name = credentials["bucket_name"]
        self.bucket = self.client.bucket(bucket_name)

    async def save_prompt_object_bucket(self, prompt_object):
        """Save a prompt object to the 'synergy_chat' bucket."""
        try:
            # Specifying the bucket name for this operation
            synergy_chat_bucket = self.client.bucket("synergy_chat")

            # Generate a unique UUID for each prompt object
            prompt_id = str(uuid.uuid4())
            filename = f"synergy_chat_{prompt_id}.json"

            # Add the generated ID to the prompt object
            prompt_object["id"] = prompt_id

            # Creating the blob object
            blob = synergy_chat_bucket.blob(filename)
            blob.upload_from_string(json.dumps(prompt_object))
            self.logger.info(f"Prompt {prompt_id} saved to bucket.")

        except Exception as e:
            self.logger.error(f"Error saving prompt object: {str(e)}")
            raise

    def _get_prompt_blob(self, prompt_num: int):
        blob_name = self.filename_pattern.format(prompt_num)
        return self.bucket.blob(blob_name)

    def get_last_prompt_num(self) -> int:
        """Retrieve the number of the latest prompt."""
        blobs = self.bucket.list_blobs()
        if blobs:
            prompt_nums = [
                int(re.search(r"\d+", os.path.basename(blob.name)).group())
                for blob in blobs
                if blob.name.endswith(".json")
            ]
            if prompt_nums:
                return max(prompt_nums)
        return 0

    async def create_prompt(
        self,
        prompt_parts: list,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        id: Optional[str] = None,
        embedding: Optional[object] = None,
        upload: Optional[bool] = None,
    ):
        """Save a prompt."""
        # Create the prompt object using the extracted or default information
        prompt_object = await self.create_prompt_object(
            prompt_parts=prompt_parts,
            prompt=prompt,
            image_url=image_url,
            revised_prompt=revised_prompt,
            id=id,
            embedding=embedding,
        )
        if upload:
            return self.upload_media_in_parallel(id)

        else:
            return await self.save_prompt_object_bucket(prompt_object)

    def upload_file(
        self,
        file_path: str,
        bucket_subdir: str,
        verbose: bool = True,
        max_retries: int = 3,
    ) -> None:
        """
        Uploads a file to a specified bucket directory with retry logic.

        Args:
            file_path (str): The local path of the file to be uploaded.
            bucket_subdir (str): The subdirectory in the bucket where the file will be uploaded.
            verbose (bool): If True, enables verbose output.
            max_retries (int): Maximum number of retries for the upload.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                blob = self.bucket.blob(
                    os.path.join(bucket_subdir, os.path.basename(file_path))
                )

                blob.upload_from_filename(file_path)
                if verbose:
                    print(f"Uploaded {file_path} to {bucket_subdir}")

                    # return gsutil UR
                return f"gs://{self.bucket.name}/{blob.name}"

            except (TimeoutError, ConnectionError) as e:
                attempt += 1
                if verbose:
                    self.logger.info(
                        f"Retry {attempt}/{max_retries} for {file_path} due to error: {e}"
                    )
                if attempt == max_retries:
                    self.logger.info(
                        f"Failed to upload {file_path} after {max_retries} attempts"
                    )
            except Exception as e:
                if verbose:
                    self.logger.error(f"Error uploading {file_path}: {e}")
                break  # Break on other types of exceptions

    def upload_batch(
        self,
        mode: str,
        batch_files: list,
        conversation_id: str,
        path: str,
        verbose: bool,
    ) -> None:
        """
        Processes and uploads a batch of files.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            batch_files (list): List of file names to be uploaded.
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            verbose (bool): If True, enables verbose output.
            rate_limit_delay (float): The delay between each file upload to respect rate limits.
        """
        bucket_subdir = os.path.join(mode, conversation_id)
        for file_name in batch_files:
            file_path = os.path.join(path, mode, conversation_id, file_name)
            self.upload_file(file_path, bucket_subdir, verbose)

    def initialize_upload(
        self, mode: str, conversation_id: str, path: str, batch_size: int, verbose: bool
    ) -> list:
        """
        Initializes the upload process by creating batches of files.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            batch_size (int): The number of files in each batch.
            verbose (bool): If True, enables verbose output.

        Returns:
            list: A list of tuples, each containing the arguments for uploading a batch of files.
        """
        directory = os.path.join(path, mode, conversation_id)
        if not os.path.exists(directory):
            if verbose:
                self.logger.info(f"Directory not found: {directory}")
            return []

        file_list = os.listdir(directory)
        return [
            (mode, file_list[i : i + batch_size], conversation_id, path, verbose)
            for i in range(0, len(file_list), batch_size)
        ]

    def upload_media_in_parallel(self, conversation_id: str, directory=None, **kwargs):
        """
        Manages the parallel upload of media files across different modalities.

        Args:
            conversation_id (str): The conversation ID associated with the files.
            **kwargs: Additional keyword arguments including:
                      - path (str): The base path where the files are located.
                      - batch_size (int): The number of files in each batch.
                      - verbose (bool): If True, enables verbose output.
                      - rate_limit_delay (float): The delay between each file upload to respect rate limits.
        """
        modality = [
            "audio",
            "text",
            "image",
            "caption",
            "prompt",
        ]
        if directory is None:
            directory = self.directory
        else:
            directory = directory

        path = kwargs.get("path", directory)
        batch_size = kwargs.get("batch_size", 10)
        verbose = kwargs.get("verbose", True)

        with ThreadPoolExecutor(max_workers=len(modality)) as executor:
            futures = []
            for mode in modality:
                tasks = self.initialize_upload(
                    mode, conversation_id, path, batch_size, verbose
                )
                for task in tasks:
                    futures.append(executor.submit(self.upload_batch, *task))

            for future in tqdm.tqdm(futures, total=len(futures)):
                future.result()

    def upload_all_media_in_parallel(self, base_path):
        """
                Manages the parallel upload of media files across different modalities for all conversations.
        r
                Args:
                    **kwargs: Additional keyword arguments including:
                              - path (str): The base path where the files are located.
                              - batch_size (int): The number of files in each batch.
                              - verbose (bool): If True, enables verbose output.
                              - rate_limit_delay (float): The delay between each file upload to respect rate limits.
        """
        title_list = os.listdir(base_path)
        title_list = [title for title in title_list if title != ".DS_Store"]
        for title in title_list:
            directory = os.path.join(base_path, title)
            for folder_id in os.listdir(os.path.join(directory, "image")):
                self.upload_media_in_parallel(folder_id, path=directory)

    def upload_from_directory(
        self, directory_path: str, bucket_subdirectory: str = "", verbose: bool = True
    ):
        """
        Uploads all files from a specified local directory to a Google Cloud Storage bucket.

        Args:
            directory_path (str): The path to the local directory.
            bucket_subdirectory (str): The subdirectory in the bucket where files will be uploaded.
        """
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                # Construct the destination blob name
                destination_blob_name = (
                    os.path.join(bucket_subdirectory, filename)
                    if bucket_subdirectory
                    else filename
                )
                # Upload the file
                self.upload_file(file_path, destination_blob_name, verbose=verbose)
        # return url
        except FileNotFoundError as e:
            self.logger.error(f"Error uploading files from directory: {e}")

        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")

        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")

        except Exception as e:
            self.logger.error(f"Error uploading files from directory: {e}")

    def download_file(
        self, source_blob_name, destination_file_name, verbose: bool = True
    ):
        download_id = uuid.uuid4()
        try:
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            if verbose:
                self.logger.info(
                    f"Download ID {download_id}: File {source_blob_name} downloaded to {destination_file_name}."
                )
        except BadRequest as e:
            self.logger.error(f"Download ID {download_id}: Bad request error: {e}")

        except GoogleAPIError as e:
            self.logger.error(f"Download ID {download_id}: Service error: {e}")

        except Exception as e:
            self.logger.error(f"Download ID {download_id}: Unexpected error: {e}")

    def process_blob(self, blob, destination_directory, verbose):
        if not blob.name.endswith("/"):
            destination_file_name = os.path.join(
                destination_directory, os.path.basename(blob.name)
            )
            self.download_file(blob.name, destination_file_name, verbose)

    def download_batch(self, mode, blobs, destination_directory, verbose):
        for blob in blobs:
            self.process_blob(blob, destination_directory, verbose)

    def initialize_download(self, mode, destination_directory, verbose):
        blobs = self.bucket.list_blobs(prefix=mode)
        return [(mode, blobs, destination_directory, verbose)]

    def download_media_in_parallel(self, **kwargs):
        modality = ["audio", "text", "image", "video"]
        destination_directory = kwargs.get("destination_directory", "chain_database")
        verbose = kwargs.get("verbose", True)

        with ThreadPoolExecutor(max_workers=len(modality)) as executor:
            futures = []
            for mode in modality:
                tasks = self.initialize_download(mode, destination_directory, verbose)
                for task in tasks:
                    future = executor.submit(self.download_batch, *task)
                    futures.append(future)

            for future in tqdm.tqdm(futures, total=len(futures)):
                future.result()

    def download_prompt(self, prompt_num: int, destination_directory: str):
        """Download a prompt object."""
        blob_name = self.filename_pattern.format(prompt_num)
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            blob.download_to_filename(os.path.join(destination_directory, blob_name))
        else:
            self.logger.warning(f"Prompt {prompt_num} file does not exist.")

    def upload_blob(self, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)
            self.logger.info(
                f"File {source_file_name} uploaded to {destination_blob_name}."
            )
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def get_url(self, blob_name: str):
        blob = self.bucket.blob(blob_name)
        return blob.public_url

    def delete_bucket(self, bucket_name: str):
        try:
            bucket = self.client.bucket(bucket_name)
            if bucket.exists():
                bucket.delete()
                self.logger.info(f"Bucket {bucket_name} successfully deleted.")
            else:
                self.logger.warning(f"Bucket {bucket_name} does not exist.")
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def list_buckets(self):
        try:
            buckets = self.client.list_buckets()
            for bucket in buckets:
                self.logger.info(bucket.name)
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def list_files(self):
        try:
            blobs = self.bucket.list_blobs()
            for blob in blobs:
                self.logger.info(blob.name)
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def load_prompt(self, prompt_num: Optional[int] = None) -> dict:
        """Load a prompt object."""
        if not prompt_num:
            prompt_num = self.get_last_prompt_num()

        blob = self._get_prompt_blob(prompt_num)
        if blob.exists():
            return json.loads(blob.download_as_string())
        else:
            self.logger.warning(f"Prompt {prompt_num} file does not exist.")
            return {"status": "NOT_FOUND"}

    def get_file(self, blob_name: str):
        try:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob
            else:
                self.logger.warning(f"File {blob_name} does not exist.")
        except BadRequest as e:
            self.logger.error(f"Bad request error: {e}")
        except GoogleAPIError as e:
            self.logger.error(f"Service error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
