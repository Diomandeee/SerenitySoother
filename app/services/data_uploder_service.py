import os
import re


class DataUploader:
    def __init__(self, directory):
        self.directory = directory
        self.logger = self.create_logger()

    def create_logger(self):
        import logging

        logger = logging.getLogger("DataUploader")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def natural_sort(self, l):
        """Sort the given list in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)

    def read_text_file(self, file_path: str) -> str:
        """Reads the content of a text file."""
        with open(file_path, "r") as file:
            return file.read()

    def process_file(
        self, file_path: str, verbose: bool = True, max_retries: int = 3
    ) -> str:
        """
        Processes a file and returns its local path.

        Args:
            file_path (str): The local path of the file.
            verbose (bool): If True, enables verbose output.
            max_retries (int): Maximum number of retries for processing the file.

        Returns:
            str: The local file path.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                if verbose:
                    print(f"Processed {file_path}")
                return file_path

            except (TimeoutError, ConnectionError) as e:
                attempt += 1
                if verbose:
                    self.logger.info(
                        f"Retry {attempt}/{max_retries} for {file_path} due to error: {e}"
                    )
                if attempt == max_retries:
                    self.logger.info(
                        f"Failed to process {file_path} after {max_retries} attempts"
                    )
            except Exception as e:
                if verbose:
                    self.logger.error(f"Error processing {file_path}: {e}")
                break  # Break on other types of exceptions

        return ""

    def process_batch(
        self,
        mode: str,
        batch_files: list,
        conversation_id: str,
        path: str,
        verbose: bool,
    ) -> dict:
        """
        Processes a batch of files and constructs the output data.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            batch_files (list): List of file names to be processed.
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            verbose (bool): If True, enables verbose output.

        Returns:
            dict: A dictionary containing the structured data for the batch.
        """
        data = {"conversation_id": conversation_id, mode: []}
        for file_name in batch_files:
            file_path = os.path.join(path, mode, conversation_id, file_name)
            if mode in ["prompt", "caption"]:
                content = self.read_text_file(file_path)
                data[mode].append(content)
            else:
                processed_file = self.process_file(file_path, verbose)
                if processed_file:
                    data[mode].append(processed_file)
        return data

    def initialize_processing(
        self, mode: str, conversation_id: str, path: str, batch_size: int, verbose: bool
    ) -> list:
        """
        Initializes the processing by creating batches of files.

        Args:
            mode (str): The modality of the files (e.g., 'audio', 'text').
            conversation_id (str): The conversation ID associated with the files.
            path (str): The base path where the files are located.
            batch_size (int): The number of files in each batch.
            verbose (bool): If True, enables verbose output.

        Returns:
            list: A list of tuples, each containing the arguments for processing a batch of files.
        """
        directory = os.path.join(path, mode, conversation_id)
        if not os.path.exists(directory):
            if verbose:
                self.logger.info(f"Directory not found: {directory}")
            return []

        file_list = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        file_list = self.natural_sort(file_list)
        file_list = [title for title in file_list if title != ".DS_Store"]

        return [
            (mode, file_list[i : i + batch_size], conversation_id, path, verbose)
            for i in range(0, len(file_list), batch_size)
        ]

    def construct_media_data(
        self, conversation_id: str, directory=None, **kwargs
    ) -> dict:
        """
        Manages the construction of media data across different modalities.

        Args:
            conversation_id (str): The conversation ID associated with the files.
            **kwargs: Additional keyword arguments including:
                      - path (str): The base path where the files are located.
                      - batch_size (int): The number of files in each batch.
                      - verbose (bool): If True, enables verbose output.

        Returns:
            dict: A dictionary containing the structured data for all modalities.
        """
        modalities = [
            "audio",
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

        media_data = {"prompt": [], "caption": [], "image": [], "audio": []}

        for mode in modalities:
            tasks = self.initialize_processing(
                mode, conversation_id, path, batch_size, verbose
            )
            for task in tasks:
                batch_data = self.process_batch(*task)
                media_data[mode].extend(batch_data[mode])

        return media_data

    def upload_all_media_in_parallel(self, base_path) -> dict:
        """
        Manages the parallel upload of media files across different modalities for all conversations.

        Args:
            base_path (str): The base path where the files are located.

        Returns:
            dict: A dictionary containing the structured data for all conversations.
        """
        all_media_data = {}
        title_list = os.listdir(base_path)
        title_list = [title for title in title_list if title != ".DS_Store"]
        for title in title_list:
            directory = os.path.join(base_path, title)
            all_media_data[title] = {
                "prompt": [],
                "caption": [],
                "image": [],
                "audio": [],
            }
            for folder_id in os.listdir(os.path.join(directory, "image")):
                folder_path = os.path.join(directory, "image", folder_id)
                if os.path.isdir(folder_path):
                    media_data = self.construct_media_data(folder_id, path=directory)
                    all_media_data[title]["prompt"].extend(media_data["prompt"])
                    all_media_data[title]["caption"].extend(media_data["caption"])
                    all_media_data[title]["image"].extend(media_data["image"])
                    all_media_data[title]["audio"].extend(media_data["audio"])
        return all_media_data
