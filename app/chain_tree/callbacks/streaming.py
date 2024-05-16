from typing import Any, Dict, Union, Callable, Optional
from chain_tree.callbacks.base import BaseCallbackHandler
from chain_tree.base import EntityAction, EntityFinish
from concurrent.futures import ThreadPoolExecutor
from chain_tree.models import LLMResult
import PIL.Image as Image
from tqdm import tqdm
import threading
import requests
import logging
import base64
import json
import sys
import re
import os



class StreamingHandler(BaseCallbackHandler):
    """Callback handler for streaming to stdout."""

    def __init__(
        self, segment_delimiter: str, cloudinary: Optional[dict] = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.full_transcript = ""
        self.segment_delimiter = segment_delimiter
        self.cloudinary = cloudinary
        self.paragraph_delimiter = "\n\n"
        self.segment_count = 0
        self.logger = logging.getLogger(__name__)
        self.audio_thread = None
        self.session = None
        self.cumulative_story = ""
        self.prompt_segments = []
        self.audio_lock = threading.Lock()
 

    def set_language(self, language: str) -> None:
        self.language = language

    def accumulate_story(self, segment: str, special: bool = False) -> None:
        # Check if the segment is tagged as markdown and contains "Part"

        if special:
            if ("**Part" in segment or "** Part" in segment) and ": " in segment:
                parts = segment.split(": ")
                cleaned_segment = parts[-1].rstrip(self.paragraph_delimiter)
                self.cumulative_story += cleaned_segment + self.paragraph_delimiter
            else:
                if segment.endswith(self.paragraph_delimiter):
                    self.cumulative_story += (
                        segment.rstrip(self.paragraph_delimiter)
                        + self.paragraph_delimiter
                    )
                else:
                    self.cumulative_story += segment + self.segment_delimiter
        else:
            self.cumulative_story += segment
        self.prompt_segments.append(segment)

    def extract_and_clean_segment(self, segment: str) -> str:
        """
        Extracts and cleans a segment of text.

        Args:
            segment: The segment of text to clean.

        Returns:
            Cleaned segment.
        """
        part_number_match = re.search(r"Part\s*\d+", segment)
        return re.sub(r"Part\s*\d+", "", segment) if part_number_match else segment

    def extract_and_p_segment(self, segment: str) -> str:
        """
        Extracts and cleans a segment of text.

        Args:
            segment: The segment of text to clean.

        Returns:
            Cleaned segment.
        """
        part_number_match = re.search(r"P\s*\d+", segment)
        return re.sub(r"P\s*\d+", "", segment) if part_number_match else segment

    def extract_and_replace_segment(self, segment: str) -> str:
        """
        Extracts 'Tapestry' from a segment of text and replaces it with 'Mosaic'.

        Args:
            segment: The segment of text to process.

        Returns:
            The segment with 'Tapestry' replaced by 'Synergy'.
        """
        # First, check if 'Tapestry' is in the segment
        tapestry_match = re.search(r"Tapestry", segment, re.IGNORECASE)

        # Replace 'Tapestry' with 'Synergy'
        return (
            re.sub(r"Tapestry", "Synergy", segment, flags=re.IGNORECASE)
            if tapestry_match
            else segment
        )

    def extract_and_replace_segment_(self, segment: str) -> str:
        """
        Extracts 'Tapestry' from a segment of text and replaces it with 'Mosaic'.

        Args:
            segment: The segment of text to process.

        Returns:
            The segment with 'Tapestry' replaced by 'Synergy'.
        """
        # First, check if 'Tapestry' is in the segment
        tapestry_match = re.search(r"Tapestries", segment, re.IGNORECASE)

        # Replace 'Tapestry' with 'Synergy'
        return (
            re.sub(r"Tapestries", "Synergies", segment, flags=re.IGNORECASE)
            if tapestry_match
            else segment
        )

    def extract_segment(self, segment: str) -> str:
        """
        Extracts 'Tapestry' from a segment of text and replaces it with 'Mosaic'.

        Args:
            segment: The segment of text to process.

        Returns:
            The segment with 'Tapestry' replaced by 'Mosaic'.
        """

        cleaned_segment = self.extract_and_clean_segment(segment)
        cleaned_segment = self.extract_and_p_segment(cleaned_segment)
        cleaned_segment = self.extract_and_replace_segment(cleaned_segment)
        cleaned_segment = self.extract_and_replace_segment_(cleaned_segment)

        return cleaned_segment

    def create_file_path_and_directories(self, base_path: str, filename: str) -> str:
        """
        Creates a file path and necessary directories.

        Args:
            base_path: The base path where the file will be saved.
            filename: The name of the file to be saved.

        Returns:
            The full path of the file.
        """
        file_path = os.path.join(base_path, filename)
        os.makedirs(base_path, exist_ok=True)
        return file_path

    def generate_audio(
        self,
        cleaned_segment: str,
        generate_func: Callable,
        file_path: str,
        language: Optional[str] = None,
    ) -> None:
        """
        Generates audio for a cleaned segment.

        Args:
            cleaned_segment: The cleaned segment of text.
            generate_func: The function used to generate audio.
            file_path: Path where the audio will be saved.
            language: Language parameter if required by the audio generation function.
        """
        if language is None:
            response = generate_func(cleaned_segment)
        else:
            response = generate_func(cleaned_segment, language)

        if hasattr(
            response, "stream_to_file"
        ):  # Check if response has a stream_to_file method
            response.stream_to_file(file_path)
        else:
            # The audio_content is binary.
            with open(file_path, "wb") as out:
                out.write(response)

    def generate_embeddings(self, segment: str, generate_func: Callable) -> Any:
        """
        Generates embeddings for a segment.

        Args:
            segment: The segment for which embeddings will be generated.
            generate_func: The function used to generate embeddings.

        Returns:
            The generated embeddings.
        """
        return generate_func(segment)

    def save_data(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Saves data to a file in JSON format.

        Args:
            file_path: The file path where data will be saved.
            data: The data to be saved.
            data_key: The key under which the data will be stored.
        """
        with open(file_path, "w") as file:
            json.dump(data, file)

    def save_prompt(self, file_path: str, text: Any) -> None:
        """
        Saves data to a file in JSON format.

        Args:
            file_path: The file path where data will be saved.
            data: The data to be saved.
            data_key: The key under which the data will be stored.
        """
        with open(file_path, "a") as text_file:
            text_file.write(text)

    def save_image_prompt(self, file_path: str, text: Any) -> None:
        """
        Saves data to a file in JSON format.

        Args:
            file_path: The file path where data will be saved.
            data: The data to be saved.
            data_key: The key under which the data will be stored.
        """
        with open(file_path, "a") as text_file:
            text_file.write(text)

    def download_and_save_image(self, source: str, file_path_image: str) -> None:
        """
        Downloads and saves an image from a given source.

        Args:
            source: The source of the image, can be a URL or a base64 string.
            file_path_image: Path where the image will be saved.
        """
        try:
            if source.startswith("http"):
                response = requests.get(source)
                response.raise_for_status()
                image_data = response.content
            else:
                image_data = base64.b64decode(source)

            with open(file_path_image, "wb") as f:
                f.write(image_data)
        except Exception as e:
            self.logger.error(f"Error in saving the image: {e}")

    def display_image(self, image_path: str) -> None:
        """Displays an image from the specified path."""
        try:
            image = Image.open(image_path)
            image.show()
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")

    def play_audio(self, file_path: str) -> None:
        """Plays audio from the specified file path."""
        import pygame
        pygame.init()

        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")

    def play_audio_lock(self, file_path: str) -> None:
        """Plays audio from the specified file path."""
        import pygame
        pygame.init()

        with self.audio_lock:
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            except Exception as e:
                self.logger.error(f"Error playing audio: {e}")

    def start_audio_thread(
        self,
        file_path: str,
        play: bool,
        create: bool,
    ) -> None:
        """
        Starts a thread to play audio from the specified file path.

        Args:
            file_path: The path of the audio file to play.
            play: Flag to indicate whether to play the audio.
        """

        if play:
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join()
            if create:
                self.audio_thread = threading.Thread(
                    target=self.play_audio, args=(file_path,)
                )

            else:
                self.audio_thread = threading.Thread(
                    target=self.play_audio_lock, args=(file_path,)
                )
            self.audio_thread.start()

    def generate_and_play_audio(self, segment: str, **kwargs: Any) -> None:
        """
        Generates and plays audio for a given segment.

        Args:
            segment: The segment for which audio will be generated.
            kwargs: Additional keyword arguments.
        """
        base_path_audio = kwargs.get("base_path_audio")
        generate_audio_func = kwargs.get("generate_audio_func")
        play = kwargs.get("play", False)
        generate = kwargs.get("generate")
        create = kwargs.get("create")
        language = kwargs.get("language")

        if generate and generate_audio_func and segment:
            try:
                cleaned_segment = self.extract_segment(segment)
                unique_filename = f"segment_{self.segment_count}.mp3"
                file_path = self.create_file_path_and_directories(
                    base_path_audio, unique_filename
                )

                self.generate_audio(
                    cleaned_segment, generate_audio_func, file_path, language
                )
                self.start_audio_thread(file_path, play, create)

            except Exception as e:
                self.logger.error(f"Error generating or playing audio: {e}")

    def generate_and_play_audios(self, segments: list, **kwargs: Any) -> None:
        """
        Generates and plays audio for a given list of segments.

        Args:
            segments: The list of segments for which audio will be generated.
            kwargs: Additional keyword arguments.
                base_path_audio: The base path where audio files will be saved.
                generate_audio_func: The function used to generate audio.
                play: Flag indicating whether to play the audio.
                generate: Flag indicating whether to generate audio.
                create: Flag indicating whether to create a new audio thread.
                language: The language for audio generation.
                start_index: Custom start index for segment processing.
        """
        base_path_audio = kwargs.get("base_path_audio")
        generate_audio_func = kwargs.get("generate_audio_func")
        play = kwargs.get("play", False)
        generate = kwargs.get("generate")
        create = kwargs.get("create")
        language = kwargs.get("language")
        start_index = kwargs.get("start_index", 0)

        if generate and generate_audio_func and segments:
            try:
                cleaned_segments = segments
                segment_length = len(cleaned_segments)
                unique_filenames = [
                    f"segment_{i}.mp3"
                    for i in range(start_index, start_index + segment_length)
                ]
                file_paths = [
                    self.create_file_path_and_directories(base_path_audio, filename)
                    for filename in unique_filenames
                ]
                for i, segment in enumerate(
                    tqdm(cleaned_segments, total=segment_length)
                ):
                    self.generate_audio(
                        segment, generate_audio_func, file_paths[i], language
                    )
                    self.start_audio_thread(file_paths[i], play, create)

            except Exception as e:
                self.logger.error(f"Error generating or playing audio: {e}")

    def generate_image(
        self,
        segment: str,
        generate_func: Callable,
        file_path_image: str,
        is_dalle: bool = False,
    ) -> str:
        """
        Generates an image for a segment and saves it.

        Args:
            segment: The segment for which the image will be generated.
            generate_func: The function used to generate the image.
            file_path_image: Path where the image will be saved.
            is_dalle: Flag to indicate if DALL-E is used for image generation.

        Returns:
            The revised prompt used for generating the image.
        """
        revised_prompt, source = generate_func(segment)
        if is_dalle:
            self.download_and_save_image(source, file_path_image)
        else:
            download_thread = threading.Thread(
                target=self.download_and_save_image, args=(source, file_path_image)
            )
            download_thread.start()
            download_thread.join()

        return revised_prompt

    def generate_and_show_image_async(self, segment: str, **kwargs: Any) -> None:
        """
        Asynchronously generates and shows an image for a given segment.

        Args:
            segment: The segment for which the image will be generated.
            kwargs: Additional keyword arguments.
        """
        base_path_image = kwargs.get("base_path_image")
        base_path_text = kwargs.get("base_path_text")
        base_path_prompt = kwargs.get("base_path_prompt")
        base_path_caption = kwargs.get("base_path_caption")
        generate_image_func = kwargs.get("generate_image_func")
        generate_revised_func = kwargs.get("generate_revised_func")
        generate_imagine_func_dalle = kwargs.get("generate_imagine_func_dalle")
        show = kwargs.get("show", False)
        revised = kwargs.get("revised", False)
        create = kwargs.get("create", False)

        if create:
            try:
                (
                    file_path_image,
                    file_path_text,
                    file_path_prompt,
                    file_path_caption,
                ) = (
                    self.create_file_path_and_directories(
                        base_path_image, f"image_{self.segment_count}.png"
                    ),
                    self.create_file_path_and_directories(
                        base_path_text, f"text_{self.segment_count}.json"
                    ),
                    self.create_file_path_and_directories(
                        base_path_prompt, f"prompt_{self.segment_count}.md"
                    ),
                    self.create_file_path_and_directories(
                        base_path_caption, f"caption_{self.segment_count}.md"
                    ),
                )

                def generate_and_save():
                    try:
                        if generate_imagine_func_dalle:
                            if revised:
                                revised_image_prompt = generate_revised_func(
                                    self.cumulative_story
                                )
                            else:
                                revised_image_prompt = self.cumulative_story

                            revised_prompt = self.generate_image(
                                revised_image_prompt,
                                generate_imagine_func_dalle,
                                file_path_image,
                                is_dalle=True,
                            )
                        else:
                            revised_prompt = self.generate_image(
                                segment,
                                lambda seg: (
                                    generate_revised_func(seg),
                                    generate_image_func(seg),
                                ),
                                file_path_image,
                            )

                        self.save_data(
                            file_path_text,
                            {
                                "prompt": {
                                    "text": self.extract_and_replace_segment(segment),
                                    # "embeddings": self.generate_embeddings(
                                    #     segment,
                                    #     kwargs.get("generate_embeddings_func"),
                                    # ),
                                },
                                "revised_prompt": {
                                    "text": revised_prompt,
                                    # "embeddings": self.generate_embeddings(
                                    #     revised_prompt,
                                    #     kwargs.get("generate_embeddings_func"),
                                    # ),
                                },
                                "revised_image_prompt": {
                                    "text": revised_image_prompt,
                                    # "embeddings": self.generate_embeddings(
                                    #     revised_image_prompt,
                                    #     kwargs.get("generate_embeddings_func"),
                                    # ),
                                },
                            },
                        )

                        self.save_prompt(
                            file_path_prompt,
                            self.extract_and_replace_segment(segment),
                        )
                        self.save_prompt(file_path_caption, revised_prompt)
                        if show:
                            self.display_image(file_path_image)

                        self.logger.info(
                            f"Image and text data saved: {file_path_image}, {file_path_text}"
                        )
                    except Exception as e:
                        self.logger.error(f"Error generating or saving image: {e}")

                # Use ThreadPoolExecutor for parallel execution
                with ThreadPoolExecutor() as executor:
                    executor.submit(generate_and_save)

            except Exception as e:
                self.logger.error(f"Error generating or processing image and text: {e}")

    def process_media_segment(self, segment: str, **kwargs: Any) -> None:
        self.segment_count += 1

        self.generate_and_play_audio(segment=segment, **kwargs)
        self.accumulate_story(segment)

        if self.segment_delimiter == "\n\n" and kwargs.get("play", False):
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor() as executor:
                executor.submit(
                    self.generate_and_show_image_async, segment=segment, **kwargs
                )
        else:
            self.generate_and_show_image_async(segment=segment, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

        self.full_transcript += token
        process_media = kwargs.get("process_media", False)
        internal = kwargs.get("internal", False)
        generate_serenity_func = kwargs.get("generate_serenity_func")

        if process_media:
            if self.segment_delimiter in self.full_transcript:
                segment, self.full_transcript = self.full_transcript.split(
                    self.segment_delimiter, 1
                )
                segment = segment.strip()

                if internal:
                    cleaned_segment = self.extract_segment(segment)
                    modified_segment = generate_serenity_func(cleaned_segment)
                    split_segment = modified_segment.split(self.segment_delimiter)
                    for seg in split_segment:
                        self.process_media_segment(segment=seg, **kwargs)
                        self.cumulative_story = split_segment[-1]
                else:
                    self.process_media_segment(segment=segment, **kwargs)

    def on_llm_new_tokens(self, tokens: list, **kwargs: Any) -> None:
        for token in tokens:
            self.on_llm_new_token(token, **kwargs)

    def on_stream_end(self, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if self.full_transcript.strip():
            self.process_media_segment(segment=self.full_transcript, **kwargs)
            self.full_transcript = ""
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join()

    def on_cycle_end(self, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        id = kwargs.get("prompt_num")
        base_path_dir = kwargs.get("base_path_dir")
        add_text = kwargs.get("add_text")
        generate = kwargs.get("generate", True)
        create = kwargs.get("create")
        generate_imagine_func_dalle = kwargs.get("generate_imagine_func_dalle")
        file_path = os.path.join(base_path_dir, f"image_{id}.png")

        if create:
            self.cumulative_story = ""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print("\n\n")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: EntityAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: EntityFinish, **kwargs: Any) -> None:
        """Run on agent end."""
