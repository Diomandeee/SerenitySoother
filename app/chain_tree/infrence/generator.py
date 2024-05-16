from typing import Dict, List, Optional, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from chain_tree.utils import convert_webm_to_flac, log_handler
from chain_tree.infrence.session import PromptSessionWrapper
from chain_tree.callbacks.streaming import StreamingHandler
from chain_tree.response.system import ReplyChainSystem
from prompt_toolkit.completion import WordCompleter
from chain_tree.infrence.artificial import AI
from functools import lru_cache, wraps
from chain_tree.models import Chain
from pydub import AudioSegment
from datetime import datetime
from chain_tree.infrence.manager import (
    CloudManager,
    ChainManager,
    PromptManager,
)
from tqdm import tqdm
import random
import uuid
import json
import time
import os


class Generator(ChainManager):
    MAX_WORKERS = 4
    directory_path = "/Users/mohameddiomande/Desktop/chain_database/message"
    base_path = "/Users/mohameddiomande/Desktop/chain_database/recording"

    def save_message_to_json(self, message_data, json_file_path: str):
        def resolve_futures(data):
            if isinstance(data, Future):
                return data.result()  # Get the result of the Future
            elif isinstance(data, dict):
                return {k: resolve_futures(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [resolve_futures(item) for item in data]
            else:
                return data

        # Resolve any futures in message_data
        resolved_message_data = resolve_futures(message_data)

        # Create the folder if it doesn't exist, handling nested directories
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path, exist_ok=True)

        # Construct the final path
        json_file_path = os.path.join(self.directory_path, json_file_path)

        # Load existing messages if the file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as file:
                existing_messages = json.load(file)
        else:
            existing_messages = []

        # Add the new message to the list
        existing_messages.append(resolved_message_data)

        # Save the messages to the file
        with open(json_file_path, "w") as file:
            json.dump(existing_messages, file, indent=4)

        # return the full path
        return json_file_path

    def save_conversation(
        self,
        conversation_id: str,
        title: str = None,
        folder_path: str = None,
    ):
        """Save the current conversation to json file"""

        conversation = self.get_conversation(conversation_id)

        if not title:
            title = conversation.conversation_id

        convo = conversation.save_conversation(title, folder_path)

        return convo

    def rewind_conversation(self, conversation_id: str, steps: int = 1):
        conversation = self.get_conversation(conversation_id)
        conversation.rewind_conversation(steps)

    def load_conversation(self, conversation_id: str, title: str = "Untitled"):
        """Load a conversation from json file"""
        conversation = self.get_conversation(conversation_id)
        conversation.load_conversation(title)

    def delete_conversation(self, conversation_id: str) -> None:
        self.delete_conversation(conversation_id)

    def _generic_creation(
        self,
        prompt: Optional[str],
        generation_function: Callable[[Optional[str], Optional[Dict[str, Any]]], Any],
        creation_function: Callable[[str], Any],
        upload: bool = False,
        response: Optional[str] = None,
        **kwargs,
    ) -> str:
        conversation_id = self.create_conversation()

        generated_parts = generation_function(
            prompt, response, conversation_id=conversation_id, **kwargs
        )
        # If the generation_function returns a Chain object, access the content
        text = generated_parts.content.raw

        # Create the prompt object with the prompt parts and embedding
        creation_function(text, prompt, conversation_id, upload, **kwargs)

        return text

    def _generic_prompt_creation(
        self,
        prompt: Optional[str],
        generation_function: Callable[[Optional[str], Optional[Dict[str, Any]]], Any],
        creation_function: Callable[[str], Any],
        **kwargs,
    ) -> str:
        conversation_id = self.create_conversation()

        # Use the generation_function to get the processed conversation parts
        text = generation_function(prompt, **kwargs)

        # Create the prompt object with the prompt parts and embedding
        creation_function(text, prompt, conversation_id, **kwargs)

        return text

    def handle_command(
        self, command: str, conversation_id: str, conversation_path: str
    ) -> str:
        if command.startswith("/save"):
            parts = command.split(maxsplit=1)  # Split the command into at most 2 parts
            if len(parts) > 1:
                title = parts[1]  # The second part is our title
                try:
                    self.save_conversation(
                        conversation_id, title, folder_path=conversation_path
                    )
                    return "Conversation saved successfully with title: " + title
                except Exception as e:
                    return f"Error saving conversation: {e}"
            else:
                return "Please 4provide a title. Use /save <title>."

        elif command.startswith("/load"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                title = parts[1]
                try:
                    self.load_conversation(conversation_id, title)
                    return "Conversation loaded successfully with title: " + title
                except Exception as e:
                    return f"Error loading conversation: {e}"

            else:
                return "Please provide a title. Use /load <title>."

        elif command.startswith("/rewind"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                steps = int(parts[1])
                self.rewind_conversation(conversation_id, steps)
                return f"Conversation rewinded by {steps} steps."
            else:
                return "Please provide the number of steps to rewind. Use /rewind <number>."

        elif command.startswith("/delete"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                title = parts[1]
                try:
                    self.delete_conversation(conversation_id, title)
                    return "Conversation deleted successfully with title: " + title
                except Exception as e:
                    return f"Error deleting conversation: {e}"
            else:
                return "Please provide a title. Use /delete <title>."

        elif command == "/restart":
            self.restart_conversation(conversation_id)
            return "Conversation restarted."

        elif command == "/history":
            return "\n\n".join(self.get_conversation(conversation_id).get_messages())

        elif command == "/quit":
            return "Quitting and saving the conversation..."

        elif command == "q u i t":
            return "Quitting and saving the conversation..."

        elif command == "/help":
            return self.help()

        else:
            return "Unknown command. Use /help for a list of commands."

    def help(self):
        commands = {
            "/save": "Save the conversation to a JSON file. Use '/save <title>' to specify a title.",
            "/load": "Load a conversation from a JSON file. Use '/load <title>' to specify a title.",
            "/delete": "Delete a conversation from a JSON file. Use '/delete <title>' to specify a title.",
            "/restart": "Restart the conversation.",
            "/history": "Show the conversation history.",
            "/quit": "Quit the conversation and save it to a JSON file.",
            "/help": "Show this help message.",
            "/rewind": "Rewind the conversation by a specified number of steps. Use '/rewind <number>' to specify the number of steps.",
        }
        return "\n".join(
            [f"{command}: {description}" for command, description in commands.items()]
        )

    def submit_task(
        self,
        executor: ThreadPoolExecutor,
        generate_prompt: Callable,
        message_data: Optional[str],
        response: Optional[str] = None,
        use_process_conversations: Optional[bool] = False,
    ) -> Future:
        return executor.submit(
            generate_prompt, message_data, response, use_process_conversations
        ).result()

    def is_conversation_finished(self, conversation_id: str) -> bool:
        return self.get_conversation(conversation_id).is_finished()

    def process_message_data(
        self, conversation_id: str, message_data: str, message_data_user: str
    ):
        last_message_id = self.get_conversation(conversation_id).get_last_message_id()

        self.handle_user_input(conversation_id, message_data_user, last_message_id)
        self.handle_agent_response(conversation_id, message_data, last_message_id)

    def handle_feedback(self, chat: AI):
        feedback_input = (
            input("Are you satisfied with the responses? (yes/no): ").strip().lower()
        )
        if feedback_input == "no":
            self.adjust_chat_temperature(chat)

    def adjust_chat_temperature(self, chat: AI):
        adjustment = (
            input(
                "Would you like the responses to be more 'random' or 'deterministic'? "
            )
            .strip()
            .lower()
        )
        if adjustment == "random":
            chat.temperature += 0.1
        elif adjustment == "deterministic":
            chat.temperature -= 0.1
        chat.temperature = min(max(chat.temperature, 0.2), 2.0)
        print(f"model_name temperature adjusted to: {chat.temperature}")

    def display_results_and_feedback(
        self, chat, message_data, display_results, feedback
    ):
        if display_results:
            for key, future in message_data.items():
                print(f"{key}: {future.result()}")
                print("\n")

            if feedback:
                self.handle_feedback(chat)

    def run_streaming(
        self,
        audio_file=None,
        energy_threshold: int = 300,
        record_timeout: int = 1,
        min_audio_duration: float = 1.0,  # Minimum duration to consider as valid audio
        sample_rate: int = 16000,
        default_microphone: str = None,
        language: str = "en-US",
    ) -> str:
        import speech_recognition as sr
        import platform

        recorder = sr.Recognizer()
        recorder.energy_threshold = energy_threshold
        recorder.dynamic_energy_threshold = False
        recorder.pause_threshold = 0.5

        if not default_microphone:
            default_microphone = (
                "pulse" if "linux" in platform.system().lower() else None
            )

        if default_microphone:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if default_microphone in name:
                    source = sr.Microphone(sample_rate=sample_rate, device_index=index)
                    break
            else:
                source = sr.Microphone(sample_rate=sample_rate)
        else:
            source = sr.Microphone(sample_rate=sample_rate)

        transcription = []
        stop_command_detected = False

        def record_callback(recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
            nonlocal stop_command_detected
            try:
                text = recognizer.recognize_google(audio, language=language)
                print(f"{text}")
                transcription.append(text)
                # Check if "stop" is detected in the recognized text
                if "stop" in text.lower():
                    stop_command_detected = True
            except sr.UnknownValueError:
                print("")
            except sr.RequestError as e:
                print(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
            # Save audio data to a file
            base = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = "Recording" + " " + base
            file_name = base_name + ".wav"  # Change extension as per your requirement
            file_path = os.path.join(self.base_path, file_name)  # Modify path
            with open(file_path, "wb") as f:
                f.write(audio.get_wav_data())
            # Calculate audio duration based on length and sample rate
            audio_duration = len(audio.frame_data) / (
                audio.sample_width * audio.sample_rate
            )
            # Delete the file if audio duration is less than min_audio_duration or empty
            if audio_duration < min_audio_duration or audio_duration < 1.0:
                os.remove(file_path)

        with source as s:
            recorder.adjust_for_ambient_noise(s, duration=1)
            print("...")
            while not stop_command_detected:
                try:
                    audio = recorder.listen(s, timeout=record_timeout)
                    record_callback(recorder, audio)
                except sr.WaitTimeoutError:
                    if not transcription:
                        continue
                    else:
                        break  # Break the loop if there's already some transcription
        if audio_file:
            try:
                with sr.AudioFile(audio_file) as source:
                    audio_data = recorder.record(source)
                    text = recorder.recognize_google(audio_data)
                    transcription.append(text)
            except Exception as e:
                print(f"An error occurred while processing the audio file: {e}")

        return " ".join(transcription)

    def run_chat(
        self,
        generate_prompt: Callable,
        chat: AI,
        directory,
        subdirectory,
        audio_file: Optional[str] = None,
        initial_prompt: str = "",
        end_letter: str = "A",
        end_roman_numeral: Optional[str] = None,
        feedback: bool = False,
        display_results: bool = False,
        streaming: bool = False,
        conversation_path: str = None,
        record_timeout: int = 5,
        language: str = "en-US",
    ):
        response = None
        base_dirs = [
            "recordings",
            "transcriptions",
            "conversations",
            "youtube",
            "modalities",
            "website",
            "image",
        ]

        # create subdirectory if it does not exist
        if not os.path.exists(os.path.join(directory, subdirectory)):
            os.makedirs(os.path.join(directory, subdirectory))

        output_directory = os.path.join(directory, subdirectory, base_dirs[0])
        transcption_directory = os.path.join(directory, subdirectory, base_dirs[1])

        conversation_path = os.path.join(directory, subdirectory, base_dirs[2])
        youtube_directory = os.path.join(directory, subdirectory, base_dirs[3])
        modalities_directory = os.path.join(directory, subdirectory, base_dirs[4])
        website_directory = os.path.join(directory, subdirectory, base_dirs[5])
        image_directory = os.path.join(directory, subdirectory, base_dirs[6])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not os.path.exists(transcption_directory):
            os.makedirs(transcption_directory)

        if not os.path.exists(conversation_path):
            os.makedirs(conversation_path)

        if not os.path.exists(youtube_directory):
            os.makedirs(youtube_directory)

        if not os.path.exists(modalities_directory):
            os.makedirs(modalities_directory)

        if not os.path.exists(website_directory):
            os.makedirs(website_directory)

        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        conversation_id = self.start_conversation(initial_prompt)
        saved_messages = []
        user_message = ""
        title = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        # Define a list of Roman numerals
        roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        filename_global = input("Enter a filename to save the conversation: ")

        # Adjust the max_roman_index calculation
        max_roman_index = None
        if end_roman_numeral:
            max_roman_index = roman_numerals.index(end_roman_numeral) + 1

        while True:
            message_data = {}
            if streaming:
                user_message = self.run_streaming(
                    audio_file=audio_file,
                    record_timeout=record_timeout,
                    language=language,
                )

            else:
                session = PromptSessionWrapper()
                user_message = session.session.prompt(
                    completer=WordCompleter(["quit", "restart"], ignore_case=True)
                )
            print("\n")
            # Command Handling

            if user_message.startswith("/"):
                try:
                    response = self.handle_command(
                        user_message.strip().lower(), conversation_id, conversation_path
                    )

                except Exception as e:
                    print(e)
                    break

                print(response)
                print("\n\n")  # New line after the command's response
                continue

            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                prev_future = None

                # Adjusting the outer loop based on the provided end_letter
                for current_char in map(chr, range(ord("A"), ord(end_letter) + 1)):
                    # If end_roman_numeral is provided, iterate over Roman numerals
                    if end_roman_numeral:
                        for roman in roman_numerals[:max_roman_index]:
                            if prev_future:
                                result_list = prev_future.result().content.raw
                                joined_result = "".join(map(str, result_list))
                                future = executor.submit(generate_prompt, joined_result)
                            else:
                                future = executor.submit(
                                    generate_prompt,
                                    user_message,
                                )

                            key = f"{current_char}-{roman}"
                            message_data[key] = future.result().content.raw
                            prev_future = future
                    else:
                        # If end_roman_numeral is not provided, skip the inner loop
                        if prev_future:
                            result_list = prev_future.result().content.raw
                            joined_result = "".join(map(str, result_list))
                            future = executor.submit(
                                generate_prompt, "Continue", joined_result
                            )
                        else:
                            future = executor.submit(
                                generate_prompt,
                                user_message,
                            )

                        key = current_char
                        message_data[key] = future.result().content.raw
                        prev_future = future

                saved_messages.append(message_data)
                # Process the message data
                self.process_message_data(conversation_id, message_data, user_message)

                conversation_path = os.path.join(directory, subdirectory, base_dirs[2])
                # Save the conversation to a JSON file
                self.save_conversation(
                    conversation_id, title, folder_path=conversation_path
                )

                # Display the results and ask for feedback
                self.display_results_and_feedback(
                    chat, message_data, display_results, feedback
                )

                self.save_message_to_json(saved_messages, filename_global + ".json")


class PromptGenerator:
    def __init__(
        self,
        model_name: Optional[str],
        api_keys: List[str] = None,
        callback: Optional[StreamingHandler] = None,
        cloudinary: Optional[dict] = None,
        credentials: Optional[dict] = None,
        segment_delimiter: Optional[str] = ".",
        media: Optional[dict] = None,
        name: Optional[str] = None,
        technique: Optional[object] = None,
        upload: Optional[bool] = None,
        path: Optional[str] = None,
        with_responses: Optional[bool] = False,
        storage: Optional[str] = None,
        prompt_path: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        target_tokens: Optional[int] = 16385,
        image_model: Optional[str] = None,
        verbose: Optional[bool] = False,
        cloud: Optional[bool] = False,
        create: Optional[bool] = False,
        play: Optional[bool] = False,
        show: Optional[bool] = False,
        process_media: Optional[bool] = False,
        internal: Optional[bool] = False,
        audio_func: Optional[bool] = False,
        stop: Optional[bool] = False,
        provider: Optional[str] = "openai",
        convert: Optional[bool] = False,
        subdirectory: Optional[str] = None,
    ):
        self.stop = stop
        self.name = name
        self.path = path
        self.media = media
        self.build = create
        self.upload = upload
        self.verbose = verbose
        self.convert = convert
        self.storage = storage
        self.cloudinary = cloudinary
        self.provider = provider
        self.with_responses = with_responses

        if cloud:
            self.prompt_manager = CloudManager(
                credentials=credentials, directory=prompt_path
            )
        else:
            self.prompt_manager = PromptManager(
                credentials=credentials, directory=prompt_path
            )

        self.reply_chain_system = ReplyChainSystem(
            name=name,
            register_synthesis_technique=technique,
            verbose=verbose,
        )
        self.generator = Generator()
        self.callback = (
            callback
            if callback
            else StreamingHandler(
                segment_delimiter=segment_delimiter,
                cloudinary=cloudinary,
            )
        )
        self.chat = AI(
            storage=storage,
            callbacks=[self.callback],
            api_keys=api_keys,
            model_name=model_name,
            media=media,
            prompt_manager=self.prompt_manager,
            max_tokens=max_tokens,
            image_model=image_model,
            create=create,
            play=play,
            process_media=process_media,
            internal=internal,
            show=show,
            audio_func=audio_func,
            provider=provider,
            subdirectory=subdirectory,
            target_tokens=target_tokens,
        )

    def generate_prompt_parts(
        self,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        conversation_id: Optional[str] = None,
        use_basic_truncation: Optional[bool] = False,
        **kwargs,
    ) -> Chain:
        """Generate prompt parts for the conversation."""
        conversation_history = self.reply_chain_system.prepare_conversation_history(
            prompt, response, **kwargs
        )

        truncated_history = self.chat._process_conversation_history(
            conversation_history,
            prompt=prompt,
            use_basic_truncation=use_basic_truncation,
            verbose=self.verbose,
        )

        return self.chat(truncated_history, conversation_id, self.stop)

    def run_speech(
        self,
        input_file,
        output_file_flac,
        language="en",
        use_google=True,
        skip_responses=False,
    ):
        # Convert the WEBM file to FLAC if it ends with .webm
        # check if the audio file is longer then 3 seconds if not skip and return empty string
        audio = AudioSegment.from_file(input_file)
        duration = len(audio) / 1000
        output_file = None
        if skip_responses is not True:
            if duration > 3:
                output_file = convert_webm_to_flac(input_file, output_file_flac)

                if use_google:
                    url = self.prompt_manager.upload_file(output_file, "audio")
                    response_text = self.chat.generate_transcript_google(url, language)
                    os.remove(output_file)
                else:
                    response_text = self.chat.generate_transcript(output_file)
                    os.remove(output_file)
                return response_text, output_file

        else:
            output_file = convert_webm_to_flac(input_file, output_file_flac)

        os.remove(input_file)

        return "", output_file

    def run_generator(
        self,
        directory: str,
        subdirectory: str,
        conversation_path: str,
        streaming: bool = False,
        audio_file: Optional[str] = None,
        record_timeout: int = 5,
        energy_threshold: int = 5000,
        initial_prompt: str = "",
        language: str = "en-US",
        custom_conversation_data: Optional[List[Tuple[str, str]]] = None,
    ):
        conversation_id = self.generator.start_conversation(initial_prompt)
        prev_response_text = None
        saved_messages = []
        title = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        base_dirs = ["recordings", "transcriptions", "conversations"]

        # create subdirectory if it does not exist
        if not os.path.exists(os.path.join(directory, subdirectory)):
            os.makedirs(os.path.join(directory, subdirectory))

        output_directory = os.path.join(directory, subdirectory, base_dirs[0])
        transcption_directory = os.path.join(directory, subdirectory, base_dirs[1])
        conversation_path = os.path.join(directory, subdirectory, base_dirs[2])

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not os.path.exists(transcption_directory):
            os.makedirs(transcption_directory)

        if not os.path.exists(conversation_path):
            os.makedirs(conversation_path)

        while True:
            if streaming:
                response_text = self.generator.run_streaming(
                    audio_file=audio_file,
                    record_timeout=record_timeout,
                    energy_threshold=energy_threshold,
                    language=language,
                )
            else:
                session = PromptSessionWrapper()
                response_text = session.session.prompt(
                    completer=WordCompleter(["quit", "restart"], ignore_case=True)
                )
            print("\n\n")

            if response_text.startswith("/") or response_text.startswith("q u i t"):
                try:
                    response = self.generator.handle_command(
                        response_text.strip().lower(),
                        conversation_id,
                        conversation_path,
                    )

                    if response_text == "quit" or response_text == "q u i t":
                        filename = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
                        self.generator.save_message_to_json(
                            saved_messages, filename + ".json"
                        )
                        print(
                            f"Saved all messages to {filename}.json in a structured format."
                        )
                        print(response)
                        break

                except Exception as e:
                    print(e)
                    break

                print(response)
                print("\n\n")  # New line after the command's response
                continue

            if prev_response_text:
                response = self.generate_prompt_parts(
                    prompt=response_text,
                    response=prev_response_text,
                    use_process_conversations=True,
                    conversation_id=conversation_id,
                )

            else:
                response = self.generate_prompt_parts(
                    prompt=response_text,
                    response=prev_response_text,
                    conversation_id=conversation_id,
                    custom_conversation_data=custom_conversation_data,
                )

            prev_response_text = response.content.raw

            saved_messages.append(response_text)

            base = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = "recording" + "_" + base

            with open(
                transcption_directory + "/" + base_name + ".md",
                "w",
            ) as f:
                f.write(response_text + "\n\n" + prev_response_text + "\n\n")

            # Process the message data
            self.generator.process_message_data(
                conversation_id, response.content.raw, response_text
            )

            conversation_path = os.path.join(directory, subdirectory, base_dirs[2])

            # Save the conversation to a JSON file
            self.generator.save_conversation(
                conversation_id, title, folder_path=conversation_path
            )

    def _create_subdirectories(self, directory, subdirectory, base_dirs):
        # create subdirectory if it does not exist
        if not os.path.exists(os.path.join(directory, subdirectory)):
            os.makedirs(os.path.join(directory, subdirectory))

        output_directory = os.path.join(directory, subdirectory, base_dirs[0])
        transcption_directory = os.path.join(directory, subdirectory, base_dirs[1])

        conversation_path = os.path.join(directory, subdirectory, base_dirs[2])
        youtube_directory = os.path.join(directory, subdirectory, base_dirs[3])
        modalities_directory = os.path.join(directory, subdirectory, base_dirs[4])
        website_directory = os.path.join(directory, subdirectory, base_dirs[5])
        image_directory = os.path.join(directory, subdirectory, base_dirs[6])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not os.path.exists(transcption_directory):
            os.makedirs(transcption_directory)

        if not os.path.exists(conversation_path):
            os.makedirs(conversation_path)

        if not os.path.exists(youtube_directory):
            os.makedirs(youtube_directory)

        if not os.path.exists(modalities_directory):
            os.makedirs(modalities_directory)

        if not os.path.exists(website_directory):
            os.makedirs(website_directory)

        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        return (
            output_directory,
            transcption_directory,
            conversation_path,
            youtube_directory,
            modalities_directory,
            website_directory,
            image_directory,
        )

    def _create_prompt_and_embedding(
        self,
        text: str,
        conversation_id: str,
        prompt: Optional[str] = None,
        upload: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Helper function to create prompt object and generate embedding."""

        embedding = self.chat.generate_embeddings(text)
        self.prompt_manager.create_prompt(
            prompt=prompt,
            prompt_parts=text.split("\n\n"),
            id=conversation_id,
            embedding=embedding,
            upload=upload,
            **kwargs,
        )

    def generate_prompt(
        self,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        **kwargs,
    ) -> str:
        text = self.generator._generic_creation(
            prompt=prompt,
            response=response,
            generation_function=self.generate_prompt_parts,
            creation_function=self._create_prompt_and_embedding,
            upload=self.upload,
            **kwargs,
        )
        return text

    def imagine(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_imagine, self._create_prompt_and_embedding
        )

    def brainstorm(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_brainstorm, self._create_prompt_and_embedding
        )

    def questions(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_questions, self._create_prompt_and_embedding
        )

    def create(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_create, self._create_prompt_and_embedding
        )

    def synergetic(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_synergetic, self._create_prompt_and_embedding
        )

    def category(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_category, self._create_prompt_and_embedding
        )

    def revised(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_revised, self._create_prompt_and_embedding
        )

    def spf(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_spf, self._create_prompt_and_embedding
        )

    def serenity(self, prompt: Optional[str] = None):
        return self.generator._generic_prompt_creation(
            prompt, self.chat.generate_serenity, self._create_prompt_and_embedding
        )

    def run_thread(self, prompt, **kwargs) -> str:
        return self.generator._generic_prompt_creation(
            prompt=prompt,
            generation_function=self.generate_prompt,
            creation_function=self._create_prompt_and_embedding,
            **kwargs,
        )

    def run_chat(self, **kwargs) -> str:
        return self.generator.run_chat(
            generate_prompt=self.generate_prompt_parts,
            chat=self.chat,
            **kwargs,
        )

    def run_streaming(
        self,
        language: str = "en-US",
    ) -> str:
        return self.generator.run_streaming(
            language=language,
        )

    def _create_future_tasks(
        self,
        executor: ThreadPoolExecutor,
        prompt: str,
        answer: str = None,
        parent_id: str = None,
        answer_split: bool = False,
        mode: str = "run",
        **kwargs,  # Accept additional keyword arguments
    ):
        futures = []
        if answer_split:
            responses = answer.split("\n\n")
        else:
            responses = [answer]

        for index, response in enumerate(responses):
            task_parent_id = f"{parent_id}_{index}" if answer_split else parent_id

            if mode == "imagine":
                future = executor.submit(
                    self.imagine, prompt=prompt, **kwargs
                )  # using run_thread
            elif mode == "brainstorm":
                future = executor.submit(
                    self.brainstorm, prompt=prompt, **kwargs
                )  # using run_thread
            elif mode == "questions":
                future = executor.submit(
                    self.questions, prompt=prompt, **kwargs
                )  # using run_thread
            elif mode == "create":
                future = executor.submit(
                    self.create, prompt=prompt, **kwargs
                )  # using run_thread
            elif mode == "synergetic":
                future = executor.submit(
                    self.synergetic, prompt=prompt, **kwargs
                )  # using run_thread
            elif mode == "category":
                future = executor.submit(
                    self.category, prompt=prompt, **kwargs
                )  # using run_thread
            elif mode == "revised":
                future = executor.submit(
                    self.revised, prompt=prompt, **kwargs
                )  # using run_chat
            elif mode == "run":
                future = executor.submit(self.run_thread, prompt=prompt, **kwargs)

            elif mode == "spf":
                future = executor.submit(self.spf, prompt=prompt, **kwargs)

            elif mode == "serenity":
                future = executor.submit(self.serenity, prompt=prompt, **kwargs)

            elif mode == "run_chat":
                future = executor.submit(self.run_chat, **kwargs)

            elif mode == "run_streaming":
                future = executor.submit(self.run_streaming, prompt=prompt, **kwargs)

            else:
                future = executor.submit(
                    self.generate_prompt,
                    prompt,
                    response,
                    **kwargs,
                )
            futures.append(future)
        return futures

    def _get_unique_prompt(self, example_pairs: List, generated_prompts: set):

        parent_id = str(uuid.uuid4())

        prompt, answer = example_pairs.pop()

        while (prompt, answer, parent_id) in generated_prompts:
            prompt, answer, parent_id = example_pairs.pop()
        generated_prompts.add((prompt, answer, parent_id))
        return prompt, answer, parent_id

    def run_interactive(
        self,
        mode: str = "run",
        answer_split: bool = False,
        max_workers: int = 1,
        batch_size: int = 4,
        example_pairs: Optional[List[Tuple[str, str]]] = None,
        with_responses: Optional[bool] = True,
    ) -> None:

        num_prompts = len(example_pairs)

        generated_prompts = set()
        total_batches = (num_prompts + batch_size - 1) // batch_size
        valid_count = 0

        for batch_num in range(total_batches):
            print(f"Generating batch {batch_num + 1} of {total_batches}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for _ in range(min(batch_size, num_prompts - valid_count)):
                    prompt, answer, parent_id = self._get_unique_prompt(
                        example_pairs, generated_prompts
                    )
                    if with_responses:
                        futures.extend(
                            self._create_future_tasks(
                                executor,
                                prompt,
                                answer,
                                parent_id,
                                answer_split,
                                mode,
                            )
                        )
                    else:
                        futures.extend(
                            self._create_future_tasks(
                                executor,
                                prompt,
                                None,
                                parent_id,
                                answer_split,
                                mode,
                            )
                        )

                for future in futures:
                    future.result()

            valid_count += len(futures)

        print(f"Generated {valid_count} prompts.")

    @lru_cache(maxsize=None)
    def cached_generate_prompt_task(
        self,
        prompt: str,
        response: str,
        use_process_conversations: bool = False,
        custom_conversation_data: dict = None,
    ) -> str:
        """Generate a prompt with the cached conversation data."""
        if use_process_conversations:
            return self.generate_prompt(
                prompt, response, custom_conversation_data=custom_conversation_data
            )
        else:
            return self.generate_prompt(prompt, response, custom_conversation_data=None)

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
