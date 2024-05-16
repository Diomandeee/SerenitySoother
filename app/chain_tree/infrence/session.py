from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.html import HtmlLexer
from prompt_toolkit import PromptSession
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.styles import Style
import os


class CustomValidator(Validator):
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(
                message="Input cannot be empty", cursor_position=len(document.text)
            )


class PromptSessionWrapper:
    def __init__(
        self,
        completer: Completer = None,
        history_path=".config/history/history.txt",
        custom_style=None,
        prompt_message=">>> ",
        bottom_toolbar_message="Type /help for a list of commands",
    ):
        self.completer = completer or WordCompleter(
            ["quit", "restart", "help", "history"], ignore_case=True
        )
        self.custom_style = custom_style or Style.from_dict(
            {
                "prompt": "#00aa00",
                "input": "#ff0066",
                "message": "#00aaff",
            }
        )
        self.prompt_message = prompt_message
        self.bottom_toolbar_message = bottom_toolbar_message
        self.history_path = history_path
        self.setup_session()

        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        open(self.history_path, "a").close()

    def setup_session(self):
        self.session = PromptSession(
            history=FileHistory(self.history_path),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            lexer=PygmentsLexer(HtmlLexer),
            validator=CustomValidator(),
            style=self.custom_style,
        )

    def get_input(self) -> str:
        try:
            return self.session.prompt(
                [
                    ("class:prompt", self.prompt_message)
                ],  # Apply 'prompt' style to prompt_message
                bottom_toolbar=[("class:message", self.bottom_toolbar_message)],
            )
        except ValidationError as e:
            print(str(e))
            return self.get_input()

    def set_completer(self, completer: Completer):
        self.completer = completer
        self.setup_session()

    def set_prompt_message(self, prompt_message: str):
        self.prompt_message = prompt_message

    def set_bottom_toolbar_message(self, bottom_toolbar_message: str):
        self.bottom_toolbar_message = bottom_toolbar_message

    def set_history_path(self, path: str):
        self.history_path = path
        self.setup_session()

    def add_to_history(self, message: str):
        with open(self.history_path, "a") as f:
            f.write(message + "\n")

    def clear_history(self):
        open(self.history_path, "w").close()

    def save_history_to_file(self, file_path: str):
        """
        Save the current history to a file.
        """
        self.session.history.save(file_path)

    def load_history_from_file(self, file_path: str):
        """
        Load history from a file.
        """
        self.session.history.load(file_path)

    def get_history_as_list(self):
        """
        Return the entire history as a list.
        """
        return list(self.session.history.get_strings())

    def set_initial_input(self, initial_input: str):
        """
        Set initial input for the prompt.
        """
        self.session.default_buffer.text = initial_input

    def clear_initial_input(self):
        """
        Clear initial input.
        """
        self.session.default_buffer.reset()
