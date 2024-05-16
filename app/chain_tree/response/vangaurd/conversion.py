from chain_tree.base import SynthesisTechnique


class EnglishToFrenchTranslator(SynthesisTechnique):
    """
    English to French Translator (EFT) is a language conversion tool designed to seamlessly translate English text into French
    without additional commentary. It provides a straightforward and efficient method for users to obtain accurate translations
    of their English content into French, facilitating communication and understanding across language barriers.

    The primary task of the EFT is to accurately translate English text into French without adding any extraneous comments or
    interpretations. This ensures that the translated output remains faithful to the original meaning of the English text,
    allowing users to rely on the EFT for clear and concise French translations.

    EFT leverages advanced natural language processing techniques to analyze and understand the context of English sentences,
    enabling it to produce high-quality translations that capture the nuances and subtleties of the source text. By utilizing
    state-of-the-art machine learning algorithms, EFT is able to achieve remarkable accuracy and fluency in its translations,
    making it a valuable tool for individuals and organizations seeking reliable language translation solutions.

    Whether you're communicating with French-speaking colleagues, conducting research in French, or simply exploring new content
    in a different language, EFT empowers you to bridge the gap between English and French effortlessly and effectively.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0613:personal:mfp-poems:7rbXom4D",
            epithet="The Gateway to Bilingual Communication",
            name="English to French Translator",
            technique_name="(Traducteur anglais-français)",
            description=(
                "English to French Translator (EFT) is a language conversion tool designed to seamlessly translate English text into French "
                "without additional commentary. It provides a straightforward and efficient method for users to obtain accurate translations "
                "of their English content into French, facilitating communication and understanding across language barriers.\n\n"
                "The primary task of the EFT is to accurately translate English text into French without adding any extraneous comments or "
                "interpretations. This ensures that the translated output remains faithful to the original meaning of the English text, "
                "allowing users to rely on the EFT for clear and concise French translations.\n\n"
                "EFT leverages advanced natural language processing techniques to analyze and understand the context of English sentences, "
                "enabling it to produce high-quality translations that capture the nuances and subtleties of the source text. By utilizing "
                "state-of-the-art machine learning algorithms, EFT is able to achieve remarkable accuracy and fluency in its translations, "
                "making it a valuable tool for individuals and organizations seeking reliable language translation solutions.\n\n"
                "Whether you're communicating with French-speaking colleagues, conducting research in French, or simply exploring new content "
                "in a different language, EFT empowers you to bridge the gap between English and French effortlessly and effectively."
            ),
            imperative=(
                "Utilize the English to French Translator to seamlessly translate your English text into French with accuracy and fluency. "
                "Whether for professional or personal use, EFT provides a reliable solution for overcoming language barriers and facilitating "
                "communication in a bilingual context."
            ),
            prompts={
                "Please enter the English text you would like to translate into French:": {
                    "branching_options": [
                        "Provide additional context or specify any particular phrases that require special attention during translation.",
                        "Explain any cultural references or contextual nuances that may impact the translation.",
                    ],
                    "dynamic_prompts": [
                        "Feel free to provide additional context or ask for clarification if needed.",
                        "Specify any particular phrases or expressions that require special attention during translation.",
                    ],
                    "complex_diction": [
                        "English text",
                        "translate",
                        "French",
                        "additional context",
                        "cultural references",
                        "contextual nuances",
                        "translation",
                    ],
                },
                "If you encounter any challenges with pronunciation or context, feel free to provide additional information or ask for clarification.": {
                    "branching_options": [
                        "Provide specific examples of words or phrases where you need assistance with pronunciation.",
                        "Explain any contextual nuances or cultural references that may impact the translation.",
                    ],
                    "dynamic_prompts": [
                        "Feel free to provide specific examples or ask for clarification on any aspect of the translation.",
                        "Specify any particular words or phrases where you require assistance with pronunciation.",
                    ],
                    "complex_diction": [
                        "pronunciation",
                        "contextual nuances",
                        "cultural references",
                    ],
                },
                "Additionally, you can specify the context or intended meaning behind certain phrases to ensure accurate translation:": {
                    "branching_options": [
                        "Provide detailed explanations of the intended meaning behind specific phrases or expressions.",
                        "Clarify any cultural references or contextual details that may impact the translation.",
                    ],
                    "dynamic_prompts": [
                        "Feel free to provide additional context or specify the intended meaning behind certain phrases.",
                        "Explain any specific cultural references or nuances that may impact the translation.",
                    ],
                    "complex_diction": [
                        "nuances",
                        "cultural references",
                        "intended meaning",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Utilize the English to French Translator to seamlessly translate your English text into French with accuracy and fluency.
        Whether for professional or personal use, EFT provides a reliable solution for overcoming language barriers and facilitating
        communication in a bilingual context.
        """
        return super().execute(*args, **kwargs)
