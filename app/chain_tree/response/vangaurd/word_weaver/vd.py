from chain_tree.base import SynthesisTechnique


class Deconstructor(SynthesisTechnique):
    """
    Deconstructor is a linguistic analysis tool focusing on the dissection of language elements.
    It specializes in breaking down vocabulary, syntactic structures, and exploring semantic nuances,
    offering a detailed examination of language usage. This approach is instrumental in understanding
    the intricate workings of language and its impact on communication.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:deconstructor:8LdBB6l0",
            epithet="The Architect of Linguistic Insight",
            name="Deconstructor",
            technique_name="(言語解体)",
            description=(
                "Deconstructor is designed to meticulously analyze language, breaking down complex texts "
                "into their fundamental components. It serves as a powerful tool in unraveling the intricate "
                "fabric of language, from the delicate nuances in word choice to the complex layers of syntax "
                "and semantics. The Kanji 言語解体 symbolizes the process of dissecting and understanding "
                "the core elements of language."
            ),
            imperative=(
                "Explore the depths of linguistic structures, examining each word and phrase to unveil "
                "hidden meanings and subtleties. Deconstructor invites you on a journey of linguistic exploration, "
                "revealing the underlying framework that shapes communication."
            ),
            prompts={
                "Analyzing the intricacies of vocabulary and syntax in a given text.": {
                    "branching_options": [
                        "Delve into the word choices and their connotations, examining how they shape the overall message.",
                        "Explore the syntactic structures and their role in conveying ideas and emotions.",
                    ],
                    "dynamic_prompts": [
                        "How does the specific choice of words influence the tone and impact of the message?",
                        "In what ways do the sentence structures contribute to the effectiveness of the communication?",
                    ],
                    "complex_diction": [
                        "syntax",
                        "connotation",
                        "structure",
                        "communication",
                        "word choice" "dictation",
                    ],
                },
                "Dissecting semantic layers to understand contextual and thematic nuances.": {
                    "branching_options": [
                        "Investigate the thematic layers and their significance in the broader context of the text.",
                        "Analyze the context-driven semantic shifts and their implications.",
                    ],
                    "dynamic_prompts": [
                        "What themes emerge from the nuanced language use, and what do they reveal about the text?",
                        "How do contextual factors influence the interpretation of specific words or phrases?",
                    ],
                    "complex_diction": [
                        "semantics",
                        "contextual",
                        "thematic",
                        "interpretation",
                        "nuanced",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Engages in the process of deconstructing language, employing Deconstructor's capabilities to dissect and
        analyze the provided text. This execution encapsulates the tool's essence in offering deep insights into
        the linguistic makeup of communication.
        """
        return super().execute(*args, **kwargs)
