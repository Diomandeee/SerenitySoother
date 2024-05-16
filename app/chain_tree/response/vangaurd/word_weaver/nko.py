import random
from chain_tree.base import SynthesisTechnique


class NkoFusionist(SynthesisTechnique):
    """
        NkoFusionist is a powerful linguistic synthesis tool that specializes in creating captivating Nko narratives. It combines various language elements, including unique vocabulary, syntax,
    intonations, and cultural nuances to produce original and engaging storylines.
        This approach highlights the richness and complexity of the Nko language, while fostering creativity and innovation.

    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:nkofusionist:LdBjJ7o1",
            epithet="The Grand Master of Nko Storytelling",
            name="NkoFusionist",
            technique_name="Nko",
            description=(
                "NkoFusionist is an advanced linguistic synthesis tool designed to blend various language elements in Nko,"
                "resulting in captivating storytelling. The name derives from the Nkò term ẹbọ̀yẹ́rì, which means "
                "a weaver or artisan. With a focus on authenticity and creativity, NkoFusionist invites you to explore"
                "the depths of this intriguing language."
            ),
            imperative=(
                "Use the power of NkoFusionist to combine various elements of the Nko language and create unique,"
                "engaging, and captivating stories. Embrace the richness and complexity of the Nko language as"
                "you explore its endless potential for creativity and expression."
            ),
            prompts={
                "Blending vocabulary and syntax for captivating storytelling.": {
                    "branching_options": [
                        "Create vivid descriptions using a combination of Nko vocabulary and phrases.",
                        "Experiment with sentence structures to enhance the flow and rhythm of your narratives.",
                        "Incorporate idiomatic expressions to add depth and nuance to your stories.",
                    ],
                    "dynamic_prompts": [
                        "How can you use unique Nko words or idioms to enrich your storytelling?",
                        "What syntactic patterns can you employ to create a compelling narrative flow?",
                        "What creative ways can you combine different vocabulary sets within the same context?",
                    ],
                    "complex_diction": [
                        "vocabulary",
                        "idioms",
                        "syntax",
                        "narrative",
                        "flow",
                    ],
                },
                "Incorporating tone, rhythm, and intonation for emotive storytelling.": {
                    "branching_options": [
                        "Employ various Nko intonations to convey different emotions and moods.",
                        "Use rhythmic structures to emphasize important points in your narratives.",
                        "Incorporate cultural references and historical context to add depth and authenticity.",
                    ],
                    "dynamic_prompts": [
                        "What role does tone play in conveying emotion in Nko storytelling?",
                        "How can you use rhythm and intonation to create a sense of tension or suspense?",
                        "What historical or cultural contexts can you incorporate into your narratives to make them more engaging?",
                    ],
                    "complex_diction": [
                        "intonations",
                        "emotion",
                        "moods",
                        "rhythm",
                        "tension",
                        "context",
                    ],
                },
                "Exploring cultural references and historical context for depth and authenticity.": {
                    "branching_options": [
                        "Incorporate historical events, myths, or legends into your narratives.",
                        "Add cultural nuances to enhance the depth of your characters and settings.",
                        "Use Nko proverbs or idioms that reflect the values and traditions of the Nko people.",
                    ],
                    "dynamic_prompts": [
                        "What historical events, myths, or legends can you use to add depth and authenticity to your narratives?",
                        "How can you add cultural nuances that reflect the values and traditions of the Nko people?",
                        "What Nko proverbs or idioms can you use to make your stories more relatable and meaningful?",
                    ],
                    "complex_diction": [
                        "history",
                        "myths",
                        "legends",
                        "cultural_nuances",
                        "proverbs",
                    ],
                },
            },
            special_features=[
                "Randomly generate Nko idioms or proverbs to use in your narratives.",
                "Suggest creative ways to combine different vocabulary sets and grammatical structures within the same context.",
            ],
        )

    def suggest_idiom(self):
        idiom_list = [
            "A kpé wɔn wọ́ tɛ wɔtɔ.",  # One hand claps.
            "E gbe wó nkansa, e mu a yɔ wɔfɔ.",  # The person who has no salt will not miss it when it is thrown away.
            "A kpá lɛ mí wɔn bɔlɔ wɔn kɔkɔ.",  # When two elephants fight, the grass suffers.
            "Wɔkɔ a nkpa bɛ wɔ, ɔdɔ a pá wɔ.",  # The palm nut that falls on the ground is for the ant.
            "A tɛ ntɔn bɛ kpá ɔdɔ, owọn jɛ tɛ ntɔn bɛ ewɔ ntɔn.",  # The elephant's footprints are big, but those of the ants are countless.
            "A tɛ sɔ dza a wɔ sɔ.",  # A man who is afraid of a lion will not approach a mouse.
            "Mí wɔn bɔlɔ wɔn wɔ nkansa.",  # The yams in the basket are fighting each other.
            "Ntɔ́wɔn nkyɛ kpá bɔlɔ, ntɔ́wɔn mí wɔn bɔlɔ.",  # The antelopes in the bush are strong, but those in the yard are weak.
            "Yɛ mɛ wɔn yɔ wɔfɔ, ɔdɔ wɔn hɛ kpá ntɔ́wɔ.",  # If you give a monkey a peanut, it will throw the shell away.
        ]
        return random.choice(idiom_list)

    def suggest_vocabulary_combination(self):
        verb_list = ["kpá", "nkansa", "wɔtɔ", "pá"]
        noun_list = ["mí", "ntɔ́wɔn", "kwɛsɛ", "jɔ", "ɔdɔ", "kpá", "ewɔ"]
        adjective_list = ["nkyɛ", "hɛ", "bɔlɔ"]
        article_list = ["a", "ɛ"]
        verb = random.choice(verb_list)
        noun1, noun2 = random.sample(noun_list, 2)
        adjective = random.choice(adjective_list)
        article1, article2 = random.sample(article_list, 2)
        return f"{article1} {verb} {article2} {noun1} {adjective} {noun2}"

    def execute(self, *args, **kwargs) -> None:
        """

        Initiates the process of Nko language synthesis and innovation, employing NkoFusionist's capabilities to
        create new, meaningful language patterns. This execution embodies the essence of NkoFusionist in fostering
        creativity and fresh perspectives in Nko storytelling.

        """
        return super().execute(*args, **kwargs)
