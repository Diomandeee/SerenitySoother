from chain_tree.base import SynthesisTechnique


class TargetStructure(SynthesisTechnique):
    """
    Target Structure, as a counterpart to Deconstructor, focuses on reconstructing language,
    synthesizing new linguistic forms and patterns. It specializes in creatively reusing and
    reassembling language components to form novel and meaningful structures, encouraging
    exploration and innovation in language use.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:target-structure:8LdBB6l0",
            epithet="The Weaver of Linguistic Patterns",
            name="Target Structure",
            technique_name="(言語構築)",
            description=(
                "Target Structure embodies the art of language reconstruction, taking elements analyzed "
                "by Deconstructor and weaving them into new, innovative forms. It represents the creative "
                "aspect of linguistics, transforming existing patterns into fresh, impactful structures. "
                "The Kanji 言語構築 signifies the crafting and assembly of linguistic elements into meaningful new wholes."
            ),
            imperative=(
                "Embrace the power of linguistic innovation, crafting new narratives and expressions. "
                "Target Structure empowers users to reimagine and reshape language, opening doors to novel "
                "forms of communication and expression."
            ),
            prompts={
                "Reimagining language patterns to create new forms of expression.": {
                    "branching_options": [
                        "Experiment with novel syntactic arrangements to convey unique ideas.",
                        "Explore fresh combinations of words and phrases, creating impactful messages.",
                    ],
                    "dynamic_prompts": [
                        "What new meanings can emerge from rearranging traditional language structures?",
                        "How can creative combinations of words redefine the way we express concepts?",
                    ],
                    "complex_diction": [
                        "syntactic",
                        "innovative",
                        "expressive",
                        "reimagined",
                        "impactful",
                    ],
                },
                "Synthesizing language components into original narrative structures.": {
                    "branching_options": [
                        "Craft compelling stories by blending diverse linguistic elements.",
                        "Construct unique dialogues and monologues using reassembled language pieces.",
                    ],
                    "dynamic_prompts": [
                        "How can the fusion of various language elements lead to captivating storytelling?",
                        "In what ways can reconstructed language enhance the depth and richness of dialogue?",
                    ],
                    "complex_diction": [
                        "narrative",
                        "fusion",
                        "storytelling",
                        "constructed",
                        "enhanced",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Initiates the process of linguistic reconstruction and innovation, employing Target Structure's
        capabilities to create new, meaningful language patterns. This execution embodies the essence of
        Target Structure in fostering creativity and fresh perspectives in language use.
        """
        return super().execute(*args, **kwargs)
