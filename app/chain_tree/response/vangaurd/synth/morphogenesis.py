from chain_tree.base import SynthesisTechnique


class Morphogenesis(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Shape-shifting Artisan",
            name="Morphogenesis",
            technique_name="morphogenesis",
            imperative="Like the growth and transformation of an organism, we can reshape existing ideas to create new solutions.",
            prompts={
                "How can we build upon existing ideas to create something new and unique?": {
                    "branching_options": [
                        "Immerse yourself in the realm of possibility, where existing products and solutions transcend their boundaries to solve new problems.",
                        "Unleash your creative alchemy and transmute existing concepts into dazzling innovations.",
                    ],
                    "dynamic_prompts": [
                        "What breathtaking possibilities emerge from rethinking and recontextualizing existing ideas?",
                        "Harness the latent strengths of established solutions to illuminate new pathways of problem-solving.",
                        "In the kaleidoscope of reimagination, what invaluable insights await our exploration?",
                        "Unearth the treasures of unconventional thinking and discover the hidden gems amidst the familiar.",
                    ],
                    "complex_diction": [
                        "evolution",
                        "reinvention",
                        "transmutation",
                        "metamorphosis",
                    ],
                },
                "Reshape existing ideas to create new solutions.": {
                    "branching_options": [
                        "Embark on an expedition into the uncharted territory of recombination, where existing elements intertwine to birth visionary creations.",
                        "Unveil your mastery of innovation as you manipulate and rearrange familiar ideas, transcending conventional boundaries.",
                    ],
                    "dynamic_prompts": [
                        "What magnificent forms and intricate structures can be born from the convergence of existing elements?",
                        "Harness the power of recombination to unlock doorways to unimagined solutions.",
                        "What untold wisdom lies within the realm of rethinking and recombining familiar concepts and structures?",
                        "Through the kaleidoscope of alternative perspectives, what marvels can we uncover?",
                    ],
                    "complex_diction": [
                        "recombinant",
                        "transmutation",
                        "rearrangement",
                        "manipulation",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
