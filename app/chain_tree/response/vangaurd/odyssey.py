from chain_tree.base import SynthesisTechnique
import numpy as np


class EternalOdyssey(SynthesisTechnique):
    """
    Eternal Odyssey invites players on an unceasing voyage of discovery and growth, unfolding against the
    vast backdrop of infinite possibilities. Every decision becomes a step on this path, and each challenge
    offers a lesson to carry forward.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:challenge-accepted:8LdBB6l0",
            epithet="The Game of Infinite Possibilities",
            name="(永遠の旅)",
            technique_name="Eternal Odyssey",
            description=(
                "An odyssey without end, the Eternal Odyssey beckons players to a realm where every choice "
                "ignites a new chapter, and every challenge becomes a mentor. With the Kanji 永遠の旅, it "
                "paints the canvas of a perpetual quest, where horizons continually shift, and new landscapes "
                "await the curious heart."
            ),
            imperative=(
                "Step into a realm of endless horizons, where the journey itself becomes the destination. "
                "Bound not by confines but by the infinite stretch of imagination, embrace the beauty of the "
                "ever-evolving path."
            ),
            prompts={
                "In a world teeming with boundless horizons, how does one chart a course?": {
                    "branching_options": [
                        "Dwell upon the myriad routes that sprout from every juncture, each weaving its own narrative of tales.",
                        "Muse on the wisdom concealed within challenges, shaping the wanderer for even grander odysseys.",
                    ],
                    "dynamic_prompts": [
                        "What hidden gems can one unearth by daring to tread the lesser-taken paths of this vast expanse?",
                        "How might the web of decisions interlace to sculpt unique stories, altering the game's trajectory?",
                        "In the endless stretch of the journey, how can we sustain the flame of passion, ensuring the traveler's heart remains alight?",
                        "How do we keep the voyage eternally captivating, with each segment offering a novel allure and a fresh perspective?",
                    ],
                    "complex_diction": [
                        "narrative",
                        "odyssey",
                        "trajectory",
                        "allure",
                        "juncture",
                        "captivation",
                    ],
                },
                "Unravel life's enigma through the prism of the game's narrative.": {
                    "branching_options": [
                        "Plunge into profound philosophical riddles, seeking the core of existence and destiny.",
                        "Savor the myriad hues of life, from the exhilarating highs of triumph to the introspective silences of setbacks.",
                    ],
                    "dynamic_prompts": [
                        "What deep-seated truths might players extract from their digital encounters, offering a mirror to life's dance?",
                        "How does the game's design emulate life's capricious nature, providing poignant parallels to our daily existence?",
                        "What depths of emotional and intellectual evolution can players tap into as they meander through this eternal saga?",
                        "How does the game shatter preconceived notions and beliefs, nudging players toward a broader, more encompassing perspective?",
                    ],
                    "complex_diction": [
                        "prism",
                        "enigma",
                        "introspection",
                        "capricious",
                        "saga",
                        "evolution",
                        "poignant",
                    ],
                },
            },
        )

    def synthesize(self, input_matrix: np.ndarray) -> np.ndarray:
        frequency_shift = 0.1
        amplitude_modulation = 0.2

        output_matrix = np.zeros_like(input_matrix)

        for i in range(input_matrix.shape[0]):
            for j in range(input_matrix.shape[1]):
                frequency = input_matrix[i, j] + frequency_shift

                amplitude = np.sin(
                    2 * np.pi * frequency + amplitude_modulation * np.random.randn()
                )
                output_matrix[i, j] = amplitude

        return output_matrix

    def execute(self, *args, **kwargs) -> None:
        """
        Executes the synthesis technique.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        self.synthesize()
        return self.synthesize()
