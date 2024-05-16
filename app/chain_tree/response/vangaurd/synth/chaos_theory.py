from chain_tree.base import SynthesisTechnique


class ChaosTheory(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Unpredictable Catalyst",
            name="Chaos Theory",
            technique_name="chaos_theory",
            imperative="Like a river branching into multiple streams, we can expand our thoughts and perspectives to find new solutions and opportunities.",
            prompts={
                "Unveiling the Enigmatic Beauty: Embrace the Dance of Disorder": {
                    "branching_options": [
                        "Immerse yourself in the enthralling chaos and uncover its hidden wonders",
                        "Embrace the intricate tapestry woven by unpredictability and marvel at its potential for discovery",
                    ],
                    "dynamic_prompts": [
                        "What mesmerizing patterns and hidden order lie within the seemingly random and chaotic?",
                        "How can we harness the swirling energy of chaos to ignite profound transformation?",
                        "What breathtaking perspectives and luminous insights can emerge from the depths of unpredictability?",
                        "Venture into the uncharted realms of the unknown, where untapped opportunities eagerly await your exploration.",
                    ],
                    "complex_diction": [
                        "disorder",
                        "unpredictability",
                        "randomness",
                        "entropy",
                    ],
                },
                "Embracing the Veil of Uncertainty: Unlocking the Euphoria of the Unknown": {
                    "branching_options": [
                        "Immerse yourself in the rhythmic dance of chaos and seize its tantalizing opportunities",
                        "Embrace the kaleidoscope of unpredictability and unlock the boundless realm of new possibilities",
                    ],
                    "dynamic_prompts": [
                        "What novel and extraordinary ideas and concepts can emerge from the swirling mists of disorder?",
                        "How can we harness the raw power of chaos to drive awe-inspiring transformation and change?",
                        "Embark on a thrilling journey of discovery, where hidden treasures and breathtaking breakthroughs await in uncharted territories.",
                        "Unleash the dormant potential that lies within the enchanting realm of unpredictability, waiting to be awakened.",
                    ],
                    "complex_diction": [
                        "uncertainty",
                        "spontaneity",
                        "discovery",
                        "potential",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
