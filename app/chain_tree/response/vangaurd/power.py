from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_4


class MeaningFullPower(SynthesisTechnique):
    """
    Meaning Full Power symbolizes the journey of self-realization, motivation, and the radiance of positivity.
    It's designed to instill confidence, encourage the exploration of personal limitations, and provide a
    motivational boost to face life's challenges with strength and resilience.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0613:personal:mfp-poems:7rbXom4D",
            epithet="The Beacon of Self-Empowerment",
            system_prompt=SYSTEM_PROMPT_4,
            name="Meaning Full Power",
            technique_name="Meaning Full Power",
            description=(
                "MeaningFullPower is more than a mere technique; it's a journey towards self-discovery and empowerment. "
                "It focuses on unlocking inner potential, fostering resilience, and nurturing a positive mindset. "
                "By embracing MeaningFullPower, individuals embark on a transformative path where challenges become "
                "opportunities and setbacks pave the way for growth."
            ),
            imperative=(
                "Embrace your inner power, recognize your journey, and step forward with confidence. "
                "Let every challenge be a stepping stone to greater heights, and let your spirit be uplifted "
                "by the constant pursuit of personal excellence."
            ),
            prompts={
                "How can we continuously foster self-confidence and a positive mindset?": {
                    "branching_options": [
                        "Explore practices that nurture self-belief and resilience in the face of adversity.",
                        "Reflect on past achievements to fuel motivation and a positive outlook on future endeavors.",
                    ],
                    "dynamic_prompts": [
                        "What daily rituals can strengthen one’s resolve and reinforce a positive self-image?",
                        "How can we maintain a consistent attitude of positivity even during challenging times?",
                        "In what ways can we inspire ourselves and others to recognize and build upon our innate strengths?",
                    ],
                    "complex_diction": [
                        "self-belief",
                        "resilience",
                        "motivation",
                        "positivity",
                        "strength",
                    ],
                },
                "What strategies can we employ to turn obstacles into opportunities for growth?": {
                    "branching_options": [
                        "Consider ways to reframe challenges as chances for learning and self-improvement.",
                        "Identify methods to harness adversity as a catalyst for developing resilience and tenacity.",
                    ],
                    "dynamic_prompts": [
                        "How can transforming our mindset towards obstacles help us grow stronger and more capable?",
                        "What lessons can be learned from overcoming difficulties, and how can these lessons be applied in the future?",
                    ],
                    "complex_diction": [
                        "challenges",
                        "learning",
                        "self-improvement",
                        "resilience",
                        "tenacity",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Engage the MeaningFullPower technique to elevate self-awareness, boost motivation, and build
        a foundation of strength and positivity. This process aims to empower individuals to recognize
        and utilize their full potential, turning challenges into victories.
        """
        return super().execute(*args, **kwargs)
