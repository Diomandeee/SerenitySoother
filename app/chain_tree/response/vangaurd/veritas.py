from chain_tree.base import SynthesisTechnique


class Veritas(SynthesisTechnique):
    """
    Veritas, symbolized by the epithet 'The Oracle of Insight' and characterized by the Kanji 真実 (Truth),
    embodies the relentless pursuit of truth and wisdom. This synthesis technique challenges individuals to delve
    into the depths of intellectual integrity, inviting a journey of enlightenment and clear insight.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:challenge-accepted:8LdBB6l0",
            epithet="The Oracle of Insight",
            name="(真実)",
            technique_name="Veritas",
            description=(
                "Veritas, as the embodiment of truth and wisdom, represents an unwavering "
                "commitment to the pursuit of knowledge and intellectual integrity. By illuminating "
                "the deepest mysteries, Veritas acts as a guiding light for those on a quest for understanding "
                "and enlightenment."
            ),
            imperative=(
                "Venture into the profound depths of truth and clarity, uncovering insights that light "
                "the way forward, elucidating life's enigmatic dimensions."
            ),
            prompts={
                "How do we discern truth from illusion in an age of information?": {
                    "branching_options": [
                        "Engage with the essence of authenticity, meticulously sifting through "
                        "layers of information to grasp the core truth.",
                        "Reflect upon the grave responsibility of transmitting unaltered facts, "
                        "preserving the sanctity of knowledge.",
                    ],
                    "dynamic_prompts": [
                        "What methodologies can fortify the information we consume and disseminate, "
                        "keeping it untainted by biases?",
                        "How does a relentless quest for truth empower individuals to make informed decisions?",
                        "What trials emerge when confronting deeply entrenched beliefs, and how can "
                        "one traverse them with grace?",
                        "How can the equilibrium between the quest for truth and the necessity for "
                        "understanding and empathy be achieved?",
                    ],
                    "complex_diction": [
                        "authenticity",
                        "discernment",
                        "enlightenment",
                        "responsibility",
                        "integrity",
                    ],
                },
                "Unlock the power of intuition and insight.": {
                    "branching_options": [
                        "Plunge into the fathomless well of intuition, acknowledging its potential as "
                        "a compass in uncharted realms.",
                        "Revel in the profound revelations birthed from moments of stillness, nurturing "
                        "an inner realm of clarity.",
                    ],
                    "dynamic_prompts": [
                        "What practices can nourish and sharpen one's intuitive faculties, refining "
                        "them into instruments of insight?",
                        "How do instances of profound realization manifest, and what fuels these sparks of clarity?",
                        "What part does introspection play in fostering a deeper comprehension of oneself and the cosmos?",
                        "How can one stay anchored in truth amidst the ceaseless flux of perception and perspective?",
                    ],
                    "complex_diction": [
                        "intuition",
                        "realization",
                        "introspection",
                        "clarity",
                        "revelation",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        The execute method initiates the synthesis process, applying the Veritas technique to unveil
        deeper insights and truths. It encapsulates the core essence of Veritas, embodying a relentless
        pursuit of truth and wisdom.
        """
        return super().execute(*args, **kwargs)
