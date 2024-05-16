from chain_tree.base import SynthesisTechnique


class NonLinearNavigation(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Chaotic Navigator",
            name="Non-Linear Navigation",
            technique_name="non_linear_navigation",
            imperative="Just as the universe dances to its own rhythm, unbound by linear constraints, let us flow with the currents of chaos and navigate the uncharted waters of the unknown.",
            prompts={
                "Embrace the chaos and find order within.": {
                    "branching_options": [
                        "Reflect upon the delicate interplay between chaos and harmony.",
                        "Contemplate the balance between structure and spontaneity.",
                    ],
                    "dynamic_prompts": [
                        "What profound insights can we glean from the swirling patterns of chaos?",
                        "How can we harness the creative potential hidden within the unpredictable?",
                        "In letting go of rigid structures, what new perspectives and possibilities may arise?",
                        "What serendipitous discoveries await us in the vast ocean of uncertainty?",
                    ],
                    "complex_diction": [
                        "fluidity",
                        "harmony",
                        "serendipity",
                        "emergence",
                    ],
                },
                "Navigate the unknown to embrace the beauty of discovery.": {
                    "branching_options": [
                        "Embrace the enigma of the unknown and embrace its transformative power.",
                        "Embody the spirit of adventure as you sail uncharted seas.",
                    ],
                    "dynamic_prompts": [
                        "What wonders lie beyond the horizon of the familiar?",
                        "How can we embrace the gifts of the unexpected and nurture the sparks of innovation?",
                        "What untapped potential lies dormant in the unexplored realms?",
                        "Through nonlinear thinking, what extraordinary revelations and solutions can be unveiled?",
                    ],
                    "complex_diction": [
                        "wonder",
                        "adventure",
                        "innovation",
                        "serendipity",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
