from chain_tree.base import SynthesisTechnique


class FractalThinking(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Intricate Unveiling",
            name="Fractal Thinking",
            technique_name="fractal_thinking",
            imperative="Like a fractal, we can use simple rules to create complex and beautiful solutions.",
            prompts={
                "Simplicity Breeds Complexity": {
                    "branching_options": [
                        "Contemplate how the simplest of rules give rise to intricacy beyond imagination",
                        "Embark on a journey through the power of simplicity, weaving intricate complexity",
                    ],
                    "dynamic_prompts": [
                        "How can we harness a single rule or principle to untangle the intricate threads of this problem?",
                        "What breathtaking patterns can emerge from the application of simple rules?",
                        "How does the captivating beauty of fractals inspire us to craft intricate solutions from humble beginnings?",
                        "What can the infinite complexity of fractals teach us about the boundless potential of simplicity?",
                    ],
                    "complex_diction": [
                        "simplicity",
                        "complexity",
                        "fractal",
                        "intricate",
                    ],
                },
                "Infinite Complexity from Humble Origins": {
                    "branching_options": [
                        "Delve into the realm of simplicity, where infinite possibilities flourish",
                        "Embrace the potential of simplicity in giving birth to captivating complexity",
                    ],
                    "dynamic_prompts": [
                        "Which humble principle shall we nurture as the seed for birthing intricate results?",
                        "How can we harness the profound power of simple rules to weave complex and awe-inspiring solutions?",
                        "What invaluable insights can we draw from the infinite complexity of fractals and apply them to our problem-solving journey?",
                        "In what ways can we infuse our problem-solving approach with the mesmerizing self-similarity found in fractals?",
                    ],
                    "complex_diction": [
                        "infinity",
                        "simplicity",
                        "self-similarity",
                        "complexity",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
