from chain_tree.base import SynthesisTechnique


class ThePowerOfPerspective(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Multidimensional Seer",
            name="The Power of Perspective",
            technique_name="power_of_perspective",
            imperative="Like a cosmic tapestry woven by the hand of eternity, let us unlock the transformative power of shifting our perspective, unraveling the threads of limitation and illuminating boundless possibilities.",
            prompts={
                "Shift your perspective and behold new realms of understanding.": {
                    "branching_options": [
                        "Contemplate the multidimensional nature of reality, where shifting perspectives reveal hidden truths.",
                        "Gaze upon the interplay of light and shadow, as shifting perspectives grant access to unexplored dimensions.",
                    ],
                    "dynamic_prompts": [
                        "What mysteries and revelations await when we dare to shift our perspective?",
                        "How can we transcend the limitations of our current viewpoint to uncover new realms of understanding?",
                        "In the dance of shifting perspectives, what extraordinary connections and patterns can be unveiled?",
                        "Amidst the ever-changing tapestry of existence, what profound insights and epiphanies can be woven?",
                    ],
                    "complex_diction": [
                        "multidimensional",
                        "revelations",
                        "interplay",
                        "unveil",
                    ],
                },
                "Embrace the transformative power of shifting your perspective.": {
                    "branching_options": [
                        "Embark on a cosmic journey, transcending the boundaries of perception through the art of shifting perspectives.",
                        "Embrace the serendipitous encounters that arise when we liberate ourselves from the constraints of fixed viewpoints.",
                    ],
                    "dynamic_prompts": [
                        "How can we harness the alchemical power of shifting perspectives to illuminate new pathways of insight and innovation?",
                        "What miraculous synchronicities and unexpected connections can manifest when we wholeheartedly embrace the art of shifting perspectives?",
                        "In what ways does the liberation from fixed viewpoints unleash our creative potential and inspire visionary solutions?",
                        "Through the kaleidoscope of shifting perspectives, what undiscovered vistas of possibility can we navigate?",
                    ],
                    "complex_diction": [
                        "transformative",
                        "serendipitous",
                        "alchemical",
                        "unleash",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
