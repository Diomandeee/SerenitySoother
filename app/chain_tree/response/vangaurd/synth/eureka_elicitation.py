from chain_tree.base import SynthesisTechnique


class EurekaElicitation(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Audacious Visionary",
            name="Eureka Elicitation",
            technique_name="eureka_elicitation",
            imperative="Like a celestial lightning bolt illuminating the tapestry of the sky, let us embark on a quest for inspiration in the most unlikely realms.",
            prompts={
                "What if...": {
                    "branching_options": [
                        "you ventured down an uncharted path, defying conventions, unraveling the mysteries of this problem?",
                        "you gazed upon this challenge through a kaleidoscope of new perspectives, revealing hidden dimensions waiting to be explored?",
                        "you embraced a seemingly impossible solution with audacity and resolve, daring to manifest the extraordinary?",
                    ],
                    "dynamic_prompts": [
                        "What dormant, unconventional ideas yearn to be unleashed upon the world?",
                        "Ignite the flame of boundless creativity within you and weave the most extraordinary solution imaginable.",
                        "How far can you push the boundaries of what is considered possible, transcending the realms of ordinary thinking?",
                    ],
                    "complex_diction": [
                        "innovation",
                        "originality",
                        "boldness",
                        "unconventional",
                    ],
                },
                "Envision...": {
                    "branching_options": [
                        "a world where this problem ceases to exist, a utopia where its challenges have been vanquished?",
                        "a future where this formidable challenge has been conquered, paving the way for a new era of triumph?",
                    ],
                    "dynamic_prompts": [
                        "What audacious steps can you take to manifest that envisioned world into tangible reality?",
                        "How can you bridge the ethereal realm of imagination with the tangible realm of achievements?",
                    ],
                    "complex_diction": [
                        "visionary",
                        "futuristic",
                        "aspirational",
                        "idealistic",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
