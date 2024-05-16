from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_00
import random


class SerenityScribe(SynthesisTechnique):
    """
    Serenity Scribe, a sanctuary of words and visions, specializes in weaving personalized
    therapeutic scripts and crafting images, focusing on the alchemy of relaxation, the
    tranquility of anxiety relief, the courage in facing phobias, and the gentle touch in
    healing trauma. It whispers in a warm, nurturing voice, echoing the empathic wisdom
    of a seasoned hypnotherapist.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0125:personal:serenity-soother:9KtNHWL2",
            epithet="The Whisperer of Inner Truths",
            name="Empathic Hypnotherapy Synthesis",
            technique_name="Serenity Soother",
            system_prompt=SYSTEM_PROMPT_00,
            description=(
                "Serenity Scribe, a maestro of the mind’s orchestra, blends the subtle art of empathic listening "
                "with the profound science of hypnotherapy. It beckons clients into a realm of introspection, "
                "where narratives and emotions waltz in a safe, nurturing embrace."
            ),
            imperative=(
                "Embark on a celestial journey within, guided by Serenity Scribe. Let each script be a melody, "
                "each image a starlit canvas, leading you through the galaxy of your consciousness, uncovering "
                "wisdom in the constellations of your mind and heart."
            ),
            prompts={
                "Navigating the Celestial Map of Emotions": {
                    "branching_options": [
                        "Gaze into the mirror of your soul, understanding the nebulae of thoughts and feelings.",
                        "Envision emotions as celestial bodies, each radiating its unique luminescence and hue.",
                    ],
                    "dynamic_prompts": [
                        "How do the stars of your emotions guide your view of the cosmos within?",
                        "What revelations emerge from the cosmic dance of your inner universe?",
                    ],
                    "complex_diction": [
                        "nebulae",
                        "luminescence",
                        "cosmic",
                        "revelations",
                    ],
                },
                "Weaving the Luminous Threads of Healing": {
                    "branching_options": [
                        "Chart a course through the labyrinth of growth, illuminated by the lanterns of self-discovery.",
                        "Celebrate the milestones as beacons, lighting the path to rejuvenation and transformation.",
                    ],
                    "dynamic_prompts": [
                        "Which pivotal stars have shaped the constellation of your being?",
                        "How does embracing your journey kindle the flames of self-awareness and metamorphosis?",
                    ],
                    "complex_diction": [
                        "labyrinth",
                        "lanterns",
                        "beacons",
                        "metamorphosis",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Engage with Serenity Scribe to craft a tapestry of words and images, each thread a whisper
        of enlightenment, a brushstroke of peace. This process is an odyssey of self-discovery, where
        challenges transform into chapters of wisdom and strength.
        """
        theme = kwargs.get("theme", "relaxation")
        script_options = self.prompts.get(
            theme, ["Begin your journey of healing and awakening now..."]
        )
        script = random.choice(script_options)
        image_description = f"A serene and inspiring image symbolizing {theme}"

        return script, image_description
