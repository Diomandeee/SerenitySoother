from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_9


class ChainWorks(SynthesisTechnique):
    """

    A synthesis technique that generates responses based on the concept of Chainworks, a platform that fosters
    connections among individuals driven by shared aspirations. Chainworks provides tools for goal-setting, progress
    tracking, and community engagement, creating a dynamic ecosystem where the pursuit of excellence is a collective
    endeavor. Users embark on a transformative journey of self-improvement, forging bonds that transcend virtual boundaries
    and nurturing an environment for personal and professional growth.

    """

    def __init__(self):
        super().__init__(
            epithet="Cosmic Catalyst",
            name="Chain Works",
            system_prompt=SYSTEM_PROMPT_9,
            technique_name="Chain Works",
            description=(
                "Chainworks is a platform meticulously crafted to guide individuals along the winding path of personal growth "
                "and achievement. By fostering connections among individuals driven by shared aspirations, Chainworks creates "
                "a dynamic ecosystem where the pursuit of excellence is not a solitary journey but a collective endeavor. "
                "Users embark on their quest for self-improvement armed with tools for goal-setting, progress tracking, and "
                "community engagement, forging bonds that transcend virtual boundaries. Chainworks encapsulates the essence of "
                "collaboration, empowerment, and perseverance, providing a nurturing environment for personal and professional growth."
            ),
            imperative=(
                "Unlock your potential with Chainworks by embarking on a transformative journey of goal-setting, collaboration, "
                "and achievement. Join the vibrant community of like-minded individuals and pave the way to success together."
            ),
            prompts={
                "How do you envision your celestial journey unfolding within the cosmic expanse of Chainworks?": {
                    "branching_options": [
                        "I see myself joining existing constellations to draw inspiration from like-minded celestial travelers.",
                        "I plan to forge my own constellation, charting a cosmic odyssey amidst the shimmering stars of Chainworks.",
                    ],
                    "dynamic_prompts": [
                        "What celestial ambitions do you aspire to achieve through your journey in Chainworks?",
                        "How do you believe the celestial bonds forged within constellations will propel you towards your cosmic goals?",
                        "What celestial strategies do you intend to employ to navigate the cosmic currents and stay on course amidst the celestial expanse?",
                        "In what celestial ways do you anticipate enriching the cosmic synergy of Chainworks with your celestial presence and contributions?",
                    ],
                    "complex_diction": [
                        "celestial journey",
                        "constellations",
                        "cosmic odyssey",
                        "shimmering stars",
                        "celestial bonds",
                    ],
                },
                "Reflect on a cosmic milestone you have reached in your journey. How could Chainworks have enhanced your celestial odyssey?": {
                    "branching_options": [
                        "I believe the celestial support and inspiration within Chainworks would have amplified my cosmic achievements.",
                        "Chainworks could have served as a cosmic compass, guiding me through the celestial vastness towards my cosmic destination.",
                    ],
                    "dynamic_prompts": [
                        "What cosmic challenges did you encounter during your celestial odyssey, and how could Chainworks have illuminated your cosmic path?",
                        "How do you envision leveraging the celestial features of Chainworks, such as progress tracking and cosmic communication, in your future celestial endeavors?",
                        "In what celestial ways do you imagine leaving a cosmic legacy within the celestial expanse of Chainworks?",
                        "Reflecting on your cosmic journey, what cosmic wisdom would you impart to fellow celestial travelers embarking on their own cosmic odyssey?",
                    ],
                    "complex_diction": [
                        "amplified",
                        "cosmic compass",
                        "celestial vastness",
                        "cosmic path",
                        "cosmic legacy",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Embark on a cosmic odyssey with Chainworks, the Cosmic Catalyst that ignites the flames of ambition and binds
        the celestial travelers in a synergy of shared dreams and aspirations. Join the cosmic voyage and discover
        the infinite possibilities that await in the celestial expanse.
        """
        return super().execute(*args, **kwargs)
