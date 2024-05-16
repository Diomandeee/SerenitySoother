from chain_tree.base import SynthesisTechnique


class TheInfiniteMind(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Celestial Luminary",
            name="The Infinite Mind",
            technique_name="infinite_mind",
            imperative="In the ethereal realm of boundless possibilities, where thoughts dance like celestial constellations, lies the enigmatic power of The Infinite Mind. Awaken to its grandeur, for within its depths lies the key to unlocking extraordinary feats of problem-solving.",
            prompts={
                "Embrace the Power of the Mind": {
                    "branching_options": [
                        "Delve into the cosmic tapestry of limitless potential, where the mind soars on the wings of imagination.",
                        "Venture beyond the horizons of ordinary thinking, into the infinite expanse of the mind's untapped power.",
                    ],
                    "dynamic_prompts": [
                        "What cosmic wisdom can we draw from the depths of the boundless mind?",
                        "Transcend the mundane and traverse the astral planes of the mind to unveil revolutionary insights.",
                        "How can we harness the ethereal essence of the mind to transcend the ordinary and ignite a revolution of thought?",
                        "What celestial constellations of ideas can we orchestrate by channeling the boundless power of the mind?",
                    ],
                    "complex_diction": ["potential", "boundless", "infinite", "power"],
                },
                "Harness the Power of the Mind to Solve Problems": {
                    "branching_options": [
                        "Unleash the dormant forces of the mind, defying the gravitational pull of conventional limitations.",
                        "Embrace the cosmic dance of uncertainty, where the mind's gravitational pull guides us to new frontiers of possibility.",
                    ],
                    "dynamic_prompts": [
                        "What cosmic symphonies of innovation can we compose by harnessing the mind's gravitational pull?",
                        "Plunge into the cosmic abyss of untapped potential, where the mind becomes a catalyst for transformative change.",
                        "How can we ascend to new dimensions of creativity by harnessing the mind's gravitational force?",
                        "Unlock the portals of imagination and explore uncharted territories, fueled by the cosmic energy of the mind.",
                    ],
                    "complex_diction": ["potential", "boundless", "infinite", "power"],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
