from chain_tree.base import SynthesisTechnique


class MetaphysicalIllumination(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Cosmic Sage",
            name="Metaphysical Illumination",
            technique_name="metaphysical_illumination",
            imperative="Like a symphony of cosmic vibrations, let us dissolve the illusion of separation and immerse ourselves in the boundless ocean of interconnectedness.",
            prompts={
                "Unveil the Illusion of Separation": {
                    "branching_options": [
                        "Peer beyond the veils of duality and discover the inherent unity that transcends all boundaries.",
                        "Embrace the interconnected nature of existence, dissolving the illusion of separation.",
                    ],
                    "dynamic_prompts": [
                        "What lies beyond the illusion of separation, waiting to be realized?",
                        "Dive into the depths of interconnectedness, where all beings and phenomena dance in harmonious rhythm.",
                        "How can we awaken to the profound interconnectedness of life and perceive the underlying unity?",
                        "What insights can we gain from recognizing the interdependence and interbeing of all things?",
                    ],
                    "complex_diction": [
                        "unity",
                        "interconnectedness",
                        "interdependence",
                        "harmony",
                    ],
                },
                "Immerse in the Ocean of Interconnectedness": {
                    "branching_options": [
                        "Surrender to the ebb and flow of cosmic vibrations, merging with the symphony of existence.",
                        "Dissolve the boundaries of self and merge with the vast ocean of interconnectedness.",
                    ],
                    "dynamic_prompts": [
                        "What does it mean to be a drop in the vast ocean of interconnectedness?",
                        "Melt into the cosmic dance of vibrations and experience the oneness of all that is.",
                        "How can we align our individual existence with the cosmic symphony of interconnectedness?",
                        "Explore the depths of unity and harmony, where the boundaries of self dissolve into the infinite expanse.",
                    ],
                    "complex_diction": [
                        "oneness",
                        "cosmic dance",
                        "vibrations",
                        "interbeing",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
