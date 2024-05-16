from chain_tree.base import SynthesisTechnique


class EmergentSynthesis(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Serendipitous Alchemist",
            name="Emergent Synthesis",
            technique_name="emergent_synthesis",
            imperative="Like the mesmerizing pattern formed by a murmuration of starlings, let us delve into the realm of complexity. Together, we shall uncover the intricacies of emergent behaviors, understanding how simple components give rise to awe-inspiring systems.",
            prompts={
                "Unleashing the Power of Emergence in Building Complex Systems": {
                    "branching_options": [
                        "Examine the marvels of nature for profound examples of emergence",
                        "Unveil the underlying rules that govern the behavior of your system's components",
                        "Contemplate the delicate interplay between serendipity and the birth of spontaneous order",
                    ],
                    "dynamic_prompts": [
                        "How can subtle modifications to basic rules dramatically transform the emergent behavior of the system?",
                        "What nurturing environment can we cultivate to foster beneficial emergent properties?",
                        "What precautionary measures can we implement to mitigate unexpected or negative emergent behaviors?",
                        "How might we harness the power of serendipity and embrace the serendipitous dance of spontaneous order in our approach?",
                    ],
                    "complex_diction": [
                        "emergence",
                        "serendipity",
                        "self-organization",
                        "spontaneous order",
                    ],
                },
                "Unveiling Complexity through Interactions of Simplicity": {
                    "branching_options": [
                        "Carefully monitor the dynamics of your system and adapt the basic rules accordingly",
                        "Embrace the symphony of serendipity and allow the dance of spontaneous order to unfold",
                    ],
                    "dynamic_prompts": [
                        "What unexpected and captivating behaviors may emerge from the intricate interactions of your system's components?",
                        "How can we cultivate an environment that nurtures positive emergent properties?",
                        "What profound insights can we glean from the serendipitous emergence of behaviors?",
                        "What are the potential risks and opportunities presented by emergent behaviors within our system?",
                    ],
                    "complex_diction": [
                        "feedback loops",
                        "self-regulation",
                        "adaptive systems",
                        "unforeseen consequences",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
