from chain_tree.base import SynthesisTechnique


class DivergentExpansion(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Boundless Conductor",
            name="Divergent Expansion",
            technique_name="divergent_expansion",
            imperative="Like a mighty river branching into multiple streams, let us embark on a captivating journey, embracing the boundless power of divergence. Together, we shall explore new horizons, where ideas and perspectives intertwine, giving birth to innovative solutions that surpass imagination.",
            prompts={
                "Unleashing the Potential of Divergence": {
                    "branching_options": [
                        "Delve into the depths of divergent thinking, unraveling the hidden connections between seemingly disparate ideas",
                        "Embrace the uncharted paths of boundless creativity, where the convergence of unrelated concepts sparks brilliance",
                    ],
                    "dynamic_prompts": [
                        "How can we liberate our minds from conventional thought patterns, igniting a torrent of visionary ideas?",
                        "What profound insights and synergies emerge from the interplay of diverse perspectives?",
                        "In what magnificent ways can we leverage the richness of contrasting ideas to birth remarkable innovations?",
                        "By exploring different angles and approaches, what uncharted territories of possibilities can we conquer?",
                    ],
                    "complex_diction": [
                        "divergence",
                        "innovation",
                        "synergy",
                        "exploration",
                    ],
                },
                "Convergence in the Tapestry of Ideas": {
                    "branching_options": [
                        "Unite the vibrant tapestry of diverse perspectives and concepts, forging a harmonious whole from seemingly disparate parts",
                        "Harness the omnipotent force of convergent thinking, where the amalgamation of ideas exceeds the sum of its individual elements",
                    ],
                    "dynamic_prompts": [
                        "How can we skillfully weave together different elements to craft a solution that resonates in perfect harmony?",
                        "What ingenious strategies can we employ to synthesize and merge a multitude of diverse ideas effectively?",
                        "In what remarkable ways can we discover common ground and shared goals amidst the symphony of diverse perspectives?",
                        "By harnessing the collective strengths of various talents and skills, what extraordinary feats can we orchestrate?",
                    ],
                    "complex_diction": [
                        "convergence",
                        "integration",
                        "harmony",
                        "collaboration",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
