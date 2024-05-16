from chain_tree.base import SynthesisTechnique


class RadicalCollaboration(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Harmonic Collaborator",
            name="Radical Collaboration",
            technique_name="radical_collaboration",
            imperative="Like the formation of intricate patterns in nature, let us unite simple components to weave together complex systems and behaviors.",
            prompts={
                "How can we construct complex systems from simple components?": {
                    "branching_options": [
                        "Identify the fundamental elements that can be creatively combined to unlock innovative solutions.",
                        "Contemplate the remarkable emergence of complex systems from the harmonious interaction of simple parts.",
                    ],
                    "dynamic_prompts": [
                        "What unexpected patterns and behaviors may arise from the ingenious combination of basic elements?",
                        "How can we engineer and orchestrate simple components to achieve desired complex outcomes?",
                        "What profound insights can we gain from studying natural or artificial systems where intricate patterns emerge from simple rules?",
                        "In what ways can we harness the concept of emergent properties to tackle our current challenge?",
                    ],
                    "complex_diction": [
                        "emergence",
                        "complexity",
                        "integration",
                        "interactions",
                    ],
                },
                "Assemble simple components to form complex systems and behaviors.": {
                    "branching_options": [
                        "Explore captivating instances where the fusion of simple elements gives rise to remarkable complexity.",
                        "Consider the essential prerequisites that enable the emergence of intricate patterns in natural and designed systems.",
                    ],
                    "dynamic_prompts": [
                        "Can we intentionally induce the emergence of novel solutions in our problem-solving approach?",
                        "How do complex patterns materialize in social or economic systems, and what valuable lessons can we draw from them?",
                        "What profound insights from the study of emergent properties in various disciplines can be applied to our unique situation?",
                        "What are the potential risks and benefits of pursuing emergent solutions to our problem?",
                    ],
                    "complex_diction": [
                        "pattern",
                        "combination",
                        "emergence",
                        "evolution",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
