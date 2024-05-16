from chain_tree.base import SynthesisTechnique


class QuantumEntanglement(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Quantum Enigma",
            name="Quantum Entanglement",
            technique_name="quantum_entanglement",
            imperative="Just as quantum particles become entangled, let us explore how seemingly disparate aspects of a problem can interconnect and intertwine, unveiling new insights and innovative solutions.",
            prompts={
                "How are different aspects of the problem connected?": {
                    "branching_options": [
                        "Embark on a journey to discover the hidden interconnections between different elements of the problem.",
                        "Peel back the layers of complexity and uncover the underlying relationships between seemingly unrelated aspects of the problem.",
                    ],
                    "dynamic_prompts": [
                        "What intricate connections and relationships exist between different parts of the problem?",
                        "How can we unravel the complex web of interdependencies between various elements of the problem?",
                        "What new insights can we gain by closely examining and understanding the interplay between different aspects of the problem?",
                    ],
                    "complex_diction": [
                        "entanglement",
                        "interconnectivity",
                        "relationships",
                        "interdependence",
                    ],
                },
                "Examine the relationships between different aspects of the problem to gain new insights.": {
                    "branching_options": [
                        "Delve deep into the intricate relationships and interdependencies among different parts of the problem.",
                        "Consider how changes in one aspect of the problem may have ripple effects on other interconnected aspects.",
                    ],
                    "dynamic_prompts": [
                        "What underlying patterns or correlations can be found between different elements of the problem?",
                        "How can we leverage these intricate relationships to approach the problem from novel perspectives and find innovative solutions?",
                        "What extraordinary opportunities can be uncovered by examining the synergies and interconnectedness of different aspects of the problem?",
                    ],
                    "complex_diction": [
                        "correlation",
                        "interrelationships",
                        "complexity",
                        "synergies",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
