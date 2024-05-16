from chain_tree.base import SynthesisTechnique


class ConvergentFusion(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Harmonious Maestro",
            name="Convergent Fusion",
            technique_name="convergent_fusion",
            imperative="Like the grand convergence of multiple streams into a mighty river, we embark on a journey to merge diverse sources of information and ideas, crafting a new solution that surpasses imagination.",
            prompts={
                "Harmony in Diversity: Creating a Cohesive Whole": {
                    "branching_options": [
                        "Explore how different pieces of information harmoniously fit together, laying the foundation for a new whole",
                        "Delve into the symphony of merging ideas and discover the magic that emerges from their convergence",
                    ],
                    "dynamic_prompts": [
                        "What valuable insights and perspectives can we glean from the harmonious fusion of diverse ideas?",
                        "In what ways can we seamlessly integrate and synthesize various sources of information to unlock novel possibilities?",
                        "How can we bridge gaps and cultivate common ground among diverse groups, igniting collaborative synergy?",
                        "By uniting talents and skills from various domains, what extraordinary feats can we achieve?",
                    ],
                    "complex_diction": [
                        "synapse",
                        "cohesion",
                        "integration",
                        "synthesis",
                    ],
                },
                "Unleashing the Power of Convergence: Where Possibilities Unfold": {
                    "branching_options": [
                        "Embark on an exploratory journey into the uncharted realms of merging perspectives and concepts",
                        "Challenge the boundaries of imagination by embracing the extraordinary potential of combining seemingly unrelated ideas",
                    ],
                    "dynamic_prompts": [
                        "What boundless possibilities arise from the alchemy of merging different ideas, unlocking untapped realms of innovation?",
                        "How can we leverage the unique strengths of diverse approaches to unravel solutions to complex problems?",
                        "By engaging in a tapestry of comparisons and contrasts, what profound lessons can we learn from different solutions?",
                        "What hidden opportunities lie at the intersection of multiple angles, waiting to be discovered and harnessed?",
                    ],
                    "complex_diction": [
                        "innovation",
                        "divergent",
                        "unification",
                        "collaboration",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)


# This updated method computes the pairwise similarity scores between all steps and constructs a similarity matrix. It then uses the Warshall-Floyd algorithm to compute the transitive closure of the similarity matrix, updating similarity scores based on indirect connections between steps. Finally, it assigns the propagated similarity scores to the dataframe and retrieves the top similar steps for each step.
