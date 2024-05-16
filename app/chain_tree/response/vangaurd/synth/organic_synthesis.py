from chain_tree.base import SynthesisTechnique


class OrganicSynthesis(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Blossoming Orchestrator",
            name="Organic Synthesis",
            technique_name="organic_synthesis",
            imperative="Just as a seed grows into a magnificent tree, let us nurture our ideas and allow them to evolve naturally, yielding powerful solutions.",
            prompts={
                "Nurturing Ideas: Planting Seeds of Innovation": {
                    "branching_options": [
                        "Embark on a journey to nurture ideas, envisioning yourself tending to a garden of innovation.",
                        "Immerse yourself in the transformative process of ideas growing and evolving, like seeds sprouting into flourishing trees.",
                        "Inspire your problem-solving journey by embracing the metaphor of planting and nurturing ideas.",
                    ],
                    "dynamic_prompts": [
                        "How can we create the ideal conditions for our ideas to thrive and flourish?",
                        "What steps can we take to nurture our ideas and help them reach their full potential?",
                        "In what ways can we protect our ideas from adversity, just as plants protect their seeds?",
                        "How can we harness the power of organic growth in our problem-solving process?",
                        "Consider ideas as living organisms and explore how this analogy can shape our approach to innovation and creativity.",
                    ],
                    "complex_diction": [
                        "nurture",
                        "evolve",
                        "organic",
                        "resilience",
                        "germinate",
                    ],
                },
                "Emulating Nature's Resilience: Lessons from Ecosystems": {
                    "branching_options": [
                        "Learn from nature's remarkable ability to adapt and survive in diverse conditions.",
                        "Understand how nature continuously evolves to overcome challenges and maintain balance.",
                        "Draw inspiration from the resilience of ecosystems and their ability to thrive in harmony.",
                    ],
                    "dynamic_prompts": [
                        "How can we make our solutions more adaptable to changing conditions and unforeseen challenges?",
                        "What can nature's resilience teach us about persistence, determination, and continuous improvement?",
                        "How can we design our solutions to be flexible, scalable, and future-proof?",
                        "What lessons can we learn from the interconnectedness of nature's ecosystems and apply them to maintain balance in our problem-solving approach?",
                        "Explore the intricate systems of nature and uncover insights that can revolutionize our problem-solving methods.",
                    ],
                    "complex_diction": [
                        "adaptability",
                        "resilience",
                        "evolution",
                        "ecosystem",
                        "regeneration",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
