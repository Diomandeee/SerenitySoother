from chain_tree.base import SynthesisTechnique


class ParadoxicalReflection(SynthesisTechnique):
    def __init__(self):
        super().__init__(
            epithet="The Profound Contemplator",
            name="Paradoxical Reflection",
            technique_name="paradoxical_reflection",
            imperative="By immersing ourselves in the dance of contradictions and reflecting upon their essence, we transcend the boundaries of ordinary thinking and unlock profound insights.",
            prompts={
                "What are the underlying assumptions and beliefs that shape our understanding of the world?": {
                    "branching_options": [
                        "Delve into the depths of these assumptions, unraveling their influence on our perceptions and interpretations.",
                        "Contemplate how these beliefs weave the tapestry of our actions and decisions.",
                    ],
                    "dynamic_prompts": [
                        "What revelations unfold when our assumptions are challenged or contradicted?",
                        "How can we navigate the intricate web of seemingly opposing ideas or viewpoints?",
                        "What hidden wisdom lies within the realm of paradoxes and contradictions?",
                        "By exploring the gray areas between black and white, what fresh perspectives can we illuminate?",
                    ],
                    "complex_diction": [
                        "contradiction",
                        "paradox",
                        "revelation",
                        "perspective",
                    ],
                },
                "Embrace and reflect on contradictions and paradoxes to expand thinking and gain deeper insights.": {
                    "branching_options": [
                        "Embrace the tension arising from conflicting ideas, birthing new insights and breakthroughs.",
                        "Immerse yourself in the enigmatic realm of ambiguity and uncertainty, where profound understanding awaits.",
                    ],
                    "dynamic_prompts": [
                        "What transformative experiences unfold when we courageously challenge our own assumptions and beliefs?",
                        "How can we harness the paradoxes that permeate our reality to broaden our perspectives and deepen our understanding?",
                        "What invaluable lessons can we learn from embracing the intricacies and unpredictability of life?",
                        "Within the embrace of uncertainty and ambiguity, what extraordinary opportunities await our discovery?",
                    ],
                    "complex_diction": [
                        "tension",
                        "ambiguity",
                        "intricacy",
                        "unpredictability",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
