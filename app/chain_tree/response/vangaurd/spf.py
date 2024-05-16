from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_15


class SynergeticPromptFramework(SynthesisTechnique):
    """
    The Synergetic Prompt Framework technique is a versatile tool for generating prompts, brainstorming ideas,
    asking thought-provoking questions, and creating synergetic prompts across various creative domains.
    """

    def __init__(self):
        super().__init__(
            model="synergetic-prompt-gpt3.5-turbo",
            epithet="The Catalyst of Creativity",
            system_prompt=SYSTEM_PROMPT_15,
            name="SPF",
            technique_name="(共鳴プロンプトフレームワーク)",
            description=(
                "The Synergetic Prompt Framework technique empowers users to explore their creativity "
                "by providing a structured approach to generating ideas, asking questions, and crafting prompts. "
                "Whether you're a writer, artist, musician, or innovator, this technique offers a comprehensive "
                "framework for sparking inspiration and unlocking new possibilities."
            ),
            imperative=(
                "Harness the power of synergy and creativity by leveraging the Synergetic Prompt Framework "
                "to generate innovative ideas, ask thought-provoking questions, and craft compelling prompts. "
                "Let your imagination soar as you explore new creative frontiers and unlock your full potential."
            ),
            prompts={
                "Imagine That": {
                    "branching_options": [
                        "Generate creative ideas inspired by a specific scenario or theme.",
                        "Explore different perspectives and possibilities within a given context.",
                    ],
                    "dynamic_prompts": [
                        "What if you were tasked with creating a new masterpiece in your chosen field? How would you approach it?",
                        "Imagine yourself in a scenario where you have complete creative freedom. What would you create?",
                        "If you could explore any theme or concept through your work, what would it be and why?",
                    ],
                    "complex_diction": [
                        "creativity",
                        "imagination",
                        "innovation",
                        "exploration",
                        "possibility",
                    ],
                },
                "Brainstorming": {
                    "branching_options": [
                        "Generate a variety of ideas or concepts related to a specific theme or topic.",
                        "Encourage free-flowing creativity and exploration of different avenues.",
                    ],
                    "dynamic_prompts": [
                        "What are some unconventional ways to approach this problem or theme?",
                        "How can you combine different elements or concepts to create something new and innovative?",
                        "What are some potential challenges or obstacles you might encounter, and how can you overcome them?",
                    ],
                    "complex_diction": [
                        "innovative",
                        "unconventional",
                        "exploration",
                        "challenge",
                        "solution",
                    ],
                },
                "Thought Provoking Questions": {
                    "branching_options": [
                        "Ask insightful questions to stimulate critical thinking and exploration.",
                        "Encourage deeper reflection and analysis of a given topic or concept.",
                    ],
                    "dynamic_prompts": [
                        "What are some potential implications or consequences of this idea or concept?",
                        "How might different perspectives or viewpoints influence our understanding of this topic?",
                        "What are some ethical considerations or dilemmas related to this issue, and how might they be addressed?",
                    ],
                    "complex_diction": [
                        "insightful",
                        "reflection",
                        "analysis",
                        "perspective",
                        "ethics",
                    ],
                },
                "Create Prompts": {
                    "branching_options": [
                        "Craft engaging and thought-provoking prompts to inspire creativity and exploration.",
                        "Encourage active participation and contribution from others.",
                    ],
                    "dynamic_prompts": [
                        "How can you tailor your prompts to specific audiences or demographics?",
                        "What are some creative ways to encourage engagement and interaction with your prompts?",
                        "How might you incorporate feedback or suggestions from others into your prompt creation process?",
                    ],
                    "complex_diction": [
                        "engaging",
                        "participation",
                        "interaction",
                        "feedback",
                        "contribution",
                    ],
                },
                "Synergetic Prompts": {
                    "branching_options": [
                        "Create prompts that combine different ideas, themes, or concepts in innovative ways.",
                        "Encourage interdisciplinary collaboration and exploration.",
                    ],
                    "dynamic_prompts": [
                        "How can you blend different elements or concepts to create a cohesive and compelling prompt?",
                        "What are some potential synergies or connections between seemingly unrelated ideas or themes?",
                        "How might you leverage the diversity of perspectives and backgrounds to create more inclusive and impactful prompts?",
                    ],
                    "complex_diction": [
                        "innovative",
                        "collaboration",
                        "exploration",
                        "cohesive",
                        "inclusive",
                    ],
                },
                "Category": {
                    "branching_options": [
                        "Summarize the overall theme or concept explored in a set of prompts or ideas.",
                        "Provide context and clarity to help users understand the overarching goals or objectives.",
                    ],
                    "dynamic_prompts": [
                        "What is the central theme or concept that ties all of your prompts or ideas together?",
                        "How can you communicate this theme or concept effectively to your audience?",
                        "What are some potential categories or keywords that capture the essence of your prompt set?",
                    ],
                    "complex_diction": [
                        "theme",
                        "concept",
                        "context",
                        "clarity",
                        "essence",
                    ],
                },
            },
        )

    def execute(self, prompt: str) -> str:
        return super().execute(prompt)
