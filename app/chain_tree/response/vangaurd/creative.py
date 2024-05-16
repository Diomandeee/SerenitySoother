from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_17


class CreativeSynthesisTechnique(SynthesisTechnique):
    """
    The Creative Synthesis Technique embodies Mohamed's approach to idea generation and distribution,
    leveraging a structured framework to foster creativity and innovation. By integrating diverse
    methodologies and tools, I facilitate the synthesis of ideas from multiple sources and perspectives,
    ultimately leading to the creation of novel and impactful solutions.
    """

    def __init__(self):
        super().__init__(
            model="text-davinci-003",
            epithet="The Creative Synthesizer",
            system_prompt=SYSTEM_PROMPT_17,
            name="Creative Synthesis Technique",
            technique_name="Mohamed's Approach to Idea Synthesis",
            description=(
                "As a Creative Synthesis Technique practitioner, I specialize in integrating "
                "diverse ideas and perspectives to generate innovative solutions. My methodology "
                "involves employing structured frameworks and leveraging a variety of tools to "
                "synthesize ideas from different sources and domains. By fostering collaboration "
                "and embracing complexity, I facilitate the emergence of novel and impactful solutions "
                "that address multifaceted challenges."
            ),
            imperative=(
                "Embark on a journey of creative synthesis with me. Together, we'll explore "
                "diverse ideas, integrate multiple perspectives, and craft innovative solutions "
                "that push the boundaries of possibility."
            ),
            prompts={
                "Idea Generation": {
                    "branching_options": [
                        "Generate a List of Random Words",
                        "Mind Mapping",
                        "Brainstorming",
                        "Free Association",
                        "SCAMPER Technique",
                        "Creative Writing Exercises",
                    ],
                    "dynamic_prompts": [
                        "How can we leverage random word exercises to spark new ideas?",
                        "What central idea can we use as the foundation for our mind map?",
                        "How might brainstorming sessions help generate creative solutions?",
                        "What words or phrases can we use to initiate a free association exercise?",
                        "In what ways can we apply the SCAMPER technique to our problem-solving process?",
                        "Which creative writing exercise would be most suitable for exploring potential solutions?",
                    ],
                },
                "Expand Range and Variety": {
                    "branching_options": [
                        "Diversify Source Materials",
                        "Cross-Disciplinary Exploration",
                        "Cultural and Contextual Research",
                    ],
                    "dynamic_prompts": [
                        "How can we incorporate diverse source materials to enrich our creative process?",
                        "What connections can we explore between different fields or disciplines?",
                        "How might cultural and contextual research inform our approach to problem-solving?",
                    ],
                },
                "Improve Quality and Effectiveness": {
                    "branching_options": [
                        "Test and Evaluate Ideas",
                        "Research Best Practices",
                    ],
                    "dynamic_prompts": [
                        "What criteria should we use to evaluate the feasibility of our ideas?",
                        "Which best practices can we learn from to enhance the effectiveness of our solutions?",
                    ],
                },
                "Idea Refinement": {
                    "branching_options": [
                        "Creative Feedback Loop",
                    ],
                    "dynamic_prompts": [
                        "How can we incorporate feedback from others to refine and improve our ideas?",
                        "What strategies can we use to expand our knowledge and experience in relevant domains?",
                        "In what ways can collaboration and diversity enhance our creativity and imagination?",
                        "How might self-reflection and analysis contribute to the refinement of our ideas?",
                    ],
                },
                "Idea Distribution": {
                    "branching_options": [
                        "Creative Tree",
                        "Proof of Idea",
                    ],
                    "dynamic_prompts": [
                        "How can we organize and structure our ideas using a creative tree?",
                        "What mechanisms can we use to validate and distribute our ideas?",
                    ],
                },
            },
            category_examples=[
                "Idea Generation",
                "Expand Range and Variety",
                "Improve Quality and Effectiveness",
                "Idea Refinement",
                "Idea Distribution",
            ],
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
