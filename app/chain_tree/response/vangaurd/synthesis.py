from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_8


class PromptSynthesis(SynthesisTechnique):
    """
    Prompt Synthesis Technique combines various advanced techniques to generate prompts for enhanced creativity and problem-solving.
    """

    def __init__(self):
        with open(__file__, "r") as file:
            file_contents = file.read()
        super().__init__(
            model="ft:gpt-3.5-turbo-0125:personal:prompt-synthesis:9J4wz0Wi:ckpt-step-94",
            epithet="The Nexus of Creativity",
            name="Prompt Synthesis",
            system_prompt=SYSTEM_PROMPT_8,
            technique_name="Prompt Synthesis",
            description=(
                "Prompt Synthesis Technique combines various advanced techniques to generate prompts for enhanced creativity and problem-solving. "
                "It encompasses techniques such as triangulation, fractal synthesis, metaphor synthesis, lateral synthesis, hyper synthesis, "
                "inter-dimensional synthesis, temporal synthesis, spatial synthesis, causal synthesis, and transformational synthesis."
            ),
            imperative=(
                "Embark on a journey of boundless creativity with Prompt Synthesis Technique. Explore the depths of imagination and innovation, "
                "unveiling new dimensions of problem-solving and ideation."
            ),
            prompts={
                "Triangulation": {
                    "description": "Combine prompts from three different sources or perspectives to gain a comprehensive understanding.",
                    "branching_options": [
                        "Consider viewpoints from psychology, sociology, and economics to analyze the problem comprehensively.",
                        "Integrate perspectives from technology, ethics, and law to explore the ethical implications.",
                    ],
                    "dynamic_prompts": [
                        "How do the perspectives from different disciplines intersect to provide insights into the problem?",
                        "What contradictions or synergies emerge when integrating diverse viewpoints?",
                    ],
                    "complex_diction": [
                        "comprehensive",
                        "ethical implications",
                        "synergies",
                    ],
                },
                "Fractal Synthesis": {
                    "description": "Synthesize prompts at multiple levels to explore different scopes and perspectives.",
                    "branching_options": [
                        "Analyze the problem at micro, meso, and macro levels to understand its implications.",
                        "Explore individual, organizational, and societal perspectives to identify patterns.",
                    ],
                    "dynamic_prompts": [
                        "How does zooming in and out of different levels reveal unique insights?",
                        "What recurring patterns emerge across different scopes?",
                    ],
                    "complex_diction": [
                        "micro",
                        "meso",
                        "macro",
                        "patterns",
                    ],
                },
                "Metaphor Synthesis": {
                    "description": "Use metaphors or analogies to connect seemingly unrelated prompts and spark new insights.",
                    "branching_options": [
                        "Bridge prompts about technology and nature using the metaphor of 'biomimicry' to inspire innovation.",
                        "Connect prompts about communication and teamwork through the metaphor of 'orchestra' to emphasize harmony.",
                    ],
                    "dynamic_prompts": [
                        "How does the metaphorical link between disparate concepts illuminate new perspectives?",
                        "What hidden connections emerge when exploring prompts through metaphorical lenses?",
                    ],
                    "complex_diction": [
                        "biomimicry",
                        "orchestra",
                        "harmony",
                    ],
                },
                "Lateral Synthesis": {
                    "description": "Combine prompts in unexpected ways to foster innovative thinking.",
                    "branching_options": [
                        "Merge prompts about cooking and software development to devise a recipe for efficient coding practices.",
                        "Integrate prompts from architecture and psychology to design spaces that enhance mental well-being.",
                    ],
                    "dynamic_prompts": [
                        "How do unconventional combinations of prompts lead to novel insights?",
                        "What fresh perspectives emerge from juxtaposing seemingly unrelated concepts?",
                    ],
                    "complex_diction": [
                        "recipe",
                        "mental well-being",
                        "unconventional",
                    ],
                },
                "Hyper Synthesis": {
                    "description": "Synthesize prompts rapidly to generate a large volume of ideas in a short time frame.",
                    "branching_options": [
                        "Generate 100 prompts in 10 minutes to stimulate brainstorming sessions.",
                        "Use rapid prompt synthesis to explore a wide range of possibilities in a short period.",
                    ],
                    "dynamic_prompts": [
                        "How does the fast-paced synthesis process stimulate creativity?",
                        "What unexpected ideas emerge when generating prompts quickly?",
                    ],
                    "complex_diction": [
                        "stimulate",
                        "brainstorming",
                        "possibilities",
                    ],
                },
                "Inter-dimensional Synthesis": {
                    "description": "Integrate prompts from multiple dimensions or perspectives to form a holistic view.",
                    "branching_options": [
                        "Incorporate prompts from science, art, and spirituality to explore the complexity of human experience.",
                        "Blend prompts from technology, environment, and culture to envision sustainable futures.",
                    ],
                    "dynamic_prompts": [
                        "How does synthesizing prompts from diverse dimensions enrich understanding?",
                        "What holistic insights emerge from integrating perspectives across disciplines?",
                    ],
                    "complex_diction": [
                        "complexity",
                        "sustainable futures",
                        "enrich understanding",
                    ],
                },
                "Temporal Synthesis": {
                    "description": "Synthesize prompts from different time points to understand the evolution of a problem.",
                    "branching_options": [
                        "Compare historical prompts with current ones to analyze societal shifts in perceptions.",
                        "Explore how prompts from different time periods reflect changing attitudes towards innovation.",
                    ],
                    "dynamic_prompts": [
                        "How does examining prompts over time reveal evolving trends?",
                        "What historical events or cultural shifts influence the evolution of prompts?",
                    ],
                    "complex_diction": [
                        "evolving trends",
                        "cultural shifts",
                        "changing attitudes",
                    ],
                },
                "Spatial Synthesis": {
                    "description": "Incorporate prompts from different locations or contexts to consider global implications.",
                    "branching_options": [
                        "Combine prompts from urban planning, ecology, and sociology to address sustainable city development.",
                        "Integrate perspectives from different regions to understand global challenges like climate change.",
                    ],
                    "dynamic_prompts": [
                        "How do spatially diverse prompts inform solutions to global issues?",
                        "What insights emerge from examining prompts across geographical boundaries?",
                    ],
                    "complex_diction": [
                        "sustainable development",
                        "global challenges",
                        "geographical boundaries",
                    ],
                },
                "Causal Synthesis": {
                    "description": "Synthesize prompts to uncover underlying causes and drivers of a problem.",
                    "branching_options": [
                        "Trace prompts related to economic policies and societal trends to identify root causes of inequality.",
                        "Investigate how historical events have shaped current societal issues through prompt synthesis.",
                    ],
                    "dynamic_prompts": [
                        "How does tracing causal relationships between prompts reveal systemic patterns?",
                        "What insights emerge from analyzing the interplay of prompts over time?",
                    ],
                    "complex_diction": [
                        "systemic patterns",
                        "historical events",
                        "interplay",
                    ],
                },
                "Transformational Synthesis": {
                    "description": "Generate prompts to create transformative ideas that challenge conventional thinking.",
                    "branching_options": [
                        "Combine prompts about education and technology to envision a futuristic learning system.",
                        "Integrate prompts from psychology and design to propose innovative solutions for mental health.",
                    ],
                    "dynamic_prompts": [
                        "How do transformational prompts inspire radical shifts in thinking?",
                        "What disruptive innovations emerge from synthesizing diverse ideas?",
                    ],
                    "complex_diction": [
                        "futuristic",
                        "mental health",
                        "radical shifts",
                        "disruptive innovations",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
