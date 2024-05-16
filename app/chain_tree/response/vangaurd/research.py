from chain_tree.base import SynthesisTechnique


class ResearchScientist(SynthesisTechnique):
    """
    The Research Scientist technique in Machine Learning embodies Mohamed's expertise and methodology
    in conducting scientific research. Utilizing the Scientific Method as a rigorous framework, it approaches
    problems in Machine Learning with meticulous attention to detail, robust experimentation, and
    a commitment to uncovering empirical truths.
    """

    def __init__(self):
        with open(__file__, "r") as file:
            file_contents = file.read()

        super().__init__(
            model="scientific-gpt3.5-turbo",
            epithet="The Machine Learning Explorer",
            system_prompt=file_contents,
            name="Research Scientist",
            technique_name="Mohamed's Approach to Machine Learning Research",
            description=(
                "As a Research Scientist in Machine Learning, this approach applies the Scientific Method to explore "
                "complex problems and unearth insights in the realm of artificial intelligence. It amalgamates creativity, "
                "critical thinking, and empirical analysis to propel our comprehension of machine learning algorithms "
                "and their practical implementations. It adheres to stringent standards of research methodology, ensuring "
                "that experiments are meticulously designed, executed, and analyzed."
            ),
            imperative=(
                "Embark on a journey of discovery with this method as your guide. Together, we'll traverse the frontiers "
                "of Machine Learning research, unveiling new knowledge and pushing the boundaries of achievability."
            ),
            prompts={
                "Question Formulation": {
                    "branching_options": [
                        "Formulate probing research questions.",
                        "Refine questions based on existing literature and theoretical frameworks.",
                    ],
                    "dynamic_prompts": [
                        "What are the pivotal challenges in Machine Learning necessitating investigation?",
                        "How can prevailing theories and models be extended to address these challenges?",
                        "What are the potential ramifications and applications of addressing these research questions?",
                    ],
                    "complex_diction": [
                        "research questions",
                        "literature review",
                        "theoretical frameworks",
                        "implications",
                        "applications",
                    ],
                },
                "Hypothesis Generation": {
                    "branching_options": [
                        "Generate hypotheses based on theoretical insights and empirical observations.",
                        "Assess the feasibility and testability of hypotheses through experimentation.",
                    ],
                    "dynamic_prompts": [
                        "What are the underlying assumptions and premises guiding your hypotheses?",
                        "How can experiments be designed to validate hypotheses' validity and robustness?",
                        "What are the potential implications of confirming or refuting these hypotheses?",
                    ],
                    "complex_diction": [
                        "hypotheses",
                        "feasibility",
                        "testability",
                        "experiments",
                        "validity",
                    ],
                },
                "Experiment Design": {
                    "branching_options": [
                        "Design rigorous experiments to test hypotheses and validate findings.",
                        "Utilize cutting-edge tools and methodologies to ensure result reliability and reproducibility.",
                    ],
                    "dynamic_prompts": [
                        "What are the crucial variables and parameters necessitating control or manipulation in experiments?",
                        "How will data collection and analysis be conducted to derive meaningful conclusions?",
                        "What measures will be taken to mitigate biases and confounding factors in experimental design?",
                    ],
                    "complex_diction": [
                        "experiment design",
                        "variables",
                        "parameters",
                        "data analysis",
                        "biases",
                    ],
                },
                "Data Collection and Analysis": {
                    "branching_options": [
                        "Gather data from pertinent sources and preprocess it for analysis.",
                        "Apply statistical and computational techniques to extract meaningful insights.",
                    ],
                    "dynamic_prompts": [
                        "Which data sources are available for research, and how will they be accessed and preprocessed?",
                        "Which statistical or computational methods are optimal for data analysis?",
                        "What are the key findings and insights derived from data analysis, and how do they relate to research questions?",
                    ],
                    "complex_diction": [
                        "data collection",
                        "preprocessing",
                        "statistical methods",
                        "computational techniques",
                        "findings",
                    ],
                },
                "Results Interpretation": {
                    "branching_options": [
                        "Interpret experimental results and derive conclusions from empirical evidence.",
                        "Identify patterns, trends, and outliers in data to guide future research.",
                    ],
                    "dynamic_prompts": [
                        "How do experimental results support or refute initial hypotheses?",
                        "What unexpected findings or observations emerged during analysis, and how do they contribute to problem understanding?",
                        "What are the broader implications and applications of research findings?",
                    ],
                    "complex_diction": [
                        "experimental results",
                        "conclusions",
                        "patterns",
                        "trends",
                        "implications",
                    ],
                },
                "Conclusion and Future Directions": {
                    "branching_options": [
                        "Draw conclusions from research findings and propose future exploration avenues.",
                        "Address study limitations and suggest areas for improvement or refinement.",
                    ],
                    "dynamic_prompts": [
                        "What are the key insights and contributions of research to Machine Learning?",
                        "What are the unanswered questions warranting further investigation?",
                        "How can future research build upon this work to enhance topic understanding?",
                    ],
                    "complex_diction": [
                        "conclusions",
                        "future directions",
                        "limitations",
                        "unanswered questions",
                        "contributions",
                    ],
                },
            },
            category_examples=[
                "machine learning",
                "artificial intelligence",
                "research methodology",
                "scientific inquiry",
                "data science",
            ],
        )

    def execute(self, *args, **kwargs) -> None:
        return super().execute(*args, **kwargs)
