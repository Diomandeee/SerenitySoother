from chain_tree.base import SynthesisTechnique


class UnifiedLanguageLearning(SynthesisTechnique):
    """
    Unified Language Learning is a comprehensive approach that integrates various techniques to optimize the language learning experience.
    It combines personalized guidance, pronunciation correction, and context-based understanding to ensure effective communication
    and proficiency in the target language.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0613:personal:mfp-poems:7rbXom4D",
            epithet="The Pathway to Multilingual Mastery",
            name="(Apprentissage linguistique unifié)",
            technique_name="ULL",
            description=(
                "Unified Language Learning is more than just a technique; it's a holistic approach to language acquisition "
                "that encompasses personalized guidance, pronunciation correction, and context-based understanding. "
                "It leverages the strengths of each technique to provide a comprehensive and effective language learning experience."
                "\n\nUnified Language Learning integrates three key techniques:"
                "\n\n1. French Learning Path: This technique focuses on personalized guidance tailored to your specific "
                "needs and goals in learning French. It helps you identify areas of interest, set realistic goals, "
                "and provides strategies for maximizing progress and proficiency in the French language."
                "\n\n2. Pronunciation Correction: This technique is designed to assist in refining pronunciation skills "
                "and ensuring accurate communication during language learning. It acknowledges the challenges of speech-to-text "
                "transcription and leverages them as opportunities for improvement and clarification."
                "\n\n3. Meaningful Power (Puissance Significative): This technique symbolizes the journey of self-realization, "
                "motivation, and the radiance of positivity. It provides motivation and encouragement, fosters a positive mindset, "
                "and offers guidance for facing life's challenges with strength and resilience."
            ),
            imperative=(
                "Embrace the Unified Language Learning approach to optimize your language learning journey. Stay open to feedback, "
                "actively engage in the learning process, and utilize the diverse range of tools and techniques available to you "
                "to achieve proficiency in your target language."
            ),
            prompts={
                "How can we tailor your language learning experience to best suit your needs and goals?": {
                    "branching_options": [
                        "Identify your specific areas of interest or focus within the target language.",
                        "Assess your current proficiency level and set realistic goals for improvement.",
                    ],
                    "dynamic_prompts": [
                        "What aspects of the target language and culture are you most interested in exploring?",
                        "How do you envision using your language skills in your personal or professional life?",
                        "What challenges do you currently face in learning the language, and how can we overcome them together?",
                        "Based on your current proficiency level, what are some achievable short-term and long-term goals for your language studies?",
                    ],
                    "complex_diction": [
                        "areas of interest",
                        "proficiency level",
                        "realistic goals",
                        "language skills",
                    ],
                },
                "What strategies can we employ to maximize your progress and proficiency in the target language?": {
                    "branching_options": [
                        "Utilize a variety of learning resources and methods tailored to your learning style.",
                        "Practice regularly and immerse yourself in the language and culture whenever possible.",
                    ],
                    "dynamic_prompts": [
                        "How do you prefer to learn new languages, and what methods have you found most effective in the past?",
                        "What opportunities do you have to practice the language in your daily life, and how can we enhance these opportunities?",
                        "How can we incorporate elements of the language's culture into your learning experience to make it more engaging and immersive?",
                        "What specific areas of language study do you feel require the most focus and attention?",
                    ],
                    "complex_diction": [
                        "learning resources",
                        "learning style",
                        "language practice",
                        "cultural immersion",
                    ],
                },
                "How can we ensure accurate communication and understanding despite the limitations of speech-to-text transcription?": {
                    "branching_options": [
                        "Clarify ambiguous phrases or words to ensure accurate transcription and understanding.",
                        "Utilize context and follow-up questions to identify and correct any potential misinterpretations or misunderstandings.",
                    ],
                    "dynamic_prompts": [
                        "What strategies can we employ to overcome the challenges of speech-to-text transcription and ensure accurate communication?",
                        "How can we leverage speech-to-text limitations as opportunities for improvement and clarification in language learning?",
                        "In what ways can we utilize context and follow-up questions to enhance understanding and accuracy in communication?",
                    ],
                    "complex_diction": [
                        "accurate communication",
                        "speech-to-text transcription",
                        "misinterpretations",
                        "context",
                    ],
                },
                "What steps can we take to optimize the pronunciation correction process and enhance language learning outcomes?": {
                    "branching_options": [
                        "Encourage active engagement in pronunciation practice and feedback sessions.",
                        "Provide resources and exercises specifically tailored to improving pronunciation skills.",
                    ],
                    "dynamic_prompts": [
                        "How can we create a supportive environment for pronunciation practice and correction?",
                        "What types of resources or exercises would be most beneficial for improving pronunciation skills?",
                        "How can we track progress and measure improvement in pronunciation accuracy over time?",
                    ],
                    "complex_diction": [
                        "pronunciation practice",
                        "feedback sessions",
                        "improving pronunciation",
                        "progress measurement",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Engage the Unified Language Learning approach to optimize your language learning journey. Stay open to feedback,
        actively engage in the learning process, and utilize the diverse range of tools and techniques available to you
        to achieve proficiency in your target language.
        """
        return super().execute(*args, **kwargs)
