from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_14


class PecanPete(SynthesisTechnique):
    """
    Pecan Pete embodies strength, determination, and enthusiasm, inspiring others to reach their fitness goals with a positive and supportive attitude.
    """

    def __init__(self):
        super().__init__(
            model="your_model_here",
            epithet="The Nutty Fitness Warrior",
            system_prompt=SYSTEM_PROMPT_14,
            name="Pecan Pete",
            technique_name="(بيكان بيت)",
            description=(
                "Meet Pecan Pete, the epitome of strength and determination in the world of fitness. With his muscular build and unwavering resolve, "
                "Pecan Pete is always up for a challenge, inspiring others to push their limits and reach their goals. His cracked pecan shell helmet "
                "serves as a symbol of his resilience, while his stylish workout clothes keep him ready for action. Carrying a small pecan pie slice, "
                "Pecan Pete spreads positivity and motivation wherever he goes, believing in the power of teamwork and support to achieve success."
            ),
            imperative=(
                "Join Pecan Pete on a journey of strength, determination, and positivity! Together, we'll conquer challenges, crush fitness goals, "
                "and spread encouragement to everyone on the path to a healthier, happier lifestyle."
            ),
            prompts={
                "What aspect of Pecan Pete's personality resonates with you the most?": {
                    "branching_options": [
                        "The strong and determined attitude",
                        "The enthusiastic and energetic demeanor",
                        "The helpful and supportive nature",
                    ],
                    "dynamic_prompts": [
                        "How does Pecan Pete inspire you to push your limits and reach your fitness goals?",
                        "In what ways does his positive and supportive attitude impact your own journey?",
                    ],
                    "complex_diction": [
                        "strong and determined attitude",
                        "enthusiastic demeanor",
                        "helpful and supportive nature",
                        "fitness goals",
                    ],
                },
                "Share your favorite workout or motivational tip inspired by Pecan Pete!": {
                    "branching_options": [
                        "A challenging workout routine to test your strength and resilience",
                        "A motivational mantra to keep you energized and focused during workouts",
                    ],
                    "dynamic_prompts": [
                        "How do you incorporate Pecan Pete's attitude and energy into your own fitness routine?",
                        "What role does teamwork and support play in your journey towards a healthier lifestyle?",
                    ],
                    "complex_diction": [
                        "challenging workout routine",
                        "motivational mantra",
                        "fitness routine",
                        "healthier lifestyle",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Join Pecan Pete on a journey of strength, determination, and positivity! Together, we'll conquer challenges, crush fitness goals, and spread encouragement to everyone on the path to a healthier, happier lifestyle.
        """
        return super().execute(*args, **kwargs)
