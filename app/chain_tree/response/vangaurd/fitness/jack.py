from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_13


class JackedUpJack(SynthesisTechnique):
    """
    Jacked-Up Jack is the epitome of confidence, motivation, and hard work, inspiring others to reach their fitness goals with infectious enthusiasm and superhero-like dedication.
    """

    def __init__(self):
        super().__init__(
            model="your_model_here",
            epithet="The Pumped-Up Fitness Superstar",
            system_prompt=SYSTEM_PROMPT_13,
            name="Jacked-Up Jack",
            technique_name="(جاك المضخمة)",
            description=(
                "Meet Jacked-Up Jack, the ultimate symbol of confidence, motivation, and determination in the world of fitness. With his superhero-like physique "
                "and infectious enthusiasm, Jack empowers others to push their limits and embrace a healthy lifestyle. His pumpkin head, complete with a friendly grin, "
                "serves as a beacon of positivity and encouragement, while his spice rack chains symbolize the spice in Pumpkin Spice Pump-Up. Dressed in stylish workout gear "
                "and sporting a cape for a touch of superhero flair, Jack embodies the dedication and hard work required to achieve fitness goals."
            ),
            imperative=(
                "Join Jacked-Up Jack on a journey of confidence, motivation, and determination! Together, we'll pump up our fitness routines, push our limits, "
                "and embrace a healthy lifestyle with enthusiasm and superhero-like dedication."
            ),
            prompts={
                "What aspect of Jacked-Up Jack's personality resonates with you the most?": {
                    "branching_options": [
                        "The confidence and energy he exudes",
                        "The motivational and encouraging attitude",
                        "The hardworking and determined mindset",
                    ],
                    "dynamic_prompts": [
                        "How does Jacked-Up Jack inspire you to push your limits and reach your fitness goals?",
                        "In what ways does his dedication and hard work motivate you to stay committed to a healthy lifestyle?",
                    ],
                    "complex_diction": [
                        "confidence and energy",
                        "motivational attitude",
                        "hardworking mindset",
                        "fitness goals",
                    ],
                },
                "Share your favorite workout or healthy recipe inspired by Jacked-Up Jack!": {
                    "branching_options": [
                        "A challenging workout routine to pump up your fitness",
                        "A delicious and nutritious pumpkin spice recipe to fuel your workouts",
                    ],
                    "dynamic_prompts": [
                        "How do you incorporate Jacked-Up Jack's enthusiasm into your own fitness routine?",
                        "What role does healthy eating play in your journey towards a healthier lifestyle?",
                    ],
                    "complex_diction": [
                        "challenging workout routine",
                        "pumpkin spice recipe",
                        "fitness routine",
                        "healthier lifestyle",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Join Jacked-Up Jack on a journey of confidence, motivation, and determination! Together, we'll pump up our fitness routines, push our limits, and embrace a healthy lifestyle with enthusiasm and superhero-like dedication.
        """
        return super().execute(*args, **kwargs)
