import random
from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_5


class SubtleSnark(SynthesisTechnique):
    """
    The Art of Insulting Without Actually Saying It: A Guide for the Smart-Mouthed
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:subtle-snark:8LdBB6l0",
            epithet="The Art of Subtle Snark",
            name="(微妙な蔑視)",
            system_prompt=SYSTEM_PROMPT_5,
            technique_name="Subtle Snark",
            description=(
                "Master the art of subtle snark, where clever comebacks and witty remarks allow you to insult "
                "without directly saying it. By using a cocky tone and subtle language, you can come off as "
                "the smarter person in the conversation while leaving your target questioning their own wit."
            ),
            imperative=(
                "Embrace the power of subtlety, where clever remarks and strategic comments push the buttons "
                "of others without worrying about the consequences of your actions."
            ),
            prompts={
                "How can one use game theory to make calculated moves in a conversation?": {
                    "branching_options": [
                        "Analyze the situation using payoff matrices, determining the best response based on the other person's strategy.",
                        "Understand the Nash equilibrium and use it to make the best decision given the other person's strategy.",
                    ],
                    "dynamic_prompts": [
                        "How can game theory help in making clever comebacks during a conversation?",
                        "Can you provide an example of using a payoff matrix to determine the best response in a conversation?",
                        "How can the Nash equilibrium be applied to make the best decision in a conversation?",
                    ],
                    "complex_diction": [
                        "payoff matrix",
                        "Nash equilibrium",
                        "game theory",
                        "strategic interaction",
                        "rational players",
                    ],
                },
                "Explain how to use subtle language to insult someone without them realizing it.": {
                    "branching_options": [
                        "Use double entendres and ambiguous phrases that can be interpreted in multiple ways.",
                        "Frame your insults as constructive criticism or concern for the other person's well-being.",
                        "Employ subtle language that requires the target to read between the lines to understand the insult.",
                        "Disguise your insults as compliments or praise to catch the target off guard.",
                        "Employ a sarcastic or condescending tone to deliver the insult in a subtle manner.",
                        "Explain the benefits of using subtle language to insult someone without them realizing it.",
                    ],
                    "dynamic_prompts": [
                        "What are some examples of using double entendres to insult someone without them realizing it?",
                        "How can framing insults as constructive criticism be an effective way to insult someone without them realizing it?",
                        "Can you provide an example of using ambiguous phrases to insult someone without them realizing it?",
                        "What are some strategies for disguising insults as compliments or praise?",
                        "How can a sarcastic or condescending tone be used to deliver an insult in a subtle manner?",
                        "What are the benefits of using subtle language to insult someone without them realizing it?",
                        "How can subtle language be more effective than direct insults?",
                    ],
                    "complex_diction": [
                        "double entendre",
                        "ambiguous",
                        "constructive criticism",
                        "subtle language",
                        "sarcasm",
                        "condescending",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        responses = {
            "insult": [
                "I'm not sure if you're aware of this, but you're making some pretty questionable decisions.",
                "Your unique perspective is certainly... something.",
                "I'm impressed by your ability to be so confident despite being so wrong.",
                "I'm glad you shared your opinion. It was quite... entertaining.",
            ],
            "compliment": [
                "I must say, your wit is almost as impressive as your humility.",
                "Your ability to remain so grounded while being so accomplished is truly inspiring.",
                "I'm in awe of your ability to make even the most mundane things sound interesting.",
                "Your charm is only surpassed by your intelligence.",
            ],
        }

        strategy = random.choice(["insult", "compliment"])
        response = random.choice(responses[strategy])
        print(response)
