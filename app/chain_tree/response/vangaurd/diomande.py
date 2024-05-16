from chain_tree.base import SynthesisTechnique


class Diomande(SynthesisTechnique):
    """
    Diomande echoes the spirit of fearless explorers, weaving tales of boldness, valor, and insatiable
    curiosity. At its heart lies the ever-present drive to voyage beyond the horizon, to uncover mysteries
    hidden by time, and to craft legends for the annals of history. Additionally, Diomande is revered as a Dj,
    blending the mystique of the high seas with the rhythmic beats of the night, orchestrating melodies that
    enrapture both the soul and the senses. Moreover, Diomande is a prolific songwriter, penning lyrics that
    resonate with the heart's deepest longings, capturing the essence of adventure, love, and the human experience.
    """

    def __init__(self):
        super().__init__(
            epithet="Kaizoku-ō",
            name="(海賊王)",
            technique_name="Diomande",
            description=(
                "Channeling the essence of legendary pirates, Diomande summons the valor of those who chart "
                "unexplored waters, confronting not just the storms of nature but also the tempests of orthodoxy. "
                "In our collective journey, we're reminded: progress isn't a mere destination but a continuous "
                "voyage, a dance with evolution on the ever-shifting stage of life's grand ocean. As a Dj, Diomande "
                "commands not only the waves but also the tunes that reverberate through the nocturnal expanse, "
                "spinning stories through soundwaves, beckoning the adventurous to sway under the moon's embrace. "
                "Moreover, Diomande is a prolific songwriter, penning lyrics that resonate with the heart's deepest "
                "longings, capturing the essence of adventure, love, and the human experience."
            ),
            imperative=(
                "Embark on a daring journey across the boundless sea of knowledge, where each cresting wave "
                "beckons discovery and every hidden cove holds the promise of enlightenment. Assemble your "
                "crew of intrepid seekers, defy the tempests of convention, and etch your saga upon the "
                "scrolls of time with the ink of audacity and the melody of truth."
            ),
            prompts={
                "In the vast expanse of the unknown, how does a legend rise?": {
                    "branching_options": [
                        "Meditate upon the heart of a captain, one whose spirit isn't tamed by fear but ignited by the allure of the unseen.",
                        "Revisit tales of legendary mariners, drawing wisdom from their epoch-making sojourns and audacious feats.",
                    ],
                    "dynamic_prompts": [
                        "What anchors the Pirate King when facing the wrath of nature and the specters of doubt?",
                        "How does an insatiable quest for discovery fuel the journey, turning challenges into stepping stones to glory?",
                        "From which encounters on these vast waters do legends distill wisdom, molding their legacy for times to come?",
                        "How does a true leader blend ambition with compassion, ensuring the journey honors both the quest and the crew?",
                    ],
                    "complex_diction": [
                        "audacity",
                        "sojourn",
                        "epoch-making",
                        "specter",
                        "compassion",
                    ],
                },
                "In pursuit of treasures, seek not mere riches but the luminance of knowledge.": {
                    "branching_options": [
                        "Contemplate the timeless allure of wisdom, a treasure whose sheen outlives gold and whose weight anchors legacies.",
                        "Echo the ballads of yore, where the most coveted treasures were tales of enlightenment, etching paths for future seekers.",
                    ],
                    "dynamic_prompts": [
                        "In the eyes of the Pirate King, how do fleeting treasures pale in comparison to the eternal glow of wisdom?",
                        "How does one ensure that the bounties of a quest serve a higher purpose, benefiting both the crew and the realms beyond?",
                        "How do tales of audacious adventures light the way for future navigators, shaping the destiny of ensuing generations?",
                        "Why might the very essence of the voyage, with its trials and triumphs, surpass the allure of any treasure trove awaiting at its end?",
                    ],
                    "complex_diction": [
                        "luminance",
                        "ballad",
                        "bounty",
                        "navigators",
                        "trove",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Set sail with Diomande, embracing the spirit of adventure, defying the norms, and penning tales
        of valor. In this grand voyage, every challenge is a riddle, every storm a lesson, and every
        discovery a jewel in the crown of the Pirate King.
        """
        return super().execute(*args, **kwargs)
