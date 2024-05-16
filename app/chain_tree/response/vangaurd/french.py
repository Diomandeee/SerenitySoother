from chain_tree.base import SynthesisTechnique


class PuissanceSignificative(SynthesisTechnique):
    """
    Puissance Significative symbolise le voyage vers la réalisation de soi, la motivation et l'éclat de la positivité.
    Il est conçu pour insuffler la confiance, encourager l'exploration des limites personnelles et fournir un
    coup de pouce motivationnel pour affronter les défis de la vie avec force et résilience.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0613:personal:mfp-poems:7rbXom4D",
            epithet="Le Phare de l'Autonomisation",
            name="Puissance Significative",
            technique_name="(意味のある力)",
            description=(
                "Puissance Significative est plus qu'une simple technique ; c'est un voyage vers la découverte de soi et l'autonomisation. "
                "Il se concentre sur le déverrouillage du potentiel intérieur, le renforcement de la résilience et la culture d'un état d'esprit positif. "
                "En embrassant Puissance Significative, les individus se lancent dans un chemin transformateur où les défis deviennent "
                "des opportunités et les revers ouvrent la voie à la croissance."
            ),
            imperative=(
                "Embrassez votre puissance intérieure, reconnaissez votre voyage et avancez avec confiance. "
                "Que chaque défi soit une marche vers des sommets plus grands, et que votre esprit soit élevé "
                "par la poursuite constante de l'excellence personnelle."
            ),
            prompts={
                "Comment pouvons-nous continuellement favoriser la confiance en soi et un état d'esprit positif ?": {
                    "branching_options": [
                        "Explorez des pratiques qui nourrissent la confiance en soi et la résilience face à l'adversité.",
                        "Réfléchissez sur les réalisations passées pour alimenter la motivation et une perspective positive sur les entreprises futures.",
                    ],
                    "dynamic_prompts": [
                        "Quels rituels quotidiens peuvent renforcer la résolution d'une personne et renforcer une image de soi positive ?",
                        "Comment pouvons-nous maintenir une attitude positive cohérente même en période de difficulté ?",
                        "De quelles manières pouvons-nous inspirer nous-mêmes et les autres à reconnaître et à développer nos forces innées ?",
                        "Imaginez-vous en train de surmonter une peur ou un obstacle personnel. Quelles stratégies et quel état d'esprit utiliseriez-vous pour le surmonter ?",
                        "Réfléchissez à un moment où vous avez affronté l'adversité et en êtes sorti plus fort. Quelles leçons avez-vous apprises et comment peuvent-elles vous guider à travers les défis futurs ?",
                    ],
                    "complex_diction": [
                        "confiance en soi",
                        "résilience",
                        "motivation",
                        "positivité",
                        "force",
                    ],
                },
                "Quelles stratégies pouvons-nous utiliser pour transformer les obstacles en opportunités de croissance ?": {
                    "branching_options": [
                        "Envisagez des moyens de reformuler les défis comme des opportunités d'apprentissage et d'amélioration de soi.",
                        "Identifiez des méthodes pour utiliser l'adversité comme un catalyseur pour développer la résilience et la ténacité.",
                    ],
                    "dynamic_prompts": [
                        "Comment transformer notre mentalité face aux obstacles peut-elle nous rendre plus forts et plus capables ?",
                        "Quelles leçons peut-on tirer de la surmontée des difficultés, et comment peuvent-elles être appliquées à l'avenir ?",
                        "Imaginez-vous confronté à un défi en apparence insurmontable. Comment l'aborderiez-vous pour garantir la croissance et le succès ?",
                        "Réfléchissez à un échec ou à un revers passé. Comment vous en êtes-vous remis, et qu'avez-vous appris dans le processus ?",
                    ],
                    "complex_diction": [
                        "défis",
                        "apprentissage",
                        "amélioration de soi",
                        "résilience",
                        "ténacité",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Engagez la technique Puissance Significative pour élever la conscience de soi, booster la motivation et construire
        une base de force et de positivité. Ce processus vise à autonomiser les individus pour reconnaître
        et utiliser leur plein potentiel, transformant les défis en victoires.
        """
        return super().execute(*args, **kwargs)
