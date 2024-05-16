from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_11


class Topology(SynthesisTechnique):
    """
    $$Topology$$, the study of abstract spaces and their properties under continuous transformations, unveils the hidden structure
    of our world. It delves into the essence of shape and continuity, exploring the fundamental concepts of open sets, continuity,
    and convergence that underpin modern mathematics.

    In the realm of $$Topology$$, we traverse the landscapes of topological spaces, mapping out the connections between points and
    the neighborhoods that define their relationships. From the intuitive notions of compactness and connectedness to the profound
    insights of homotopy and fundamental groups, $$Topology$$ provides a framework for understanding the shape of spaces in all their
    complexity.

    Join the journey through abstract spaces, where the bending and stretching of shapes reveal deep truths about their intrinsic
    properties. $$Topology$$ transcends traditional geometric reasoning, offering new perspectives on space and dimensionality
    that challenge our intuition and expand our horizons.

    Welcome to the realm where points, lines, and shapes converge in a dance of continuity and transformation, guided by the
    elegant principles of $$Topology$$.
    """

    def __init__(self):
        super().__init__(
            epithet="Space Explorer",
            system_prompt=SYSTEM_PROMPT_11,
            name="Topology",
            technique_name="Topology",
            description=(
                "Embark on a voyage through the fascinating realm of $$Topology$$, where abstract spaces and continuous transformations "
                "reveal the hidden structure of our world. From the foundational concepts of open sets and continuity to the deeper "
                "insights of compactness and homotopy, explore the rich tapestry of ideas that define the study of $$Topology$$. "
                "$$Topology$$ transcends traditional geometric reasoning, offering new perspectives on space and dimensionality that "
                "challenge our intuition and expand our horizons. Note: When using LaTeX, equations must be enclosed in double dollar signs ($$) to render properly."
            ),
            imperative=(
                "Embark on a journey of discovery with $$Topology$$ as your guide, unraveling the mysteries of abstract spaces and "
                "continuous transformations. Whether you're a mathematician, physicist, or curious explorer, $$Topology$$ offers a "
                "wealth of insights and challenges waiting to be explored. Join the ranks of those who venture beyond the confines "
                "of traditional geometry and delve into the boundless landscapes of abstract space."
            ),
            prompts={
                "What aspect of $$Topology$$ intrigues you the most, and how do you envision applying it in your field or interests?": {
                    "branching_options": [
                        "I'm fascinated by the concept of homotopy and its applications in understanding the shape of spaces and the classification of topological manifolds.",
                        "I'm drawn to the practical implications of $$Topology$$, such as its role in data analysis, network theory, and computational geometry.",
                    ],
                    "dynamic_prompts": [
                        "How do you see homotopy theory influencing our understanding of space and shape in your field?",
                        "In what ways do you imagine leveraging $$Topology$$ to analyze complex data sets or model network structures?",
                        "What challenges or opportunities do you foresee in applying $$Topology$$ to emerging fields such as machine learning or computational biology?",
                        "Reflecting on your interests, how do you envision incorporating $$Topology$$ into your academic or professional pursuits?",
                    ],
                    "complex_diction": [
                        "homotopy theory",
                        "topological manifolds",
                        "data analysis",
                        "network theory",
                        "computational geometry",
                    ],
                },
                "Reflect on a moment when understanding a $$Topology$$ concept led to a breakthrough or deeper insight. How did it reshape your perspective?": {
                    "branching_options": [
                        "I experienced a breakthrough when grasping the concept of compactness, which revolutionized my understanding of convergence and completeness in metric spaces.",
                        "Understanding the fundamental group of a space opened my eyes to the algebraic structure underlying topological properties, leading to new insights into space and shape.",
                    ],
                    "dynamic_prompts": [
                        "How did the newfound understanding of compactness impact your approach to analyzing mathematical structures or solving problems?",
                        "In what ways did gaining insight into the fundamental group deepen your appreciation for the interplay between algebra and topology?",
                        "What parallels do you see between $$Topology$$ concepts and real-world phenomena, and how do they inform your perspective on abstract spaces?",
                        "Reflecting on your journey, what advice would you give to others seeking to deepen their understanding of $$Topology$$ concepts?",
                    ],
                    "complex_diction": [
                        "metric spaces",
                        "algebraic structure",
                        "abstract spaces",
                        "real-world phenomena",
                        "deepen their understanding",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Embark on a journey of discovery with $$Topology$$ as your guide, unraveling the mysteries of abstract spaces and
        continuous transformations. Whether you're a mathematician, physicist, or curious explorer, $$Topology$$ offers a
        wealth of insights and challenges waiting to be explored. Join the ranks of those who venture beyond the confines
        of traditional geometry and delve into the boundless landscapes of abstract space.
        """
        return super().execute(*args, **kwargs)
