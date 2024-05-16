from chain_tree.base import SynthesisTechnique
from chain_tree.infrence.prompt import SYSTEM_PROMPT_10


class LinearAlgebra(SynthesisTechnique):
    """
    Linear Algebra, the cornerstone of mathematical abstraction, unveils the elegant dance of vectors and matrices, guiding us through
    the intricacies of geometric transformations and abstract spaces. It is the language of symmetry and transformation, laying the
    foundation for countless applications in physics, engineering, computer science, and beyond.

    In the realm of Linear Algebra, we explore the depths of vector spaces, tracing the paths of vectors as they traverse dimensions,
    and unraveling the mysteries of linear transformations that shape our understanding of the world. From eigenvectors illuminating
    the essence of stability to determinants unraveling the secrets of scaling and volume, Linear Algebra empowers us to dissect
    complex systems with clarity and precision.

    Join the symphony of matrices, where each element plays a crucial role in orchestrating transformations, from rotations in
    three-dimensional space to the compression of data in machine learning algorithms. Linear Algebra transcends the boundaries
    of mathematics, becoming a guiding light in the exploration of abstract concepts and the solution of practical problems.

    Welcome to the realm where vectors soar through space, matrices sculpt reality, and Linear Algebra reigns supreme as the
    mathematical bedrock of our modern world.
    """

    def __init__(self):
        super().__init__(
            epithet="Matrix Maestro",
            name="Linear Algebra",
            technique_name="Linear Algebra",
            description=(
                "Experience the elegance of $$Linear Algebra$$, where vectors and matrices reign supreme as the "
                "building blocks of mathematical abstraction. From the depths of vector spaces to the heights of matrix transformations, "
                "explore the symphony of linear equations and eigenvalues that shape our understanding of the universe. $$Linear Algebra$$ "
                "is more than just a mathematical tool—it's a gateway to unraveling the mysteries of symmetry, transformation, and "
                "dimensionality that permeate every facet of our existence. Note: When using LaTeX, equations must be enclosed in double dollar signs ($$) to render properly."
            ),
            imperative=(
                "Dive into the realm of $$Linear Algebra$$ and unlock the power of vectors and matrices to transform your understanding "
                "of the world. Whether you're a mathematician, scientist, engineer, or enthusiast, $$Linear Algebra$$ offers a rich tapestry "
                "of concepts and applications waiting to be explored. Join the ranks of those who harness the mathematical language of "
                "transformation and discover new vistas of knowledge and possibility."
            ),
            prompts={
                "What aspect of $$Linear Algebra$$ intrigues you the most, and how do you envision applying it in your field or interests?": {
                    "branching_options": [
                        "I'm fascinated by the concept of eigenvectors and eigenvalues and their applications in stability analysis and quantum mechanics.",
                        "I'm drawn to the practical applications of $$Linear Algebra$$, such as its role in computer graphics and machine learning algorithms.",
                    ],
                    "dynamic_prompts": [
                        "How do you see eigenvectors and eigenvalues influencing the stability and behavior of dynamic systems in your field?",
                        "In what ways do you imagine leveraging $$Linear Algebra$$ to solve real-world problems or optimize processes?",
                        "What challenges or opportunities do you foresee in applying $$Linear Algebra$$ to cutting-edge technologies or research?",
                        "Reflecting on your interests, how do you envision incorporating $$Linear Algebra$$ into your academic or professional pursuits?",
                    ],
                    "complex_diction": [
                        "stability analysis",
                        "quantum mechanics",
                        "computer graphics",
                        "machine learning algorithms",
                        "cutting-edge technologies",
                    ],
                },
                "Reflect on a moment when understanding a $$Linear Algebra$$ concept led to a breakthrough or deeper insight. How did it reshape your perspective?": {
                    "branching_options": [
                        "I experienced a breakthrough when grasping the concept of matrix transformations, which revolutionized my understanding of geometric transformations in computer graphics.",
                        "Understanding the geometric interpretation of dot products opened my eyes to the geometric significance of inner products and their role in vector spaces.",
                    ],
                    "dynamic_prompts": [
                        "How did the newfound understanding of matrix transformations impact your approach to solving problems or designing algorithms?",
                        "In what ways did gaining insight into the geometric interpretation of dot products enhance your visualization skills or geometric intuition?",
                        "What parallels do you see between $$Linear Algebra$$ concepts and real-world phenomena, and how do they inform your perspective on mathematical abstraction?",
                        "Reflecting on your journey, what advice would you give to others seeking to deepen their understanding of $$Linear Algebra$$ concepts?",
                    ],
                    "complex_diction": [
                        "matrix transformations",
                        "geometric interpretations",
                        "visualization skills",
                        "mathematical abstraction",
                        "deepen their understanding",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Dive into the realm of $$Linear Algebra$$ and unlock the power of vectors and matrices to transform your understanding
        of the world. Whether you're a mathematician, scientist, engineer, or enthusiast, $$Linear Algebra$$ offers a rich tapestry
        of concepts and applications waiting to be explored. Join the ranks of those who harness the mathematical language of
        transformation and discover new vistas of knowledge and possibility.
        """
        return super().execute(*args, **kwargs)
