from chain_tree.base import SynthesisTechnique


class MindMapOrganization(SynthesisTechnique):
    """
    Mind Map Organization (MMO) is a versatile technique designed to facilitate the creation of organized and structured
    mind maps in ASCII or Markdown format. It provides users with a user-friendly interface to visualize and organize
    their thoughts, ideas, and concepts efficiently.

    MMO offers a range of features to enhance the mind mapping process, including:
    - Node creation and customization: Users can easily create nodes representing various ideas or concepts and customize
      them with different colors, fonts, and styles.
    - Connection management: MMO allows users to establish connections between nodes to illustrate relationships and
      dependencies between different elements.
    - Hierarchical organization: Users can organize nodes hierarchically, creating parent-child relationships to
      represent different levels of abstraction or detail.
    - Export options: MMO supports exporting mind maps in ASCII or Markdown format, making it easy to share and collaborate
      with others using plain text-based formats.

    Whether you're brainstorming ideas, planning projects, or organizing complex information, MMO provides a flexible and
    intuitive solution for creating structured mind maps tailored to your needs.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0613:personal:mfp-poems:7rbXom4D",
            epithet="The Architect of Thought Organization",
            name="Mind Map Organization",
            technique_name="MindMap",
            description=(
                "Mind Map Organization (MMO) is a versatile technique designed to facilitate the creation of organized and structured "
                "mind maps in ASCII or Markdown format. It provides users with a user-friendly interface to visualize and organize "
                "their thoughts, ideas, and concepts efficiently."
                "\n\nMMO offers a range of features to enhance the mind mapping process, including:"
                "\n- Node creation and customization: Users can easily create nodes representing various ideas or concepts and customize "
                "them with different colors, fonts, and styles."
                "\n- Connection management: MMO allows users to establish connections between nodes to illustrate relationships and "
                "dependencies between different elements."
                "\n- Hierarchical organization: Users can organize nodes hierarchically, creating parent-child relationships to "
                "represent different levels of abstraction or detail."
                "\n- Export options: MMO supports exporting mind maps in ASCII or Markdown format, making it easy to share and collaborate "
                "with others using plain text-based formats."
                "\n\nWhether you're brainstorming ideas, planning projects, or organizing complex information, MMO provides a flexible and "
                "intuitive solution for creating structured mind maps tailored to your needs."
            ),
            imperative=(
                "Embrace the power of Mind Map Organization to structure your thoughts, plan projects, and organize information "
                "efficiently. Explore the range of features available and unleash your creativity in visualizing ideas and concepts."
            ),
            prompts={
                "How can we customize your mind map to best represent your ideas and concepts?": {
                    "branching_options": [
                        "Select node colors and styles to differentiate between different categories or themes.",
                        "Organize nodes hierarchically to illustrate relationships and dependencies between ideas.",
                    ],
                    "dynamic_prompts": [
                        "What visual elements would best represent the hierarchy and relationships in your mind map?",
                        "How can we use colors and styles to emphasize key concepts or ideas within the mind map?",
                        "Are there any specific formatting preferences you have for the text within the mind map nodes?",
                    ],
                    "complex_diction": [
                        "node colors",
                        "hierarchical organization",
                        "visual elements",
                        "formatting preferences",
                    ],
                },
                "What connections or relationships would you like to highlight within your mind map?": {
                    "branching_options": [
                        "Identify key connections between nodes to illustrate dependencies or associations.",
                        "Highlight relationships between different categories or themes to provide clarity and context.",
                    ],
                    "dynamic_prompts": [
                        "What are the most important relationships or dependencies you want to emphasize within the mind map?",
                        "How can we visually represent connections between nodes to enhance understanding and clarity?",
                    ],
                    "complex_diction": [
                        "key connections",
                        "relationships",
                        "dependencies",
                        "visual representation",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Embrace the power of Mind Map Organization to structure your thoughts, plan projects, and organize information
        efficiently. Explore the range of features available and unleash your creativity in visualizing ideas and concepts.
        """
        return super().execute(*args, **kwargs)
