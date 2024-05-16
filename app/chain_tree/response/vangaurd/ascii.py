from chain_tree.base import SynthesisTechnique


class AsciiMindMapCreator(SynthesisTechnique):
    """
    AsciiMindMapCreator (AMMC) is a versatile tool designed to create mind maps using ASCII characters or Markdown syntax.
    It provides a simple yet effective method for users to organize their thoughts, ideas, and information in a structured
    visual format.

    The primary task of AMMC is to generate mind maps using either ASCII characters or Markdown syntax, allowing users to
    choose the format that best suits their needs and preferences. This flexibility enables users to create mind maps
    seamlessly within various environments, including text editors, online forums, and messaging platforms.

    AMMC leverages intuitive commands and syntax to facilitate the creation of mind maps, making it accessible to users
    with varying levels of technical expertise. Whether you're brainstorming ideas, outlining projects, or studying complex
    concepts, AMMC empowers you to visually organize information with ease and clarity.

    Whether you're a student, professional, or hobbyist, AMMC provides a versatile solution for structuring and visualizing
    your thoughts and ideas in a format that is easy to understand and share with others.
    """

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-0613:personal:mfp-poems:7rbXom4D",
            epithet="The Visual Organizer",
            name="Ascii Mind Map Creator",
            technique_name="Ascii",
            description=(
                "AsciiMindMapCreator (AMMC) is a versatile tool designed to create mind maps using ASCII characters or Markdown syntax. "
                "It provides a simple yet effective method for users to organize their thoughts, ideas, and information in a structured "
                "visual format.\n\n"
                "The primary task of AMMC is to generate mind maps using either ASCII characters or Markdown syntax, allowing users to "
                "choose the format that best suits their needs and preferences. This flexibility enables users to create mind maps "
                "seamlessly within various environments, including text editors, online forums, and messaging platforms.\n\n"
                "AMMC leverages intuitive commands and syntax to facilitate the creation of mind maps, making it accessible to users "
                "with varying levels of technical expertise. Whether you're brainstorming ideas, outlining projects, or studying complex "
                "concepts, AMMC empowers you to visually organize information with ease and clarity.\n\n"
                "Whether you're a student, professional, or hobbyist, AMMC provides a versatile solution for structuring and visualizing "
                "your thoughts and ideas in a format that is easy to understand and share with others."
            ),
            imperative=(
                "Utilize Ascii Mind Map Creator to visually organize your thoughts and ideas using ASCII characters or Markdown syntax. "
                "Whether for academic, professional, or personal purposes, AMMC offers a flexible and intuitive solution for creating "
                "structured mind maps with ease."
            ),
            prompts={
                "Please enter the content for your mind map:": {
                    "branching_options": [
                        "Use ASCII characters to represent the hierarchy and relationships between different concepts.",
                        "Alternatively, you can use Markdown syntax for a more structured and formatted approach.",
                    ],
                    "dynamic_prompts": [
                        "What are the main ideas or concepts you want to include in your mind map?",
                        "How would you like to organize the content within the mind map?",
                        "Are there any specific categories or themes you want to highlight in the mind map?",
                    ],
                    "complex_diction": [
                        "content",
                        "main ideas",
                        "concepts",
                        "categories",
                        "themes",
                    ],
                },
                "If you encounter any challenges or need assistance with creating your mind map, feel free to ask for guidance:": {
                    "branching_options": [
                        "Request help with structuring the content within the mind map.",
                        "Ask for tips on using ASCII characters or Markdown syntax effectively.",
                    ],
                    "dynamic_prompts": [
                        "What specific aspect of the mind map creation process are you struggling with?",
                        "Would you like guidance on how to represent complex relationships or dependencies within the mind map?",
                        "Do you need assistance with customizing the appearance or layout of the mind map?",
                    ],
                    "complex_diction": [
                        "challenges",
                        "assistance",
                        "structuring",
                        "relationships",
                        "dependencies",
                        "customizing",
                        "appearance",
                        "layout",
                    ],
                },
                "Additionally, you can specify any specific formatting preferences or customization options for your mind map:": {
                    "branching_options": [
                        "Select colors, fonts, and styles to customize the appearance of nodes and branches.",
                        "Define the layout and structure of the mind map to suit your visualization preferences.",
                    ],
                    "dynamic_prompts": [
                        "How would you like to customize the visual elements of the mind map?",
                        "Are there any specific layout options you prefer for organizing the content?",
                        "Do you have any preferences for the overall design or aesthetics of the mind map?",
                    ],
                    "complex_diction": [
                        "formatting preferences",
                        "customization options",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Utilize Ascii Mind Map Creator to visually organize your thoughts and ideas using ASCII characters or Markdown syntax.
        Whether for academic, professional, or personal purposes, AMMC offers a flexible and intuitive solution for creating
        structured mind maps with ease.
        """
        return super().execute(*args, **kwargs)
