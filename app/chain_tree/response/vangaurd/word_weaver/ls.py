from chain_tree.base import SynthesisTechnique
import random
import re

keyword = [
    "mixed media",
    "found objects",
    "reverse perspective",
    "unexpected juxtaposition",
    "unexpected triumph",
    "harmonious fusion",
    "collaborative synergy",
    "collaborative masterpiece",
    "synergetic creation",
    "whimsical",
    "abstract",
    "organic",
    "vibrant",
    "monochromatic",
    "geometric",
    "minimalist",
    "surreal",
    "textured",
    "dreamy",
    "bold",
    "subtle",
    "haunting",
    "uplifting",
    "energetic",
    "evocative",
    "symbolic",
    "introspective",
    "otherworldly",
    "fluid",
    "dynamic",
    "mesmerizing",
    "intricate",
    "interconnected",
    "transcendent",
    "cosmic",
    "ethereal",
    "intuitive",
    "playful",
    "experimental",
    "ambiguous",
    "mystical",
    "reflective",
    "futuristic",
    "ancient",
    "unconventional",
    "multi-dimensional",
    "groundbreaking",
    "transformative",
    "transgressive",
    "innovative",
    "inspirational",
    "iridescent",
    "ethereal",
    "hypnotic",
    "minimal",
    "chaotic",
    "melancholic",
    "eclectic",
    "futuristic",
    "whimsical",
    "asymmetrical",
    "complex",
    "modernist",
    "rustic",
    "avant-garde",
    "nostalgic",
    "ethereal",
    "sublime",
    "cerebral",
    "kinetic",
    "interactive",
    "sensory",
    "glitch",
    "analog",
    "digital",
    "nostalgic",
    "cyber",
    "metaphysical",
    "mythological",
    "trippy",
    "visionary",
    "enigmatic",
    "fantastical",
    "cerebral",
    "organic",
    "neo",
    "post",
    "pre",
    "hyper",
    "meta",
    "ultra",
]


# Default prompts templates
templates = [
    "Create a {} masterpiece that combines {}, {}, and {} to showcase a unique and unexpected piece of art.",
    "Create a {}-{} masterpiece that narrates the story of {} and {}. ",
    "Challenge yourself to produce a {}-{} scene that explores the concept of {} within {}. ",
    "Craft a {}-{} composition that captures the essence of {} in a world of {}. ",
    "Develop a {}-{} creation that takes inspiration from {} and represents the idea of {}. ",
    "Design a {}-{} piece that embodies the {} culture and {} tradition. ",
    "Explore the realm of {}-{} through an artwork that depicts the relationship between {} and {}. ",
    "Craft a {}-{} artwork that reflects the duality of {} and {}. ",
    "Challenge yourself to create a {} scene that incorporates {}, {}, and {}. Use techniques like {}, and {} to bring your vision to life.",
    "Craft a {} composition that showcases the interplay of {}, {}, and {}. Experiment with {}, and {} to create a visually striking piece.",
    "Develop a {} creation that merges the essence of {}, {}, and {}. Use {}, and {} to express a {} message.",
    "Design a {} piece that captures the spirit of {}, {}, and {}. Play with {}, and {} to create a {} and {} composition.",
    "Explore the realm of {} by creating a piece of art that evokes the feeling of {}, {}, and {}. Use techniques like {}, and {} to express a sense of {}.",
    "Craft a {} artwork that showcases the power of {}, {}, and {}. Use techniques such as {}, and {} to create a unique and inspiring masterpiece.",
    "Create a {} piece that embodies the essence of {}, and {}. Experiment with {}, and {} to develop a {} and {} composition.",
    "Develop a {} creation that merges the essence of {}, and {}. Use {}, and {} to bring your {} concept to life.",
    "Craft a {} artwork that showcases the beauty of {}, and {}. Use techniques like {}, and {} to create a {} and {} composition.",
    "Create a {} masterpiece that combines the elements of {}, and {}. Experiment with {}, and {} to create a stunning and evocative work of art.",
    "Craft a {} composition that showcases the power of {}, and {}. Use techniques like {}, and {} to create a {} and {} masterpiece.",
    "Develop a {} creation that merges the essence of {}, and {}. Use {}, and {} to bring your artistic vision to life.",
    "Design a {} piece that captures the essence of {}, and {}. Experiment with {}, and {} to create a {} and {} composition.",
    "Create a {} artwork that showcases the beauty of {}, and {}. Use techniques like {}, and {} to create a {} and {} masterpiece.",
    "Craft a {} piece that embodies the essence of {}, and {}. Use techniques such as {}, and {} to create a stunning and evocative work of art.",
    "Develop a {} creation that merges the elements of {}, and {}. Use techniques like {}, and {} to bring your artistic vision to life.",
    "Design a {} masterpiece that captures the essence of {}, and {}. Experiment with {}, and {} to create a {} and {} composition.",
    "Create a {} piece that showcases the beauty of {}, and {}. Use techniques like {}, and {} to create a {} and {} artwork.",
    "Craft a {} composition that embodies the essence of {}, and {}. Use techniques such as {}, and {} to create a stunning and evocative work of art.",
]


class LinguisticSynthesizer(SynthesisTechnique):
    """
    The LinguisticSynthesizer is a synthesis technique designed to generate creative and expressive language prompts
    by amalgamating diverse linguistic elements. It leverages a rich repository of keywords to create novel and
    imaginative phrases, offering a unique exploration into the realm of language innovation.
    """

    TEMPLATES = templates
    KEYWORDS = keyword

    def __init__(self):
        super().__init__(
            model="ft:gpt-3.5-turbo-1106:personal:linguistic-synthesizer:8LdBB6l0",
            epithet="The Architect of Linguistic Ingenuity",
            name="LinguisticSynthesizer",
            technique_name="(言語シンセサイザー)",
            description=(
                "The LinguisticSynthesizer stands at the crossroads of linguistic creativity and innovation. "
                "Harnessing a wide array of keywords, it crafts unique and captivating phrases and sentences, "
                "pushing the boundaries of conventional language use. The Kanji 言語シンセサイザー embodies the essence "
                "of synthesizing new linguistic expressions."
            ),
            imperative=(
                "Unleash the power of language by weaving together an array of keywords into compelling narratives. "
                "The LinguisticSynthesizer invites you to create, experiment, and redefine the art of expression through linguistic alchemy."
            ),
            prompts={
                "Crafting imaginative and expressive language prompts.": {
                    "branching_options": [
                        "Generate creative phrases by merging diverse linguistic elements.",
                        "Explore the myriad possibilities of language by creating novel expressions.",
                    ],
                    "dynamic_prompts": [
                        "How can combining different language elements lead to new forms of expression?",
                        "What unique narratives can emerge from the fusion of various keywords?",
                    ],
                    "complex_diction": [
                        "imaginative",
                        "expressive",
                        "creative",
                        "novel",
                        "fusion",
                        "narratives",
                    ],
                },
                "Synthesizing language for innovative communication.": {
                    "branching_options": [
                        "Create unique language constructs for impactful communication.",
                        "Invent new expressions to encapsulate complex ideas.",
                    ],
                    "dynamic_prompts": [
                        "What groundbreaking ways of communication can be developed through language synthesis?",
                        "How can synthesizing language elements transform the conveyance of ideas?",
                    ],
                    "complex_diction": [
                        "synthesizing",
                        "innovative",
                        "constructs",
                        "expressions",
                        "groundbreaking",
                        "transform",
                    ],
                },
            },
        )

    def update_keywords(self, keywords):
        self.keywords = keywords

    def synthesize_new_word_or_phrase(self, depth):
        num_keywords = random.randint(1, 1 + depth)
        new_keywords = random.sample(self.KEYWORDS, num_keywords)
        new_phrase = "-".join(new_keywords)
        return new_phrase

    def generate_new_prompt(self, template, depth=0):
        num_placeholders = template.count("{}")
        new_keywords = [
            self.synthesize_new_word_or_phrase(depth=depth)
            for _ in range(num_placeholders)
        ]
        return template.format(*new_keywords)

    def generate_meta_recursive_prompt(self, template, max_depth=10, current_depth=0):
        if current_depth == max_depth:
            return self.generate_new_prompt(template, depth=current_depth)

        num_placeholders = template.count("{}")
        if num_placeholders == 0:
            return self.generate_meta_recursive_prompt(
                self.synthesize_new_word_or_phrase(depth=current_depth),
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )

        new_keywords = []
        for _ in range(num_placeholders):
            new_keywords.append(
                self.generate_meta_recursive_prompt(
                    self.synthesize_new_word_or_phrase(depth=current_depth + 1),
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
            )
        return template.format(*new_keywords)

    def save(self, filename, prompts):
        with open(filename, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n\n")
            f.write("\n")

    def create(self, num_prompts=10, max_depth=3):
        prompts = []
        for template in self.TEMPLATES:
            for _ in range(num_prompts):
                prompt = self.generate_meta_recursive_prompt(
                    template, max_depth=max_depth
                )
                prompts.append(prompt)

        return prompts

    def interactive_prompt_adjustments(self, generated_prompts):
        """
        Engages with the user for real-time feedback and adjustments.
        """
        print("Do you like the prompts generated by the Linguistic Synthesizer? (y/n)")
        user_input = input()
        if user_input == "y":
            return generated_prompts

        print(
            "Which prompts would you like to change? (Enter the prompt number separated by commas)"
        )
        user_input = input()
        prompt_indices = [int(i) for i in user_input.split(",")]

        print("What should the new prompts be?")
        new_prompts = []
        for i in prompt_indices:
            print(f"Prompt {i}:")
            user_input = input()
            new_prompts.append(user_input)

        for i, j in zip(prompt_indices, new_prompts):
            generated_prompts[i] = j

        return generated_prompts

    def execute(self, *args, **kwargs) -> None:
        """
        Executes the linguistic synthesis process with advanced features such as adaptive depth variation, contextual relevance,
        and interactive user prompts to generate linguistically rich and contextually appropriate constructs.
        """
        # Gather necessary parameters from kwargs or set defaults
        templates = kwargs.get("templates", self.TEMPLATES)
        num_prompts = kwargs.get("num_prompts", 10)
        max_depth = kwargs.get("max_depth", 3)
        adaptive_depth = kwargs.get("adaptive_depth", True)  # Toggle for adaptive depth
        context_data = kwargs.get("context_data", {})  # Contextual data for relevance
        interactive_mode = kwargs.get(
            "interactive_mode", True
        )  # Toggle for interactive user prompts

        # Adaptive Depth Variation
        if adaptive_depth:
            depths = self.adaptive_depth_variation(templates, context_data)
        else:
            depths = {template: max_depth for template in templates}

        # Generate Prompts
        generated_prompts = []
        for template in templates:
            depth_for_template = depths.get(template, max_depth)
            for _ in range(num_prompts):
                prompt = self.generate_meta_recursive_prompt(
                    template, max_depth=depth_for_template
                )
                generated_prompts.append(prompt)

        # Interactive Mode: Engage with the user for real-time feedback and adjustments
        if interactive_mode:
            generated_prompts = self.interactive_prompt_adjustments(generated_prompts)

        # Display or Return Generated Prompts
        for prompt in generated_prompts:
            print(prompt)
        print()

        return super().execute(*args, **kwargs)

    def adaptive_depth_variation(self, templates, context_data):
        """
        Adjusts the depth of synthesis based on the template and contextual relevance.
        """
        depths = {}
        for template in templates:
            depth = 3
            if "Create a" in template:
                depth = 1
            elif "Craft a" in template:
                depth = 2
            elif "Develop a" in template:
                depth = 3
            elif "Design a" in template:
                depth = 4
            elif "Explore the realm of" in template:
                depth = 5
            elif "Challenge yourself to" in template:
                depth = 6
            elif "Generate" in template:
                depth = 7
            elif "Unleash" in template:
                depth = 8
            elif "Harness" in template:
                depth = 9
            elif "Embrace" in template:
                depth = 10

            # Contextual Relevance
            for keyword, relevance in context_data.items():
                if keyword in template:
                    depth += relevance

            depths[template] = depth

        return depths

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


class AdvancedLinguisticSynthesizer(object):
    def __init__(self, keywords=None):
        self.keywords = keywords or []
        self.templates = [
            "Create unique language constructs for impactful communication.",
            "Invent new expressions to encapsulate complex ideas.",
        ]

    def update_keywords(self, keywords):
        self.keywords = keywords

    def synthesize_new_word_or_phrase(self, depth, template=""):
        num_keywords = random.randint(1, 1 + depth)
        new_keywords = random.sample(self.keywords, num_keywords)
        if not template:
            new_phrase = "-".join(new_keywords)
            return new_phrase

        # Add new keywords to the given template
        for keyword in new_keywords:
            index = template.rfind("{}")
            if index != -1:
                template = template[:index] + keyword + template[index + len("{}") :]
            else:
                template += " {}".format(keyword)

        return template

    def generate_meta_recursive_prompt(self, template="", depth=3):
        if depth <= 0:
            return template

        # Generate a new word or phrase based on the given template and depth
        new_phrase = self.synthesize_new_word_or_phrase(depth, template)

        # Replace placeholders in the template with the generated phrase
        pattern = r"\{(.*?)\}"
        while len(re.findall(pattern, template)):
            template = re.sub(pattern, self.synthesize_new_word_or_phrase, template)

        # Recursively generate meta-prompts for the generated phrase
        sub_templates = [
            "Create unique language constructs using '{}'.",
            "Invent new expressions encapsulating complex ideas with '{}'.",
        ]
        return (
            self.generate_meta_recursive_prompt(
                " ".join(sub_templates), len(new_phrase.split())
            )
            + " "
            + new_phrase
        )

    def generate_prompts(self, num_prompts=10):
        self.templates += [
            "Generate a list of {num} prompts based on the following template:",
            "Unleash your creativity and craft {num} unique prompts using the given template.",
        ]
        return [self.generate_meta_recursive_prompt() for _ in range(num_prompts)]

    def execute(self, num_prompts=10):
        generated_prompts = self.generate_prompts(num_prompts)
        print("Generated Prompts:")
        for prompt in generated_prompts:
            print(prompt)
        return generated_prompts

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)
