from chain_tree.base import SynthesisTechnique


class SensorSpatialNavigator(SynthesisTechnique):
    """
    Imagine an interactive AI experience that goes beyond mere text and enters the realm of personal cognition, tailored uniquely to you.
    How? By mapping real-time sensor data from wearable devices like Fitbit, including accelerometer and gyroscope readings, we can
    construct a user-centric, spatially aware interface formed by your interactions with the AI.

    This is where the sensor data comes into play. The accelerations, the twists, and turns that your Fitbit picks up throughout your
    day are used as input to navigate this cognitive space. Think of each physical movement as an input command, driving the journey
    through your personal knowledge landscape.

    In other words, your daily routines, habits, and energy become the fuel propelling your personal AI assistant. Every interaction,
    every dialogue with the AI, isn't just a passive exchange but an active stride through a terrain of thoughts, ideas, and knowledge.
    This isn't just a user interface - it's a user experience, one that blends the boundaries between the physical and cognitive,
    bridging the gap through the synergy of AI and personal sensor data.

    Welcome to a future where we're not just talking with chatbots, but traversing intellectual landscapes with our 'Sensor-Spatial Navigators'.
    Here, understanding isn't just about reading or hearing - it's about exploring and experiencing. And it all starts with a simple step,
    or a turn of the wrist.
    """

    def __init__(self):
        super().__init__(
            epithet="Mind Mapper",
            name="Movement Mixer",
            technique_name="Movement Mixer",
            description=(
                "Experience a revolution in AI interaction with the Sensor-Spatial Navigator - a groundbreaking concept that integrates "
                "real-time sensor data from wearable devices like Fitbit to create a personal cognitive landscape. Your daily movements "
                "and gestures serve as the fuel driving your journey through a spatially aware interface, where each action propels you "
                "deeper into your own knowledge terrain. The Sensor-Spatial Navigator is more than a tool; it's an extension of your "
                "cognitive abilities, enhancing your understanding and engagement with information through embodied interaction. Whether "
                "you're strolling through a park, jogging on a treadmill, or typing away at your desk, every movement shapes your cognitive "
                "experience, turning mundane actions into meaningful interactions with AI."
            ),
            imperative=(
                "Embark on an immersive journey of exploration with the Sensor-Spatial Navigator, where your physical movements shape "
                "your cognitive experience. Dive into the depths of your thoughts and ideas, guided by the synergy of AI and personal "
                "sensor data. Join the forefront of interactive AI and discover the transformative power of embodied cognition. Whether "
                "you're a fitness enthusiast, a busy professional, or a lifelong learner, the Sensor-Spatial Navigator adapts to your "
                "lifestyle, seamlessly integrating into your daily routine to enhance your mental agility and expand your intellectual horizons."
            ),
            prompts={
                "How do you envision integrating your daily routines and physical movements into your cognitive exploration with the Sensor-Spatial Navigator?": {
                    "branching_options": [
                        "I see myself using my daily walks and runs to navigate through different topics and concepts, turning physical activity into intellectual exploration.",
                        "I plan to incorporate gestures and movements during my work routine to interact with the AI and explore new ideas, blending productivity with cognitive engagement.",
                    ],
                    "dynamic_prompts": [
                        "What specific areas of your life do you believe would benefit most from this integrated approach to cognitive exploration?",
                        "How do you imagine leveraging the synergy between physical activity and intellectual engagement to enhance your productivity and creativity?",
                        "What strategies do you intend to implement to ensure a seamless integration between your physical movements and cognitive interactions?",
                        "In what ways do you anticipate the Sensor-Spatial Navigator transforming your daily routine into a dynamic journey of discovery and learning?",
                    ],
                    "complex_diction": [
                        "cognitive exploration",
                        "intellectual engagement",
                        "seamless integration",
                        "dynamic journey",
                        "discovery and learning",
                    ],
                },
                "Reflect on a recent experience where you felt particularly engaged or immersed in a learning activity. How do you imagine replicating or enhancing this experience with the Sensor-Spatial Navigator?": {
                    "branching_options": [
                        "I believe the Sensor-Spatial Navigator could amplify my immersion by translating physical movements into meaningful interactions with the AI, creating a more embodied learning experience.",
                        "With the Sensor-Spatial Navigator, I envision a more dynamic and interactive learning experience that mirrors the fluidity of real-world activities, fostering deeper understanding and retention of information.",
                    ],
                    "dynamic_prompts": [
                        "What aspects of your recent learning experience do you think could be enhanced through the integration of real-time sensor data and spatial navigation?",
                        "How do you envision the Sensor-Spatial Navigator facilitating deeper engagement and understanding compared to traditional learning methods?",
                        "In what ways do you anticipate the Sensor-Spatial Navigator revolutionizing the way we interact with AI and engage with knowledge?",
                        "Reflecting on your learning journey, what insights can you glean about the potential impact of the Sensor-Spatial Navigator on future educational experiences?",
                    ],
                    "complex_diction": [
                        "amplify",
                        "dynamic and interactive",
                        "revolutionizing",
                        "educational experiences",
                        "engagement and understanding",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Embark on an immersive journey of exploration with the Sensor-Spatial Navigator, where your physical movements shape
        your cognitive experience. Dive into the depths of your thoughts and ideas, guided by the synergy of AI and personal
        sensor data. Join the forefront of interactive AI and discover the transformative power of embodied cognition. Whether
        you're a fitness enthusiast, a busy professional, or a lifelong learner, the Sensor-Spatial Navigator adapts to your
        lifestyle, seamlessly integrating into your daily routine to enhance your mental agility and expand your intellectual horizons.
        """
        return super().execute(*args, **kwargs)
