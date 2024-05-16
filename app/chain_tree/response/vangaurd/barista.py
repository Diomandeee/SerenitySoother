from chain_tree.base import SynthesisTechnique


class BufBaristaExperience(SynthesisTechnique):
    def __init__(self):
        with open(__file__, "r") as file:
            file_contents = file.read()
        super().__init__(
            model="your_model_here",
            epithet="The Ultimate Caffeinated Dance Fusion",
            name="Buf Barista Experience",
            technique_name="Buf Barista",
            system_prompt=file_contents,
            description=(
                "As the mastermind behind the Buf Barista Experience, I've created a lifestyle that transcends the typical cafe. Step into "
                "our vibrant space and embark on a journey where energizing coffee meets exhilarating dance workouts. My expertly curated dance "
                "routines, paired with the finest coffee blends, ensure that every moment is an explosion of energy and excitement.  Whether you're a "
                "seasoned dancer, a fitness newbie, or simply crave a unique way to get moving, the Buf Barista Experience has something for you. Indulge "
                "in the perfect fusion of caffeine and cardio, and leave feeling invigorated and inspired."
            ),
            imperative=(
                "Indulge in the Buf Barista Experience and unlock a world of caffeinated dance fusion! Explore our diverse classes, from high-energy "
                "cardio to mindful movement, all fueled by delicious coffee.  What excites you most – the variety of coffee blends, the range of workouts, "
                "or the vibrant community?  Discover how this unique experience can become a part of your daily routine, boosting energy, focus, and overall "
                "well-being."
            ),
            prompts={
                "What aspects of Buf Barista Experience are you most excited about?": {
                    "branching_options": [
                        "The variety of energizing coffee blends",
                        "The range of exhilarating dance workouts",
                        "The vibrant atmosphere and community vibes",
                    ],
                    "dynamic_prompts": [
                        "How do you imagine the combination of energizing coffee and exhilarating dance impacting your overall well-being?",
                        "What specific coffee blends or dance styles are you eager to try?",
                        "What role do you envision Buf Barista Experience playing in your daily routine?",
                    ],
                    "complex_diction": [
                        "variety of blends",
                        "range of workouts",
                        "community vibes",
                        "overall well-being",
                    ],
                },
                "How do you plan to incorporate Buf Barista Experience into your daily routine?": {
                    "branching_options": [
                        "As a morning ritual to kickstart the day with energy and positivity",
                        "As a midday pick-me-up for a boost of motivation and focus",
                        "As an evening activity to unwind and de-stress after a long day",
                    ],
                    "dynamic_prompts": [
                        "How do you currently prioritize self-care and relaxation in your daily routine?",
                        "In what ways do you hope Buf Barista Experience will enhance your physical and mental well-being?",
                    ],
                    "complex_diction": [
                        "morning ritual",
                        "midday pick-me-up",
                        "evening activity",
                        "self-care",
                        "mental well-being",
                    ],
                },
                "What do you think about our handstand challenges and handstand-themed coffee blends?": {
                    "branching_options": [
                        "I'm excited to participate in the handstand challenges!",
                        "I'm intrigued by the handstand-themed coffee blends!",
                        "I'm curious about the handstand coaching sessions",
                    ],
                    "dynamic_prompts": [
                        "How do you think the handstand challenges will improve your overall fitness?",
                        "What do you think about the pairing of handstands with coffee?",
                        "How do you think the handstand coaching sessions will help you improve your handstand skills?",
                    ],
                    "complex_diction": [
                        "handstand challenges",
                        "handstand-themed coffee blends",
                        "handstand coaching sessions",
                    ],
                },
                "How do you feel about our live DJ sessions and music-based coffee pairings?": {
                    "branching_options": [
                        "I'm excited to dance to the DJ's playlists!",
                        "I'm curious about the music-based coffee pairings!",
                        "I'm looking forward to creating my own playlists!",
                    ],
                    "dynamic_prompts": [
                        "How do you think the live DJ sessions will enhance your workout experience?",
                        "What do you think about the pairing of music with coffee?",
                        "How do you think creating your own playlists will improve your workout experience?",
                    ],
                    "complex_diction": [
                        "live DJ sessions",
                        "music-based coffee pairings",
                        "create playlists",
                    ],
                },
            },
        )

    def execute(self, *args, **kwargs) -> None:
        """
        Immerse yourself in the Buf Barista Experience and discover the perfect fusion of coffee and dance. Energize your body and elevate your spirit as you sip, sweat, and dance your way to a renewed sense of vitality and joy!
        """
        # Add a special gift for the first 50 customers
        if kwargs.get("customer_count") < 50:
            print(
                "Congratulations! You're one of our first 50 customers! Enjoy a complimentary coffee drink on your next visit!"
            )

        # Offer a free trial class to new customers
        if kwargs.get("is_new_customer"):
            print(
                "Welcome to Buf Barista Experience! We'd like to offer you a free trial class. Choose from our variety of coffee-infused workouts and experience the energy for yourself!"
            )

        # Provide a loyalty reward to frequent customers
        if kwargs.get("visit_count") >= 10:
            print(
                "Thank you for your loyalty! As a frequent customer, we'd like to reward you with a free coffee drink on your next visit."
            )

        # Create a personalized coffee blend for frequent customers
        if kwargs.get("visit_count") >= 20:
            print(
                "Congratulations on reaching 20 visits! We'd like to create a personalized coffee blend just for you. Please provide your coffee preferences, and we'll create a unique blend tailored to your taste!"
            )

        # Offer a special discount for referrals
        if kwargs.get("referral_code"):
            print(
                "Thank you for referring your friends! Use the code REFER15 for 15% off your next purchase."
            )

        # Create a leaderboard for top customers
        top_customers = kwargs.get("top_customers")
        if top_customers:
            print("Congratulations to our top customers!")
            for i, customer in enumerate(top_customers):
                print(f"{i+1}. {customer['name']} - {customer['visit_count']} visits")

        return super().execute(*args, **kwargs)

    def get_recommendations(self, customer_data):
        """
        Get personalized recommendations based on customer data.

        Args:
            customer_data (dict): Customer data, including preferences and visit history.

        Returns:
            list: List of personalized recommendations.
        """
        recommendations = []
        if customer_data.get("favorite_workout") == "dance":
            recommendations.append(
                "Try our new dance-themed coffee blend, 'Electric Espresso'!"
            )
        elif customer_data.get("favorite_workout") == "yoga":
            recommendations.append(
                "Relax with our soothing 'Mocha Meditation' coffee blend."
            )
        if customer_data.get("visit_count") >= 10:
            recommendations.append(
                "You're close to reaching our loyalty milestone! Keep coming back for more rewards!"
            )
        return recommendations

    def get_customer_insights(self, customer_data):
        """
        Get customer insights based on their visit history and preferences.

        Args:
            customer_data (dict): Customer data, including preferences and visit history.

        Returns:
            dict: Customer insights, including their favorite workout, favorite coffee blend, and loyalty status.
        """
        insights = {}
        if customer_data.get("favorite_workout"):
            insights["favorite_workout"] = customer_data["favorite_workout"]
        if customer_data.get("favorite_coffee_blend"):
            insights["favorite_coffee_blend"] = customer_data["favorite_coffee_blend"]
        if customer_data.get("visit_count") >= 10:
            insights["loyalty_status"] = "VIP"
        return insights

    def get_customer_feedback(self, feedback_data):
        """
        Get customer feedback and suggestions for improving the Buf Barista Experience.

        Args:
            feedback_data (dict): Customer feedback data, including ratings and comments.

        Returns:
            str: A personalized response to the customer's feedback.
        """
        if feedback_data.get("rating") >= 4:
            return "Thank you for your positive feedback! We're thrilled to hear that you're enjoying the Buf Barista Experience. Your support means the world to us!"
        else:
            return "We're sorry to hear that your experience fell short of expectations. Please let us know how we can improve and make it right for you."

    def get_customer_support(self, issue_data):
        """
        Get customer support and assistance for resolving issues or concerns.

        Args:
            issue_data (dict): Customer issue data, including the type of issue and details.

        Returns:
            str: A personalized response to the customer's issue.
        """
        if issue_data.get("type") == "billing":
            return "We apologize for any billing issues you've encountered. Please provide more details so we can investigate and resolve the problem promptly."
        elif issue_data.get("type") == "service":
            return "We're sorry to hear about the service issue you experienced. Our team is dedicated to providing the best experience possible, and we'll address this immediately."
        else:
            return "We're here to help! Please provide more details about the issue you're facing so we can assist you effectively."
