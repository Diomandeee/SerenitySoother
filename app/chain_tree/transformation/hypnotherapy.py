import umap
import uuid
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.metrics.pairwise import cosine_similarity
from chain_tree.transformation.collection import (
    CoordinateHandler,
    CoordinateTree,
)
from scipy.spatial.distance import euclidean


class Hypnotherapy:
    def __init__(
        self,
        transcript: str,
        num_state_dimensions: int = 5,
        embedder=None,
        message_splitter: str = "\n\n",
        separate_columns: bool = True,
        weights: np.ndarray = np.array([1, 1, 1, 1]),
    ):
        self.weights = weights

        self.embedder = embedder
        self.previous_states = []
        self.current_segment_index = 0
        self.segments = self.segment_transcript(transcript, message_splitter)
        self.num_segments = len(self.segments)
        self.state_dimensions = num_state_dimensions
        self.state = np.zeros(self.state_dimensions)
        self.current_path = [self.current_segment_index]
        self.initialize_data_engineering(embedder, separate_columns)

    def update_weights(self, new_weights: np.ndarray):
        """
        Update the weights for the influence factors.

        Args:
        new_weights (np.ndarray): The new weights for the factors.
        """
        self.weights = new_weights

    @staticmethod
    def segment_transcript(transcript: str, message_splitter: str = "\n\n"):
        segments = transcript.strip().split(message_splitter)
        return [segment.strip() for segment in segments]

    @staticmethod
    def generate_cartesian_product_by_number(num_segments):
        # Generate Cartesian product based on the number of segments
        return [(i, j) for i in range(num_segments) for j in range(num_segments)]

    @staticmethod
    def generate_cartesian_product_by_content(segments):
        # Generate Cartesian product based on segment content
        return [(i, j) for i in segments for j in segments]

    @staticmethod
    def parse_coordinate_string(coord_tuple):
        id = str(uuid.uuid4())  # Generate a unique ID for each coordinate
        x, y, z, t = (
            coord_tuple[0],
            coord_tuple[1],
            coord_tuple[2],
            coord_tuple[3] if len(coord_tuple) > 3 else 0,
        )
        return CoordinateTree(id=id, x=x, y=y, z=z, t=t)

    def reset_session(self):
        self.current_segment_index = 0
        self.state = np.zeros(self.state_dimensions)

    def initialize_session(self, total_time):
        """
        Initializes the session by resetting the state and current segment index.
        """
        self.reset_session()

        # Initialize the state
        self.state = self.conduct_session(total_time)

        # Present the final segment
        print(
            f"Segment {self.current_segment_index + 1}: {self.segments[self.current_segment_index]}"
        )

    def generate_cartesian_product_by_number(self, num_segments):
        # Generate Cartesian product based on the number of segments
        return [(i, j) for i in range(num_segments) for j in range(num_segments)]
    def initialize_data_engineering(self, embedder, separate_columns):
        """
        Initializes the data engineering components of the session, including embeddings,
        transition matrix, coordinate matrix, and coordinate handler.
        """
        self.embeddings = embedder.embed(self.segments)
        self.transition_matrix = self.generate_cartesian_product_by_content(
            self.segments
        )
        self.coordinate_matrix = self.generate_cartesian_product_by_number(
            self.num_segments
        )
        self.data_eng = embedder.compute_embeddings(
            pd.DataFrame(self.transition_matrix, columns=["Segment 1", "Segment 2"]),
            separate_columns=separate_columns,
        )

        self.data_eng.rename(
            columns={"Segment 1": "prompt", "Segment 2": "response"},
            inplace=True,
        )

        self.data_eng.rename(
            columns={
                "Segment 1 embedding": "prompt_embedding",
                "Segment 2 embedding": "response_embedding",
            },
            inplace=True,
        )

        # Add the coordinate data as a column in the data_eng DataFrame
        self.data_eng["coordinate"] = [
            (index,) + coord for index, coord in enumerate(self.coordinate_matrix)
        ]

        # Generate UMAP projections for the embeddings
        umap_projections = embedder.generate_umap_projections(self.embeddings)

        # Create a dictionary to map each segment to its UMAP projection
        umap_dict = {
            segment: projection
            for segment, projection in zip(self.segments, umap_projections)
        }

        # Add the UMAP projections to the 'prompt_embedding' and 'response_embedding' columns in the data_eng DataFrame
        self.data_eng["prompt_umap"] = self.data_eng["prompt"].apply(

            lambda x: umap_dict[x]
        )

        self.data_eng["response_umap"] = self.data_eng["response"].apply(
            lambda x: umap_dict[x]
        )

        # Stack the UMAP projections and the coordinate data into a unified space
        self.data_eng["unified_space"] = list(
            zip(
                self.data_eng["coordinate"],
                self.data_eng["prompt_umap"],
                self.data_eng["response_umap"],
            )
        )

        # Flatten the unified space
        self.data_eng["unified_space"] = self.data_eng["unified_space"].apply(

            lambda x: np.concatenate(x)
        )

        # Compute the Euclidean distance between the unified space of the prompt and the response
        self.data_eng["euclidean_distance"] = self.data_eng.apply(
            lambda row: np.linalg.norm(row["prompt_umap"] - row["response_umap"]), axis=1
        )

        # create a new column called prompt_id and response_id, we populate it with uuid
        self.data_eng["prompt_id"] = self.data_eng["prompt"].apply(
            lambda x: str(uuid.uuid4())
        )


        self.data_eng["response_id"] = self.data_eng["response"].apply(
            lambda x: str(uuid.uuid4())
        )
        # Convert the coordinate data from tuples to Coordinate objects
        self.coordinates_list = [
            self.parse_coordinate_string(coord) for coord in self.data_eng["coordinate"]
        ]

        # Initialize the CoordinateHandler with the list of Coordinate objects
        self.handler = CoordinateHandler(coordinates=self.coordinates_list)

    def find_next_coherent_segment(
        self, current_embedding: np.ndarray, exclude: list = None
    ) -> int:
        """
        Identifies the next coherent segment based on cosine similarity.

        Args:
        current_embedding (np.ndarray): The embedding of the current segment.
        exclude (list, optional): List of indices to exclude from consideration.

        Returns:
        int: Index of the next coherent segment.
        """
        exclude_set = set(exclude or [])
        valid_indices = [i for i in range(len(self.embeddings)) if i not in exclude_set]

        # Compute cosine similarities only for valid indices
        similarities = [
            cosine_similarity([current_embedding], [self.embeddings[i]])[0][0]
            for i in valid_indices
        ]

        # Find the index of the highest similarity
        max_index = valid_indices[np.argmax(similarities)]
        return max_index

    def _find_alternative_segment(
        self, last_segment_index: int, current_path: list
    ) -> int:
        """
        Finds a contextually coherent segment different from the ones in the current path.

        Args:
        last_segment_index (int): Index of the last visited segment.
        current_path (list): The current path of segments.

        Returns:
        int: Index of the next segment.
        """
        current_embedding = self.embeddings[last_segment_index]
        if len(current_path) >= self.num_segments:
            return self.find_next_coherent_segment(current_embedding)
        return self.find_next_coherent_segment(current_embedding, exclude=current_path)

    def _find_next_linear_segment(self, last_segment_index: int) -> int:
        """
        Finds the next segment in a linear fashion.

        Args:
        last_segment_index (int): Index of the last visited segment.

        Returns:
        int: Index of the next segment.
        """
        return (last_segment_index + 1) % self.num_segments

    def find_homotopic_path(self, current_path: list, feedback: str) -> list:
        """
        Modifies the narrative path based on feedback. Finds a contextually coherent next segment for 'negative' feedback, and
        progresses linearly for 'positive' or other feedback.

        Args:
        current_path (list): The current path taken in the narrative.
        feedback (str): The feedback received ('negative' or 'positive').

        Returns:
        list: The updated path incorporating the feedback.
        """
        if not current_path:
            raise ValueError("Current path is empty")

        last_segment_index = current_path[-1]

        if feedback == "negative":
            next_segment = self._find_alternative_segment(
                last_segment_index, current_path
            )
       
        elif feedback == "positive":
            next_segment = self._find_next_linear_segment(last_segment_index)

        else:
            next_segment = self._find_next_linear_segment(last_segment_index)

        print(f"Moving to segment {next_segment}.")
        return current_path + [next_segment]

    def find_homotopic_path(self, current_path: list, feedback: str) -> list:
        """
        Modifies the narrative path based on feedback. Finds a contextually coherent next segment for 'negative' feedback, and
        progresses linearly for 'positive' or other feedback.

        Args:
        current_path (list): The current path taken in the narrative.
        feedback (str): The feedback received ('negative' or 'positive').

        Returns:
        list: The updated path incorporating the feedback.
        """
        if not current_path:
            raise ValueError("Current path is empty")

        print(f"Current path: {current_path}")

        # Get the last segment index
        last_segment_index = current_path[-1]

        # Find the next segment based on the feedback
        if feedback == "0":
            # Find the next coherent segment
            next_segment_index = self._find_alternative_segment(
                last_segment_index, current_path
            )

        else:
            # Find the next linear segment
            next_segment_index = self._find_next_linear_segment(last_segment_index)

        # Update the path
        current_path.append(next_segment_index)

        return current_path

    def _should_end_session(self, current_time: float, total_time: float) -> bool:
        """
        Determines whether the therapy session should end based on certain conditions.

        Args:
        current_time (float): The current time in the session.
        total_time (float): The total time allocated for the session.

        Returns:
        bool: True if the session should end, False otherwise.
        """
        all_segments_visited = len(set(self.current_path)) >= self.num_segments
        time_threshold_reached = current_time >= total_time * 0.9
        max_segments_visited = (
            len(self.current_path) > self.num_segments * 1.5
        )  # Example threshold

        return all_segments_visited or time_threshold_reached or max_segments_visited


    def calculate_segment_influence(
        self, segment_index: int, t: float, weights: np.ndarray = None
    ) -> np.ndarray:
        """
        Calculate the influence of a segment on the state at time t.

        Args:
        segment_index (int): Index of the segment.
        t (float): The current time.
        weights (np.ndarray): The weights for the different factors.

        Returns:
        np.ndarray: The influence of the segment on the state at time t.
        """

        if weights is None:
            weights = self.weights

        # Calculate the relationship factor
        relationship_factor = self.calculate_relationship_factor(segment_index)

        # Calculate the Euclidean distance between the segment's UMAP projection and the current state
        euclidean_distance = euclidean(
            self.data_eng["response_umap"].iloc[segment_index], self.state
        )

        # Calculate the segment's length factor
        segment_length = len(self.segments[segment_index])
        length_factor = segment_length / max(len(segment) for segment in self.segments)

        # Calculate the segment's position factor
        position_factor = (segment_index + 1) / self.num_segments

        # Calculate the segment's similarity to the current segment
        current_segment_embedding = self.embeddings[self.current_segment_index]
        similarity_factor = cosine_similarity(
            [current_segment_embedding], [self.embeddings[segment_index]]
        )[0][0]

        # Combine the factors using the weights
        influence = (
            weights[0] * relationship_factor
            + weights[1] * length_factor
            + weights[2] * position_factor
            + weights[3] * similarity_factor
        ) / (euclidean_distance + 1)

        return influence
    
    def calculate_relationship_factor(self, segment_index: int) -> float:
        """
        Calculate a factor based on the segment's relationship with other segments in the path.

        Args:
        segment_index (int): Index of the segment.

        Returns:
        float: Relationship factor.
        """
        # Example: Calculate relationship factor based on the segment's distance from the current segment
        current_segment = self.current_segment_index
        distance = abs(current_segment - segment_index)
        return 1 / (distance + 1)

    def influence_function(self, segment_index, state, t):
        embedding = self.embeddings[segment_index]
        segment_influence = np.linalg.norm(embedding)

        # Adjust the influence based on the segment's position in the narrative
        segment_influence *= 1 + 0.1 * (segment_index / self.num_segments)

        return segment_influence    
    
    def system_dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Calculates the system dynamics at time t, considering the influence of each segment.

        Args:
        state (np.ndarray): The current state of the system.
        t (float): The current time.

        Returns:
        np.ndarray: The total influence on the state at time t.
        """
        # Compute the influence of each segment on the state
        influences = np.array(
            [self.influence_function(i, state, t) for i in range(self.num_segments)]
        )

        # Update the state based on the influences
        state += np.sum(influences, axis=0)

        # Normalize the state
        state /= np.linalg.norm(state)

        return state
    def evaluate_segment_influence(self, segment_index: int) -> float:
        """
        Evaluate the influence of a segment based on its embedding and its position in the narrative.

        Args:
        segment_index (int): Index of the segment.

        Returns:
        float: A calculated influence value.
        """
        embedding = self.embeddings[segment_index]
        segment_influence = np.linalg.norm(embedding)

        # Adjust the influence based on the segment's position in the narrative
        segment_influence *= 1 + 0.1 * (segment_index / self.num_segments)

        return segment_influence

    def calculate_state_rate_of_change(self, state: np.ndarray) -> float:
        """
        Calculate a more nuanced rate of change of the state, considering both immediate and recent changes,
        as well as the characteristics of the segments involved.

        Args:
        state (np.ndarray): The current state of the system.

        Returns:
        float: A calculated rate of change value.
        """
        if self.previous_states:
            # Calculate the immediate rate of change
            immediate_change = np.linalg.norm(state - self.previous_states[-1])

            # Calculate the average rate of change over the recent history
            recent_changes = [
                np.linalg.norm(current - previous)
                for current, previous in zip(
                    self.previous_states[1:], self.previous_states[:-1]
                )
            ]
            average_recent_change = np.mean(recent_changes) if recent_changes else 0

            # Factor in the characteristics of the segments
            segment_influence_factors = [
                self.evaluate_segment_influence(segment)
                for segment in self.current_path[-len(self.previous_states) :]
            ]
            weighted_segment_influence = (
                np.mean(segment_influence_factors) if segment_influence_factors else 0
            )

            # Combine these factors into a single rate of change measure
            return (
                immediate_change + average_recent_change + weighted_segment_influence
            ) / 3
        else:
            # Default to a basic measure if there are no previous states to compare with
            return 0

    def determine_next_time_step(
        self,
        current_time: float,
        current_state: np.ndarray,
        min_max=(0.05, 0.2),
    ) -> float:
        """
        Determine the next time step, dynamically adjusting based on the state's rate of change.

        Args:
        current_time (float): Current time.
        current_state (np.ndarray): Current state of the system.

        Returns:
        float: Next time step.
        """
        rate_of_change = self.calculate_state_rate_of_change(current_state)
        min_time_step = min_max[0]
        max_time_step = min_max[1]

        # Adjust time step based on rate of change
        dynamic_time_step = max(
            min_time_step, min(max_time_step, 1 / (1 + rate_of_change))
        )
        return current_time + dynamic_time_step
 
 
        
    def conduct_session(
        self, total_time: float, time_step_resolution: int = 1000
    ) -> np.ndarray:
        """
        Conducts a therapy session over a specified total time with dynamic feedback integration and state updates.

        Args:
        total_time (float): The total time to conduct the session.
        time_step_resolution (int): The number of time points to consider in the session.

        Returns:
        np.ndarray: The final state of the session.
        """
        if total_time <= 0 or time_step_resolution <= 0:
            raise ValueError("Total time and time step resolution must be positive.")

        # Initialize the state and time variables
        state = self.state
        current_time = 0.0
        time_points = np.linspace(0, total_time, time_step_resolution)

        # Conduct the session over time
        for t in time_points:

            # Store the current state for future comparisons
            self.previous_states.append(state)

            # Determine the next coherent segment based on feedback
            feedback = input().lower()

            # Update the current path based on feedback
            self.current_path = self.find_homotopic_path(self.current_path, feedback)
            self.current_segment_index = self.current_path[-1]

            # Present the current segment
            print(
                f"Segment {self.current_segment_index + 1}: {self.segments[self.current_segment_index]}"
            )


            # Determine the next time step
            next_time = self.determine_next_time_step(current_time, state)

            # Calculate the state dynamics over the next time step
            state = odeint(self.system_dynamics, state, [current_time, next_time])[-1]

            # Check if the session should end
            if self._should_end_session(t, total_time):
                break

            # Update the current time
            current_time = next_time

        return state
 
    