from app.services.embedding_service import TextEmbeddingService
from app.models import HypnotherapyScript, HypnotherapySession
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from scipy.spatial.distance import euclidean
from sqlalchemy.future import select
from scipy.integrate import odeint
from fastapi import HTTPException
from app.helper import log_handler
from typing import List
import pandas as pd
import numpy as np
import uuid


class Hypnotherapy:
    def __init__(
        self,
        transcript: str,
        num_state_dimensions: int = 5,
        embedder=None,
        prompt_generator=None,
        message_splitter: str = "\n\n",
        separate_columns: bool = True,
        weights: np.ndarray = np.array([1, 1, 1, 1]),
    ):
        self.weights = weights
        self.embedder = embedder
        self.prompt_generator = prompt_generator
        self.previous_states = []
        self.current_segment_index = 0
        self.segments = self.segment_transcript(transcript, message_splitter)
        self.num_segments = len(self.segments)
        self.state_dimensions = num_state_dimensions
        self.state = np.zeros(self.state_dimensions)
        self.current_path = [self.current_segment_index]
        self.initialize_data_engineering(embedder, separate_columns)

    def update_weights(self, new_weights: np.ndarray):
        self.weights = new_weights

    @staticmethod
    def segment_transcript(transcript: str, message_splitter: str = "\n\n"):
        segments = transcript.strip().split(message_splitter)
        return [segment.strip() for segment in segments]

    @staticmethod
    def generate_cartesian_product_by_number(num_segments):
        return [(i, j) for i in range(num_segments) for j in range(num_segments)]

    @staticmethod
    def generate_cartesian_product_by_content(segments):
        return [(i, j) for i in segments for j in segments]

    @staticmethod
    def parse_coordinate_string(coord_tuple):
        id = str(uuid.uuid4())
        x, y, z, t = (
            coord_tuple[0],
            coord_tuple[1],
            coord_tuple[2],
            coord_tuple[3] if len(coord_tuple) > 3 else 0,
        )
        return (id, x, y, z, t)

    def reset_session(self):
        self.current_segment_index = 0
        self.state = np.zeros(self.state_dimensions)

    def initialize_session(self, total_time):
        self.reset_session()
        self.state = self.conduct_session(total_time)
        print(
            f"Segment {self.current_segment_index + 1}: {self.segments[self.current_segment_index]}"
        )

    def initialize_data_engineering(self, embedder, separate_columns):
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

        self.data_eng["coordinate"] = [
            (index,) + coord for index, coord in enumerate(self.coordinate_matrix)
        ]

        umap_projections = embedder.generate_umap_projections(self.embeddings)
        umap_dict = {
            segment: projection
            for segment, projection in zip(self.segments, umap_projections)
        }

        self.data_eng["prompt_umap"] = self.data_eng["prompt"].apply(
            lambda x: umap_dict[x]
        )
        self.data_eng["response_umap"] = self.data_eng["response"].apply(
            lambda x: umap_dict[x]
        )

        self.data_eng["unified_space"] = list(
            zip(
                self.data_eng["coordinate"],
                self.data_eng["prompt_umap"],
                self.data_eng["response_umap"],
            )
        )

        self.data_eng["unified_space"] = self.data_eng["unified_space"].apply(
            lambda x: np.concatenate(x)
        )
        self.data_eng["euclidean_distance"] = self.data_eng.apply(
            lambda row: np.linalg.norm(row["prompt_umap"] - row["response_umap"]),
            axis=1,
        )

        self.data_eng["prompt_id"] = self.data_eng["prompt"].apply(
            lambda x: str(uuid.uuid4())
        )
        self.data_eng["response_id"] = self.data_eng["response"].apply(
            lambda x: str(uuid.uuid4())
        )
        self.coordinates_list = [
            self.parse_coordinate_string(coord) for coord in self.data_eng["coordinate"]
        ]

    def find_next_coherent_segment(
        self, current_embedding: np.ndarray, exclude: list = None
    ) -> int:
        exclude_set = set(exclude or [])
        valid_indices = [i for i in range(len(self.embeddings)) if i not in exclude_set]
        similarities = [
            cosine_similarity([current_embedding], [self.embeddings[i]])[0][0]
            for i in valid_indices
        ]
        max_index = valid_indices[np.argmax(similarities)]
        return max_index

    def _find_alternative_segment(
        self, last_segment_index: int, current_path: list
    ) -> int:
        current_embedding = self.embeddings[last_segment_index]
        if len(current_path) >= self.num_segments:
            return self.find_next_coherent_segment(current_embedding)
        return self.find_next_coherent_segment(current_embedding, exclude=current_path)

    def _find_next_linear_segment(self, last_segment_index: int) -> int:
        return (last_segment_index + 1) % self.num_segments

    def find_homotopic_path(self, current_path: list, feedback: str) -> list:
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

    def _should_end_session(self, current_time: float, total_time: float) -> bool:
        all_segments_visited = len(set(self.current_path)) >= self.num_segments
        time_threshold_reached = current_time >= total_time * 0.9
        max_segments_visited = len(self.current_path) > self.num_segments * 1.5
        return all_segments_visited or time_threshold_reached or max_segments_visited

    def calculate_segment_influence(
        self, segment_index: int, t: float, weights: np.ndarray = None
    ) -> np.ndarray:
        if weights is None:
            weights = self.weights
        relationship_factor = self.calculate_relationship_factor(segment_index)
        euclidean_distance = euclidean(
            self.data_eng["response_umap"].iloc[segment_index], self.state
        )
        segment_length = len(self.segments[segment_index])
        length_factor = segment_length / max(len(segment) for segment in self.segments)
        position_factor = (segment_index + 1) / self.num_segments
        current_segment_embedding = self.embeddings[self.current_segment_index]
        similarity_factor = cosine_similarity(
            [current_segment_embedding], [self.embeddings[segment_index]]
        )[0][0]
        influence = (
            weights[0] * relationship_factor
            + weights[1] * length_factor
            + weights[2] * position_factor
            + weights[3] * similarity_factor
        ) / (euclidean_distance + 1)
        return influence

    def calculate_relationship_factor(self, segment_index: int) -> float:
        current_segment = self.current_segment_index
        distance = abs(current_segment - segment_index)
        return 1 / (distance + 1)

    def update_memory(self, segment_index, weight, decay_factor):
        self.memory[segment_index] = weight
        self.memory = [w * decay_factor for w in self.memory]

    def generate_new_path(self, current_path: list) -> list:
        print("Generating a new path based on memory...")

        memory_prompt = " ".join([self.segments[i] for i in current_path])
        new_path_text = self.prompt_generator.generate_prompt(memory_prompt)

        new_path_segments = self.segment_transcript(new_path_text)
        self.segments += new_path_segments

        new_coordinates = self.generate_cartesian_product_by_number(
            len(new_path_segments)
        )
        new_coordinates_list = [
            self.parse_coordinate_string(coord) for coord in new_coordinates
        ]
        self.coordinates_list += new_coordinates_list

        self.num_segments = len(self.segments)

        new_transition_matrix = self.generate_cartesian_product_by_content(
            new_path_segments
        )
        self.transition_matrix += new_transition_matrix

        new_coordinate_matrix = self.generate_cartesian_product_by_number(
            len(new_path_segments)
        )
        self.coordinate_matrix += new_coordinate_matrix

        new_path = [self.segments.index(segment) for segment in new_path_segments]

        for index in new_path:
            self.update_memory(index, 1, 0.9)

        return new_path

    def influence_function(self, segment_index, state, t):
        embedding = self.embeddings[segment_index]
        segment_influence = np.linalg.norm(embedding)
        segment_influence *= 1 + 0.1 * (segment_index / self.num_segments)
        return segment_influence

    def system_dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        influences = np.array(
            [self.influence_function(i, state, t) for i in range(self.num_segments)]
        )
        state += np.sum(influences, axis=0)
        state /= np.linalg.norm(state)
        return state

    def evaluate_segment_influence(self, segment_index: int) -> float:
        embedding = self.embeddings[segment_index]
        segment_influence = np.linalg.norm(embedding)
        segment_influence *= 1 + 0.1 * (segment_index / self.num_segments)
        return segment_influence

    def calculate_state_rate_of_change(self, state: np.ndarray) -> float:
        if self.previous_states:
            immediate_change = np.linalg.norm(state - self.previous_states[-1])
            recent_changes = [
                np.linalg.norm(current - previous)
                for current, previous in zip(
                    self.previous_states[1:], self.previous_states[:-1]
                )
            ]
            average_recent_change = np.mean(recent_changes) if recent_changes else 0
            segment_influence_factors = [
                self.evaluate_segment_influence(segment)
                for segment in self.current_path[-len(self.previous_states) :]
            ]
            weighted_segment_influence = (
                np.mean(segment_influence_factors) if segment_influence_factors else 0
            )
            return (
                immediate_change + average_recent_change + weighted_segment_influence
            ) / 3
        else:
            return 0

    def determine_next_time_step(
        self, current_time: float, current_state: np.ndarray, min_max=(0.05, 0.2)
    ) -> float:
        rate_of_change = self.calculate_state_rate_of_change(current_state)
        min_time_step = min_max[0]
        max_time_step = min_max[1]
        dynamic_time_step = max(
            min_time_step, min(max_time_step, 1 / (1 + rate_of_change))
        )
        return current_time + dynamic_time_step

    def conduct_session(
        self, total_time: float, time_step_resolution: int = 1000
    ) -> np.ndarray:
        if total_time <= 0 or time_step_resolution <= 0:
            raise ValueError("Total time and time step resolution must be positive.")
        state = self.state
        current_time = 0.0
        time_points = np.linspace(0, total_time, time_step_resolution)
        for t in time_points:
            self.previous_states.append(state)
            feedback = input().lower()
            self.current_path = self.find_homotopic_path(self.current_path, feedback)
            self.current_segment_index = self.current_path[-1]
            print(
                f"Segment {self.current_segment_index + 1}: {self.segments[self.current_segment_index]}"
            )
            next_time = self.determine_next_time_step(current_time, state)
            state = odeint(self.system_dynamics, state, [current_time, next_time])[-1]
            if self._should_end_session(t, total_time):
                break
            current_time = next_time
        return state

    def analyze_session(self):
        return {
            "segments": self.segments,
            "embeddings": self.embeddings,
            "transition_matrix": self.transition_matrix,
            "coordinates": self.coordinates_list,
            "data_engineering": self.data_eng,
        }


class HypnotherapyService:
    def __init__(self, db: AsyncSession, api_key: str, num_state_dimensions: int = 5):
        self.db = db
        self.embedder = TextEmbeddingService(api_key=api_key)
        self.num_state_dimensions = num_state_dimensions

    async def get_hypnotherapy_script(self, script_id: int) -> HypnotherapyScript:
        try:
            result = await self.db.execute(
                select(HypnotherapyScript).filter(HypnotherapyScript.id == script_id)
            )
            script = result.scalars().first()
            if not script:
                raise HTTPException(
                    status_code=404, detail="Hypnotherapy script not found"
                )
            return script
        except Exception as e:
            log_handler(f"Error getting hypnotherapy script {script_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while fetching hypnotherapy script.",
            )

    async def create_hypnotherapy_script(self, user_id: int, title: str, content: str):
        try:
            new_script = HypnotherapyScript(
                user_id=user_id, title=title, content=content
            )
            self.db.add(new_script)
            await self.db.commit()
            await self.db.refresh(new_script)
            return new_script
        except Exception as e:
            log_handler(f"Error creating hypnotherapy script for user {user_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while creating the hypnotherapy script.",
            )

    async def get_hypnotherapy_scripts(self, user_id: int) -> List[HypnotherapyScript]:
        try:
            result = await self.db.execute(
                select(HypnotherapyScript).filter(HypnotherapyScript.user_id == user_id)
            )
            return result.scalars().all()
        except Exception as e:
            log_handler(f"Error getting hypnotherapy scripts for user {user_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while fetching hypnotherapy scripts.",
            )

    async def create_hypnotherapy_session(
        self, user_id: int, script_id: int, session_notes: str
    ):
        try:
            new_session = HypnotherapySession(
                user_id=user_id, script_id=script_id, session_notes=session_notes
            )
            self.db.add(new_session)
            await self.db.commit()
            await self.db.refresh(new_session)
            return new_session
        except Exception as e:
            log_handler(f"Error creating hypnotherapy session for user {user_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while creating the hypnotherapy session.",
            )

    async def get_hypnotherapy_sessions(
        self, user_id: int
    ) -> List[HypnotherapySession]:
        try:
            result = await self.db.execute(
                select(HypnotherapySession).filter(
                    HypnotherapySession.user_id == user_id
                )
            )
            return result.scalars().all()
        except Exception as e:
            log_handler(f"Error getting hypnotherapy sessions for user {user_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while fetching hypnotherapy sessions.",
            )

    async def get_hypnotherapy_session(self, session_id: int) -> HypnotherapySession:
        try:
            result = await self.db.execute(
                select(HypnotherapySession).filter(HypnotherapySession.id == session_id)
            )
            session = result.scalars().first()
            if not session:
                raise HTTPException(
                    status_code=404, detail="Hypnotherapy session not found"
                )
            return session
        except Exception as e:
            log_handler(f"Error getting hypnotherapy session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while fetching hypnotherapy session.",
            )

    async def update_hypnotherapy_session(
        self, session_id: int, session_notes: str
    ) -> HypnotherapySession:
        try:
            result = await self.db.execute(
                select(HypnotherapySession).filter(HypnotherapySession.id == session_id)
            )
            session = result.scalars().first()
            if not session:
                raise HTTPException(
                    status_code=404, detail="Hypnotherapy session not found"
                )
            session.session_notes = session_notes
            await self.db.commit()
            await self.db.refresh(session)
            return session
        except Exception as e:
            log_handler(f"Error updating hypnotherapy session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while updating hypnotherapy session.",
            )

    async def delete_hypnotherapy_session(self, session_id: int):
        try:
            result = await self.db.execute(
                select(HypnotherapySession).filter(HypnotherapySession.id == session_id)
            )
            session = result.scalars().first()
            if not session:
                raise HTTPException(
                    status_code=404, detail="Hypnotherapy session not found"
                )
            self.db.delete(session)
            await self.db.commit()
        except Exception as e:
            log_handler(f"Error deleting hypnotherapy session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while deleting hypnotherapy session.",
            )

    async def initialize_session(self, user_id: int, script_id: int):
        try:
            script = await self.get_hypnotherapy_script(script_id)
            hypnotherapy = Hypnotherapy(
                script.content, self.num_state_dimensions, self.embedder
            )
            return hypnotherapy
        except Exception as e:
            log_handler(
                f"Error initializing hypnotherapy session for user {user_id}, script {script_id}: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail="An error occurred while initializing the hypnotherapy session.",
            )

    async def process_feedback(
        self, session_id: int, feedback: str, current_path: List[int]
    ):
        try:
            session = await self.get_hypnotherapy_session(session_id)
            hypnotherapy = Hypnotherapy(
                session.script.content, self.num_state_dimensions, self.embedder
            )
            hypnotherapy.current_path = current_path
            updated_path = hypnotherapy.find_homotopic_path(current_path, feedback)
            return updated_path
        except Exception as e:
            log_handler(f"Error processing feedback for session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing feedback for the hypnotherapy session.",
            )

    async def conduct_session(self, session_id: int, total_time: float):
        try:
            session = await self.get_hypnotherapy_session(session_id)
            hypnotherapy = Hypnotherapy(
                session.script.content, self.num_state_dimensions, self.embedder
            )
            final_state = hypnotherapy.conduct_session(total_time)
            return final_state
        except Exception as e:
            log_handler(f"Error conducting hypnotherapy session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while conducting the hypnotherapy session.",
            )

    async def analyze_hypnotherapy_session(self, session_id: int):
        try:
            session = await self.get_hypnotherapy_session(session_id)
            hypnotherapy = Hypnotherapy(
                session.script.content, self.num_state_dimensions, self.embedder
            )
            analysis_result = hypnotherapy.analyze_session()
            return analysis_result
        except Exception as e:
            log_handler(f"Error analyzing hypnotherapy session {session_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while analyzing the hypnotherapy session.",
            )

    async def generate_new_path(self, session_id: int):
        try:
            session = await self.get_hypnotherapy_session(session_id)
            hypnotherapy = Hypnotherapy(
                session.script.content, self.num_state_dimensions, self.embedder
            )
            new_path = hypnotherapy.generate_new_path()
            return new_path
        except Exception as e:
            log_handler(
                f"Error generating new path for hypnotherapy session {session_id}: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail="An error occurred while generating a new path for the hypnotherapy session.",
            )
