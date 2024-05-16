from typing import List, Optional, Any, Tuple
from pydantic import BaseModel


class BaseParams(BaseModel):
    api_keys: Optional[List[str]] = None
    replicate_api_key: Optional[str] = None
    system_message_text: str = ""
    retrieve_model_name: bool = False
    fine_tune: bool = False
    output_name: Optional[str] = None
    name: Optional[str] = None
    technique: Optional[dict] = None
    verbose: bool = True
    model_name: Optional[str] = "gpt-3.5-turbo-16k"


class DataParams(BaseModel):
    path: Optional[str] = None
    database_uri: Optional[str] = None
    storage: Optional[str] = None
    gcloud: Optional[dict] = None
    cloudinary: Optional[dict] = None
    media: Optional[dict] = None
    base_persist_dir: str = "chain_database/"


class ModelParams(BaseModel):
    use_embeddings: bool = False
    use_openai: bool = False
    exact_match: bool = False
    use_semantic_similarity: bool = False
    parametric: bool = False
    train_model: bool = False
    stable_matching: bool = False
    create: bool = False


class ChainParams(BaseModel):
    tree_range: Optional[Tuple[int, Optional[int]]] = (0, None)
    visualize: bool = True
    target_number: int = 10
    animate: bool = False
    top_k: int = 10
    combine: bool = False
    strategy: Optional[str] = None
    phrases: Optional[List[str]] = None
    message_contains: Optional[List[str]] = None
    alpha_scale: float = 1.0
    alpha_final_z: float = 0.0
    method: str = "both"

    class Config:
        arbitrary_types_allowed = True


class EngineParams(BaseModel):
    dataframe: Optional[object] = (None,)
    local_dataset_path: Optional[str] = (None,)
    huggingface_dataset_name: Optional[str] = (None,)
    prompt_subdir: Optional[str] = None
    prompt_col: str = ("prompt",)
    conversation_dir: str = ("conversation/",)
    response_col: str = ("response",)
    root_directory: Optional[str] = (None,)
    verbose: bool = (False,)
    api_key: Optional[str] = (None,)


class MiscParams(BaseModel):
    verbose: bool = True
    spatial_similarity: Optional[Any] = None
    skip_trees: bool = False
    end_letter: str = "B"
    end_roman_numeral: Optional[str] = None
    local_dataset_path: Optional[str] = None
    feedback: bool = (False,)
    display_results: bool = (False,)
    prompt: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class ProcessTreeInput(BaseModel):
    generator_params: dict
    data_params: dict
    model_params: dict
    chain_params: dict
    misc_params: dict
    filter_params: dict
    engine_params: dict = None


class ProcessTreeOutput(BaseModel):
    generator_params: dict
    data_params: dict
    model_params: dict
    chain_params: dict
    misc_params: dict
    filter_params: dict
    engine_params: dict = None


class ChainRunnerInput(BaseModel):
    generator_params: dict
    data_params: dict
    media_params: dict = None
    conversation_path: str = None
    conversation_data: str = None
    conversation_dir: str = None
    relation_data: str = None


class ChainRunnerInitInput(BaseModel):
    input_data: ChainRunnerInput
