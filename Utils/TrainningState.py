import json
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class TrainningState:
    title: str
    author: str
    price: float
    tags: List[str]


def load_training_state(file_path: str) -> TrainningState:
    pass


def save_training_state(
    subset_name,
    model_name,
    max_length,
    batch_size,
    leaning_rate,
    weight_decay,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    GAMMA,
    TEMPERATURE_T,
    M0,
    S,
    LAMBDA_PROTO,
    USE_MOMENTUM_ENCODER,
    ALPHA,
    MOMENTUM,
    DEVICE,
    UMAP_MAX_POINTS,
):
    pass
