"""Edit your training settings here. Paths are relative to the project root."""
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    root: str = "datasets/window_frame/train"
    # Common image formats. The loader does not yet use this setting.
    image_type: list[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff", "avif",
    ])
    image_size: int = 512
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    # Optional JSON mapping sample IDs to scene/property IDs.
    scene_groups: str = ""
    class_meanings_confirmed: bool = True


@dataclass
class ModelConfig:
    base_channels: int = 32
    max_correction: float = 0.25
    confidence: float = 0.8


@dataclass
class TrainingConfig:
    epochs: int = 10
    # Set to None to use all examples. The held-out test split is unchanged.
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    # 0 disables early stopping. Preparatory stages are never stopped early.
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0001
    # Both 0: learn all tasks jointly from epoch 1.
    segmentation_epochs: int = 0
    enhancement_epochs: int = 0
    batch_size: int = 10
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    # Auto balances CPU computation and image-loading workers.
    workers: int | str = "auto"
    cpu_threads: int | str = "auto"
    audit_workers: int | str = "auto"
    # Only skip validation when you have already checked the dataset.
    skip_audit: bool = True
    # GPU speed options (used only on CUDA).
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    channels_last: bool = True
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = True
    accumulation_steps: int = 1
    # Requires: uv sync --extra dashboard
    tensorboard: bool = False


@dataclass
class OutputConfig:
    root: str = "output/window_frame/ai_guide"


@dataclass
class Settings:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

settings = Settings()
