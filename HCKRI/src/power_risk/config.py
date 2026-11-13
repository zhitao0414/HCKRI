from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: str = "data/raw/TON_IoT"
    file_glob: str = "*.csv"
    auto_download: bool = True
    download_url: str = "https://huggingface.co/datasets/codymlewis/TON_IoT_network/resolve/main/train_test_network.csv?download=true"
    download_filename: str = "train_test_network.csv"
    download_sha256: str = "26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974"
    download_timeout: int = 120
    max_rows_per_file: int = 20000
    max_rows_per_source: int = 0
    align_sources_by_rank: bool = False
    split_group_column: str = ""
    excluded_feature_columns: list[str] = field(default_factory=list)
    sequence_length: int = 12
    stride: int = 3
    val_size: float = 0.15
    test_size: float = 0.20
    labeled_fraction: float = 0.30
    numeric_hash_dim: int = 48
    categorical_hash_dim: int = 32
    numeric_observed_only: bool = False
    allow_synthetic_fallback: bool = True
    synthetic_samples: int = 2500
    seed: int = 42
    risk_level_names: list[str] = field(
        default_factory=lambda: [
            "正常",
            "一般风险",
            "较大风险",
            "重大风险",
            "特别重大风险",
        ]
    )
    risk_type_level_map: dict[str, int] = field(
        default_factory=lambda: {
            "normal": 0,
            "scanning": 1,
            "scan": 1,
            "password": 2,
            "xss": 2,
            "injection": 3,
            "dos": 3,
            "ddos": 3,
            "backdoor": 4,
            "mitm": 4,
            "ransomware": 4,
            "malware": 4,
            "data_exfiltration": 4,
        }
    )
    unknown_attack_types: list[str] = field(default_factory=list)
    label_columns: list[str] = field(
        default_factory=lambda: [
            "label",
            "Label",
            "attack",
            "Attack",
            "is_attack",
            "is_anomaly",
        ]
    )
    type_columns: list[str] = field(
        default_factory=lambda: [
            "type",
            "Type",
            "attack_type",
            "Attack_type",
            "category",
            "Category",
        ]
    )
    timestamp_columns: list[str] = field(
        default_factory=lambda: [
            "ts",
            "time",
            "timestamp",
            "Timestamp",
            "date",
            "Date",
            "datetime",
        ]
    )
    include_source_names: list[str] = field(default_factory=list)
    exclude_source_names: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    model_dim: int = 64
    source_embedding_dim: int = 16
    cnn_kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    lstm_hidden: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.20


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    semi_supervised_weight: float = 0.35
    contrastive_weight: float = 0.10
    pseudo_label_threshold: float = 0.90
    prototype_temperature: float = 0.20
    anomaly_threshold: float = 0.50
    unknown_confidence_threshold: float = 0.55
    device: str = "auto"
    seed: int = 42


@dataclass
class OutputConfig:
    results_path: str = "outputs/ton_iot_risk_results.json"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output: OutputConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return ExperimentConfig(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        train=TrainConfig(**payload.get("train", {})),
        output=OutputConfig(**payload.get("output", {})),
    )
