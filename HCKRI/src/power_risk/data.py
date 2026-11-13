from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig
from .download import ensure_ton_iot_downloaded


NORMAL_NAMES = {"0", "normal", "benign", "none", "no", "false"}
META_COLUMNS = {
    "__event_id",
    "__source_id",
    "__source_name",
    "__row_order",
    "__timestamp_order",
    "__y_binary",
    "__risk_type",
    "__risk_level",
    "__y_class",
    "__y_unknown",
}


@dataclass
class RiskDatasetBundle:
    train_x: np.ndarray
    train_sources: np.ndarray
    train_y_binary: np.ndarray
    train_y_class: np.ndarray
    train_y_unknown: np.ndarray
    train_labeled_mask: np.ndarray
    val_x: np.ndarray
    val_sources: np.ndarray
    val_y_binary: np.ndarray
    val_y_class: np.ndarray
    val_y_unknown: np.ndarray
    test_x: np.ndarray
    test_sources: np.ndarray
    test_y_binary: np.ndarray
    test_y_class: np.ndarray
    test_y_unknown: np.ndarray
    feature_dim: int
    n_sources: int
    n_classes: int
    source_names: list[str]
    class_names: list[str]
    unknown_attack_types: list[str]
    used_synthetic_data: bool
    split_summary: dict[str, object] | None = None


class HashedFeatureEncoder:
    """Map heterogeneous TON_IoT columns into one fixed-size feature vector."""

    def __init__(self, numeric_hash_dim: int, categorical_hash_dim: int,
                 *, numeric_observed_only: bool = False) -> None:
        self.numeric_hash_dim = numeric_hash_dim
        self.categorical_hash_dim = categorical_hash_dim
        self.feature_dim = numeric_hash_dim + categorical_hash_dim
        self.numeric_observed_only = numeric_observed_only
        self.numeric_stats: dict[str, tuple[float, float]] = {}
        self.categorical_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        feature_columns = [col for col in df.columns if col not in META_COLUMNS]
        for column in feature_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if self.numeric_observed_only:
                # Other sources' absent fields must not change the field's data type.
                observed = df[column].notna()
                if not observed.any():
                    continue
                numeric_ratio = float(values[observed].notna().mean())
            else:
                numeric_ratio = float(values.notna().mean())
            if numeric_ratio >= 0.70:
                arr = values.to_numpy(dtype=np.float64, copy=True)
                arr[~np.isfinite(arr)] = np.nan
                arr = np.clip(arr, -1e12, 1e12)
                mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
                std = float(np.nanstd(arr)) if np.isfinite(arr).any() else 1.0
                self.numeric_stats[column] = (mean, max(std, 1e-6))
            else:
                self.categorical_columns.append(column)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = np.zeros((len(df), self.feature_dim), dtype=np.float32)

        for column, (mean, std) in self.numeric_stats.items():
            if column not in df:
                continue
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            values[~np.isfinite(values)] = mean
            values = np.clip(values, -1e12, 1e12)
            values = np.clip(values, mean - 10.0 * std, mean + 10.0 * std)
            index = _stable_hash(f"num:{column}", self.numeric_hash_dim)
            x[:, index] += ((values - mean) / std).astype(np.float32)

        for column in self.categorical_columns:
            if column not in df:
                continue
            values = df[column].fillna("<missing>").astype(str).to_numpy()
            base = self.numeric_hash_dim
            for row, value in enumerate(values):
                index = base + _stable_hash(f"cat:{column}={value}", self.categorical_hash_dim)
                x[row, index] += 1.0
        return x


def load_risk_dataset(config: DataConfig) -> RiskDatasetBundle:
    data_dir = Path(config.data_dir)
    ensure_ton_iot_downloaded(config)
    files = sorted(data_dir.rglob(config.file_glob)) if data_dir.exists() else []
    if files:
        events, source_names, used_synthetic = _load_ton_iot_events(files, config), [], False
    elif config.allow_synthetic_fallback:
        events, source_names, used_synthetic = _synthetic_multisource_events(config), [], True
    else:
        raise FileNotFoundError(
            f"No TON_IoT CSV files found under {data_dir}. "
            "Put the dataset there or enable allow_synthetic_fallback."
        )

    if "__source_name" not in events:
        raise ValueError("No usable source data was loaded.")
    events = _filter_sources(events, config)
    if events.empty:
        raise ValueError("No events remain after applying source filters.")
    events = _limit_rows_per_source(events, config)
    if config.align_sources_by_rank:
        events = _align_source_timestamps_by_rank(events)
    source_names = sorted(events["__source_name"].unique().tolist())
    source_to_id = {name: idx for idx, name in enumerate(source_names)}
    events["__source_id"] = events["__source_name"].map(source_to_id).astype(int)

    class_names = list(config.risk_level_names)
    events["__risk_level"] = _map_risk_levels(events["__risk_type"], config)
    events["__y_class"] = events["__risk_level"].astype(int)
    events["__y_unknown"] = _mark_unknown_attacks(events["__risk_type"], config)

    train_df, val_df, test_df = _split_events(events, config)
    encoder = HashedFeatureEncoder(
        numeric_hash_dim=config.numeric_hash_dim,
        categorical_hash_dim=config.categorical_hash_dim,
        numeric_observed_only=config.numeric_observed_only,
    )
    excluded = list(config.excluded_feature_columns)
    if config.split_group_column:
        excluded.append(config.split_group_column)
    encoder.fit(train_df.drop(columns=excluded, errors="ignore"))

    train_x, train_sources, train_y_binary, train_y_class, train_y_unknown = _make_windows(
        train_df, encoder, config
    )
    val_x, val_sources, val_y_binary, val_y_class, val_y_unknown = _make_windows(
        val_df, encoder, config
    )
    test_x, test_sources, test_y_binary, test_y_class, test_y_unknown = _make_windows(
        test_df, encoder, config
    )
    train_labeled_mask = _make_labeled_mask(
        train_y_class,
        fraction=config.labeled_fraction,
        seed=config.seed,
    )
    split_summary: dict[str, object] = {
        "selected_rows": len(events),
        "excluded_feature_columns": excluded,
        "encoded_columns": list(encoder.numeric_stats) + encoder.categorical_columns,
        "numeric_observed_only": config.numeric_observed_only,
        "numeric_columns": list(encoder.numeric_stats),
        "categorical_columns": encoder.categorical_columns,
        "record_counts": {name: len(frame) for name, frame in
                          zip(["train", "val", "test"], [train_df, val_df, test_df])},
    }
    if config.split_group_column:
        group_sets = [set(frame["__event_id"]) for frame in [train_df, val_df, test_df]]
        overlaps = {"train_val": len(group_sets[0] & group_sets[1]),
                    "train_test": len(group_sets[0] & group_sets[2]),
                    "val_test": len(group_sets[1] & group_sets[2])}
        if any(overlaps.values()):
            raise AssertionError(f"Event-group leakage: {overlaps}")
        split_summary["group_overlaps"] = overlaps
        split_summary["group_counts"] = dict(zip(["train", "val", "test"], map(len, group_sets)))

    return RiskDatasetBundle(
        train_x=train_x,
        train_sources=train_sources,
        train_y_binary=train_y_binary,
        train_y_class=train_y_class,
        train_y_unknown=train_y_unknown,
        train_labeled_mask=train_labeled_mask,
        val_x=val_x,
        val_sources=val_sources,
        val_y_binary=val_y_binary,
        val_y_class=val_y_class,
        val_y_unknown=val_y_unknown,
        test_x=test_x,
        test_sources=test_sources,
        test_y_binary=test_y_binary,
        test_y_class=test_y_class,
        test_y_unknown=test_y_unknown,
        feature_dim=encoder.feature_dim,
        n_sources=len(source_names),
        n_classes=len(class_names),
        source_names=source_names,
        class_names=class_names,
        unknown_attack_types=_normalized_unknown_attack_types(config),
        used_synthetic_data=used_synthetic,
        split_summary=split_summary,
    )


def _load_ton_iot_events(files: Iterable[Path], config: DataConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    included_sources = _normalized_source_filter(config.include_source_names)
    excluded_sources = _normalized_source_filter(config.exclude_source_names)
    for path in files:
        source_name = _infer_source_name(path)
        if included_sources and source_name not in included_sources:
            continue
        if source_name in excluded_sources:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"Skip unreadable file {path}: {exc}")
            continue
        if df.empty:
            continue

        label_column = _find_column(df, config.label_columns)
        type_column = _find_column(df, config.type_columns)
        if label_column is None and type_column is None:
            print(f"Skip file without label/type columns: {path}")
            continue
        df = _limit_rows_by_stratified_sample(
            df,
            limit=config.max_rows_per_file,
            seed=config.seed,
            label_column=label_column,
            type_column=type_column,
        )

        y_binary, risk_type = _infer_targets(df, label_column, type_column, source_name)
        timestamp = _infer_timestamp_order(df, config.timestamp_columns)
        work = df.copy()
        for column in [label_column, type_column]:
            if column is not None and column in work:
                work = work.drop(columns=[column])
        work["__source_name"] = source_name
        work["__row_order"] = np.arange(len(work), dtype=float)
        work["__timestamp_order"] = timestamp
        work["__y_binary"] = y_binary
        work["__risk_type"] = risk_type
        frames.append(work)

    if not frames:
        raise ValueError("Found CSV files, but none had usable TON_IoT labels.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _limit_rows_by_stratified_sample(
    df: pd.DataFrame,
    limit: int,
    seed: int,
    label_column: str | None,
    type_column: str | None,
) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)

    stratify_column = type_column or label_column
    if stratify_column is None or stratify_column not in df:
        return df.sample(n=limit, random_state=seed).reset_index(drop=True)

    stratify = df[stratify_column].fillna("<missing>").astype(str)
    if stratify.value_counts().min() < 2:
        return df.sample(n=limit, random_state=seed).reset_index(drop=True)

    indices = np.arange(len(df))
    try:
        sampled_idx, _ = train_test_split(
            indices,
            train_size=limit,
            random_state=seed,
            stratify=stratify.to_numpy(),
        )
    except ValueError:
        sampled_idx = df.sample(n=limit, random_state=seed).index.to_numpy()
    return df.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def _limit_rows_per_source(events: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    limit = config.max_rows_per_source
    if limit <= 0:
        return events.reset_index(drop=True)

    sampled_frames: list[pd.DataFrame] = []
    for source_id, (_, source_events) in enumerate(
        events.groupby("__source_name", sort=True)
    ):
        sampled_frames.append(
            _limit_rows_by_stratified_sample(
                source_events,
                limit=limit,
                seed=config.seed + source_id,
                label_column="__y_binary",
                type_column="__risk_type",
            )
        )
    return pd.concat(sampled_frames, axis=0, ignore_index=True)


def _align_source_timestamps_by_rank(events: pd.DataFrame) -> pd.DataFrame:
    aligned = events.copy()
    aligned["__event_id"] = aligned.groupby(
        "__source_name", sort=False
    ).cumcount()
    aligned["__timestamp_order"] = aligned.groupby(
        "__source_name", sort=False
    )["__timestamp_order"].rank(method="first", pct=True)
    return aligned


def _synthetic_multisource_events(config: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    source_names = [
        "device_alarm_log",
        "network_flow",
        "vulnerability_scan",
        "system_state",
    ]
    attack_types = ["normal", "scan", "dos", "data_exfiltration", "malware"]
    rows = []
    per_source = max(config.synthetic_samples // len(source_names), 1)
    for source_id, source_name in enumerate(source_names):
        for row in range(per_source):
            is_attack = rng.random() < 0.08
            risk_type = str(rng.choice(attack_types[1:])) if is_attack else "normal"
            severity_shift = attack_types.index(risk_type) * 0.65
            base = rng.normal(0.0, 1.0, size=6)
            if is_attack:
                base += severity_shift + rng.normal(0.0, 0.25, size=6)
            record = {
                "time": row + rng.normal(0.0, 0.05),
                "src_bytes": base[0] * (source_id + 1) + 3.0 * is_attack,
                "dst_bytes": base[1] + 1.5 * (risk_type == "data_exfiltration"),
                "duration": abs(base[2]) + 2.0 * (risk_type == "dos"),
                "cpu_load": base[3] + 1.2 * (risk_type == "malware"),
                "memory_load": base[4] + 1.0 * (risk_type == "malware"),
                "vuln_score": abs(base[5]) + 2.0 * (risk_type == "scan"),
                "protocol": rng.choice(["tcp", "udp", "icmp", "modbus", "mqtt"]),
                "alarm_level": rng.choice(["info", "warn", "critical"] if is_attack else ["info", "warn"]),
                "__source_name": source_name,
                "__row_order": float(row),
                "__timestamp_order": float(row),
                "__y_binary": int(is_attack),
                "__risk_type": risk_type,
                "__risk_level": 0,
                "__y_unknown": False,
            }
            rows.append(record)
    return pd.DataFrame(rows)


def _map_risk_levels(risk_type: pd.Series, config: DataConfig) -> np.ndarray:
    max_level = max(len(config.risk_level_names) - 1, 0)
    level_map = {
        str(name).strip().lower(): int(level)
        for name, level in config.risk_type_level_map.items()
    }
    values = risk_type.fillna("normal").astype(str).str.strip().str.lower()
    levels = values.map(level_map).fillna(1).astype(int).clip(0, max_level)
    return levels.to_numpy(dtype=np.int64)


def _mark_unknown_attacks(risk_type: pd.Series, config: DataConfig) -> np.ndarray:
    unknown_types = set(_normalized_unknown_attack_types(config))
    if not unknown_types:
        return np.zeros(len(risk_type), dtype=bool)
    values = risk_type.fillna("normal").astype(str).str.strip().str.lower()
    return values.isin(unknown_types).to_numpy(dtype=bool)


def _normalized_unknown_attack_types(config: DataConfig) -> list[str]:
    return sorted(
        {
            str(name).strip().lower()
            for name in config.unknown_attack_types
            if str(name).strip()
        }
    )


def _filter_sources(events: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    source_names = events["__source_name"].fillna("").astype(str).str.strip().str.lower()
    include_names = _normalized_source_filter(config.include_source_names)
    exclude_names = _normalized_source_filter(config.exclude_source_names)
    mask = np.ones(len(events), dtype=bool)
    if include_names:
        mask &= source_names.isin(include_names).to_numpy()
    if exclude_names:
        mask &= ~source_names.isin(exclude_names).to_numpy()
    return events.loc[mask].reset_index(drop=True)


def _normalized_source_filter(values: list[str]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


def _infer_targets(
    df: pd.DataFrame,
    label_column: str | None,
    type_column: str | None,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if type_column is not None:
        raw_type = df[type_column].fillna("normal").astype(str).str.strip().str.lower()
        risk_type = raw_type.where(~raw_type.isin(NORMAL_NAMES), "normal").to_numpy()
    else:
        risk_type = np.array(["attack"] * len(df), dtype=object)

    if label_column is not None:
        raw_label = df[label_column]
        numeric = pd.to_numeric(raw_label, errors="coerce")
        if numeric.notna().mean() >= 0.70:
            y_binary = (numeric.fillna(0).to_numpy() > 0).astype(int)
        else:
            label_text = raw_label.fillna("normal").astype(str).str.strip().str.lower()
            y_binary = (~label_text.isin(NORMAL_NAMES)).astype(int).to_numpy()
    else:
        y_binary = np.array([0 if item in NORMAL_NAMES else 1 for item in risk_type], dtype=int)

    risk_type = np.where(y_binary == 0, "normal", risk_type)
    risk_type = np.where(risk_type == "attack", f"{source_name}_attack", risk_type)
    return y_binary.astype(int), risk_type.astype(str)


def _split_events(
    events: pd.DataFrame,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.split_group_column:
        if config.align_sources_by_rank:
            raise ValueError("Choose explicit event grouping or rank alignment, not both.")
        column = config.split_group_column
        if column not in events or events[column].isna().any():
            raise ValueError(f"Missing event grouping values: {column}")
        events = events.copy()
        events["__event_id"] = events[column]
        return _split_aligned_events(events, config)
    if config.align_sources_by_rank and "__event_id" in events:
        return _split_aligned_events(events, config)

    unknown_mask = events["__y_unknown"].astype(bool).to_numpy()
    known_events = events.loc[~unknown_mask].copy()
    unknown_events = events.loc[unknown_mask].copy()
    if known_events.empty:
        raise ValueError(
            "All attack types were configured as unknown. Keep at least one known class for training."
        )

    y = known_events["__y_class"].to_numpy()
    indices = np.arange(len(known_events))
    stratify = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=stratify,
    )
    temp_y = y[train_idx]
    stratify_temp = (
        temp_y
        if len(np.unique(temp_y)) > 1 and pd.Series(temp_y).value_counts().min() >= 2
        else None
    )
    val_ratio = config.val_size / max(1.0 - config.test_size, 1e-8)
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_ratio,
        random_state=config.seed + 1,
        stratify=stratify_temp,
    )
    train_df = known_events.iloc[train_idx].copy()
    val_df = known_events.iloc[val_idx].copy()
    test_df = known_events.iloc[test_idx].copy()
    if not unknown_events.empty:
        test_df = pd.concat([test_df, unknown_events], axis=0, ignore_index=True)
    return train_df, val_df, test_df


def _split_aligned_events(
    events: pd.DataFrame,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep aligned source views of one event in the same data partition."""
    group_info = events.groupby("__event_id", sort=True).agg(
        is_unknown=("__y_unknown", "max"),
        risk_level=("__y_class", _majority_value),
    )
    known_groups = group_info.index[~group_info["is_unknown"].astype(bool)].to_numpy()
    unknown_groups = group_info.index[group_info["is_unknown"].astype(bool)].to_numpy()
    if len(known_groups) == 0:
        raise ValueError(
            "All aligned events were configured as unknown. Keep at least one known class."
        )

    group_y = group_info.loc[known_groups, "risk_level"].to_numpy(dtype=np.int64)
    stratify = _stratify_or_none(group_y)
    train_groups, test_groups = train_test_split(
        known_groups,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=stratify,
    )
    train_y = group_info.loc[train_groups, "risk_level"].to_numpy(dtype=np.int64)
    val_ratio = config.val_size / max(1.0 - config.test_size, 1e-8)
    train_groups, val_groups = train_test_split(
        train_groups,
        test_size=val_ratio,
        random_state=config.seed + 1,
        stratify=_stratify_or_none(train_y),
    )

    train_df = events.loc[events["__event_id"].isin(train_groups)].copy()
    val_df = events.loc[events["__event_id"].isin(val_groups)].copy()
    test_group_ids = np.concatenate([test_groups, unknown_groups])
    test_df = events.loc[events["__event_id"].isin(test_group_ids)].copy()
    return train_df, val_df, test_df


def _majority_value(values: pd.Series) -> int:
    counts = values.astype(int).value_counts()
    return int(counts.index[0])


def _stratify_or_none(values: np.ndarray) -> np.ndarray | None:
    counts = pd.Series(values).value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return values
    return None


def _make_windows(
    df: pd.DataFrame,
    encoder: HashedFeatureEncoder,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = df.sort_values(["__timestamp_order", "__row_order"]).reset_index(drop=True)
    features = encoder.transform(ordered)
    source_ids = ordered["__source_id"].to_numpy(dtype=np.int64)
    y_binary = ordered["__y_binary"].to_numpy(dtype=np.int64)
    y_class = ordered["__y_class"].to_numpy(dtype=np.int64)
    y_unknown = ordered["__y_unknown"].to_numpy(dtype=bool)

    seq_len = max(config.sequence_length, 1)
    stride = max(config.stride, 1)
    windows_x: list[np.ndarray] = []
    windows_source: list[np.ndarray] = []
    windows_binary: list[int] = []
    windows_class: list[int] = []
    windows_unknown: list[bool] = []

    if len(ordered) < seq_len:
        pad = seq_len - len(ordered)
        x = np.pad(features, ((0, pad), (0, 0)), mode="constant")
        s = np.pad(source_ids, (0, pad), mode="edge") if len(source_ids) else np.zeros(seq_len)
        windows_x.append(x.astype(np.float32))
        windows_source.append(s.astype(np.int64))
        windows_binary.append(int(y_binary.max()) if len(y_binary) else 0)
        windows_class.append(_window_class(y_binary, y_class))
        windows_unknown.append(bool(y_unknown.any()) if len(y_unknown) else False)
    else:
        for start in range(0, len(ordered) - seq_len + 1, stride):
            end = start + seq_len
            windows_x.append(features[start:end].astype(np.float32))
            windows_source.append(source_ids[start:end].astype(np.int64))
            windows_binary.append(int(y_binary[start:end].max()))
            windows_class.append(_window_class(y_binary[start:end], y_class[start:end]))
            windows_unknown.append(bool(y_unknown[start:end].any()))

    return (
        np.stack(windows_x),
        np.stack(windows_source),
        np.array(windows_binary, dtype=np.int64),
        np.array(windows_class, dtype=np.int64),
        np.array(windows_unknown, dtype=bool),
    )


def _window_class(y_binary: np.ndarray, y_class: np.ndarray) -> int:
    attacks = y_class[y_binary == 1]
    if len(attacks) == 0:
        return 0
    values, counts = np.unique(attacks, return_counts=True)
    return int(values[counts.argmax()])


def _make_labeled_mask(y_class: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y_class), dtype=bool)
    fraction = min(max(float(fraction), 0.0), 1.0)
    for label in np.unique(y_class):
        indices = np.where(y_class == label)[0]
        if len(indices) == 0:
            continue
        n_label = max(1, int(round(len(indices) * fraction)))
        chosen = rng.choice(indices, size=min(n_label, len(indices)), replace=False)
        mask[chosen] = True
    return mask


def _build_class_names(risk_type: pd.Series) -> list[str]:
    attacks = sorted({str(value) for value in risk_type.unique() if str(value) != "normal"})
    return ["normal", *attacks]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_column = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        column = lower_to_column.get(candidate.lower())
        if column is not None:
            return column
    return None


def _infer_timestamp_order(df: pd.DataFrame, timestamp_columns: list[str]) -> np.ndarray:
    column = _find_column(df, timestamp_columns)
    if column is None:
        return np.arange(len(df), dtype=float)
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.notna().mean() >= 0.70:
        return numeric.ffill().fillna(0).to_numpy(dtype=float)
    try:
        parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(df[column], errors="coerce")
    if parsed.notna().any():
        return (parsed.ffill().fillna(parsed.min()).astype("int64") / 1e9).to_numpy()
    return np.arange(len(df), dtype=float)


def _infer_source_name(path: Path) -> str:
    name = path.stem.lower()
    if "network_layer" in name:
        return "network_layer"
    if "transport_layer" in name:
        return "transport_layer"
    if "application_protocol" in name:
        return "application_protocol"
    if "application_service" in name:
        return "application_service"
    if "industrial_protocol" in name:
        return "industrial_protocol"
    if "network" in name or "flow" in name:
        return "network_flow"
    if "application_log" in name or "service_log" in name:
        return "application_service_log"
    if "vul" in name or "scan" in name:
        return "vulnerability_scan"
    if "log" in name or "windows" in name or "linux" in name:
        return "device_alarm_log"
    if "iot" in name or "sensor" in name or "fridge" in name or "garage" in name:
        return "system_state"
    return name.replace("train_test_", "").replace("ton_iot_", "")


def _stable_hash(value: str, modulo: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(modulo, 1)
