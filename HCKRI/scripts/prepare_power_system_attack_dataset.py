from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.io import arff


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "PowerSystemAttack" / "multiclass"
OUT_DIR = ROOT / "data" / "raw" / "PowerSystemAttack" / "processed_multisource"


NATURAL_FAULT_MARKERS = {1, 2, 3, 4, 5, 6}
LINE_MAINTENANCE_MARKERS = {13, 14}
DATA_INJECTION_MARKERS = {7, 8, 9, 10, 11, 12}
REMOTE_TRIPPING_MARKERS = {15, 16, 17, 18, 19, 20}
RELAY_SETTING_MARKERS = set(range(21, 31)) | set(range(35, 41))
NORMAL_MARKER = 41


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_frames: dict[str, list[pd.DataFrame]] = {
        "power_pmu": [],
        "control_panel": [],
        "protective_relay": [],
        "snort_alert": [],
    }

    global_offset = 0
    for path in sorted(RAW_DIR.glob("*.arff"), key=_dataset_sort_key):
        print(f"Loading {path.name}")
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        df["marker"] = df["marker"].map(_decode_marker).astype(int)
        n_rows = len(df)
        meta = pd.DataFrame(
            {
                "time": range(global_offset, global_offset + n_rows),
                "dataset_part": path.stem.replace(" Sampled Scenarios.csv", ""),
                "Attack_label": (df["marker"] != NORMAL_MARKER).astype(int),
                "Attack_type": df["marker"].map(_marker_to_type),
                "marker": df["marker"],
            }
        )
        global_offset += n_rows

        groups = {
            "power_pmu": _prefix_columns(df, ("R1", "R2", "R3", "R4")),
            "control_panel": [c for c in df.columns if c.startswith("control_panel")],
            "protective_relay": [c for c in df.columns if c.startswith("relay")],
            "snort_alert": [c for c in df.columns if c.startswith("snort")],
        }
        for source_name, columns in groups.items():
            source_df = pd.concat([meta, df[columns].reset_index(drop=True)], axis=1)
            source_frames[source_name].append(source_df)

    for source_name, frames in source_frames.items():
        out_path = OUT_DIR / f"{source_name}.csv"
        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(combined)} rows, {len(combined.columns)} columns)")


def _dataset_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem.split()[0] if ch.isdigit())
    return int(digits or 0)


def _prefix_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [column for column in df.columns if column.startswith(prefixes)]


def _decode_marker(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def _marker_to_type(marker: int) -> str:
    if marker == NORMAL_MARKER:
        return "normal"
    if marker in NATURAL_FAULT_MARKERS:
        return "natural_fault"
    if marker in LINE_MAINTENANCE_MARKERS:
        return "line_maintenance"
    if marker in DATA_INJECTION_MARKERS:
        return "data_injection"
    if marker in REMOTE_TRIPPING_MARKERS:
        return "remote_tripping"
    if marker in RELAY_SETTING_MARKERS:
        return "relay_setting_change"
    return f"scenario_{marker}"


if __name__ == "__main__":
    main()
