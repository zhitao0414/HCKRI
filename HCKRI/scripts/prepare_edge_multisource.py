from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_COLUMNS = ["Attack_label", "Attack_type"]
TIME_COLUMNS = ["frame.time"]
SOURCE_PREFIXES = {
    "edge_network_layer.csv": ("ip.", "arp.", "icmp."),
    "edge_transport_layer.csv": ("tcp.", "udp."),
    "edge_application_protocol.csv": ("http.", "dns.", "mqtt.", "mbtcp."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the merged Edge-IIoTset table into protocol-source views."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("data/raw/Edge_IIoTset/ML-EdgeIIoT-dataset.csv"),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data/raw/Edge_IIoTset/layer_multisource"),
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def selected_columns(columns: list[str], prefixes: tuple[str, ...]) -> list[str]:
    shared = [column for column in [*TIME_COLUMNS, *LABEL_COLUMNS] if column in columns]
    features = [column for column in columns if column.startswith(prefixes)]
    return list(dict.fromkeys([*shared, *features]))


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    selections = {
        filename: selected_columns(header, prefixes)
        for filename, prefixes in SOURCE_PREFIXES.items()
    }
    for path in (args.output_dir / filename for filename in selections):
        if path.exists():
            path.unlink()

    totals = {filename: 0 for filename in selections}
    for chunk_index, chunk in enumerate(
        pd.read_csv(args.input_csv, chunksize=args.chunk_size, low_memory=False)
    ):
        for filename, columns in selections.items():
            # Preserve every original event in every protocol view. The grouped
            # splitter then keeps all views of an event in one data partition.
            source_frame = chunk.loc[:, columns].copy()
            if source_frame.empty:
                continue
            output_path = args.output_dir / filename
            source_frame.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )
            totals[filename] += len(source_frame)
        print(f"Processed chunk {chunk_index + 1}")

    print("Edge-IIoTset protocol-source views:")
    for filename, count in totals.items():
        print(f"  {filename}: {count} rows")


if __name__ == "__main__":
    main()
