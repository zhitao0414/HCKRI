from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_COLUMNS = ["label", "type"]
SOURCE_COLUMNS = {
    "ton_network_layer.csv": [
        "src_ip",
        "dst_ip",
        "proto",
        "conn_state",
        "missed_bytes",
    ],
    "ton_transport_layer.csv": [
        "src_port",
        "dst_port",
        "duration",
        "src_bytes",
        "dst_bytes",
        "src_pkts",
        "dst_pkts",
        "src_ip_bytes",
        "dst_ip_bytes",
    ],
    "ton_application_protocol.csv": [
        "service",
        "dns_query",
        "dns_qclass",
        "dns_qtype",
        "dns_rcode",
        "dns_AA",
        "dns_RD",
        "dns_RA",
        "dns_rejected",
        "ssl_version",
        "ssl_cipher",
        "ssl_resumed",
        "ssl_established",
        "ssl_subject",
        "ssl_issuer",
        "http_trans_depth",
        "http_method",
        "http_uri",
        "http_version",
        "http_request_body_len",
        "http_response_body_len",
        "http_status_code",
        "http_user_agent",
        "http_orig_mime_types",
        "http_resp_mime_types",
        "weird_name",
        "weird_addl",
        "weird_notice",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split TON-IoT network events into synchronized feature views."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("data/raw/TON_IoT/train_test_network.csv"),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data/raw/TON_IoT/network_multiview"),
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    selections = {
        filename: [
            *[column for column in LABEL_COLUMNS if column in header],
            *[column for column in columns if column in header],
        ]
        for filename, columns in SOURCE_COLUMNS.items()
    }
    targets = {name: args.output_dir / name for name in selections}
    for target in targets.values():
        if target.exists():
            target.unlink()

    totals = {name: 0 for name in selections}
    for chunk_index, chunk in enumerate(
        pd.read_csv(args.input_csv, chunksize=args.chunk_size, low_memory=False), start=1
    ):
        for filename, columns in selections.items():
            chunk.loc[:, columns].to_csv(
                targets[filename],
                mode="a",
                header=not targets[filename].exists(),
                index=False,
            )
            totals[filename] += len(chunk)
        print(f"Processed chunk {chunk_index}")

    print("TON-IoT synchronized feature views:")
    for filename, count in totals.items():
        print(f"  {filename}: {count} rows")


if __name__ == "__main__":
    main()
