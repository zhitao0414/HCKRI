from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def format_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    headers: list[str] | None = None,
    digits: int = 4,
) -> str:
    headers = headers or columns
    rendered_rows = [[_format_cell(row.get(col, ""), digits) for col in columns] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rendered_rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        str(header).ljust(widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(widths))) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def write_markdown_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    headers: list[str] | None = None,
    digits: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_markdown_table(rows, columns, headers=headers, digits=digits) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def with_average_row(
    rows: list[dict[str, Any]],
    label_column: str,
    metric_columns: list[str],
    label: str = "Average",
) -> list[dict[str, Any]]:
    table_rows = [dict(row) for row in rows]
    average = {label_column: label}
    for column in metric_columns:
        values = [
            float(row[column])
            for row in rows
            if isinstance(row.get(column), int | float)
        ]
        average[column] = sum(values) / len(values) if values else ""
    table_rows.append(average)
    return table_rows


def _format_cell(value: Any, digits: int) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value.is_integer():
            return str(int(value))
        return f"{value:.{digits}f}"
    return str(value)
