from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable


Record = dict[str, Any]


def parquet_available() -> bool:
    return importlib.util.find_spec("pandas") is not None and importlib.util.find_spec("pyarrow") is not None


def materialize_table_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.suffix == ".parquet" and not parquet_available():
        return requested.with_suffix(requested.suffix + ".jsonl")
    return requested


def write_table(records: Iterable[Record], path: str | Path) -> Path:
    output = materialize_table_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if output.suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output, index=False)
        return output
    if output.suffix == ".csv":
        write_csv(rows, output)
        return output
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return output


def read_table(path: str | Path) -> list[Record]:
    source = Path(path)
    if not source.exists() and source.suffix == ".parquet":
        fallback = source.with_suffix(source.suffix + ".jsonl")
        if fallback.exists():
            source = fallback
    if source.suffix == ".parquet":
        import pandas as pd

        return json.loads(pd.read_parquet(source).to_json(orient="records"))
    if source.suffix == ".csv":
        return read_csv(source)
    rows: list[Record] = []
    if not source.exists():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: str | Path) -> list[Record]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_coerce_record(row) for row in reader]


def write_csv(rows: list[Record], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _coerce_record(row: dict[str, str]) -> Record:
    return {key: _coerce_cell(value) for key, value in row.items()}


def _coerce_cell(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in value and "e" not in lowered:
            return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
