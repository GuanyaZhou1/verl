#!/usr/bin/env python3
"""Rewrite a string prefix recursively inside parquet rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _rewrite_value(value: Any, old: str, new: str) -> tuple[Any, int]:
    """Recursively rewrite string values and return replacement count."""
    if isinstance(value, str):
        rewritten = value.replace(old, new)
        return rewritten, int(rewritten != value)

    if isinstance(value, list):
        rewritten_items = []
        replaced = 0
        for item in value:
            rewritten_item, item_replaced = _rewrite_value(item, old, new)
            rewritten_items.append(rewritten_item)
            replaced += item_replaced
        return rewritten_items, replaced

    if isinstance(value, tuple):
        rewritten_items = []
        replaced = 0
        for item in value:
            rewritten_item, item_replaced = _rewrite_value(item, old, new)
            rewritten_items.append(rewritten_item)
            replaced += item_replaced
        return tuple(rewritten_items), replaced

    if isinstance(value, dict):
        rewritten_dict = {}
        replaced = 0
        for key, item in value.items():
            rewritten_item, item_replaced = _rewrite_value(item, old, new)
            rewritten_dict[key] = rewritten_item
            replaced += item_replaced
        return rewritten_dict, replaced

    return value, 0


def rewrite_parquet(input_path: Path, output_path: Path, old: str, new: str, force: bool = False) -> tuple[int, int]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    writer: pq.ParquetWriter | None = None

    total_rows = 0
    total_replacements = 0

    try:
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx)
            rows = table.to_pylist()

            rewritten_rows = []
            for row in rows:
                rewritten_row, replaced = _rewrite_value(row, old, new)
                rewritten_rows.append(rewritten_row)
                total_replacements += replaced

            rewritten_table = pa.Table.from_pylist(rewritten_rows, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema=schema, compression="snappy")
            writer.write_table(rewritten_table)
            total_rows += len(rewritten_rows)
    finally:
        if writer is not None:
            writer.close()

    return total_rows, total_replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a path prefix inside parquet files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input parquet file paths.")
    parser.add_argument("--output-dir", required=True, help="Directory to write rewritten parquet files into.")
    parser.add_argument("--old-prefix", default="/softhome/", help="String prefix to replace.")
    parser.add_argument("--new-prefix", default="/data_gpu/", help="Replacement prefix.")
    parser.add_argument("--force", action="store_true", help="Overwrite output files if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for input_name in args.inputs:
        input_path = Path(input_name).expanduser().resolve()
        output_path = output_dir / input_path.name
        rows, replacements = rewrite_parquet(
            input_path=input_path,
            output_path=output_path,
            old=args.old_prefix,
            new=args.new_prefix,
            force=args.force,
        )
        print(
            f"rewrote {input_path} -> {output_path} "
            f"(rows={rows}, string_replacements={replacements})"
        )


if __name__ == "__main__":
    main()
