"""Copy the exact raw-image cohort recorded by inference JSONL files."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


def _load_sources(record_paths: Sequence[Path], source_root: Path) -> list[tuple[Path, Path]]:
    source_root = source_root.resolve()
    selected: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for record_path in record_paths:
        with record_path.open("r", encoding="utf-8-sig") as records:
            for line_number, line in enumerate(records, start=1):
                if not line.strip():
                    continue
                try:
                    record: Any = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{record_path}:{line_number}: invalid JSON: {error}") from None
                source_value = record.get("source") if isinstance(record, dict) else None
                if not isinstance(source_value, str) or not source_value.strip():
                    raise ValueError(f"{record_path}:{line_number}: missing source path")
                source_path = Path(source_value).resolve()
                try:
                    relative_path = source_path.relative_to(source_root)
                except ValueError:
                    raise ValueError(
                        f"{record_path}:{line_number}: source is outside source root: {source_path}"
                    ) from None
                key = relative_path.as_posix().casefold()
                if key in seen:
                    continue
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                seen.add(key)
                selected.append((source_path, relative_path))
    return selected


def export_inference_cohort(
    *,
    record_paths: Sequence[Path],
    source_root: Path,
    output_dir: Path,
    expected_count: int | None = None,
    cumulative_manifest: bool = False,
) -> list[dict[str, str]]:
    """Copy unique sources from inference manifests/errors while preserving paths."""
    if not record_paths:
        raise ValueError("At least one record JSONL is required")
    if expected_count is not None and expected_count <= 0:
        raise ValueError("expected_count must be positive")
    resolved_source_root = source_root.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == resolved_source_root or resolved_source_root in resolved_output_dir.parents:
        raise ValueError("output directory must be outside source root")
    selected = _load_sources(record_paths, source_root)
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} unique sources, found {len(selected)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, str]] = []
    for source_path, relative_path in selected:
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        status = "copied"
        if destination.exists():
            if not destination.is_file() or not filecmp.cmp(source_path, destination, shallow=False):
                raise FileExistsError(f"Destination exists with different content: {destination}")
            status = "skipped_identical"
        else:
            shutil.copy2(source_path, destination)
        exported.append(
            {
                "source": source_path.as_posix(),
                "relative_path": relative_path.as_posix(),
                "destination": destination.resolve().as_posix(),
                "status": status,
            }
        )

    manifest_path = output_dir.parent / f"{output_dir.name}_cohort_manifest.jsonl"
    manifest_records = exported
    if cumulative_manifest and manifest_path.is_file():
        combined: dict[str, dict[str, str]] = {}
        with manifest_path.open("r", encoding="utf-8-sig") as existing_manifest:
            for line_number, line in enumerate(existing_manifest, start=1):
                if not line.strip():
                    continue
                try:
                    existing: Any = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{manifest_path}:{line_number}: invalid JSON: {error}") from None
                relative_path = existing.get("relative_path") if isinstance(existing, dict) else None
                if not isinstance(relative_path, str) or not relative_path.strip():
                    raise ValueError(f"{manifest_path}:{line_number}: missing relative_path")
                combined[relative_path.casefold()] = existing
        for record in exported:
            combined[record["relative_path"].casefold()] = record
        manifest_records = list(combined.values())

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as manifest:
        for record in manifest_records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the exact raw-image cohort from inference JSONL files")
    parser.add_argument("--record", type=Path, action="append", required=True, help="Manifest/error JSONL; repeatable")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--cumulative-manifest",
        action="store_true",
        help="Keep previously exported sources in the cohort manifest on later runs",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        records = export_inference_cohort(
            record_paths=args.record,
            source_root=args.source_root,
            output_dir=args.output,
            expected_count=args.expected_count,
            cumulative_manifest=args.cumulative_manifest,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cohort export failed:\n{error}") from None
    copied = sum(record["status"] == "copied" for record in records)
    print(f"Exported cohort: total={len(records)}, copied={copied}, already_identical={len(records) - copied}")


if __name__ == "__main__":  # pragma: no cover
    main()
