"""Assemble multiple read-only image trees into a separate training root."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path
from typing import Sequence

from .prepare import iter_image_paths


def assemble_image_roots(input_dirs: Sequence[Path], output_dir: Path) -> tuple[int, int]:
    """Copy image trees without changing inputs or allowing path collisions."""
    if len(input_dirs) < 2:
        raise ValueError("Provide at least two input image roots")
    selected: list[tuple[Path, Path]] = []
    owners: dict[str, Path] = {}
    for input_dir in input_dirs:
        resolved_root = input_dir.resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(resolved_root)
        for source_path in iter_image_paths(resolved_root):
            relative_path = source_path.relative_to(resolved_root)
            key = relative_path.as_posix().casefold()
            previous = owners.get(key)
            if previous is not None:
                raise ValueError(
                    f"Image path collision between input roots: {previous} and {source_path}"
                )
            owners[key] = source_path
            selected.append((source_path, relative_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for source_path, relative_path in selected:
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or not filecmp.cmp(source_path, destination, shallow=False):
                raise FileExistsError(f"Destination exists with different content: {destination}")
            skipped += 1
            continue
        shutil.copy2(source_path, destination)
        copied += 1
    return copied, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy image roots into a separate collision-safe training root")
    parser.add_argument("--input", type=Path, action="append", required=True, help="Image root; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        copied, skipped = assemble_image_roots(args.input, args.output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Training image assembly failed:\n{error}") from None
    print(f"Training image root ready: copied={copied}, already_identical={skipped}, output={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
