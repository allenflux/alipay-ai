"""Group-aware train/validation/test split for COCO receipt annotations."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_groups(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise ValueError("--groups must be a JSON object: {image_file_name: source_group}")
    return data


def split_coco(
    document: dict[str, Any],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    groups: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Split by source group so variants of one receipt never leak across sets."""
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must each be > 0 and sum to 1")
    images = document.get("images")
    if not isinstance(images, list) or len(images) < 3:
        raise ValueError("At least three images are required for a train/val/test split")
    groups = groups or {}
    grouped_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        if not isinstance(image, dict):
            continue
        file_name = str(image.get("file_name", ""))
        group_name = groups.get(
            file_name,
            groups.get(Path(file_name).name, groups.get(f"images/{file_name}", f"image:{image['id']}")),
        )
        grouped_images[group_name].append(image)
    if len(grouped_images) < 3:
        raise ValueError("At least three independent source groups are required; update --groups")

    desired = {name: len(images) * ratio for name, ratio in ratios.items()}
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in ratios}
    counts = {name: 0 for name in ratios}
    group_items = list(grouped_images.items())
    random.Random(seed).shuffle(group_items)
    # Large source groups are placed first. The objective chooses the split that
    # moves its image count closest to the desired ratio while preserving groups.
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    for _, members in group_items:
        size = len(members)
        split_name = min(
            ratios,
            key=lambda name: (
                abs((counts[name] + size) - desired[name]) - abs(counts[name] - desired[name]),
                counts[name] / desired[name],
            ),
        )
        assigned[split_name].extend(members)
        counts[split_name] += size

    annotations = document.get("annotations", [])
    output: dict[str, dict[str, Any]] = {}
    for split_name, split_images in assigned.items():
        image_ids = {int(image["id"]) for image in split_images}
        output[split_name] = {
            key: value
            for key, value in document.items()
            if key not in {"images", "annotations"}
        }
        output[split_name]["images"] = split_images
        output[split_name]["annotations"] = [
            annotation
            for annotation in annotations
            if isinstance(annotation, dict) and int(annotation.get("image_id", -1)) in image_ids
        ]
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a group-safe train/val/test split from COCO annotations")
    parser.add_argument("--annotations", type=Path, required=True, help="Full COCO annotations JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for train.json, val.json and test.json")
    parser.add_argument("--groups", type=Path, help="Optional {file_name: original_receipt_or_source_group} JSON")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    document = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit("--annotations must contain a COCO JSON object")
    splits = split_coco(
        document,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        groups=_load_groups(args.groups),
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in splits.items():
        path = args.output_dir / f"{split_name}.json"
        path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{split_name}: {len(split['images'])} images, {len(split['annotations'])} boxes -> {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
