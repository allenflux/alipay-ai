"""Merge complete COCO detection sets without changing frozen val/test splits."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .labels import DETECTION_CLASSES, LABEL_TO_ID


def _canonical_file_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every COCO image needs a non-empty file_name")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"COCO file_name must be a relative image path: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("Every COCO image needs a non-empty file_name")
    return normalized


def _category_names(document: dict[str, Any], source: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for category in document.get("categories", []):
        if not isinstance(category, dict) or not isinstance(category.get("name"), str):
            continue
        category_id = int(category["id"])
        if category_id in mapping:
            raise ValueError(f"{source}: duplicate category id {category_id}")
        mapping[category_id] = category["name"]
    expected_mapping = {LABEL_TO_ID[label]: label for label in DETECTION_CLASSES}
    if mapping != expected_mapping:
        raise ValueError(f"{source}: categories must be exactly {expected_mapping}; found {mapping}")
    return mapping


def _validated_bbox(annotation: dict[str, Any], image: dict[str, Any], source: str, file_name: str) -> list[float]:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{source}: {file_name}: every annotation needs bbox=[x,y,width,height]")
    try:
        x, y, width, height = (float(value) for value in bbox)
        image_width = float(image["width"])
        image_height = float(image["height"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{source}: {file_name}: invalid bbox or image dimensions") from None
    if not all(math.isfinite(value) for value in (x, y, width, height, image_width, image_height)):
        raise ValueError(f"{source}: {file_name}: bbox and image dimensions must be finite")
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"{source}: {file_name}: bbox and image dimensions must be positive")
    tolerance = 1e-3
    if x < -tolerance or y < -tolerance or x + width > image_width + tolerance or y + height > image_height + tolerance:
        raise ValueError(f"{source}: {file_name}: bbox lies outside the image")
    return [x, y, width, height]


def merge_coco_documents(
    documents: Sequence[dict[str, Any]],
    *,
    source_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge COCO documents, normalise IDs and reject duplicates/incomplete images."""
    if not documents:
        raise ValueError("At least one COCO document is required")
    if source_names is None:
        source_names = [f"input[{index}]" for index in range(len(documents))]
    if len(source_names) != len(documents):
        raise ValueError("source_names must match documents")

    merged: dict[str, Any] = {
        "info": {
            "description": "Transfer receipt LRCNN merged training set",
            "version": "2.0",
            "sources": list(source_names),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": LABEL_TO_ID[label], "name": label, "supercategory": "transfer_receipt"}
            for label in DETECTION_CLASSES
        ],
    }
    seen_file_names: dict[str, str] = {}
    next_image_id = 1
    next_annotation_id = 1

    for document, source in zip(documents, source_names):
        if not isinstance(document, dict):
            raise ValueError(f"{source}: COCO root must be an object")
        category_names = _category_names(document, source)
        images = document.get("images", [])
        annotations = document.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError(f"{source}: images and annotations must be arrays")

        source_images: dict[int, dict[str, Any]] = {}
        for image in images:
            if not isinstance(image, dict):
                raise ValueError(f"{source}: every image entry must be an object")
            image_id = int(image["id"])
            if image_id in source_images:
                raise ValueError(f"{source}: duplicate image id {image_id}")
            source_images[image_id] = image

        annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in source_images}
        source_annotation_ids: set[int] = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"{source}: every annotation entry must be an object")
            annotation_id = int(annotation["id"])
            if annotation_id in source_annotation_ids:
                raise ValueError(f"{source}: duplicate annotation id {annotation_id}")
            source_annotation_ids.add(annotation_id)
            image_id = int(annotation["image_id"])
            if image_id not in source_images:
                raise ValueError(f"{source}: annotation references missing image id {image_id}")
            category_id = int(annotation["category_id"])
            if category_id not in category_names:
                raise ValueError(f"{source}: annotation uses unknown category id {category_id}")
            annotations_by_image[image_id].append(annotation)

        for old_image_id, image in source_images.items():
            file_name = _canonical_file_name(image.get("file_name"))
            duplicate_key = file_name.casefold()
            if duplicate_key in seen_file_names:
                raise ValueError(
                    f"Duplicate image {file_name!r} appears in both {seen_file_names[duplicate_key]} and {source}"
                )
            image_annotations = annotations_by_image[old_image_id]
            label_counts = Counter(category_names[int(annotation["category_id"])] for annotation in image_annotations)
            missing = [label for label in DETECTION_CLASSES if label_counts[label] == 0]
            duplicates = [f"{label}×{label_counts[label]}" for label in DETECTION_CLASSES if label_counts[label] > 1]
            if missing or duplicates:
                details = []
                if missing:
                    details.append("missing=" + ",".join(missing))
                if duplicates:
                    details.append("duplicates=" + ",".join(duplicates))
                raise ValueError(f"{source}: {file_name}: " + "; ".join(details))

            seen_file_names[duplicate_key] = source
            new_image = {key: value for key, value in image.items() if key != "id"}
            new_image.update({"id": next_image_id, "file_name": file_name})
            merged["images"].append(new_image)
            for annotation in image_annotations:
                new_annotation = {key: value for key, value in annotation.items() if key not in {"id", "image_id", "category_id"}}
                label = category_names[int(annotation["category_id"])]
                bbox = _validated_bbox(annotation, image, source, file_name)
                new_annotation.update(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": LABEL_TO_ID[label],
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                    }
                )
                merged["annotations"].append(new_annotation)
                next_annotation_id += 1
            next_image_id += 1

    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge complete COCO sets into one training annotation file")
    parser.add_argument("--input", type=Path, action="append", required=True, help="COCO JSON; repeat for each source")
    parser.add_argument(
        "--holdout",
        type=Path,
        action="append",
        default=[],
        help="Frozen val/test COCO file that must not overlap training; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(args.input) < 2:
        raise SystemExit("Provide at least two --input COCO files")
    output = args.output.resolve()
    inputs = [path.resolve() for path in args.input]
    holdouts = [path.resolve() for path in args.holdout]
    if output in inputs or output in holdouts:
        raise SystemExit("--output must not overwrite an input or holdout file")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output}. Pass --overwrite only when replacement is intentional.")
    try:
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
        merged = merge_coco_documents(documents, source_names=[path.as_posix() for path in inputs])
        training_names = {
            _canonical_file_name(image.get("file_name")).casefold()
            for image in merged["images"]
            if isinstance(image, dict)
        }
        seen_holdout_names: dict[str, str] = {}
        for holdout_path in holdouts:
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
            if not isinstance(holdout, dict) or not isinstance(holdout.get("images"), list):
                raise ValueError(f"{holdout_path}: holdout must be a COCO object with an images array")
            for image in holdout["images"]:
                if not isinstance(image, dict):
                    raise ValueError(f"{holdout_path}: every holdout image must be an object")
                file_name = _canonical_file_name(image.get("file_name"))
                key = file_name.casefold()
                if key in training_names:
                    raise ValueError(f"Training image {file_name!r} also appears in holdout {holdout_path}")
                if key in seen_holdout_names:
                    raise ValueError(
                        f"Holdout image {file_name!r} appears in both {seen_holdout_names[key]} and {holdout_path}"
                    )
                seen_holdout_names[key] = holdout_path.as_posix()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"COCO merge failed:\n{error}") from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    counts = Counter(
        DETECTION_CLASSES[int(annotation["category_id"]) - 1] for annotation in merged["annotations"]
    )
    print(f"Wrote {len(merged['images'])} images and {len(merged['annotations'])} boxes to {args.output}")
    print("Per class: " + ", ".join(f"{label}={counts[label]}" for label in DETECTION_CLASSES))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
