"""Convert simple LabelMe rectangles/polygons into the detector's COCO format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .labels import DETECTION_CLASSES, LABEL_TO_ID, validate_label


def _image_dimensions(image_path: Path, labelme_data: dict[str, Any]) -> tuple[int, int]:
    width = labelme_data.get("imageWidth")
    height = labelme_data.get("imageHeight")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    with Image.open(image_path) as image:
        return image.size


def _resolve_image_path(label_path: Path, labels_dir: Path, images_dir: Path, data: dict[str, Any]) -> Path:
    declared = data.get("imagePath")
    if isinstance(declared, str) and declared:
        declared_path = Path(declared)
        candidates = [images_dir / declared_path, images_dir / declared_path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    relative = label_path.relative_to(labels_dir).with_suffix("")
    candidates = [
        images_dir / relative.with_suffix(extension)
        for extension in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find image for LabelMe file {label_path}")


def _shape_bbox(shape: dict[str, Any], image_width: int, image_height: int) -> list[float] | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(image_width), max(xs))
    y2 = min(float(image_height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def convert_labelme_to_coco(
    labels_dir: Path,
    images_dir: Path,
    output_path: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Build one COCO detection JSON from LabelMe JSON files.

    Labels must be the five fixed class names documented in ``README.md``.
    A LabelMe ``description`` value is retained as optional OCR ground truth.
    """
    labels_dir = labels_dir.resolve()
    images_dir = images_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_files = sorted(labels_dir.rglob("*.json"))
    if not label_files:
        raise ValueError(f"No LabelMe JSON files under {labels_dir}")

    coco: dict[str, Any] = {
        "info": {"description": "Transfer receipt LRCNN dataset", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": LABEL_TO_ID[name], "name": name, "supercategory": "transfer_receipt"}
            for name in DETECTION_CLASSES
        ],
    }
    annotation_id = 1
    for image_id, label_path in enumerate(label_files, start=1):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{label_path} must contain a JSON object")
        image_path = _resolve_image_path(label_path, labels_dir, images_dir, data)
        image_width, image_height = _image_dimensions(image_path, data)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(images_dir).as_posix(),
                "width": image_width,
                "height": image_height,
            }
        )
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            raise ValueError(f"{label_path}: shapes must be a list")
        label_counts: Counter[str] = Counter()
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = shape.get("label")
            # Keep screen geometry in the correction manifest rather than train a
            # sixth detector class.  It is intentionally ignored here.
            if label == "screen_quad":
                continue
            if not isinstance(label, str):
                raise ValueError(f"{label_path}: every shape needs a string label")
            validate_label(label)
            label_counts[label] += 1
            bbox = _shape_bbox(shape, image_width, image_height)
            if bbox is None:
                raise ValueError(f"{label_path}: invalid or empty box for {label!r}")
            annotation: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": LABEL_TO_ID[label],
                "bbox": [round(value, 3) for value in bbox],
                "area": round(bbox[2] * bbox[3], 3),
                "iscrowd": 0,
            }
            description = shape.get("description")
            if isinstance(description, str) and description.strip():
                annotation["text"] = description.strip()
            coco["annotations"].append(annotation)
            annotation_id += 1

        if require_complete:
            missing = [label for label in DETECTION_CLASSES if label_counts[label] == 0]
            duplicates = [f"{label}×{label_counts[label]}" for label in DETECTION_CLASSES if label_counts[label] > 1]
            if missing or duplicates:
                details: list[str] = []
                if missing:
                    details.append("missing=" + ",".join(missing))
                if duplicates:
                    details.append("duplicates=" + ",".join(duplicates))
                raise ValueError(
                    f"{label_path}: expected exactly one box for each of the five labels; "
                    + "; ".join(details)
                )

    output_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return coco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert LabelMe boxes to the fixed transfer receipt COCO schema")
    parser.add_argument("--labels", type=Path, required=True, help="Folder containing LabelMe JSON files")
    parser.add_argument("--images", type=Path, required=True, help="Folder containing the corresponding rectified images")
    parser.add_argument("--output", type=Path, required=True, help="COCO annotations JSON to create")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require exactly one box for each of the five labels in every image",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    coco = convert_labelme_to_coco(
        args.labels,
        args.images,
        args.output,
        require_complete=args.require_complete,
    )
    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} boxes to {args.output}")
    category_names = {category["id"]: category["name"] for category in coco["categories"]}
    counts = Counter(category_names[annotation["category_id"]] for annotation in coco["annotations"])
    print("Per class: " + ", ".join(f"{label}={counts[label]}" for label in DETECTION_CLASSES))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
