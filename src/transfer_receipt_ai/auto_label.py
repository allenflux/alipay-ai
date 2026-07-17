"""Generate reviewable LabelMe rectangles from an LRCNN checkpoint.

The input must be the already-rectified images used by LabelMe.  Running the
normal receipt pipeline here would rectify a second time and make the generated
box coordinates disagree with the annotation images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

from tqdm import tqdm

from .geometry import load_upright_rgb
from .labels import DETECTION_CLASSES
from .model import Detection, LRCNNPredictor
from .prepare import iter_image_paths


class DetectionPredictor(Protocol):
    def predict(self, image_rgb) -> list[Detection]:
        """Return at most one preferred detection per semantic label."""


def _label_path(output_dir: Path, relative_image: Path) -> Path:
    return (output_dir / relative_image).with_suffix(".json")


def _existing_label_path(labels_dir: Path, relative_image: Path) -> Path | None:
    candidates = (
        (labels_dir / relative_image).with_suffix(".json"),
        labels_dir / relative_image.with_suffix(".json").name,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _clamp_detection(detection: Detection, image_width: int, image_height: int) -> Detection | None:
    coordinates = detection.bbox_xyxy
    if not all(math.isfinite(value) for value in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(detection.label, detection.score, (x1, y1, x2, y2))


def _labelme_document(
    *,
    image_path: Path,
    label_path: Path,
    image_width: int,
    image_height: int,
    detections: Sequence[Detection],
) -> dict[str, Any]:
    by_label = {detection.label: detection for detection in detections}
    shapes: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for label in DETECTION_CLASSES:
        detection = by_label.get(label)
        if detection is None:
            continue
        x1, y1, x2, y2 = detection.bbox_xyxy
        shapes.append(
            {
                "label": label,
                "points": [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)]],
                "group_id": None,
                # Do not put confidence here. The converter treats description
                # as manually verified OCR truth.
                "description": "",
                "shape_type": "rectangle",
                "flags": {"auto_generated": True},
                "mask": None,
            }
        )
        scores[label] = round(float(detection.score), 6)
    missing = [label for label in DETECTION_CLASSES if label not in by_label]
    return {
        "version": "6.3.0",
        "flags": {"auto_generated": True},
        "shapes": shapes,
        "imagePath": os.path.relpath(image_path, label_path.parent),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "_auto_label": {
            "reviewed": False,
            "scores": scores,
            "missing": missing,
        },
    }


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_auto_labels(
    *,
    predictor: DetectionPredictor,
    images_path: Path,
    output_dir: Path,
    existing_labels_dirs: Sequence[Path] = (),
    review_threshold: float = 0.80,
    require_complete: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Predict images and create LabelMe JSON files for human review.

    Existing JSON files are skipped unless ``overwrite`` is explicitly enabled.
    Incomplete predictions are normally written so the reviewer only needs to add
    the missing boxes; ``require_complete`` can instead omit them.
    """
    if not 0.0 <= review_threshold <= 1.0:
        raise ValueError("review_threshold must be between 0 and 1")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    image_paths = list(iter_image_paths(images_path))
    if not image_paths:
        raise ValueError(f"No supported images found under {images_path}")
    root = images_path.parent if images_path.is_file() else images_path
    records: list[dict[str, Any]] = []
    predicted_count = 0

    for image_path in tqdm(image_paths, desc="Auto-labelling images"):
        relative_image = image_path.relative_to(root)
        output_path = _label_path(output_dir, relative_image)
        protected_path = output_path if output_path.is_file() else None
        if protected_path is None:
            for labels_dir in existing_labels_dirs:
                protected_path = _existing_label_path(labels_dir, relative_image)
                if protected_path is not None:
                    break
        if protected_path is not None and not overwrite:
            records.append(
                {
                    "image": relative_image.as_posix(),
                    "status": "skipped_existing",
                    "existing_label": protected_path.resolve().as_posix(),
                }
            )
            continue
        if limit is not None and predicted_count >= limit:
            break
        predicted_count += 1

        image_rgb = load_upright_rgb(image_path)
        image_height, image_width = image_rgb.shape[:2]
        raw_detections = predictor.predict(image_rgb)
        valid_by_label: dict[str, Detection] = {}
        invalid_labels: list[str] = []
        for detection in raw_detections:
            if detection.label not in DETECTION_CLASSES:
                invalid_labels.append(detection.label)
                continue
            clamped = _clamp_detection(detection, image_width, image_height)
            if clamped is None:
                invalid_labels.append(detection.label)
                continue
            previous = valid_by_label.get(clamped.label)
            if previous is None or clamped.score > previous.score:
                valid_by_label[clamped.label] = clamped

        detections = [valid_by_label[label] for label in DETECTION_CLASSES if label in valid_by_label]
        missing = [label for label in DETECTION_CLASSES if label not in valid_by_label]
        scores = {detection.label: round(float(detection.score), 6) for detection in detections}
        low_confidence = [label for label, score in scores.items() if score < review_threshold]
        if require_complete and missing:
            status = "skipped_incomplete"
        elif missing or low_confidence or invalid_labels:
            status = "written_needs_review"
        else:
            status = "written_complete"

        if status != "skipped_incomplete" and not dry_run:
            document = _labelme_document(
                image_path=image_path.resolve(),
                label_path=output_path.resolve(),
                image_width=image_width,
                image_height=image_height,
                detections=detections,
            )
            _atomic_write_json(output_path, document)
        records.append(
            {
                "image": relative_image.as_posix(),
                "label": output_path.resolve().as_posix(),
                "status": "dry_run_" + status if dry_run else status,
                "found": [detection.label for detection in detections],
                "missing": missing,
                "low_confidence": low_confidence,
                "invalid": invalid_labels,
                "scores": scores,
                "min_score": min(scores.values()) if scores else None,
            }
        )
    return records


def _write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reviewable LabelMe boxes from an LRCNN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Already-rectified LabelMe image folder")
    parser.add_argument("--output", type=Path, required=True, help="LabelMe JSON output folder")
    parser.add_argument(
        "--existing-labels",
        type=Path,
        action="append",
        default=[],
        help="Additional manual-label folder to protect; may be repeated",
    )
    parser.add_argument("--manifest", type=Path, help="JSONL review manifest outside the label folder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--review-threshold", type=float, default=0.80)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing LabelMe JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Run predictions without writing LabelMe JSON")
    parser.add_argument("--limit", type=int, help="Predict at most this many previously-unlabelled images")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if not 0.0 <= args.review_threshold <= 1.0:
        raise SystemExit("--review-threshold must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    predictor = LRCNNPredictor(
        args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
    )
    records = generate_auto_labels(
        predictor=predictor,
        images_path=args.images,
        output_dir=args.output,
        existing_labels_dirs=args.existing_labels,
        review_threshold=args.review_threshold,
        require_complete=args.require_complete,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    manifest_path = args.manifest or args.output.parent / f"{args.output.name}_auto_manifest.jsonl"
    _write_manifest(manifest_path, records)
    counts = Counter(record["status"] for record in records)
    print(f"Auto-label manifest: {manifest_path}")
    print("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))


if __name__ == "__main__":  # pragma: no cover
    main()
