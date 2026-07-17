"""Evaluate one trained LRCNN checkpoint on an untouched COCO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .labels import DETECTION_CLASSES


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _format_score(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def format_metrics(metrics: dict[str, Any]) -> str:
    """Return a compact console report with every required receipt field."""
    lines = [f"mAP@IoU=0.50 (score>=0.05): {_format_score(metrics.get('map50'))}"]
    ap_by_class = metrics.get("ap50", {})
    recall_by_class = metrics.get("recall50", {})
    for label in DETECTION_CLASSES:
        ap = ap_by_class.get(label) if isinstance(ap_by_class, dict) else None
        recall = recall_by_class.get(label) if isinstance(recall_by_class, dict) else None
        lines.append(f"  {label:<22} AP50={_format_score(ap)}  Recall50={_format_score(recall)}")
    return "\n".join(lines)


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    images: Path,
    annotations: Path,
    output: Path,
    device: str = "auto",
    batch_size: int = 2,
    workers: int = 0,
) -> dict[str, Any]:
    """Run detector metrics and persist a reproducible JSON report."""
    import torch
    from torch.utils.data import DataLoader

    from .dataset import ReceiptCocoDataset, collate_detection_batch
    from .model import LRCNNConfig, build_lrcnn, choose_device, validate_checkpoint_classes
    from .train import evaluate_detector

    selected_device = choose_device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_classes(payload)
    checkpoint_config = payload.get("model_config") if isinstance(payload, dict) else None
    config = LRCNNConfig(**checkpoint_config) if isinstance(checkpoint_config, dict) else LRCNNConfig()
    model = build_lrcnn(config, load_pretrained_weights=False)
    state_dict = payload.get("model_state") if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state dictionary")
    model.load_state_dict(state_dict)
    model.to(selected_device)

    dataset = ReceiptCocoDataset(images, annotations, training=False)
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_detection_batch,
        "num_workers": workers,
        "pin_memory": selected_device.startswith("cuda"),
    }
    if workers > 0:
        loader_options["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_options)
    metrics = evaluate_detector(model, loader, selected_device)
    ap_by_class = metrics.get("ap50", {})
    missing_ground_truth = [
        label for label in DETECTION_CLASSES if not isinstance(ap_by_class, dict) or ap_by_class.get(label) is None
    ]
    if missing_ground_truth:
        raise ValueError(f"Test annotations contain no ground truth for: {', '.join(missing_ground_truth)}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": checkpoint.resolve().as_posix(),
        "images": images.resolve().as_posix(),
        "annotations": annotations.resolve().as_posix(),
        "device": selected_device,
        "image_count": len(dataset),
        "batch_size": batch_size,
        "workers": workers,
        "metrics": metrics,
    }
    _atomic_write_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained receipt LRCNN on a COCO split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("--batch-size must be positive and --workers cannot be negative")
    report = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        images=args.images,
        annotations=args.annotations,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(f"Evaluated {report['image_count']} image(s) on {report['device']}")
    print(format_metrics(report["metrics"]))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
