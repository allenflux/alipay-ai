"""Train the three-way transfer-status check-mark style classifier.

The detector/OCR model remains unchanged.  This small classifier only sees a
clean crop of the already detected ``transfer_status`` field and predicts one
of ``check_offset``, ``check_aligned`` or ``check_absent``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import choose_device
from .status_style import (
    STATUS_STYLE_CLASSES,
    StatusStyleConfig,
    build_status_style_model,
    preprocess_status_crop,
    validate_status_style_checkpoint,
)


@dataclass(frozen=True)
class StatusStyleRecord:
    crop: Path
    source: Path
    result_json: Path
    group_id: str
    label: str


@dataclass(frozen=True)
class StatusStyleTrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    pretrained: bool = True
    seed: int = 42
    val_ratio: float = 0.20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_reviewed_records(manifest_path: Path) -> tuple[list[StatusStyleRecord], int]:
    """Load clear reviewed records and return ``(records, skipped_unclear)``.

    Exported manifests intentionally start with ``label: null``.  Those rows,
    plus rows explicitly marked ``unclear``, are ignored rather than silently
    becoming a trainable class.  Any other unknown label is a hard error.
    """

    records: list[StatusStyleRecord] = []
    skipped = 0
    with manifest_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON: {error}") from None
            if not isinstance(payload, dict):
                raise ValueError(f"{manifest_path}:{line_number}: each line must be a JSON object")
            missing = [key for key in ("crop", "source", "result_json", "group_id", "label") if key not in payload]
            if missing:
                raise ValueError(f"{manifest_path}:{line_number}: missing {', '.join(missing)}")
            label = payload["label"]
            if label is None or label == "unclear":
                skipped += 1
                continue
            if label not in STATUS_STYLE_CLASSES:
                raise ValueError(
                    f"{manifest_path}:{line_number}: unknown label {label!r}; "
                    f"expected one of {list(STATUS_STYLE_CLASSES)} or unclear"
                )
            path_values = {key: payload[key] for key in ("crop", "source", "result_json")}
            if any(not isinstance(value, str) or not value.strip() for value in path_values.values()):
                raise ValueError(f"{manifest_path}:{line_number}: crop/source/result_json must be non-empty strings")
            group_id = payload["group_id"]
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(f"{manifest_path}:{line_number}: group_id must be a non-empty string")
            crop = _manifest_path(path_values["crop"], manifest_path)
            if not crop.is_file():
                raise FileNotFoundError(f"{manifest_path}:{line_number}: crop not found: {crop}")
            records.append(
                StatusStyleRecord(
                    crop=crop,
                    source=_manifest_path(path_values["source"], manifest_path),
                    result_json=_manifest_path(path_values["result_json"], manifest_path),
                    group_id=group_id,
                    label=label,
                )
            )
    if not records:
        raise ValueError(f"No reviewed clear labels found in {manifest_path}")
    return records, skipped


def _group_order_key(group_id: str, seed: int) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return digest, group_id


def split_records_by_group(
    records: Sequence[StatusStyleRecord],
    *,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[StatusStyleRecord], list[StatusStyleRecord]]:
    """Make a deterministic, label-stratified split without group leakage."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    grouped: dict[str, list[StatusStyleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    groups_by_label: dict[str, list[str]] = {label: [] for label in STATUS_STYLE_CLASSES}
    for group_id, members in grouped.items():
        labels = {member.label for member in members}
        if len(labels) != 1:
            raise ValueError(f"group_id {group_id!r} has conflicting labels: {sorted(labels)}")
        groups_by_label[next(iter(labels))].append(group_id)

    missing = [label for label, groups in groups_by_label.items() if not groups]
    if missing:
        raise ValueError(f"Training data has no reviewed examples for: {', '.join(missing)}")

    validation_groups: set[str] = set()
    for label, group_ids in groups_by_label.items():
        ordered = sorted(group_ids, key=lambda value: _group_order_key(value, seed))
        # A singleton must stay in training.  With two or more groups, retain at
        # least one for both train and validation regardless of rounding.
        if len(ordered) >= 2:
            count = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_ratio))))
            validation_groups.update(ordered[:count])
    if not validation_groups:
        raise ValueError("A validation split needs at least two distinct groups in one class")

    train = [record for record in records if record.group_id not in validation_groups]
    validation = [record for record in records if record.group_id in validation_groups]
    train_groups = {record.group_id for record in train}
    validation_group_ids = {record.group_id for record in validation}
    if train_groups & validation_group_ids:  # pragma: no cover - defensive invariant
        raise AssertionError("group leakage in status-style split")
    return train, validation


def class_weights(records: Sequence[StatusStyleRecord]) -> torch.Tensor:
    counts = Counter(record.label for record in records)
    missing = [label for label in STATUS_STYLE_CLASSES if counts[label] == 0]
    if missing:
        raise ValueError(f"Training split has no examples for: {', '.join(missing)}")
    total = len(records)
    return torch.tensor(
        [total / (len(STATUS_STYLE_CLASSES) * counts[label]) for label in STATUS_STYLE_CLASSES],
        dtype=torch.float32,
    )


class StatusStyleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: Sequence[StatusStyleRecord], model_config: StatusStyleConfig) -> None:
        self.records = list(records)
        self.model_config = model_config
        self.label_to_id = {label: index for index, label in enumerate(STATUS_STYLE_CLASSES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.crop) as image:
            image_rgb = np.asarray(image.convert("RGB"))
        # Deliberately no flips, rotations, crops or other geometry-changing
        # augmentation: check-mark alignment is the feature being learned.
        tensor = preprocess_status_crop(image_rgb, self.model_config)
        return tensor, torch.tensor(self.label_to_id[record.label], dtype=torch.long)


def _metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    confusion = confusion.to(torch.float64)
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    true_positive = confusion.diag()
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator))
    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(STATUS_STYLE_CLASSES)
    }
    total = int(confusion.sum())
    return {
        "accuracy": round(float(true_positive.sum() / total), 6) if total else 0.0,
        "macro_f1": round(float(f1.mean()), 6),
        "per_class": per_class,
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


@torch.inference_mode()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros((len(STATUS_STYLE_CLASSES), len(STATUS_STYLE_CLASSES)), dtype=torch.int64)
    running_loss = 0.0
    examples = 0
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += float(loss.detach().cpu()) * labels.shape[0]
        examples += labels.shape[0]
        for actual, predicted in zip(labels.detach().cpu(), logits.argmax(dim=1).detach().cpu()):
            confusion[int(actual), int(predicted)] += 1
    metrics = _metrics_from_confusion(confusion)
    metrics["loss"] = running_loss / max(1, examples)
    return metrics


def _config_dict(config: StatusStyleConfig) -> dict[str, Any]:
    if hasattr(config, "as_dict"):
        return dict(config.as_dict())
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_status_style_classifier(
    *,
    reviewed_manifest: Path,
    output_dir: Path,
    model_config: StatusStyleConfig | None = None,
    training_config: StatusStyleTrainingConfig | None = None,
    device: str = "auto",
) -> Path:
    training_config = training_config or StatusStyleTrainingConfig()
    model_config = model_config or StatusStyleConfig(pretrained=training_config.pretrained)
    seed_everything(training_config.seed)
    selected_device = choose_device(device)
    records, skipped = load_reviewed_records(reviewed_manifest)
    train_records, validation_records = split_records_by_group(
        records,
        val_ratio=training_config.val_ratio,
        seed=training_config.seed,
    )
    weights = class_weights(train_records)
    train_dataset = StatusStyleDataset(train_records, model_config)
    validation_dataset = StatusStyleDataset(validation_records, model_config)
    loader_options: dict[str, Any] = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "pin_memory": selected_device.startswith("cuda"),
    }
    if training_config.num_workers > 0:
        loader_options["persistent_workers"] = True
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = build_status_style_model(model_config).to(selected_device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(selected_device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.epochs)
    amp_enabled = selected_device.startswith("cuda") and training_config.amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_metric = -math.inf
    print(
        f"Training status style on {selected_device}: train={len(train_records)}, "
        f"validation={len(validation_records)}, skipped_unreviewed_or_unclear={skipped}"
    )
    for epoch in range(training_config.epochs):
        model.train()
        running_loss = 0.0
        examples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_config.epochs}")
        for images, labels in progress:
            images = images.to(selected_device, non_blocking=True)
            labels = labels.to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: {loss_value}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss_value * labels.shape[0]
            examples += labels.shape[0]
            progress.set_postfix(loss=f"{running_loss / max(1, examples):.4f}")
        scheduler.step()

        validation = evaluate_classifier(model, validation_loader, criterion, selected_device)
        metrics: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, examples),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        payload = {
            "epoch": epoch,
            "classes": list(STATUS_STYLE_CLASSES),
            "input_size": [model_config.input_width, model_config.input_height],
            "model_config": _config_dict(model_config),
            "training_config": asdict(training_config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "class_weights": weights.tolist(),
            "metrics": metrics,
        }
        # Validate our own semantic metadata before committing a deployable file.
        validate_status_style_checkpoint(payload)
        _save_checkpoint(last_path, payload)
        metric = float(validation["macro_f1"])
        if metric > best_metric:
            best_metric = metric
            _save_checkpoint(best_path, payload)
            print(f"Saved new best checkpoint: {best_path} (macro_f1={best_metric:.4f})")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train transfer-status check-mark style classifier")
    parser.add_argument("--records", type=Path, required=True, help="Reviewed status-crop JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=128)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("epochs and batch-size must be positive; workers cannot be negative")
    if args.input_width <= 0 or args.input_height <= 0:
        raise SystemExit("input dimensions must be positive")
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1")
    best = train_status_style_classifier(
        reviewed_manifest=args.records,
        output_dir=args.output,
        model_config=StatusStyleConfig(
            input_width=args.input_width,
            input_height=args.input_height,
            pretrained=args.pretrained,
        ),
        training_config=StatusStyleTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.workers,
            amp=args.amp,
            pretrained=args.pretrained,
            seed=args.seed,
            val_ratio=args.val_ratio,
        ),
        device=args.device,
    )
    print(f"Training complete. Best checkpoint: {best}")


if __name__ == "__main__":  # pragma: no cover
    import multiprocessing

    multiprocessing.freeze_support()
    main()
