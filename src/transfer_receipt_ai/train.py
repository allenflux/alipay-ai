"""GPU-ready training CLI for the LRCNN transfer receipt detector."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ReceiptCocoDataset, collate_detection_batch
from .labels import DETECTION_CLASSES
from .metrics import evaluate_map50
from .model import LRCNNConfig, build_lrcnn, choose_device, validate_checkpoint_classes


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 0.005
    weight_decay: float = 0.0005
    num_workers: int = 4
    amp: bool = True
    pretrained: bool = True
    seed: int = 42
    save_every: int = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_targets(targets: list[dict[str, torch.Tensor]], device: str) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


@torch.inference_mode()
def evaluate_detector(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, object]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_on_cpu: list[dict[str, torch.Tensor]] = []
    for images, targets in tqdm(loader, desc="Validation", leave=False):
        outputs = model([image.to(device) for image in images])
        predictions.extend({key: value.detach().cpu() for key, value in output.items()} for output in outputs)
        targets_on_cpu.extend({key: value.detach().cpu() for key, value in target.items()} for target in targets)
    return evaluate_map50(predictions, targets_on_cpu)


def _checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: LRCNNConfig,
    training_config: TrainingConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model_config.as_dict(),
        "training_config": asdict(training_config),
        "classes": list(DETECTION_CLASSES),
        "metrics": metrics,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def train_detector(
    *,
    train_images: Path,
    train_annotations: Path,
    output_dir: Path,
    validation_images: Path | None = None,
    validation_annotations: Path | None = None,
    model_config: LRCNNConfig | None = None,
    training_config: TrainingConfig | None = None,
    device: str = "auto",
    resume: Path | None = None,
) -> Path:
    """Train and return the path to the best checkpoint."""
    training_config = training_config or TrainingConfig()
    model_config = model_config or LRCNNConfig(pretrained=training_config.pretrained)
    seed_everything(training_config.seed)
    selected_device = choose_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ReceiptCocoDataset(train_images, train_annotations, training=True)
    val_dataset = (
        ReceiptCocoDataset(validation_images, validation_annotations, training=False)
        if validation_images is not None and validation_annotations is not None
        else None
    )
    loader_options: dict[str, Any] = {
        "batch_size": training_config.batch_size,
        "collate_fn": collate_detection_batch,
        "num_workers": training_config.num_workers,
        "pin_memory": selected_device.startswith("cuda"),
    }
    if training_config.num_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options) if val_dataset is not None else None

    start_epoch = 0
    resume_payload: dict[str, Any] | None = None
    if resume is not None:
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
        validate_checkpoint_classes(resume_payload)
        checkpoint_config = resume_payload.get("model_config", {})
        if isinstance(checkpoint_config, dict):
            model_config = LRCNNConfig(**checkpoint_config)

    # A resumed model obtains all its weights from the checkpoint, while keeping
    # the original normalization architecture recorded in model_config.
    model = build_lrcnn(model_config, load_pretrained_weights=resume_payload is None).to(selected_device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training_config.learning_rate,
        momentum=0.9,
        weight_decay=training_config.weight_decay,
    )
    milestones = sorted({max(1, int(training_config.epochs * 0.65)), max(1, int(training_config.epochs * 0.85))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    amp_enabled = selected_device.startswith("cuda") and training_config.amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        start_epoch = int(resume_payload.get("epoch", -1)) + 1

    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_metric = -math.inf
    print(f"Training on {selected_device}: {len(train_dataset)} train image(s), {len(val_dataset) if val_dataset else 0} validation image(s)")
    for epoch in range(start_epoch, training_config.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_config.epochs}")
        for images, targets in progress:
            images = [image.to(selected_device, non_blocking=True) for image in images]
            targets = _move_targets(targets, selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss_by_name = model(images, targets)
                total_loss = sum(loss_by_name.values())
            loss_value = float(total_loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: {loss_by_name}")
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss_value
            progress.set_postfix(loss=f"{running_loss / (progress.n or 1):.4f}", lr=optimizer.param_groups[0]["lr"])
        scheduler.step()

        metrics: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            validation = evaluate_detector(model, val_loader, selected_device)
            metrics["validation"] = validation
            metric_to_compare = float(validation["map50"])
        else:
            metric_to_compare = -metrics["train_loss"]
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            training_config=training_config,
            metrics=metrics,
        )
        _save_checkpoint(last_path, payload)
        if metric_to_compare > best_metric:
            best_metric = metric_to_compare
            _save_checkpoint(best_path, payload)
            print(f"Saved new best checkpoint: {best_path} (metric={best_metric:.4f})")
        if (epoch + 1) % training_config.save_every == 0:
            _save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pt", payload)
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-FPN LRCNN on transfer receipt boxes")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto (CUDA first), cuda, cuda:0, mps, or cpu")
    parser.add_argument("--min-size", type=int, default=768)
    parser.add_argument("--max-size", type=int, default=1536)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Start from COCO weights (default: true)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA mixed precision (default: true)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=Path, help="Resume from last.pt or another checkpoint")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if bool(args.val_images) != bool(args.val_annotations):
        raise SystemExit("--val-images and --val-annotations must be supplied together")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0 or args.save_every <= 0:
        raise SystemExit("epochs, batch-size and save-every must be positive; workers cannot be negative")
    best = train_detector(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        validation_images=args.val_images,
        validation_annotations=args.val_annotations,
        output_dir=args.output,
        model_config=LRCNNConfig(min_size=args.min_size, max_size=args.max_size, pretrained=args.pretrained),
        training_config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.workers,
            amp=args.amp,
            pretrained=args.pretrained,
            seed=args.seed,
            save_every=args.save_every,
        ),
        device=args.device,
        resume=args.resume,
    )
    print(f"Training complete. Best checkpoint: {best}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # Harmless on Linux; required for safe multiprocessing startup in some
    # Windows Server deployment environments.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
