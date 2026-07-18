"""Find and safely quarantine incomplete LabelMe annotations."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .labels import DETECTION_CLASSES


def _incomplete_reason(document: Any) -> str | None:
    """Return why a LabelMe document is not exactly five required boxes."""
    if not isinstance(document, dict) or "shapes" not in document:
        return None
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        return "invalid_shapes"
    counts: Counter[str] = Counter()
    unknown: list[str] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            unknown.append("<invalid_shape>")
            continue
        label = shape.get("label")
        if label == "screen_quad":
            continue
        if isinstance(label, str) and label in DETECTION_CLASSES:
            counts[label] += 1
        else:
            unknown.append(str(label))
    missing = [label for label in DETECTION_CLASSES if counts[label] == 0]
    duplicates = [f"{label}x{counts[label]}" for label in DETECTION_CLASSES if counts[label] > 1]
    details: list[str] = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if duplicates:
        details.append("duplicates=" + ",".join(duplicates))
    if unknown:
        details.append("unknown=" + ",".join(unknown))
    return "; ".join(details) if details else None


def clean_incomplete_labels(
    *,
    labels_dir: Path,
    action: str,
    rejected_dir: Path | None = None,
    report_path: Path | None = None,
) -> list[dict[str, str]]:
    """Dry-run, quarantine, or delete incomplete JSON without touching images."""
    if action not in {"dry-run", "quarantine", "delete"}:
        raise ValueError("action must be dry-run, quarantine, or delete")
    labels_dir = labels_dir.resolve()
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)
    if action == "quarantine" and rejected_dir is None:
        raise ValueError("quarantine action requires rejected_dir")
    resolved_rejected = rejected_dir.resolve() if rejected_dir is not None else None
    if resolved_rejected is not None:
        try:
            resolved_rejected.relative_to(labels_dir)
        except ValueError:
            pass
        else:
            raise ValueError("rejected_dir must be outside labels_dir")

    records: list[dict[str, str]] = []
    unreadable: list[str] = []
    candidates: list[tuple[Path, Path, str]] = []
    for label_path in sorted(labels_dir.rglob("*.json")):
        relative_path = label_path.relative_to(labels_dir)
        try:
            document = json.loads(label_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            unreadable.append(f"{relative_path.as_posix()}: {error}")
            continue
        reason = _incomplete_reason(document)
        if reason is None:
            continue
        candidates.append((label_path, relative_path, reason))

    if unreadable:
        details = "\n".join(f"  - {item}" for item in unreadable)
        raise ValueError(f"Unreadable JSON files were not touched:\n{details}")

    if action == "quarantine" and resolved_rejected is not None:
        collisions = [
            resolved_rejected / relative_path
            for _, relative_path, _ in candidates
            if (resolved_rejected / relative_path).exists()
        ]
        if collisions:
            raise FileExistsError(f"Rejected destination already exists: {collisions[0]}")

    for label_path, relative_path, reason in candidates:
        destination = ""
        status = "would_quarantine"
        if action == "quarantine" and resolved_rejected is not None:
            target = resolved_rejected / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label_path), str(target))
            destination = target.as_posix()
            status = "quarantined"
        elif action == "delete":
            label_path.unlink()
            status = "deleted"
        elif action == "dry-run":
            status = "would_quarantine"
        records.append(
            {
                "label": label_path.as_posix(),
                "relative_path": relative_path.as_posix(),
                "reason": reason,
                "status": status,
                "destination": destination,
            }
        )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as report:
            for record in records:
                report.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(report_path)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find LabelMe JSON files that are not exactly one box for each of the five required labels"
    )
    parser.add_argument("--labels", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", dest="action", action="store_const", const="dry-run")
    action.add_argument("--quarantine", dest="action", action="store_const", const="quarantine")
    action.add_argument("--delete", dest="action", action="store_const", const="delete")
    parser.add_argument("--rejected", type=Path, help="Destination used with --quarantine")
    parser.add_argument("--report", type=Path, help="Optional JSONL action report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "quarantine" and args.rejected is None:
        raise SystemExit("--quarantine requires --rejected")
    if args.action != "quarantine" and args.rejected is not None:
        raise SystemExit("--rejected is only valid with --quarantine")
    try:
        records = clean_incomplete_labels(
            labels_dir=args.labels,
            action=args.action,
            rejected_dir=args.rejected,
            report_path=args.report,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Incomplete-label cleanup failed:\n{error}") from None
    for record in records:
        print(f"{record['status']}: {record['relative_path']}: {record['reason']}")
    print(f"Summary: action={args.action}, incomplete={len(records)}")
    print("Only LabelMe JSON files were considered; image files were not changed.")


if __name__ == "__main__":  # pragma: no cover
    main()
