"""Add status-style sidecars to existing five-field v1 inference results.

This is a non-destructive second pass: it reads the source image and stored v1
geometry, reconstructs the clean ``transfer_status`` crop, then writes a new
JSON tree.  Neither source images nor v1 result JSON files are edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tqdm import tqdm

from .geometry import load_upright_rgb
from .status_crops import crop_status_region, reconstruct_rectified
from .status_style import StatusStylePredictor


SIDECAR_SCHEMA_VERSION = 2
BUSINESS_RULE_VERSION = "status-style-v2"


class UnsafeOutputPathError(ValueError):
    """Raised when tag output could pollute input results or source images."""


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _selection_key(relative_path: Path) -> bytes:
    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).digest()


def _shard_for(relative_path: Path, shard_count: int) -> int:
    return int.from_bytes(_selection_key(relative_path)[:8], "big") % shard_count


def _manifest_paths(output_dir: Path, shard_index: int, shard_count: int) -> tuple[Path, Path, Path]:
    suffix = "" if shard_count == 1 else f".shard-{shard_index:03d}-of-{shard_count:03d}"
    return (
        output_dir / f"status_style_manifest{suffix}.json",
        output_dir / f"status_style_manifest{suffix}.jsonl",
        output_dir / f"status_style_errors{suffix}.jsonl",
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _iter_result_paths(input_path: Path) -> tuple[Path, Iterable[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a v1 result JSON: {input_path}")
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    # The only aggregate .json written by v1 starts with
    # ``inference_manifest``. JSONL logs are naturally excluded.
    paths = sorted(
        path
        for path in input_path.rglob("*.json")
        if not path.name.startswith("inference_manifest")
        and not path.name.endswith(".status_style.json")
    )
    return input_path, paths


def _load_v1_result(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid result JSON: {error}") from None
    if not isinstance(payload, dict):
        raise ValueError("v1 result must be a JSON object")
    for key in ("source", "geometry", "detections"):
        if key not in payload:
            raise ValueError(f"not a v1 receipt result: missing {key}")
    if not isinstance(payload["source"], str) or not payload["source"].strip():
        raise ValueError("v1 result source must be a non-empty path")
    if not isinstance(payload["geometry"], dict) or not isinstance(payload["detections"], list):
        raise ValueError("v1 result geometry/detections have invalid types")
    return payload


def _source_path(payload: Mapping[str, object], result_path: Path) -> Path:
    value = payload.get("source")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("v1 result source must be a non-empty path")
    source_path = Path(value)
    if not source_path.is_absolute():
        source_path = result_path.parent / source_path
    return source_path.resolve()


def _status_bbox(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    matches = [
        detection
        for detection in payload["detections"]
        if isinstance(detection, dict) and detection.get("label") == "transfer_status"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one transfer_status detection; found {len(matches)}")
    bbox = matches[0].get("bbox_rectified")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("transfer_status bbox_rectified must contain four numbers")
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        raise ValueError("transfer_status bbox_rectified must contain four numbers") from None
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("transfer_status bbox_rectified is empty or inverted")
    return values  # type: ignore[return-value]


def _sidecar_path(output_dir: Path, relative_result: Path) -> Path:
    return (output_dir / relative_result).with_suffix(".status_style.json")


def _checkpoint_signature(checkpoint: Path) -> dict[str, object]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": checkpoint.as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": checkpoint.stat().st_size,
    }


def _business_tags(prediction: Mapping[str, object]) -> dict[str, object]:
    label = prediction.get("label")
    if label == "check_offset":
        return {
            "platform": "android",
            "authenticity": "not_assessed",
            "review_tag": None,
            "requires_manual_review": False,
            "reason": "status_check_offset",
            "rule_version": BUSINESS_RULE_VERSION,
        }
    if label == "check_aligned":
        return {
            "platform": "ios",
            "authenticity": "not_assessed",
            "review_tag": None,
            "requires_manual_review": False,
            "reason": "status_check_aligned",
            "rule_version": BUSINESS_RULE_VERSION,
        }
    if label == "check_absent":
        return {
            "platform": None,
            "authenticity": "not_assessed",
            "review_tag": "suspected_fake",
            "requires_manual_review": True,
            "reason": "status_check_absent",
            "rule_version": BUSINESS_RULE_VERSION,
        }
    return {
        "platform": None,
        "authenticity": "not_assessed",
        "review_tag": "review",
        "requires_manual_review": True,
        "reason": "status_style_low_confidence",
        "rule_version": BUSINESS_RULE_VERSION,
    }


def _committed_sidecar_exists(
    sidecar: Path,
    result_path: Path,
    source_path: Path,
    *,
    model_signature: Mapping[str, object],
    inference_config: Mapping[str, object],
) -> bool:
    if not sidecar.is_file():
        return False
    if sidecar.stat().st_mtime_ns < max(result_path.stat().st_mtime_ns, source_path.stat().st_mtime_ns):
        return False
    try:
        payload: Any = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == SIDECAR_SCHEMA_VERSION
        and payload.get("result_json") == result_path.resolve().as_posix()
        and payload.get("source") == source_path.resolve().as_posix()
        and isinstance(payload.get("status_style"), dict)
        and isinstance(payload.get("tags"), dict)
        and payload["tags"].get("rule_version") == BUSINESS_RULE_VERSION
        and payload.get("model") == dict(model_signature)
        and payload.get("inference_config") == dict(inference_config)
    )


def _record(result_path: Path, sidecar: Path, source_path: Path, status: str) -> dict[str, str]:
    return {
        "result_json": result_path.resolve().as_posix(),
        "source": source_path.resolve().as_posix(),
        "sidecar": sidecar.resolve().as_posix(),
        "status": status,
    }


def enrich_status_tags(
    *,
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    device: str = "auto",
    confidence_threshold: float = 0.80,
    absent_confidence_threshold: float = 0.95,
    margin_ratio: float = 0.30,
    skip_existing: bool = False,
    continue_on_error: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
) -> list[dict[str, str]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_count must be positive and shard_index must be in [0, shard_count)")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if margin_ratio < 0:
        raise ValueError("margin_ratio cannot be negative")
    result_root, candidates_iterable = _iter_result_paths(input_path)
    candidates = list(candidates_iterable)
    if not candidates:
        raise ValueError(f"No v1 result JSON files found under {input_path}")
    resolved_root = result_root.resolve()
    resolved_output = output_dir.resolve()
    if _paths_overlap(resolved_output, resolved_root):
        raise UnsafeOutputPathError(
            "output directory and the v1 result directory must be separate, non-overlapping trees"
        )

    model_signature = _checkpoint_signature(checkpoint)
    inference_config: dict[str, object] = {
        "confidence_threshold": confidence_threshold,
        "absent_confidence_threshold": absent_confidence_threshold,
        "margin_ratio": margin_ratio,
    }

    selected = sorted(
        (
            path
            for path in candidates
            if _shard_for(path.relative_to(result_root), shard_count) == shard_index
        ),
        key=lambda path: (_selection_key(path.relative_to(result_root)), path.relative_to(result_root).as_posix()),
    )
    if limit is not None:
        selected = selected[:limit]

    # Validate every referenced source tree before creating the output
    # directory or manifest.  This extra JSON-only pass is intentional: a bad
    # --output value must never leave even metadata files inside the raw tree.
    for result_path in selected:
        try:
            source_path = _source_path(_load_v1_result(result_path), result_path)
        except (OSError, ValueError):
            # Invalid inputs are recorded by the normal processing pass below.
            continue
        if _paths_overlap(resolved_output, source_path.parent):
            raise UnsafeOutputPathError(
                "output directory overlaps a source-image directory; choose a separate tag directory"
            )

    predictor = StatusStylePredictor(
        checkpoint,
        device=device,
        confidence_threshold=confidence_threshold,
        absent_confidence_threshold=absent_confidence_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_jsonl_path, errors_path = _manifest_paths(output_dir, shard_index, shard_count)
    manifest: list[dict[str, str]] = []
    failures = 0
    with manifest_jsonl_path.open("w", encoding="utf-8") as manifest_jsonl, errors_path.open(
        "w", encoding="utf-8"
    ) as errors_jsonl:
        for result_path in tqdm(selected, desc=f"Status tags shard {shard_index + 1}/{shard_count}"):
            relative_result = result_path.relative_to(result_root)
            sidecar = _sidecar_path(output_dir, relative_result)
            try:
                payload = _load_v1_result(result_path)
                source_path = _source_path(payload, result_path)
                if not source_path.is_file():
                    raise FileNotFoundError(f"source image not found: {source_path}")
                if skip_existing and _committed_sidecar_exists(
                    sidecar,
                    result_path,
                    source_path,
                    model_signature=model_signature,
                    inference_config=inference_config,
                ):
                    record = _record(result_path, sidecar, source_path, "skipped_existing")
                    manifest.append(record)
                    manifest_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest_jsonl.flush()
                    continue

                source_rgb = load_upright_rgb(source_path)
                rectified_rgb = reconstruct_rectified(payload, source_rgb)
                bbox = _status_bbox(payload)
                crop = crop_status_region(rectified_rgb, bbox, margin_ratio=margin_ratio)
                prediction = predictor.predict(crop)
                prediction_payload = prediction.as_dict()
                prediction_payload["state"] = (
                    "classified" if prediction_payload.get("label") != "unknown" else "review"
                )
                output = {
                    "schema_version": SIDECAR_SCHEMA_VERSION,
                    "result_json": result_path.resolve().as_posix(),
                    "source": source_path.resolve().as_posix(),
                    "group_id": relative_result.with_suffix("").as_posix(),
                    "transfer_status_bbox_rectified": [round(value, 3) for value in bbox],
                    "model": model_signature,
                    "inference_config": inference_config,
                    "status_style": prediction_payload,
                    "tags": _business_tags(prediction_payload),
                }
                _write_json_atomic(sidecar, output)
                record = _record(result_path, sidecar, source_path, "written")
                manifest.append(record)
                manifest_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_jsonl.flush()
            except Exception as error:
                if isinstance(error, UnsafeOutputPathError):
                    raise
                failures += 1
                error_record = {
                    "result_json": result_path.resolve().as_posix(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                errors_jsonl.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                errors_jsonl.flush()
                if not continue_on_error:
                    raise
    _write_json_atomic(manifest_path, manifest)
    if failures:
        print(f"WARNING: {failures} result(s) failed; see {errors_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich v1 result JSONs with transfer-status style tags")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="v1 result JSON or result directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate sidecar output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--absent-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--margin-ratio", type=float, default=0.30)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.absent_confidence_threshold <= 1.0:
        raise SystemExit("--absent-confidence-threshold must be between 0 and 1")
    if args.absent_confidence_threshold < args.confidence_threshold:
        raise SystemExit("--absent-confidence-threshold must be at least --confidence-threshold")
    if args.margin_ratio < 0:
        raise SystemExit("--margin-ratio cannot be negative")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-count must be positive and --shard-index must be in [0, shard-count)")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    records = enrich_status_tags(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        absent_confidence_threshold=args.absent_confidence_threshold,
        margin_ratio=args.margin_ratio,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit=args.limit,
    )
    written = sum(record["status"] == "written" for record in records)
    skipped = len(records) - written
    print(f"Status enrichment complete: written={written}, skipped_existing={skipped}, output={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
