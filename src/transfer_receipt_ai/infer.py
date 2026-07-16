"""Batch inference CLI: raw image → circles + extracted transfer fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .geometry import RectificationOptions
from .model import LRCNNPredictor
from .ocr import PaddleOCRReader
from .pipeline import ReceiptPipeline, write_receipt_result
from .prepare import iter_image_paths, load_corrections


def _parse_orientation(value: str) -> int | None:
    if value == "auto":
        return None
    degrees = int(value)
    if degrees not in {0, 90, 180, 270}:
        raise argparse.ArgumentTypeError("orientation must be auto, 0, 90, 180 or 270")
    return degrees


def _override_for(corrections: dict[str, dict[str, Any]], relative_path: Path, source_path: Path) -> dict[str, Any]:
    return corrections.get(relative_path.as_posix(), corrections.get(source_path.name, {}))


def run_inference(
    *,
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    score_threshold: float = 0.50,
    device: str = "auto",
    use_ocr: bool = True,
    corrections: dict[str, dict[str, Any]] | None = None,
    orientation_degrees: int | None = None,
    prefer_portrait: bool = True,
    auto_screen: bool = True,
    max_side: int = 1600,
    use_ocr_orientation: bool = False,
) -> list[dict[str, str]]:
    """Process an image or image tree and write one result bundle per raw image."""
    image_paths = list(iter_image_paths(input_path))
    if not image_paths:
        raise ValueError(f"No supported images found under {input_path}")
    corrections = corrections or {}
    ocr = PaddleOCRReader() if use_ocr else None
    predictor = LRCNNPredictor(checkpoint, device=device, score_threshold=score_threshold)
    root = input_path.parent if input_path.is_file() else input_path
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for source_path in tqdm(image_paths, desc="Running LRCNN inference"):
        relative_path = source_path.relative_to(root)
        override = _override_for(corrections, relative_path, source_path)
        if override.get("skip") is True:
            continue
        options = RectificationOptions(
            orientation_degrees=override.get("orientation_degrees", orientation_degrees),
            prefer_portrait=bool(override.get("prefer_portrait", prefer_portrait)),
            auto_screen=bool(override.get("auto_screen", auto_screen)),
            screen_quad=override.get("screen_quad"),
            max_side=int(override.get("max_side", max_side)),
            orientation_scorer=ocr.orientation_score if ocr and use_ocr_orientation else None,
        )
        pipeline = ReceiptPipeline(predictor, ocr=ocr, rectification_options=options)
        result = pipeline.run(source_path)
        output_stem = (output_dir / relative_path).with_suffix("")
        written = write_receipt_result(result, output_stem)
        manifest.append(
            {
                "source": source_path.resolve().as_posix(),
                "result": written["json"].resolve().as_posix(),
                "annotated_rectified": written["rectified_annotation"].resolve().as_posix(),
                "annotated_original": written["original_annotation"].resolve().as_posix(),
            }
        )
    (output_dir / "inference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, OCR and circle transfer receipt fields")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Raw image or raw-image directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--ocr", choices=("paddle", "none"), default="paddle")
    parser.add_argument("--corrections", type=Path, help="Optional per-image manual correction JSON")
    parser.add_argument("--orientation", type=_parse_orientation, default=None)
    parser.add_argument("--landscape-ok", action="store_true")
    parser.add_argument("--no-auto-screen", action="store_true")
    parser.add_argument("--ocr-orientation", action="store_true", help="Use OCR to distinguish all four orientations")
    parser.add_argument("--max-side", type=int, default=1600)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if args.max_side < 2:
        raise SystemExit("--max-side must be at least 2")
    outputs = run_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        score_threshold=args.score_threshold,
        use_ocr=args.ocr == "paddle",
        corrections=load_corrections(args.corrections),
        orientation_degrees=args.orientation,
        prefer_portrait=not args.landscape_ok,
        auto_screen=not args.no_auto_screen,
        max_side=args.max_side,
        use_ocr_orientation=args.ocr_orientation,
    )
    print(f"Wrote {len(outputs)} inference result bundle(s) to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
