"""Render a non-destructive HTML review page for status-style sidecars.

The status-style enrichment pass intentionally writes small JSON sidecars.
This module turns a deterministic subset of those sidecars into an easy to
scan HTML page.  It reconstructs each clean transfer-status crop from the
original source plus the v1 geometry; source images, v1 results, and sidecars
are never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .enrich_status_tags import (
    BUSINESS_RULE_VERSION,
    SIDECAR_SCHEMA_VERSION,
    _business_tags,
)
from .geometry import load_upright_rgb, save_rgb
from .status_crops import crop_status_region, reconstruct_rectified
from .status_style import STATUS_STYLE_CLASSES, UNKNOWN_STATUS_STYLE


class UnsafeStatusTagReviewOutputError(ValueError):
    """Raised when review output overlaps a protected input tree."""


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _selection_key(relative_path: Path) -> tuple[bytes, str]:
    relative = relative_path.as_posix()
    return hashlib.sha256(relative.encode("utf-8")).digest(), relative


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {error}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _record_path(value: object, *, sidecar_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sidecar {field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = sidecar_path.parent / path
    return path.resolve()


def _status_bbox(sidecar: Mapping[str, Any]) -> tuple[float, float, float, float]:
    value = sidecar.get("transfer_status_bbox_rectified")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("sidecar transfer_status_bbox_rectified must contain four numbers")
    try:
        bbox = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        raise ValueError("sidecar transfer_status_bbox_rectified must contain four numbers") from None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("sidecar transfer_status_bbox_rectified is empty or inverted")
    return bbox  # type: ignore[return-value]


def _margin_ratio(sidecar: Mapping[str, Any]) -> float:
    config = sidecar.get("inference_config")
    value = config.get("margin_ratio", 0.30) if isinstance(config, Mapping) else 0.30
    try:
        margin = float(value)
    except (TypeError, ValueError):
        raise ValueError("sidecar inference_config.margin_ratio must be numeric") from None
    if margin < 0:
        raise ValueError("sidecar inference_config.margin_ratio cannot be negative")
    return margin


def _source_from_v1(payload: Mapping[str, Any], result_path: Path) -> Path:
    value = payload.get("source")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("v1 result source must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = result_path.parent / path
    return path.resolve()


def _load_sidecar_record(sidecar_path: Path, tags_root: Path) -> dict[str, Any]:
    sidecar = _load_json(sidecar_path, "status-style sidecar")
    if sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            "stale status-style sidecar schema; rerun status enrichment with the current code"
        )
    status_style = sidecar.get("status_style")
    tags = sidecar.get("tags")
    if not isinstance(status_style, Mapping):
        raise ValueError("sidecar has no status_style object")
    if not isinstance(tags, Mapping):
        raise ValueError("sidecar has no tags object")
    label = status_style.get("label")
    if label not in {*STATUS_STYLE_CLASSES, UNKNOWN_STATUS_STYLE}:
        raise ValueError(f"sidecar has invalid status_style label: {label!r}")
    if dict(tags) != _business_tags(status_style):
        raise ValueError(
            "stale or inconsistent business tags; rerun status enrichment with rule "
            f"{BUSINESS_RULE_VERSION}"
        )
    model = sidecar.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("sidecar has no model signature")
    digest = model.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise ValueError("sidecar model.sha256 must be a 64-character hexadecimal digest")
    if not isinstance(sidecar.get("inference_config"), Mapping):
        raise ValueError("sidecar has no inference_config object")
    source_path = _record_path(sidecar.get("source"), sidecar_path=sidecar_path, field="source")
    result_path = _record_path(
        sidecar.get("result_json"), sidecar_path=sidecar_path, field="result_json"
    )
    if source_path.is_file() and result_path.is_file() and sidecar_path.stat().st_mtime_ns < max(
        source_path.stat().st_mtime_ns,
        result_path.stat().st_mtime_ns,
    ):
        raise ValueError("stale status-style sidecar is older than its source or v1 result")
    return {
        "sidecar_path": sidecar_path.resolve(),
        "relative_sidecar": sidecar_path.relative_to(tags_root),
        "source_path": source_path,
        "result_path": result_path,
        "bbox": _status_bbox(sidecar),
        "margin_ratio": _margin_ratio(sidecar),
        "status_style": dict(status_style),
        "tags": dict(tags),
    }


def _output_layout(output: Path) -> tuple[Path, Path, Path]:
    """Return ``(html_path, assets_dir, safety_root)`` without creating it."""
    output = output.resolve()
    if output.suffix.lower() in {".html", ".htm"}:
        if output.exists() and output.is_dir():
            raise ValueError(f"HTML output is a directory: {output}")
        assets_dir = output.parent / f"{output.stem}_assets"
        return output, assets_dir, assets_dir
    if output.exists() and not output.is_dir():
        raise ValueError(f"review output directory is a file: {output}")
    return output / "index.html", output / "assets", output


def _common_input_roots(paths: Sequence[Path]) -> list[Path]:
    """Infer a conservative root for one logical collection of input files."""
    if not paths:
        return []
    resolved = [path.resolve() for path in paths]
    try:
        common = Path(os.path.commonpath([str(path) for path in resolved])).resolve()
    except ValueError:  # Different Windows drives.
        return sorted({path.parent for path in resolved}, key=lambda path: path.as_posix())
    if common in resolved:
        common = common.parent
    # A filesystem root is too broad to be useful.  Protect each direct parent
    # in the uncommon case that one batch spans unrelated source directories.
    if common == Path(common.anchor):
        return sorted({path.parent for path in resolved}, key=lambda path: path.as_posix())
    return [common]


def _validate_output_paths(
    *,
    html_path: Path,
    assets_dir: Path,
    safety_root: Path,
    tags_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    protected = [tags_root.resolve()]
    protected.extend(_common_input_roots([Path(record["source_path"]) for record in records]))
    protected.extend(_common_input_roots([Path(record["result_path"]) for record in records]))
    output_paths = {html_path.resolve(), assets_dir.resolve(), safety_root.resolve()}
    for output_path in output_paths:
        for protected_path in protected:
            if _paths_overlap(output_path, protected_path):
                raise UnsafeStatusTagReviewOutputError(
                    "review output must be outside, and must not be an ancestor of, "
                    "the tags, v1-result, or source-image trees: "
                    f"output={output_path}, protected={protected_path}"
                )


def _format_confidence(value: object) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "—"
    if not 0.0 <= confidence <= 1.0:
        return "—"
    return f"{confidence:.1%}"


def _text(value: object) -> str:
    return "—" if value is None or value == "" else str(value)


def _card_html(record: Mapping[str, Any]) -> str:
    status_style = record["status_style"]
    tags = record["tags"]
    probabilities = status_style.get("probabilities")
    probabilities = probabilities if isinstance(probabilities, Mapping) else {}
    probability_rows = "".join(
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{html.escape(_format_confidence(probabilities.get(label)))}</td>"
        "</tr>"
        for label in STATUS_STYLE_CLASSES
    )
    label = _text(status_style.get("label"))
    platform = _text(tags.get("platform"))
    review_tag = _text(tags.get("review_tag"))
    return f"""
      <article class="card" data-label="{html.escape(label, quote=True)}">
        <img src="{html.escape(str(record['asset_url']), quote=True)}" loading="lazy"
             alt="transfer status crop">
        <div class="details">
          <h2>{html.escape(Path(record['source_path']).name)}</h2>
          <div class="badges">
            <span class="badge label">{html.escape(label)}</span>
            <span class="badge">platform: {html.escape(platform)}</span>
            <span class="badge review">review: {html.escape(review_tag)}</span>
          </div>
          <p><b>confidence:</b> {html.escape(_format_confidence(status_style.get('confidence')))}</p>
          <table><tbody>{probability_rows}</tbody></table>
          <p class="path">{html.escape(str(record['relative_sidecar']))}</p>
        </div>
      </article>
    """


def _error_html(errors: Sequence[Mapping[str, str]]) -> str:
    if not errors:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(error['sidecar'])}</td>"
        f"<td>{html.escape(error['error_type'])}</td>"
        f"<td>{html.escape(error['message'])}</td>"
        "</tr>"
        for error in errors
    )
    return f"""
      <details class="errors" open>
        <summary>错误摘要（{len(errors)}）</summary>
        <table><thead><tr><th>sidecar</th><th>类型</th><th>信息</th></tr></thead>
        <tbody>{rows}</tbody></table>
      </details>
    """


def _page_html(
    *,
    cards: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, str]],
    considered: int,
) -> str:
    labels = Counter(_text(card["status_style"].get("label")) for card in cards)
    platforms = Counter(_text(card["tags"].get("platform")) for card in cards)
    reviews = Counter(_text(card["tags"].get("review_tag")) for card in cards)
    summary_parts = [
        f"候选 {considered}",
        f"成功渲染 {len(cards)}",
        f"错误 {len(errors)}",
        "labels " + ", ".join(f"{key}={value}" for key, value in sorted(labels.items())),
        "platforms " + ", ".join(f"{key}={value}" for key, value in sorted(platforms.items())),
        "review_tags " + ", ".join(f"{key}={value}" for key, value in sorted(reviews.items())),
    ]
    cards_html = "\n".join(_card_html(card) for card in cards)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Status tag review</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; background: #f3f5f8; color: #18202a; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 18px 24px; background: #fff;
              border-bottom: 1px solid #d9dee7; box-shadow: 0 2px 8px #12213b14; }}
    h1 {{ margin: 0 0 8px; font-size: 23px; }}
    .summary {{ margin: 0; color: #4b596a; line-height: 1.6; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr)); gap: 16px; }}
    .card {{ overflow: hidden; background: #fff; border: 1px solid #d9dee7; border-radius: 12px;
             box-shadow: 0 3px 12px #12213b10; }}
    .card img {{ display: block; width: 100%; height: 185px; object-fit: contain; background: #0c1727; }}
    .details {{ padding: 14px 16px 16px; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; overflow-wrap: anywhere; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .badge {{ padding: 4px 8px; border-radius: 999px; background: #e8edf4; font-size: 13px; }}
    .badge.label {{ background: #dceaff; color: #064ca4; font-weight: 700; }}
    .badge.review {{ background: #fff0d8; color: #8b4b00; }}
    p {{ margin: 10px 0 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 4px 7px; border-bottom: 1px solid #edf0f4; text-align: left; }}
    .path {{ color: #677486; font: 12px ui-monospace, Consolas, monospace; overflow-wrap: anywhere; }}
    .errors {{ margin: 0 0 18px; padding: 12px; background: #fff4f4; border: 1px solid #f1bebe;
               border-radius: 9px; }}
    .errors summary {{ cursor: pointer; font-weight: 700; }}
    .errors td {{ overflow-wrap: anywhere; }}
    @media (max-width: 520px) {{ main {{ padding: 10px; }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>状态样式预测 Review</h1><p class="summary">{html.escape(' · '.join(summary_parts))}</p></header>
  <main>
    {_error_html(errors)}
    <section class="grid">{cards_html}</section>
  </main>
</body>
</html>
"""


def _error_record(sidecar_path: Path, error: Exception) -> dict[str, str]:
    return {
        "sidecar": sidecar_path.resolve().as_posix(),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def render_status_tag_review(
    *,
    tags_dir: Path,
    output: Path,
    limit: int | None = 200,
) -> dict[str, object]:
    """Render clean status crops and an HTML index without changing inputs."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    tags_root = tags_dir.resolve()
    if not tags_root.is_dir():
        raise NotADirectoryError(tags_root)
    html_path, assets_dir, safety_root = _output_layout(output)
    # Reject the most dangerous typo before reading any sidecars.
    _validate_output_paths(
        html_path=html_path,
        assets_dir=assets_dir,
        safety_root=safety_root,
        tags_root=tags_root,
        records=[],
    )

    candidates = sorted(
        tags_root.rglob("*.status_style.json"),
        key=lambda path: _selection_key(path.relative_to(tags_root)),
    )
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    considered = 0
    for sidecar_path in candidates:
        if limit is not None and len(records) >= limit:
            break
        considered += 1
        try:
            records.append(_load_sidecar_record(sidecar_path, tags_root))
        except (OSError, ValueError) as error:
            errors.append(_error_record(sidecar_path, error))

    # This second preflight protects the source and v1 trees named by the
    # selected sidecars.  No output directory exists before this completes.
    _validate_output_paths(
        html_path=html_path,
        assets_dir=assets_dir,
        safety_root=safety_root,
        tags_root=tags_root,
        records=records,
    )

    assets_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    for record in records:
        sidecar_path = Path(record["sidecar_path"])
        try:
            result_path = Path(record["result_path"])
            source_path = Path(record["source_path"])
            if not result_path.is_file():
                raise FileNotFoundError(f"v1 result not found: {result_path}")
            if not source_path.is_file():
                raise FileNotFoundError(f"source image not found: {source_path}")
            v1_payload = _load_json(result_path, "v1 result")
            result_source = _source_from_v1(v1_payload, result_path)
            if result_source != source_path:
                raise ValueError(
                    "sidecar source does not match v1 result source: "
                    f"sidecar={source_path}, v1={result_source}"
                )
            source_rgb = load_upright_rgb(source_path)
            rectified_rgb = reconstruct_rectified(v1_payload, source_rgb)
            crop_rgb = crop_status_region(
                rectified_rgb,
                record["bbox"],
                margin_ratio=float(record["margin_ratio"]),
            )
            digest = hashlib.sha256(record["relative_sidecar"].as_posix().encode("utf-8")).hexdigest()[:20]
            asset_path = assets_dir / f"{digest}.jpg"
            temporary_asset = assets_dir / f".{digest}.tmp.jpg"
            save_rgb(temporary_asset, crop_rgb, quality=92)
            temporary_asset.replace(asset_path)
            asset_url = os.path.relpath(asset_path, start=html_path.parent).replace(os.sep, "/")
            cards.append({**record, "asset_url": asset_url})
        except Exception as error:
            errors.append(_error_record(sidecar_path, error))

    html_path.parent.mkdir(parents=True, exist_ok=True)
    page = _page_html(cards=cards, errors=errors, considered=considered)
    temporary_html = html_path.with_suffix(html_path.suffix + ".tmp")
    temporary_html.write_text(page, encoding="utf-8")
    temporary_html.replace(html_path)
    return {
        "html": html_path,
        "assets": assets_dir,
        "rendered": len(cards),
        "errors": errors,
        "considered": considered,
        "sidecars": [card["sidecar_path"] for card in cards],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an HTML review page for status-style sidecars")
    parser.add_argument("--tags", type=Path, required=True, help="Status sidecar root directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .html path, or a directory that will contain index.html",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum rendered samples (default: 200)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = render_status_tag_review(tags_dir=args.tags, output=args.output, limit=args.limit)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Status tag review failed:\n{error}") from None
    print(
        "Status tag review complete: "
        f"rendered={summary['rendered']}, errors={len(summary['errors'])}, html={summary['html']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
