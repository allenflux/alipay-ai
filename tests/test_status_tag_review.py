import json
from pathlib import Path

import numpy as np
import pytest

from transfer_receipt_ai.enrich_status_tags import BUSINESS_RULE_VERSION, SIDECAR_SCHEMA_VERSION
from transfer_receipt_ai.geometry import save_rgb
from transfer_receipt_ai.status_tag_review import (
    UnsafeStatusTagReviewOutputError,
    _selection_key,
    render_status_tag_review,
)


def _write_inputs(root: Path, name: str, *, label: str = "check_aligned") -> tuple[Path, Path, Path]:
    source = root / "raw" / f"{name}.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(source, np.full((80, 120, 3), 80, dtype=np.uint8))
    result = root / "v1" / f"{name}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "source": source.resolve().as_posix(),
                "geometry": {
                    "source_size": {"width": 120, "height": 80},
                    "rectified_size": {"width": 120, "height": 80},
                    "H_original_to_rectified": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    tags = root / "tags"
    sidecar = tags / f"{name}.status_style.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    platform = "ios" if label == "check_aligned" else "android"
    reason = "status_check_aligned" if label == "check_aligned" else "status_check_offset"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "result_json": result.resolve().as_posix(),
                "source": source.resolve().as_posix(),
                "transfer_status_bbox_rectified": [20, 20, 100, 50],
                "model": {"path": "status.pt", "sha256": "a" * 64, "size_bytes": 123},
                "inference_config": {
                    "confidence_threshold": 0.80,
                    "absent_confidence_threshold": 0.95,
                    "margin_ratio": 0.25,
                },
                "status_style": {
                    "label": label,
                    "confidence": 0.96,
                    "probabilities": {
                        "check_offset": 0.02,
                        "check_aligned": 0.96,
                        "check_absent": 0.02,
                    },
                },
                "tags": {
                    "platform": platform,
                    "authenticity": "not_assessed",
                    "review_tag": None,
                    "requires_manual_review": False,
                    "reason": reason,
                    "rule_version": BUSINESS_RULE_VERSION,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return source, result, sidecar


def test_render_review_writes_html_and_crop_without_changing_inputs(tmp_path) -> None:
    source, result, sidecar = _write_inputs(tmp_path, "receipt")
    before = {path: path.read_bytes() for path in (source, result, sidecar)}

    summary = render_status_tag_review(
        tags_dir=tmp_path / "tags",
        output=tmp_path / "review" / "status.html",
        limit=10,
    )

    html_path = Path(summary["html"])
    page = html_path.read_text(encoding="utf-8")
    assert summary["rendered"] == 1
    assert "receipt.jpg" in page
    assert "check_aligned" in page
    assert "platform: ios" in page
    assert "96.0%" in page
    assert "check_absent" in page
    assert len(list(Path(summary["assets"]).glob("*.jpg"))) == 1
    assert {path: path.read_bytes() for path in before} == before


def test_limit_uses_deterministic_hash_order(tmp_path) -> None:
    sidecars = [_write_inputs(tmp_path, f"receipt-{index}")[2] for index in range(5)]
    tags = tmp_path / "tags"
    expected = sorted(sidecars, key=lambda path: _selection_key(path.relative_to(tags)))[:2]

    first = render_status_tag_review(tags_dir=tags, output=tmp_path / "review-a", limit=2)
    second = render_status_tag_review(tags_dir=tags, output=tmp_path / "review-b", limit=2)

    assert first["rendered"] == 2
    assert first["sidecars"] == [path.resolve() for path in expected]
    assert second["sidecars"] == first["sidecars"]


@pytest.mark.parametrize("output_relative", ["tags/review", "."])
def test_review_refuses_output_inside_or_above_tags_tree(tmp_path, output_relative) -> None:
    _write_inputs(tmp_path, "receipt")
    output = (tmp_path / output_relative).resolve()

    with pytest.raises(UnsafeStatusTagReviewOutputError, match="must be outside"):
        render_status_tag_review(tags_dir=tmp_path / "tags", output=output, limit=1)


@pytest.mark.parametrize("protected_tree", ["raw", "v1"])
def test_review_refuses_output_in_source_or_v1_tree(tmp_path, protected_tree) -> None:
    _write_inputs(tmp_path, "receipt")
    output = tmp_path / protected_tree / "review"

    with pytest.raises(UnsafeStatusTagReviewOutputError, match="must be outside"):
        render_status_tag_review(tags_dir=tmp_path / "tags", output=output, limit=1)
    assert not output.exists()


def test_review_continues_after_bad_sidecar_and_shows_error_summary(tmp_path) -> None:
    _write_inputs(tmp_path, "good")
    bad = tmp_path / "tags" / "bad.status_style.json"
    bad.write_text("{broken", encoding="utf-8")

    summary = render_status_tag_review(tags_dir=tmp_path / "tags", output=tmp_path / "review", limit=10)

    assert summary["rendered"] == 1
    assert len(summary["errors"]) == 1
    page = Path(summary["html"]).read_text(encoding="utf-8")
    assert "错误摘要（1）" in page
    assert "JSONDecodeError" not in page
    assert "ValueError" in page
    assert "bad.status_style.json" in page


def test_review_marks_old_rule_sidecar_as_error(tmp_path) -> None:
    _, _, sidecar_path = _write_inputs(tmp_path, "stale")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["tags"]["rule_version"] = "status-style-v1"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    summary = render_status_tag_review(tags_dir=tmp_path / "tags", output=tmp_path / "review", limit=10)

    assert summary["rendered"] == 0
    assert len(summary["errors"]) == 1
    assert "stale or inconsistent business tags" in summary["errors"][0]["message"]
