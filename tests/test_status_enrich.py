import json
from pathlib import Path

import numpy as np
import pytest

import transfer_receipt_ai.enrich_status_tags as enrich_module
from transfer_receipt_ai.train_status_style import (
    StatusStyleRecord,
    class_weights,
    load_reviewed_records,
    split_records_by_group,
)


class _Prediction:
    def as_dict(self) -> dict[str, object]:
        return {
            "label": "check_aligned",
            "confidence": 0.96,
            "business_tag": "ios",
            "candidate_label": "check_aligned",
            "probabilities": {
                "check_offset": 0.02,
                "check_aligned": 0.96,
                "check_absent": 0.02,
            },
        }


class _Predictor:
    def __init__(self, *args, **kwargs) -> None:
        self.crops: list[np.ndarray] = []

    def predict(self, crop: np.ndarray) -> _Prediction:
        self.crops.append(crop)
        return _Prediction()


def _write_result(result_path: Path, source_path: Path) -> bytes:
    payload = {
        "source": source_path.as_posix(),
        "geometry": {
            "rectified_size": {"width": 100, "height": 200},
            "H_original_to_rectified": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "fields": {},
        "detections": [
            {
                "label": "transfer_status",
                "score": 0.98,
                "bbox_rectified": [20, 30, 80, 60],
            }
        ],
    }
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(content)
    return content


def test_enrichment_writes_separate_sidecar_and_preserves_inputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "receipt.jpg"
    source.parent.mkdir()
    source_bytes = b"immutable source"
    source.write_bytes(source_bytes)
    results = tmp_path / "v1"
    result_path = results / "nested" / "receipt.json"
    result_bytes = _write_result(result_path, source)
    output = tmp_path / "tags"

    monkeypatch.setattr(enrich_module, "StatusStylePredictor", _Predictor)
    monkeypatch.setattr(
        enrich_module,
        "_checkpoint_signature",
        lambda path: {"path": str(path), "sha256": "model-v1", "size_bytes": 1},
    )
    monkeypatch.setattr(enrich_module, "load_upright_rgb", lambda path: np.zeros((200, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(
        enrich_module,
        "reconstruct_rectified",
        lambda payload, image: np.zeros((200, 100, 3), dtype=np.uint8),
    )
    seen_margin: list[float] = []

    def crop_status(image, bbox, margin_ratio=0.30):
        seen_margin.append(margin_ratio)
        return np.zeros((32, 96, 3), dtype=np.uint8)

    monkeypatch.setattr(enrich_module, "crop_status_region", crop_status)

    manifest = enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
    )

    assert len(manifest) == 1
    assert manifest[0]["status"] == "written"
    sidecar = output / "nested" / "receipt.status_style.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["status_style"]["label"] == "check_aligned"
    assert payload["status_style"]["business_tag"] == "ios"
    assert payload["status_style"]["state"] == "classified"
    assert payload["tags"] == {
        "platform": "ios",
        "authenticity": "not_assessed",
        "review_tag": None,
        "requires_manual_review": False,
        "reason": "status_check_aligned",
        "rule_version": "status-style-v2",
    }
    assert payload["group_id"] == "nested/receipt"
    assert seen_margin == [0.30]
    assert result_path.read_bytes() == result_bytes
    assert source.read_bytes() == source_bytes


def test_enrichment_skip_existing_is_resumable(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "receipt.jpg"
    source.parent.mkdir()
    source.write_bytes(b"source")
    results = tmp_path / "results"
    _write_result(results / "one.json", source)
    output = tmp_path / "tags"

    monkeypatch.setattr(enrich_module, "StatusStylePredictor", _Predictor)
    monkeypatch.setattr(
        enrich_module,
        "_checkpoint_signature",
        lambda path: {"path": str(path), "sha256": "model-v1", "size_bytes": 1},
    )
    monkeypatch.setattr(enrich_module, "load_upright_rgb", lambda path: np.zeros((20, 20, 3), dtype=np.uint8))
    monkeypatch.setattr(
        enrich_module,
        "reconstruct_rectified",
        lambda payload, image: np.zeros((20, 20, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        enrich_module,
        "crop_status_region",
        lambda image, bbox, margin_ratio=0.30: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
    )

    def should_not_load(path):
        raise AssertionError("a committed sidecar should skip image reconstruction")

    monkeypatch.setattr(enrich_module, "load_upright_rgb", should_not_load)
    rerun = enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
        skip_existing=True,
    )

    assert rerun[0]["status"] == "skipped_existing"
    jsonl = (output / "status_style_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1


def test_enrichment_records_error_and_continues(tmp_path, monkeypatch) -> None:
    missing_source = tmp_path / "raw" / "missing.jpg"
    results = tmp_path / "results"
    _write_result(results / "bad.json", missing_source)
    output = tmp_path / "tags"
    monkeypatch.setattr(enrich_module, "StatusStylePredictor", _Predictor)
    monkeypatch.setattr(
        enrich_module,
        "_checkpoint_signature",
        lambda path: {"path": str(path), "sha256": "model-v1", "size_bytes": 1},
    )

    manifest = enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
        continue_on_error=True,
    )

    assert manifest == []
    errors = (output / "status_style_errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(errors) == 1
    assert json.loads(errors[0])["error_type"] == "FileNotFoundError"


def test_enrichment_refuses_output_inside_v1_results(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw.jpg"
    source.write_bytes(b"source")
    results = tmp_path / "results"
    _write_result(results / "one.json", source)

    with pytest.raises(ValueError, match="separate, non-overlapping"):
        enrich_module.enrich_status_tags(
            checkpoint=tmp_path / "unused.pt",
            input_path=results,
            output_dir=results / "tags",
        )


def test_enrichment_refuses_output_that_is_an_ancestor_of_v1_results(tmp_path) -> None:
    source = tmp_path / "raw" / "receipt.jpg"
    source.parent.mkdir()
    source.write_bytes(b"source")
    results = tmp_path / "nested" / "results"
    _write_result(results / "one.json", source)

    with pytest.raises(ValueError, match="separate, non-overlapping"):
        enrich_module.enrich_status_tags(
            checkpoint=tmp_path / "unused.pt",
            input_path=results,
            output_dir=tmp_path,
        )


def test_enrichment_refuses_output_in_source_image_tree(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "receipt.jpg"
    source.parent.mkdir()
    source.write_bytes(b"source")
    results = tmp_path / "results"
    _write_result(results / "one.json", source)
    monkeypatch.setattr(enrich_module, "StatusStylePredictor", _Predictor)
    monkeypatch.setattr(
        enrich_module,
        "_checkpoint_signature",
        lambda path: {"path": str(path), "sha256": "model-v1", "size_bytes": 1},
    )

    with pytest.raises(ValueError, match="source-image directory"):
        enrich_module.enrich_status_tags(
            checkpoint=tmp_path / "unused.pt",
            input_path=results,
            output_dir=source.parent,
            continue_on_error=True,
        )


def test_absent_signal_is_review_tag_not_final_authenticity() -> None:
    tags = enrich_module._business_tags({"label": "check_absent"})

    assert tags["authenticity"] == "not_assessed"
    assert tags["review_tag"] == "suspected_fake"
    assert tags["requires_manual_review"] is True


def test_enrichment_reprocesses_when_checkpoint_signature_changes(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "receipt.jpg"
    source.parent.mkdir()
    source.write_bytes(b"source")
    results = tmp_path / "results"
    _write_result(results / "one.json", source)
    output = tmp_path / "tags"

    monkeypatch.setattr(enrich_module, "StatusStylePredictor", _Predictor)
    monkeypatch.setattr(enrich_module, "load_upright_rgb", lambda path: np.zeros((20, 20, 3), dtype=np.uint8))
    monkeypatch.setattr(
        enrich_module,
        "reconstruct_rectified",
        lambda payload, image: np.zeros((20, 20, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        enrich_module,
        "crop_status_region",
        lambda image, bbox, margin_ratio=0.30: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    signatures = iter(
        (
            {"path": "model.pt", "sha256": "model-v1", "size_bytes": 1},
            {"path": "model.pt", "sha256": "model-v2", "size_bytes": 1},
        )
    )
    monkeypatch.setattr(enrich_module, "_checkpoint_signature", lambda path: next(signatures))

    enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
    )
    rerun = enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
        skip_existing=True,
    )

    assert rerun[0]["status"] == "written"
    sidecar = json.loads((output / "one.status_style.json").read_text(encoding="utf-8"))
    assert sidecar["model"]["sha256"] == "model-v2"


def test_enrichment_reprocesses_stale_business_rule_sidecar(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "receipt.jpg"
    source.parent.mkdir()
    source.write_bytes(b"source")
    results = tmp_path / "results"
    _write_result(results / "one.json", source)
    output = tmp_path / "tags"

    monkeypatch.setattr(enrich_module, "StatusStylePredictor", _Predictor)
    monkeypatch.setattr(
        enrich_module,
        "_checkpoint_signature",
        lambda path: {"path": "model.pt", "sha256": "model-v1", "size_bytes": 1},
    )
    monkeypatch.setattr(enrich_module, "load_upright_rgb", lambda path: np.zeros((20, 20, 3), dtype=np.uint8))
    monkeypatch.setattr(
        enrich_module,
        "reconstruct_rectified",
        lambda payload, image: np.zeros((20, 20, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        enrich_module,
        "crop_status_region",
        lambda image, bbox, margin_ratio=0.30: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
    )
    sidecar_path = output / "one.status_style.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["tags"]["rule_version"] = "obsolete-rule"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    rerun = enrich_module.enrich_status_tags(
        checkpoint=tmp_path / "unused.pt",
        input_path=results,
        output_dir=output,
        skip_existing=True,
    )

    assert rerun[0]["status"] == "written"
    refreshed = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert refreshed["tags"]["rule_version"] == enrich_module.BUSINESS_RULE_VERSION


def _record(tmp_path: Path, label: str, group: str, index: int = 0) -> StatusStyleRecord:
    return StatusStyleRecord(
        crop=tmp_path / f"{group}_{index}.jpg",
        source=tmp_path / f"{group}.source.jpg",
        result_json=tmp_path / f"{group}.json",
        group_id=group,
        label=label,
    )


def test_group_split_is_deterministic_stratified_and_has_no_leakage(tmp_path) -> None:
    records: list[StatusStyleRecord] = []
    for label in ("check_offset", "check_aligned", "check_absent"):
        for group_index in range(4):
            group = f"{label}/{group_index}"
            records.append(_record(tmp_path, label, group, 0))
            records.append(_record(tmp_path, label, group, 1))

    first_train, first_val = split_records_by_group(records, val_ratio=0.25, seed=9)
    second_train, second_val = split_records_by_group(records, val_ratio=0.25, seed=9)

    assert [record.group_id for record in first_train] == [record.group_id for record in second_train]
    assert [record.group_id for record in first_val] == [record.group_id for record in second_val]
    assert {record.group_id for record in first_train}.isdisjoint(record.group_id for record in first_val)
    assert {record.label for record in first_train} == {"check_offset", "check_aligned", "check_absent"}
    assert {record.label for record in first_val} == {"check_offset", "check_aligned", "check_absent"}


def test_load_reviewed_records_skips_unreviewed_and_unclear(tmp_path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    for name in ("good.jpg", "unclear.jpg", "pending.jpg"):
        (crops / name).write_bytes(b"crop")
    manifest = tmp_path / "review.jsonl"
    base = {"source": "raw.jpg", "result_json": "result.json"}
    rows = [
        {**base, "crop": "crops/good.jpg", "group_id": "good", "label": "check_offset"},
        {**base, "crop": "crops/unclear.jpg", "group_id": "unclear", "label": "unclear"},
        {**base, "crop": "crops/pending.jpg", "group_id": "pending", "label": None},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    records, skipped = load_reviewed_records(manifest)

    assert len(records) == 1
    assert records[0].crop == crops / "good.jpg"
    assert skipped == 2


def test_class_weights_downweight_the_more_frequent_class(tmp_path) -> None:
    records = [
        _record(tmp_path, "check_offset", "a", 0),
        _record(tmp_path, "check_offset", "b", 0),
        _record(tmp_path, "check_aligned", "c", 0),
        _record(tmp_path, "check_absent", "d", 0),
    ]

    weights = class_weights(records)

    assert weights.shape == (3,)
    assert float(weights[0]) < float(weights[1])
    assert float(weights[1]) == pytest.approx(float(weights[2]))
