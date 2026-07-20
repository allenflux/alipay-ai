import json
from pathlib import Path

import pytest

import transfer_receipt_ai.infer as infer_module
from transfer_receipt_ai.model import Detection


class _DetectionItem:
    def __init__(self, label: str) -> None:
        self.detection = Detection(label, 0.9, (0, 0, 10, 10))


def _write_committed_bundle(source: Path, output_stem: Path, payload: dict[str, object]) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_stem.with_name(output_stem.name + "_rectified_annotated.jpg").write_bytes(b"image")
    output_stem.with_name(output_stem.name + "_original_annotated.jpg").write_bytes(b"image")


def test_parser_status_style_options_are_optional_and_keep_production_defaults(tmp_path) -> None:
    args = infer_module.build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "receipt.pt"),
            "--input",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert args.status_style_checkpoint is None
    assert args.status_confidence_threshold == 0.80
    assert args.status_absent_confidence_threshold == 0.95


def test_parser_accepts_status_style_checkpoint_and_thresholds(tmp_path) -> None:
    status_checkpoint = tmp_path / "status.pt"
    args = infer_module.build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "receipt.pt"),
            "--status-style-checkpoint",
            str(status_checkpoint),
            "--status-confidence-threshold",
            "0.84",
            "--status-absent-confidence-threshold",
            "0.97",
            "--input",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert args.status_style_checkpoint == status_checkpoint
    assert args.status_confidence_threshold == 0.84
    assert args.status_absent_confidence_threshold == 0.97


def test_stable_shards_assign_every_path_once() -> None:
    paths = [Path(f"batch/image_{index:04d}.jpg") for index in range(100)]

    assignments = {path: infer_module._shard_for(path, 7) for path in paths}

    assert all(0 <= shard < 7 for shard in assignments.values())
    assert len(assignments) == len(paths)
    assert infer_module._shard_for(Path("batch/image_0001.jpg"), 7) == assignments[Path("batch/image_0001.jpg")]


def test_require_five_fields_rejects_incomplete_result() -> None:
    result = type("Result", (), {"detections": [_DetectionItem("time"), _DetectionItem("amount")]})()

    try:
        infer_module._require_five_fields(result)
    except ValueError as error:
        assert "found=2" in str(error)
        assert "transfer_status" in str(error)
    else:
        raise AssertionError("incomplete detection should be rejected")


def test_require_five_fields_accepts_complete_result() -> None:
    labels = ("time", "amount", "transfer_status", "recipient_field", "payment_method_field")
    result = type("Result", (), {"detections": [_DetectionItem(label) for label in labels]})()

    infer_module._require_five_fields(result)


def test_skip_existing_rebuilds_incremental_manifest(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "receipt.jpg"
    source.write_bytes(b"not read because the result already exists")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "receipt.json").write_text('{"fields": {}}\n', encoding="utf-8")
    (output_dir / "receipt_rectified_annotated.jpg").write_bytes(b"image")
    (output_dir / "receipt_original_annotated.jpg").write_bytes(b"image")

    monkeypatch.setattr(infer_module, "LRCNNPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer_module, "PaddleOCRReader", lambda: object())

    manifest = infer_module.run_inference(
        checkpoint=tmp_path / "unused.pt",
        input_path=input_dir,
        output_dir=output_dir,
        skip_existing=True,
    )

    assert len(manifest) == 1
    assert manifest[0]["status"] == "skipped_existing"
    assert json.loads((output_dir / "inference_manifest.json").read_text(encoding="utf-8"))[0]["source"].endswith(
        "receipt.jpg"
    )
    jsonl = (output_dir / "inference_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1


def test_status_aware_skip_existing_requires_same_model_and_thresholds(tmp_path) -> None:
    source = tmp_path / "input" / "receipt.jpg"
    source.parent.mkdir()
    source.write_bytes(b"source")
    output_stem = tmp_path / "output" / "receipt"
    model = {
        "path": "D:/models/status_style_v1/best.pt",
        "sha256": "same-checkpoint",
        "size_bytes": 123,
    }
    config = {
        "confidence_threshold": 0.80,
        "absent_confidence_threshold": 0.95,
        "margin_ratio": 0.30,
    }

    # A valid old v1 bundle can still be resumed by v1, but must be rebuilt
    # when the optional classifier is requested.
    _write_committed_bundle(source, output_stem, {"fields": {}})
    assert infer_module._committed_result_exists(source, output_stem)
    assert not infer_module._committed_result_exists(
        source,
        output_stem,
        status_style_model=model,
        status_style_inference_config=config,
    )

    status_style = {
        "schema_version": infer_module.STATUS_STYLE_SCHEMA_VERSION,
        "state": "classified",
        "label": "check_offset",
        "confidence": 0.98,
        "business_tag": "android",
        "candidate_label": "check_offset",
        "probabilities": {
            "check_offset": 0.98,
            "check_aligned": 0.01,
            "check_absent": 0.01,
        },
        "model": model,
        "inference_config": config,
    }
    _write_committed_bundle(
        source,
        output_stem,
        {
            "fields": {},
            "status_style": status_style,
            "tags": infer_module.status_style_tags(status_style),
        },
    )
    assert infer_module._committed_result_exists(
        source,
        output_stem,
        status_style_model=model,
        status_style_inference_config=config,
    )

    changed_model = {**model, "sha256": "new-checkpoint"}
    assert not infer_module._committed_result_exists(
        source,
        output_stem,
        status_style_model=changed_model,
        status_style_inference_config=config,
    )
    changed_config = {**config, "confidence_threshold": 0.84}
    assert not infer_module._committed_result_exists(
        source,
        output_stem,
        status_style_model=model,
        status_style_inference_config=changed_config,
    )


def test_output_stem_collisions_are_rejected(tmp_path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    first = root / "same.jpg"
    second = root / "same.png"

    try:
        infer_module._validate_output_names([first, second], root)
    except ValueError as error:
        assert "same output stem" in str(error)
    else:
        raise AssertionError("colliding output stems should be rejected")


def test_continue_on_error_records_bad_image(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "broken.jpg").write_bytes(b"broken")
    output_dir = tmp_path / "output"

    class BrokenPipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, source_path):
            raise ValueError("bad image")

    monkeypatch.setattr(infer_module, "LRCNNPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer_module, "ReceiptPipeline", BrokenPipeline)

    manifest = infer_module.run_inference(
        checkpoint=tmp_path / "unused.pt",
        input_path=input_dir,
        output_dir=output_dir,
        use_ocr=False,
        continue_on_error=True,
    )

    assert manifest == []
    errors = (output_dir / "inference_errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(errors) == 1
    assert json.loads(errors[0])["message"] == "bad image"


def test_limit_processes_only_requested_images(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(5):
        (input_dir / f"receipt_{index}.jpg").write_bytes(b"unused")
    output_dir = tmp_path / "output"
    processed: list[str] = []

    class RecordingPipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, source_path):
            processed.append(source_path.name)
            return object()

    monkeypatch.setattr(infer_module, "LRCNNPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer_module, "ReceiptPipeline", RecordingPipeline)
    monkeypatch.setattr(infer_module, "write_receipt_result", lambda *args, **kwargs: None)

    manifest = infer_module.run_inference(
        checkpoint=tmp_path / "unused.pt",
        input_path=input_dir,
        output_dir=output_dir,
        use_ocr=False,
        limit=2,
    )

    expected = sorted(
        (Path(f"receipt_{index}.jpg") for index in range(5)),
        key=lambda path: (infer_module._selection_key(path), path.as_posix()),
    )[:2]
    assert processed == [path.name for path in expected]
    assert len(manifest) == 2


def test_status_style_predictor_is_constructed_once_per_batch(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(3):
        (input_dir / f"receipt_{index}.jpg").write_bytes(b"unused")
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "status.pt"
    model_signature = {
        "path": checkpoint.resolve().as_posix(),
        "sha256": "model-signature",
        "size_bytes": 321,
    }
    constructed: list[object] = []
    pipeline_predictors: list[object] = []

    class FakeStatusStylePredictor:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            constructed.append(self)

    class RecordingPipeline:
        def __init__(self, *args, status_style_predictor=None, **kwargs) -> None:
            pipeline_predictors.append(status_style_predictor)

        def run(self, source_path):
            return object()

    monkeypatch.setattr(infer_module, "LRCNNPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer_module, "StatusStylePredictor", FakeStatusStylePredictor)
    monkeypatch.setattr(infer_module, "status_style_checkpoint_signature", lambda _path: model_signature)
    monkeypatch.setattr(infer_module, "ReceiptPipeline", RecordingPipeline)
    monkeypatch.setattr(infer_module, "write_receipt_result", lambda *args, **kwargs: None)

    manifest = infer_module.run_inference(
        checkpoint=tmp_path / "receipt.pt",
        status_style_checkpoint=checkpoint,
        input_path=input_dir,
        output_dir=output_dir,
        device="cuda",
        use_ocr=False,
        status_confidence_threshold=0.84,
        status_absent_confidence_threshold=0.97,
    )

    assert len(manifest) == 3
    assert len(constructed) == 1
    assert pipeline_predictors == [constructed[0], constructed[0], constructed[0]]
    assert constructed[0].args == (checkpoint,)
    assert constructed[0].kwargs == {
        "device": "cuda",
        "confidence_threshold": 0.84,
        "absent_confidence_threshold": 0.97,
    }


def test_limit_must_be_positive(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "receipt.jpg").write_bytes(b"unused")

    try:
        infer_module.run_inference(
            checkpoint=tmp_path / "unused.pt",
            input_path=input_dir,
            output_dir=tmp_path / "output",
            use_ocr=False,
            limit=0,
        )
    except ValueError as error:
        assert "limit must be positive" in str(error)
    else:
        raise AssertionError("a non-positive limit should be rejected")


@pytest.mark.parametrize(
    ("confidence_threshold", "absent_threshold", "message"),
    [
        (-0.01, 0.95, "status_confidence_threshold must be between 0 and 1"),
        (0.80, 1.01, "status_absent_confidence_threshold must be between 0 and 1"),
        (0.90, 0.89, "status_absent_confidence_threshold must be at least status_confidence_threshold"),
    ],
)
def test_status_style_thresholds_are_validated_before_model_loading(
    tmp_path,
    confidence_threshold,
    absent_threshold,
    message,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "receipt.jpg").write_bytes(b"unused")

    with pytest.raises(ValueError, match=message):
        infer_module.run_inference(
            checkpoint=tmp_path / "receipt.pt",
            status_style_checkpoint=tmp_path / "status.pt",
            input_path=input_dir,
            output_dir=tmp_path / "output",
            use_ocr=False,
            status_confidence_threshold=confidence_threshold,
            status_absent_confidence_threshold=absent_threshold,
        )
