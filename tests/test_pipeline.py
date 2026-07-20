import json
from pathlib import Path

from PIL import Image

from transfer_receipt_ai.geometry import RectificationOptions
from transfer_receipt_ai.model import Detection
from transfer_receipt_ai.ocr import OCRResult
from transfer_receipt_ai.pipeline import ReceiptPipeline, write_receipt_result


class _FakePredictor:
    def predict(self, _image):
        return [
            Detection("time", 0.99, (10, 5, 45, 15)),
            Detection("amount", 0.99, (10, 20, 70, 40)),
            Detection("transfer_status", 0.98, (10, 45, 80, 60)),
            Detection("recipient_field", 0.96, (10, 65, 90, 78)),
            Detection("payment_method_field", 0.95, (10, 80, 90, 94)),
        ]


class _FakeOcr:
    def __init__(self):
        self._texts = iter(["00:01:09", "¥199.93", "✓ 转账成功", "收款方 上平(**平)", "付款方式 账户余额"])

    def recognize(self, _image):
        return OCRResult(next(self._texts), 0.99)


class _FakeStatusStylePredictor:
    def __init__(self) -> None:
        self.crops = []

    def predict(self, image):
        self.crops.append(image.copy())
        return {
            "label": "check_aligned",
            "confidence": 0.97,
            "business_tag": "ios",
            "candidate_label": "check_aligned",
            "probabilities": {
                "check_offset": 0.01,
                "check_aligned": 0.97,
                "check_absent": 0.02,
            },
        }


def test_pipeline_writes_circles_and_structured_fields(tmp_path) -> None:
    source = tmp_path / "raw.jpg"
    Image.new("RGB", (100, 100), "#3377ee").save(source)
    pipeline = ReceiptPipeline(
        _FakePredictor(),
        ocr=_FakeOcr(),
        rectification_options=RectificationOptions(auto_screen=False, orientation_degrees=0),
    )
    result = pipeline.run(source)
    written = write_receipt_result(result, tmp_path / "result")
    assert result.fields["time"]["value"] == "00:01:09"
    assert result.fields["amount"]["amount_fen"] == 19993
    assert result.fields["transfer_status"]["normalized"] == "success"
    assert result.fields["recipient"]["value"] == "上平(**平)"
    assert result.fields["payment_method"]["normalized"] == "balance"
    assert all(Path(path).is_file() for path in written.values())
    # Omitting the optional classifier preserves the original v1 schema.
    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert "status_style" not in payload
    assert "tags" not in payload


def test_pipeline_classifies_status_crop_and_adds_tags_without_a_sixth_detection(tmp_path) -> None:
    source = tmp_path / "raw.jpg"
    Image.new("RGB", (100, 100), "#3377ee").save(source)
    status_predictor = _FakeStatusStylePredictor()
    model_signature = {
        "path": "D:/models/status_style_v1/best.pt",
        "sha256": "abc123",
        "size_bytes": 42,
    }
    inference_config = {
        "confidence_threshold": 0.80,
        "absent_confidence_threshold": 0.95,
        "margin_ratio": 0.30,
    }
    pipeline = ReceiptPipeline(
        _FakePredictor(),
        ocr=_FakeOcr(),
        rectification_options=RectificationOptions(auto_screen=False, orientation_degrees=0),
        status_style_predictor=status_predictor,
        status_style_model=model_signature,
        status_style_inference_config=inference_config,
        status_style_margin_ratio=0.30,
    )

    result = pipeline.run(source)
    written = write_receipt_result(result, tmp_path / "integrated")
    payload = json.loads(written["json"].read_text(encoding="utf-8"))

    # transfer_status=(10,45,80,60), plus 30% margin, clipped to 100x100.
    assert len(status_predictor.crops) == 1
    assert status_predictor.crops[0].shape == (25, 100, 3)
    assert len(result.detections) == 5
    assert len(payload["detections"]) == 5
    assert payload["status_style"]["label"] == "check_aligned"
    assert payload["status_style"]["state"] == "classified"
    assert payload["status_style"]["model"] == model_signature
    assert payload["status_style"]["inference_config"] == inference_config
    assert payload["status_style"]["transfer_status_bbox_rectified"] == [10.0, 45.0, 80.0, 60.0]
    assert payload["tags"] == {
        "platform": "ios",
        "authenticity": "not_assessed",
        "review_tag": None,
        "requires_manual_review": False,
        "reason": "status_check_aligned",
        "rule_version": "status-style-v2",
    }
