from pathlib import Path

from PIL import Image

from transfer_receipt_ai.geometry import RectificationOptions
from transfer_receipt_ai.model import Detection
from transfer_receipt_ai.ocr import OCRResult
from transfer_receipt_ai.pipeline import ReceiptPipeline, write_receipt_result


class _FakePredictor:
    def predict(self, _image):
        return [
            Detection("amount", 0.99, (10, 10, 70, 30)),
            Detection("success_icon", 0.98, (10, 35, 25, 50)),
            Detection("success_text", 0.97, (30, 35, 80, 50)),
            Detection("recipient_value", 0.96, (25, 55, 90, 70)),
            Detection("payment_method_value", 0.95, (25, 75, 90, 90)),
        ]


class _FakeOcr:
    def __init__(self):
        self._texts = iter(["¥199.93", "转账成功", "上平(**平)", "账户余额"])

    def recognize(self, _image):
        return OCRResult(next(self._texts), 0.99)


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
    assert result.fields["amount"]["amount_fen"] == 19993
    assert result.fields["transfer_success"]["confirmed"] is True
    assert result.fields["recipient"]["raw"] == "上平(**平)"
    assert result.fields["payment_method"]["normalized"] == "balance"
    assert all(Path(path).is_file() for path in written.values())
