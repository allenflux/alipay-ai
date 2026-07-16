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
