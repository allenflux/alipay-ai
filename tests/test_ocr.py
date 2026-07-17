import sys
from types import SimpleNamespace

import numpy as np
import pytest

from transfer_receipt_ai.ocr import (
    PaddleOCRReader,
    _extract_paddle_lines,
    extract_field_value,
    normalize_amount,
    normalize_payment_method,
    normalize_status,
    normalize_time,
)


def test_extract_paddle_v3_result_object_with_numpy_scores() -> None:
    result = SimpleNamespace(
        json={"res": {"rec_texts": ["收款方", "张三"], "rec_scores": np.array([0.9, 0.8])}}
    )

    assert _extract_paddle_lines([result]) == [("收款方", 0.9), ("张三", 0.8)]


def test_paddle_reader_uses_v3_predict_api(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeV3OCR:
        def __init__(self, **options) -> None:
            calls["options"] = options

        def predict(self, image):
            calls["image"] = image
            return [SimpleNamespace(json={"res": {"rec_texts": ["转账成功"], "rec_scores": [0.97]}})]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakeV3OCR, __version__="3.4.0"))
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    result = PaddleOCRReader().recognize(image)

    assert calls["options"] == {
        "lang": "ch",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
    }
    assert calls["image"] is image
    assert result.text == "转账成功"
    assert result.confidence == 0.97


def test_paddle_reader_falls_back_to_v2_api(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeV2OCR:
        def __init__(self, **options) -> None:
            calls.append(options)
            if "use_doc_orientation_classify" in options:
                raise ValueError("Unknown argument: use_doc_orientation_classify")

        def ocr(self, image, cls):
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("账户余额", 0.88)]]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakeV2OCR, __version__="2.10.0"))

    result = PaddleOCRReader().recognize(np.zeros((10, 20, 3), dtype=np.uint8))

    assert calls[-1] == {"lang": "ch", "use_angle_cls": True, "show_log": False}
    assert result.text == "账户余额"
    assert result.confidence == 0.88


def test_paddle_v3_initialisation_error_is_not_misreported_as_v2(monkeypatch) -> None:
    class BrokenV3OCR:
        def __init__(self, **options) -> None:
            raise ValueError("model download failed")

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=BrokenV3OCR, __version__="3.4.0"),
    )

    with pytest.raises(ValueError, match="model download failed"):
        PaddleOCRReader()


def test_normalize_amount_prefers_currency_value() -> None:
    result = normalize_amount("¥ 1,299.9")
    assert result == {
        "raw": "¥ 1,299.9",
        "normalized": "¥1299.90",
        "amount_fen": 129990,
        "currency": "CNY",
    }


def test_status_and_payment_normalization() -> None:
    assert normalize_status("转 账 成 功") == "success"
    assert normalize_status("支付成功") == "success"
    assert normalize_status("转账未成功") == "failed"
    assert normalize_payment_method("账户余额") == {"raw": "账户余额", "normalized": "balance"}
    assert normalize_payment_method("余额宝(转出资金付款)") == {
        "raw": "余额宝(转出资金付款)",
        "normalized": "yuebao",
    }


def test_extract_value_from_complete_red_line_row() -> None:
    assert extract_field_value("收款方 富森(**森)", "recipient") == "富森(**森)"
    assert extract_field_value("交易方式 余额宝(转出资金付款)", "payment_method") == "余额宝(转出资金付款)"
    assert extract_field_value("富森(**森) 收款方", "recipient") == "富森(**森)"


def test_normalize_visible_status_bar_time() -> None:
    assert normalize_time("00:01:09 5G 14%") == "00:01:09"
    assert normalize_time("时间00：02") == "00:02"
