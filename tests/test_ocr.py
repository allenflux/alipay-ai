from transfer_receipt_ai.ocr import extract_field_value, normalize_amount, normalize_payment_method, normalize_status, normalize_time


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
