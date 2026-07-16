from transfer_receipt_ai.ocr import normalize_amount, normalize_payment_method, normalize_status


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
    assert normalize_payment_method("账户余额") == {"raw": "账户余额", "normalized": "balance"}
