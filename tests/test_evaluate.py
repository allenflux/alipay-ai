from transfer_receipt_ai.evaluate import format_metrics


def test_format_metrics_lists_all_five_fields() -> None:
    metrics = {
        "map50": 0.8,
        "ap50": {"time": 1.0, "amount": 0.5},
        "recall50": {"time": 0.9, "amount": 0.4},
    }
    report = format_metrics(metrics)
    assert "mAP@IoU=0.50 (score>=0.05): 0.8000" in report
    assert "time" in report and "AP50=1.0000" in report
    assert "amount" in report and "Recall50=0.4000" in report
    assert "transfer_status" in report
    assert "recipient_field" in report
    assert "payment_method_field" in report
