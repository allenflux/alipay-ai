import numpy as np

from transfer_receipt_ai.metrics import evaluate_map50


def test_perfect_detection_has_perfect_amount_ap() -> None:
    predictions = [
        {
            "boxes": np.array([[1, 1, 11, 11]], dtype=np.float32),
            "labels": np.array([1]),
            "scores": np.array([0.99]),
        }
    ]
    targets = [
        {
            "boxes": np.array([[1, 1, 11, 11]], dtype=np.float32),
            "labels": np.array([1]),
            "image_id": np.array([1]),
        }
    ]
    result = evaluate_map50(predictions, targets)
    assert result["ap50"]["amount"] == 1.0
    assert result["recall50"]["amount"] == 1.0
