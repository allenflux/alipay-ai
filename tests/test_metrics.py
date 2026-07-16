import numpy as np

from transfer_receipt_ai.labels import LABEL_TO_ID
from transfer_receipt_ai.metrics import evaluate_map50


def test_perfect_detection_has_perfect_amount_ap() -> None:
    amount_id = LABEL_TO_ID["amount"]
    predictions = [
        {
            "boxes": np.array([[1, 1, 11, 11]], dtype=np.float32),
            "labels": np.array([amount_id]),
            "scores": np.array([0.99]),
        }
    ]
    targets = [
        {
            "boxes": np.array([[1, 1, 11, 11]], dtype=np.float32),
            "labels": np.array([amount_id]),
            "image_id": np.array([1]),
        }
    ]
    result = evaluate_map50(predictions, targets)
    assert result["ap50"]["amount"] == 1.0
    assert result["recall50"]["amount"] == 1.0
