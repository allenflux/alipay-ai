import pytest

from transfer_receipt_ai.model import validate_checkpoint_classes


def test_checkpoint_class_contract_rejects_legacy_labels() -> None:
    with pytest.raises(ValueError, match="five-field schema"):
        validate_checkpoint_classes(
            {
                "classes": [
                    "amount",
                    "success_icon",
                    "success_text",
                    "recipient_value",
                    "payment_method_value",
                ]
            }
        )


def test_lrcnn_forward_smoke_without_pretrained_download() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from transfer_receipt_ai.model import LRCNNConfig, build_lrcnn

    model = build_lrcnn(LRCNNConfig(min_size=128, max_size=128, pretrained=False)).eval()
    with torch.inference_mode():
        result = model([torch.rand(3, 96, 80)])[0]
    assert {"boxes", "labels", "scores"}.issubset(result)
