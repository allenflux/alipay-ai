import pytest


def test_lrcnn_forward_smoke_without_pretrained_download() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from transfer_receipt_ai.model import LRCNNConfig, build_lrcnn

    model = build_lrcnn(LRCNNConfig(min_size=128, max_size=128, pretrained=False)).eval()
    with torch.inference_mode():
        result = model([torch.rand(3, 96, 80)])[0]
    assert {"boxes", "labels", "scores"}.issubset(result)
