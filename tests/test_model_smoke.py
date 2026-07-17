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

    model = build_lrcnn(
        LRCNNConfig(min_size=128, max_size=128, pretrained=True),
        load_pretrained_weights=False,
    ).eval()
    assert model.rpn.anchor_generator.num_anchors_per_location() == [25, 25, 25]
    assert model.rpn.head.cls_logits.out_channels == 25
    assert model.rpn.head.bbox_pred.out_channels == 100
    assert model.roi_heads.box_predictor.cls_score.out_features == 6
    backbone_grad_flags = [parameter.requires_grad for parameter in model.backbone.body.parameters()]
    assert any(backbone_grad_flags) and not all(backbone_grad_flags)
    with torch.inference_mode():
        result = model([torch.rand(3, 96, 80)])[0]
    assert {"boxes", "labels", "scores"}.issubset(result)

    model.train()
    target = {
        "boxes": torch.tensor(
            [
                [3.0, 3.0, 22.0, 12.0],
                [20.0, 18.0, 62.0, 38.0],
                [18.0, 42.0, 60.0, 52.0],
                [3.0, 58.0, 75.0, 70.0],
                [3.0, 74.0, 75.0, 88.0],
            ]
        ),
        "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
    }
    losses = model([torch.rand(3, 96, 80)], [target])
    total_loss = sum(losses.values())
    assert {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"} == set(losses)
    assert torch.isfinite(total_loss)
    total_loss.backward()
