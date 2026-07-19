import numpy as np
import pytest

from transfer_receipt_ai.status_style import (
    STATUS_STYLE_CLASSES,
    StatusStyleConfig,
    business_tag_for_status_style,
    letterbox_status_crop,
    prediction_from_probabilities,
    validate_status_style_checkpoint,
)


def test_status_style_class_order_is_an_exact_checkpoint_contract() -> None:
    assert STATUS_STYLE_CLASSES == ("check_offset", "check_aligned", "check_absent")
    validate_status_style_checkpoint({"classes": list(STATUS_STYLE_CLASSES)})

    with pytest.raises(ValueError, match="status-style schema"):
        validate_status_style_checkpoint(
            {"classes": ["check_aligned", "check_offset", "check_absent"]}
        )


@pytest.mark.parametrize(
    ("source_shape", "content_slice"),
    [
        ((100, 400, 3), (slice(24, 104), slice(0, 320))),
        ((400, 100, 3), (slice(0, 128), slice(144, 176))),
    ],
)
def test_letterbox_preserves_aspect_and_keeps_entire_crop_visible(
    source_shape: tuple[int, int, int],
    content_slice: tuple[slice, slice],
) -> None:
    source = np.full(source_shape, (90, 120, 180), dtype=np.uint8)
    result = letterbox_status_crop(source)

    assert result.shape == (128, 320, 3)
    mask = np.any(result != 0, axis=2)
    expected = np.zeros((128, 320), dtype=bool)
    expected[content_slice] = True
    assert np.array_equal(mask, expected)


def test_letterbox_does_not_crop_edge_evidence() -> None:
    source = np.zeros((40, 400, 3), dtype=np.uint8)
    source[:, :20] = (255, 0, 0)
    source[:, -20:] = (0, 255, 0)

    result = letterbox_status_crop(source)
    content = result[48:80]

    assert content[:, 0, 0].mean() > 200
    assert content[:, -1, 1].mean() > 200


def test_low_confidence_becomes_unknown_and_routes_to_review() -> None:
    prediction = prediction_from_probabilities((0.60, 0.25, 0.15))

    assert prediction.label == "unknown"
    assert prediction.candidate_label == "check_offset"
    assert prediction.business_tag == "review"
    assert prediction.confidence == pytest.approx(0.60)


def test_check_absent_uses_its_separate_higher_confidence_gate() -> None:
    inconclusive = prediction_from_probabilities(
        (0.05, 0.10, 0.85),
        confidence_threshold=0.70,
        absent_confidence_threshold=0.90,
    )
    confident = prediction_from_probabilities(
        (0.03, 0.05, 0.92),
        confidence_threshold=0.70,
        absent_confidence_threshold=0.90,
    )

    assert inconclusive.candidate_label == "check_absent"
    assert inconclusive.label == "unknown"
    assert inconclusive.business_tag == "review"
    assert confident.label == "check_absent"
    assert confident.business_tag == "suspected_fake"


@pytest.mark.parametrize(
    ("probabilities", "expected_label", "expected_tag"),
    [
        ((0.80, 0.10, 0.10), "check_offset", "android"),
        ((0.10, 0.80, 0.10), "check_aligned", "ios"),
        ((0.03, 0.05, 0.92), "check_absent", "suspected_fake"),
    ],
)
def test_business_mapping_is_objective_and_auditable(
    probabilities: tuple[float, float, float],
    expected_label: str,
    expected_tag: str,
) -> None:
    prediction = prediction_from_probabilities(probabilities)
    assert prediction.label == expected_label
    assert prediction.business_tag == expected_tag
    assert prediction.as_dict()["probabilities"] == dict(zip(STATUS_STYLE_CLASSES, probabilities))


def test_unknown_or_unrecognised_styles_are_never_treated_as_authenticity_proof() -> None:
    assert business_tag_for_status_style("unknown") == "review"
    assert business_tag_for_status_style("unrecognised") == "review"


def test_preprocess_has_default_wide_canvas_without_warping() -> None:
    torch = pytest.importorskip("torch")
    from transfer_receipt_ai.status_style import preprocess_status_crop

    tensor = preprocess_status_crop(np.ones((40, 200, 3), dtype=np.uint8), StatusStyleConfig())
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 128, 320)
    assert tensor.dtype == torch.float32


def test_mobilenet_v3_small_has_exactly_three_logits() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from transfer_receipt_ai.status_style import build_status_style_model

    model = build_status_style_model(load_pretrained_weights=False)
    assert model.classifier[-1].out_features == len(STATUS_STYLE_CLASSES)
