import numpy as np
import pytest

from transfer_receipt_ai.render import (
    RenderItem,
    StatusStyleRenderItem,
    _status_style_caption,
    draw_original_circles,
    draw_rectified_circles,
)


def test_render_keeps_source_area_and_places_text_in_side_legend() -> None:
    image = np.full((400, 240, 3), 235, dtype=np.uint8)
    item = RenderItem("amount", 0.96, (60.0, 130.0, 180.0, 190.0), "¥99.97")

    rendered = draw_rectified_circles(image, [item])

    assert rendered.shape[0] == image.shape[0]
    assert rendered.shape[1] > image.shape[1]
    # The untouched top-left receipt area stays identical; captions live in the
    # added panel rather than being drawn over source pixels.
    assert np.array_equal(rendered[:80, :120], image[:80, :120])
    assert not np.array_equal(rendered[:, image.shape[1] :], np.full_like(rendered[:, image.shape[1] :], 235))


def test_render_without_status_style_is_pixel_compatible_with_explicit_none() -> None:
    image = np.full((500, 300, 3), 235, dtype=np.uint8)
    items = [RenderItem("transfer_status", 0.98, (80.0, 90.0, 220.0, 140.0), "转账成功")]

    legacy = draw_rectified_circles(image, items)
    explicit_none = draw_rectified_circles(image, items, status_style=None)

    assert np.array_equal(legacy, explicit_none)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("check_offset", "设备/风险标签 Android (96%)"),
        ("check_aligned", "设备/风险标签 iOS (96%)"),
        ("check_absent", "设备/风险标签 疑似假图 (96%)"),
        ("unknown", "设备/风险标签 待复核 (96%)"),
        ("future_label", "设备/风险标签 待复核 (96%)"),
    ],
)
def test_status_style_caption_uses_business_facing_values(label: str, expected: str) -> None:
    assert _status_style_caption(StatusStyleRenderItem(label, 0.956)) == expected


def test_status_style_is_appended_to_legend_without_drawing_on_source() -> None:
    image = np.full((700, 360, 3), 235, dtype=np.uint8)
    items = [RenderItem("transfer_status", 0.98, (90.0, 110.0, 260.0, 165.0), "转账成功")]

    without_status = draw_rectified_circles(image, items)
    with_status = draw_rectified_circles(
        image,
        items,
        status_style=StatusStyleRenderItem("check_aligned", 0.97),
    )

    assert with_status.shape == without_status.shape
    assert np.array_equal(with_status[:, : image.shape[1]], without_status[:, : image.shape[1]])
    assert not np.array_equal(with_status[:, image.shape[1] :], without_status[:, image.shape[1] :])


def test_original_render_accepts_same_optional_status_style() -> None:
    image = np.full((700, 360, 3), 235, dtype=np.uint8)
    items = [RenderItem("transfer_status", 0.98, (90.0, 110.0, 260.0, 165.0), "转账成功")]

    rendered = draw_original_circles(
        image,
        items,
        np.eye(3),
        status_style=StatusStyleRenderItem("check_absent", 0.99),
    )

    assert rendered.shape[0] == image.shape[0]
    assert rendered.shape[1] > image.shape[1]
    assert np.array_equal(rendered[:80, :200], image[:80, :200])
