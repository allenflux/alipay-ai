import numpy as np

from transfer_receipt_ai.render import RenderItem, draw_rectified_circles


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
