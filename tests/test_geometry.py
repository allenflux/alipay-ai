import numpy as np

from transfer_receipt_ai.geometry import (
    RectificationOptions,
    order_quad,
    rectify_receipt,
    rotation_homography,
    transform_points,
)


def test_rotation_homography_round_trip() -> None:
    points = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=np.float32)
    homography = rotation_homography(width=4, height=3, degrees=90)
    rotated = transform_points(points, homography)
    assert np.allclose(rotated, [[2, 0], [2, 3], [0, 3], [0, 0]])
    assert np.allclose(transform_points(rotated, np.linalg.inv(homography)), points)


def test_order_quad_accepts_arbitrary_order() -> None:
    arbitrary = np.array([[90, 80], [10, 10], [10, 80], [90, 10]], dtype=np.float32)
    assert np.allclose(order_quad(arbitrary), [[10, 10], [90, 10], [90, 80], [10, 80]])


def test_manual_rectification_preserves_coordinate_mapping() -> None:
    image = np.zeros((80, 40, 3), dtype=np.uint8)
    result = rectify_receipt(
        image,
        RectificationOptions(orientation_degrees=0, auto_screen=False, max_side=200),
    )
    assert result.rectified_rgb.shape[:2] == (80, 40)
    original_point = np.array([[12, 34]], dtype=np.float32)
    rectified_point = transform_points(original_point, result.original_to_rectified)
    assert np.allclose(transform_points(rectified_point, result.rectified_to_original), original_point)
