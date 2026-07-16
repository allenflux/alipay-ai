import cv2
import numpy as np

from transfer_receipt_ai.geometry import (
    RectificationOptions,
    SCREEN_SCORE_FLOOR,
    _quad_score,
    find_screen_quad,
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


def test_wide_recipient_card_is_not_considered_a_screen() -> None:
    # A 1,200×360 in-app card inside a 1,280×1,000 receipt screenshot must not
    # be perspective-cropped as though it were the phone display.
    card = np.array([[30, 300], [1230, 300], [1230, 660], [30, 660]], dtype=np.float32)
    assert _quad_score(card, image_width=1280, image_height=1000) < 0


def test_full_receipt_screenshot_does_not_crop_its_wide_recipient_card() -> None:
    image = np.full((1600, 900, 3), (52, 123, 240), dtype=np.uint8)
    # This resembles the large white recipient card in the lower half of the
    # screenshot. It is deliberately the strongest visible rectangle.
    cv2.rectangle(image, (24, 1030), (876, 1410), (255, 255, 255), thickness=8)
    cv2.rectangle(image, (70, 1090), (290, 1350), (220, 235, 255), thickness=-1)

    assert find_screen_quad(image) is None
    result = rectify_receipt(
        image,
        RectificationOptions(orientation_degrees=0, auto_screen=True, max_side=0),
    )
    assert result.screen_detected is False
    assert result.rectified_rgb.shape == image.shape
    assert np.array_equal(result.rectified_rgb, image)


def test_large_portrait_phone_quad_can_be_auto_detected() -> None:
    image = np.zeros((1200, 1000, 3), dtype=np.uint8)
    phone = np.array([[230, 90], [740, 150], [700, 1100], [190, 1040]], dtype=np.int32)
    cv2.fillConvexPoly(image, phone, (52, 123, 240))

    detected = find_screen_quad(image)
    assert detected is not None
    assert _quad_score(detected, image_width=1000, image_height=1200) >= SCREEN_SCORE_FLOOR
