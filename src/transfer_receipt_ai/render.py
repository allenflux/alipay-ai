"""Draw perspective-safe circles around detected receipt fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import transform_points

DISPLAY_NAMES = {
    "amount": "金额",
    "success_icon": "成功勾",
    "success_text": "转账成功",
    "recipient_value": "收款人",
    "payment_method_value": "付款方式",
}
COLORS = {
    "amount": (255, 80, 80),
    "success_icon": (255, 210, 0),
    "success_text": (255, 155, 60),
    "recipient_value": (72, 202, 128),
    "payment_method_value": (80, 160, 255),
}


@dataclass(frozen=True)
class RenderItem:
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    text: str | None = None


def ellipse_polygon(bbox_xyxy: Sequence[float], samples: int = 40) -> np.ndarray:
    """Approximate the circle/ellipse that encloses one bounding box."""
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius_x, radius_y = max(2.0, (x2 - x1) / 2.0 + 3.0), max(2.0, (y2 - y1) / 2.0 + 3.0)
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=True)
    return np.column_stack((center_x + radius_x * np.cos(angles), center_y + radius_y * np.sin(angles))).astype(np.float32)


def _find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _item_caption(item: RenderItem) -> str:
    name = DISPLAY_NAMES.get(item.label, item.label)
    suffix = f" {item.text}" if item.text else ""
    return f"{name}{suffix} ({item.score:.0%})"


def _draw_items(image_rgb: np.ndarray, item_polygons: Iterable[tuple[RenderItem, np.ndarray]]) -> np.ndarray:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    font = _find_font(19)
    for item, polygon in item_polygons:
        color = COLORS.get(item.label, (255, 0, 255))
        points = [tuple(float(value) for value in point) for point in polygon]
        draw.line(points, fill=color, width=4, joint="curve")
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        caption = _item_caption(item)
        text_left = max(0, min(x_values))
        text_top = max(0, min(y_values) - 25)
        try:
            text_box = draw.textbbox((text_left, text_top), caption, font=font)
            draw.rounded_rectangle(text_box, radius=3, fill=(0, 0, 0))
            draw.text((text_left, text_top), caption, fill=color, font=font)
        except UnicodeEncodeError:
            # Minimal Linux containers may not contain a CJK font. Keep rendering
            # the circle and fall back to the stable machine-readable label.
            caption = f"{item.label} ({item.score:.0%})"
            text_box = draw.textbbox((text_left, text_top), caption, font=font)
            draw.rounded_rectangle(text_box, radius=3, fill=(0, 0, 0))
            draw.text((text_left, text_top), caption, fill=color, font=font)
        except AttributeError:  # Older Pillow still supports the actual text draw.
            draw.text((text_left, text_top), caption, fill=color, font=font)
    return np.asarray(image)


def draw_rectified_circles(image_rgb: np.ndarray, items: Sequence[RenderItem]) -> np.ndarray:
    """Draw ellipse outlines in the detector's rectified coordinate system."""
    return _draw_items(image_rgb, ((item, ellipse_polygon(item.bbox_xyxy)) for item in items))


def draw_original_circles(
    image_rgb: np.ndarray,
    items: Sequence[RenderItem],
    rectified_to_original: np.ndarray,
) -> np.ndarray:
    """Project rectified ellipses back into the photo before drawing them.

    A circle therefore becomes the correct perspective curve in the original
    photo instead of an inaccurate axis-aligned rectangle.
    """
    return _draw_items(
        image_rgb,
        (
            (item, transform_points(ellipse_polygon(item.bbox_xyxy), rectified_to_original))
            for item in items
        ),
    )
