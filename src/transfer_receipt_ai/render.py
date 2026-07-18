"""Draw perspective-safe circles around detected receipt fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import transform_points

DISPLAY_NAMES = {
    "time": "时间",
    "amount": "金额",
    "transfer_status": "转账状态",
    "recipient_field": "收款方",
    "payment_method_field": "付款方式",
}
COLORS = {
    "time": (222, 82, 255),
    "amount": (255, 80, 80),
    "transfer_status": (255, 210, 0),
    "recipient_field": (72, 202, 128),
    "payment_method_field": (80, 160, 255),
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
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
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


def _wrap_caption(draw: ImageDraw.ImageDraw, caption: str, font: ImageFont.ImageFont, max_width: int) -> str:
    """Wrap mixed Chinese/ASCII captions without relying on whitespace."""
    lines: list[str] = []
    current = ""
    for character in caption:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _draw_items(image_rgb: np.ndarray, item_polygons: Iterable[tuple[RenderItem, np.ndarray]]) -> np.ndarray:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    height, width = image_rgb.shape[:2]
    font_size = max(30, min(58, int(round(max(height, width) * 0.022))))
    font = _find_font(font_size)
    line_width = max(4, min(10, int(round(max(height, width) * 0.003))))
    padding = max(4, font_size // 7)
    for item, polygon in item_polygons:
        color = COLORS.get(item.label, (255, 0, 255))
        points = [tuple(float(value) for value in point) for point in polygon]
        draw.line(points, fill=color, width=line_width, joint="curve")
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        caption = _item_caption(item)
        text_left = max(0, min(x_values))
        try:
            caption = _wrap_caption(draw, caption, font, max(80, width - padding * 2))
            provisional_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=2)
        except UnicodeEncodeError:
            # Minimal Linux containers may not contain a CJK font.
            caption = f"{item.label} ({item.score:.0%})"
            provisional_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=2)
        text_width = provisional_box[2] - provisional_box[0]
        text_height = provisional_box[3] - provisional_box[1]
        text_left = min(text_left, max(0, width - text_width - padding * 2))
        text_top = max(0, min(y_values) - text_height - padding * 2)
        try:
            text_box = draw.multiline_textbbox((text_left, text_top), caption, font=font, spacing=2)
            background = (
                text_box[0] - padding,
                text_box[1] - padding,
                text_box[2] + padding,
                text_box[3] + padding,
            )
            draw.rounded_rectangle(background, radius=padding, fill=(0, 0, 0), outline=color, width=2)
            draw.multiline_text((text_left, text_top), caption, fill=(255, 255, 255), font=font, spacing=2)
        except UnicodeEncodeError:
            # Minimal Linux containers may not contain a CJK font. Keep rendering
            # the circle and fall back to the stable machine-readable label.
            caption = f"{item.label} ({item.score:.0%})"
            text_box = draw.textbbox((text_left, text_top), caption, font=font)
            background = (
                text_box[0] - padding,
                text_box[1] - padding,
                text_box[2] + padding,
                text_box[3] + padding,
            )
            draw.rounded_rectangle(background, radius=padding, fill=(0, 0, 0), outline=color, width=2)
            draw.text((text_left, text_top), caption, fill=(255, 255, 255), font=font)
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
