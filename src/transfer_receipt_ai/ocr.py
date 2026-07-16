"""OCR abstraction and conservative normalisation for detected receipt fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None
    lines: tuple[tuple[str, float], ...] = ()


class TextRecognizer(Protocol):
    def recognize(self, image_rgb: np.ndarray) -> OCRResult:
        """Read text from an RGB image crop."""


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_paddle_lines(payload: Any) -> list[tuple[str, float]]:
    """Support the common PaddleOCR 2.x result shape and its dict variants."""
    lines: list[tuple[str, float]] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts")
            scores = node.get("rec_scores") or node.get("scores")
            if isinstance(texts, (list, tuple)):
                scores = scores if isinstance(scores, (list, tuple)) else [None] * len(texts)
                for text, score in zip(texts, scores):
                    if isinstance(text, str):
                        lines.append((text, float(score) if score is not None else 0.0))
                return
            for value in node.values():
                visit(value)
            return
        if isinstance(node, (list, tuple)):
            # PaddleOCR v2 line: [[x1,y1], ...], ("text", confidence)
            if (
                len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
                and isinstance(node[1][0], str)
            ):
                try:
                    lines.append((node[1][0], float(node[1][1])))
                except (TypeError, ValueError):
                    lines.append((node[1][0], 0.0))
                return
            for value in node:
                visit(value)

    visit(payload)
    return lines


class PaddleOCRReader:
    """Chinese OCR backed by PaddleOCR, imported only when the feature is used."""

    def __init__(self, language: str = "ch", use_angle_cls: bool = True) -> None:
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as error:
            raise ImportError(
                "PaddleOCR is not installed. Install a PaddlePaddle wheel for this platform, "
                "then run `pip install -r requirements-ocr.txt`."
            ) from error
        self._use_angle_cls = use_angle_cls
        options = {"lang": language, "use_angle_cls": use_angle_cls, "show_log": False}
        try:
            self._engine = PaddleOCR(**options)
        except TypeError:  # PaddleOCR versions that no longer expose show_log.
            options.pop("show_log")
            self._engine = PaddleOCR(**options)

    def recognize(self, image_rgb: np.ndarray) -> OCRResult:
        raw = self._engine.ocr(image_rgb, cls=self._use_angle_cls)
        lines = _extract_paddle_lines(raw)
        if not lines:
            return OCRResult(text="", confidence=None)
        text = " ".join(clean_text(line[0]) for line in lines if clean_text(line[0]))
        confidence = sum(line[1] for line in lines) / len(lines)
        return OCRResult(text=text, confidence=confidence, lines=tuple(lines))

    def orientation_score(self, image_rgb: np.ndarray) -> float:
        """Return a text-quality score for one image orientation.

        The image preparation command evaluates this for 0/90/180/270 degrees.
        It is slower than EXIF/geometry correction but can distinguish upside-down
        Chinese text, which a rectangle-only method cannot do.
        """
        result = self.recognize(image_rgb)
        if not result.text or result.confidence is None:
            return float("-inf")
        return result.confidence + min(len(result.lines), 10) * 0.01


_AMOUNT_PATTERN = re.compile(r"(?:[¥￥]\s*)?([0-9OoIl]{1,3}(?:,[0-9OoIl]{3})*(?:\.\d{1,2})?)")


def normalize_amount(raw_text: str) -> dict[str, object] | None:
    """Normalise a CNY amount while retaining the raw OCR text separately."""
    raw_text = clean_text(raw_text)
    matches = list(_AMOUNT_PATTERN.finditer(raw_text))
    if not matches:
        return None
    # Currency-marked and decimal candidates are much more likely to be the value
    # than unrelated UI digits such as a time in the status bar.
    match = max(
        matches,
        key=lambda item: (
            1 if item.group(0).lstrip().startswith(("¥", "￥")) else 0,
            1 if "." in item.group(1) else 0,
            len(item.group(1)),
        ),
    )
    numeric = match.group(1).translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})).replace(",", "")
    try:
        value = Decimal(numeric).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    if value < 0:
        return None
    fen = int(value * 100)
    return {
        "raw": raw_text,
        "normalized": f"¥{value:.2f}",
        "amount_fen": fen,
        "currency": "CNY",
    }


def normalize_status(raw_text: str) -> str:
    compact = re.sub(r"\s+", "", raw_text)
    if any(token in compact for token in ("转账成功", "交易成功", "付款成功", "转帐成功")):
        return "success"
    if any(token in compact for token in ("失败", "未成功", "已撤销")):
        return "failed"
    if any(token in compact for token in ("处理中", "待处理", "进行中")):
        return "pending"
    return "unknown"


def normalize_payment_method(raw_text: str) -> dict[str, str]:
    raw = clean_text(raw_text)
    compact = re.sub(r"\s+", "", raw)
    if "余额" in compact:
        kind = "balance"
    elif "花呗" in compact:
        kind = "huabei"
    elif any(token in compact for token in ("银行卡", "储蓄卡", "信用卡")):
        kind = "bank_card"
    elif "余额宝" in compact:
        kind = "yuebao"
    else:
        kind = "other"
    return {"raw": raw, "normalized": kind}
