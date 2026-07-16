"""End-to-end rectification, LRCNN detection, OCR and structured extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    RectificationOptions,
    RectificationResult,
    bbox_to_polygon,
    load_upright_rgb,
    save_rgb,
    transform_points,
    rectify_receipt,
)
from .model import Detection, LRCNNPredictor
from .ocr import OCRResult, TextRecognizer, clean_text, normalize_amount, normalize_payment_method, normalize_status
from .render import RenderItem, draw_original_circles, draw_rectified_circles


@dataclass(frozen=True)
class ExtractedDetection:
    detection: Detection
    ocr: OCRResult | None
    original_polygon: np.ndarray

    def render_item(self) -> RenderItem:
        return RenderItem(
            label=self.detection.label,
            score=self.detection.score,
            bbox_xyxy=self.detection.bbox_xyxy,
            text=self.ocr.text if self.ocr and self.ocr.text else None,
        )

    def as_dict(self) -> dict[str, object]:
        output = self.detection.as_dict()
        output["quad_original"] = np.round(self.original_polygon, 3).tolist()
        if self.ocr is not None:
            output["ocr"] = {
                "text": self.ocr.text,
                "confidence": round(self.ocr.confidence, 6) if self.ocr.confidence is not None else None,
            }
        return output


@dataclass
class ReceiptResult:
    source_path: str
    rectification: RectificationResult
    detections: list[ExtractedDetection]
    fields: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source_path,
            "geometry": self.rectification.manifest(),
            "fields": self.fields,
            "detections": [detection.as_dict() for detection in self.detections],
        }


def _crop_with_margin(image_rgb: np.ndarray, bbox_xyxy: tuple[float, float, float, float], margin_ratio: float = 0.08) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    height, width = image_rgb.shape[:2]
    margin_x = max(2.0, (x2 - x1) * margin_ratio)
    margin_y = max(2.0, (y2 - y1) * margin_ratio)
    left = max(0, int(np.floor(x1 - margin_x)))
    top = max(0, int(np.floor(y1 - margin_y)))
    right = min(width, int(np.ceil(x2 + margin_x)))
    bottom = min(height, int(np.ceil(y2 + margin_y)))
    return image_rgb[top:bottom, left:right]


def _field_from_ocr(detection: ExtractedDetection | None) -> dict[str, object]:
    if detection is None:
        return {"state": "absent", "raw": None}
    if detection.ocr is None or not detection.ocr.text:
        return {"state": "unreadable", "raw": None, "score": round(detection.detection.score, 6)}
    return {
        "state": "read",
        "raw": detection.ocr.text,
        "ocr_confidence": round(detection.ocr.confidence, 6) if detection.ocr.confidence is not None else None,
        "detector_score": round(detection.detection.score, 6),
    }


def _build_fields(detections: list[ExtractedDetection]) -> dict[str, Any]:
    by_label = {item.detection.label: item for item in detections}
    amount = _field_from_ocr(by_label.get("amount"))
    if isinstance(amount.get("raw"), str):
        normalized_amount = normalize_amount(amount["raw"])
        if normalized_amount:
            amount.update(normalized_amount)

    recipient = _field_from_ocr(by_label.get("recipient_value"))
    payment_method = _field_from_ocr(by_label.get("payment_method_value"))
    if isinstance(payment_method.get("raw"), str):
        payment_method.update(normalize_payment_method(payment_method["raw"]))

    success_text = _field_from_ocr(by_label.get("success_text"))
    status = normalize_status(success_text["raw"]) if isinstance(success_text.get("raw"), str) else "unknown"
    icon_detected = "success_icon" in by_label
    text_detected = "success_text" in by_label
    return {
        "amount": amount,
        "transfer_success": {
            "state": status,
            "success_icon_detected": icon_detected,
            "success_text_detected": text_detected,
            # The detector still supplies a useful confirmation if a blurred crop
            # makes OCR unreadable; the raw OCR state remains in success_text.
            "confirmed": bool(icon_detected and (status == "success" or text_detected)),
            "text": success_text,
        },
        "recipient": recipient,
        "payment_method": payment_method,
    }


class ReceiptPipeline:
    def __init__(
        self,
        predictor: LRCNNPredictor,
        *,
        ocr: TextRecognizer | None = None,
        rectification_options: RectificationOptions | None = None,
    ) -> None:
        self.predictor = predictor
        self.ocr = ocr
        self.rectification_options = rectification_options or RectificationOptions()

    def run(self, source_path: str | Path) -> ReceiptResult:
        source_path = Path(source_path)
        source_rgb = load_upright_rgb(source_path)
        rectification = rectify_receipt(source_rgb, self.rectification_options)
        raw_detections = self.predictor.predict(rectification.rectified_rgb)
        detections: list[ExtractedDetection] = []
        for detection in raw_detections:
            ocr_result: OCRResult | None = None
            if self.ocr is not None and detection.label != "success_icon":
                crop = _crop_with_margin(rectification.rectified_rgb, detection.bbox_xyxy)
                if crop.size:
                    ocr_result = self.ocr.recognize(crop)
            original_polygon = transform_points(
                bbox_to_polygon(detection.bbox_xyxy),
                rectification.rectified_to_original,
            )
            detections.append(ExtractedDetection(detection, ocr_result, original_polygon))
        return ReceiptResult(
            source_path=source_path.resolve().as_posix(),
            rectification=rectification,
            detections=detections,
            fields=_build_fields(detections),
        )


def write_receipt_result(result: ReceiptResult, output_stem: str | Path) -> dict[str, Path]:
    """Write JSON, rectified image, and perspective-correct original annotation."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    items = [item.render_item() for item in result.detections]
    rectified_path = output_stem.with_name(output_stem.name + "_rectified_annotated.jpg")
    original_path = output_stem.with_name(output_stem.name + "_original_annotated.jpg")
    json_path = output_stem.with_suffix(".json")
    save_rgb(rectified_path, draw_rectified_circles(result.rectification.rectified_rgb, items))
    save_rgb(
        original_path,
        draw_original_circles(result.rectification.source_rgb, items, result.rectification.rectified_to_original),
    )
    json_path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"json": json_path, "rectified_annotation": rectified_path, "original_annotation": original_path}
