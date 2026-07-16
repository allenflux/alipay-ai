"""Small dependency-free detector evaluation used during training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from .labels import DETECTION_CLASSES, ID_TO_LABEL


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and many boxes in ``[x1,y1,x2,y2]`` form."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection_size = np.maximum(0.0, right_bottom - left_top)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0
    recalled = np.concatenate(([0.0], recalls, [1.0]))
    precision = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    change_points = np.where(recalled[1:] != recalled[:-1])[0]
    return float(np.sum((recalled[change_points + 1] - recalled[change_points]) * precision[change_points + 1]))


def evaluate_map50(predictions: Sequence[dict[str, Any]], targets: Sequence[dict[str, Any]], score_threshold: float = 0.05) -> dict[str, object]:
    """Compute per-class AP@0.50 and mAP@0.50 without pycocotools.

    It is intentionally transparent for this five-field task.  COCO mAP across
    IoU thresholds can be added later, but AP50 and field-level recall are the
    most useful early signal for receipt extraction.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    ground_truth: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    candidates: dict[str, list[tuple[float, int, np.ndarray]]] = defaultdict(list)
    for prediction, target in zip(predictions, targets):
        image_id_raw = target["image_id"]
        image_id = int(image_id_raw.item() if hasattr(image_id_raw, "item") else image_id_raw[0])
        target_boxes = target["boxes"].detach().cpu().numpy() if hasattr(target["boxes"], "detach") else np.asarray(target["boxes"])
        target_labels = target["labels"].detach().cpu().numpy() if hasattr(target["labels"], "detach") else np.asarray(target["labels"])
        for box, class_id in zip(target_boxes, target_labels):
            label = ID_TO_LABEL.get(int(class_id))
            if label:
                ground_truth[image_id][label].append(np.asarray(box, dtype=np.float32))

        prediction_boxes = prediction["boxes"].detach().cpu().numpy() if hasattr(prediction["boxes"], "detach") else np.asarray(prediction["boxes"])
        prediction_labels = prediction["labels"].detach().cpu().numpy() if hasattr(prediction["labels"], "detach") else np.asarray(prediction["labels"])
        prediction_scores = prediction["scores"].detach().cpu().numpy() if hasattr(prediction["scores"], "detach") else np.asarray(prediction["scores"])
        for box, class_id, score in zip(prediction_boxes, prediction_labels, prediction_scores):
            label = ID_TO_LABEL.get(int(class_id))
            if label and float(score) >= score_threshold:
                candidates[label].append((float(score), image_id, np.asarray(box, dtype=np.float32)))

    ap_by_class: dict[str, float | None] = {}
    recall_by_class: dict[str, float | None] = {}
    for label in DETECTION_CLASSES:
        total_gt = sum(len(per_image.get(label, [])) for per_image in ground_truth.values())
        if total_gt == 0:
            ap_by_class[label] = None
            recall_by_class[label] = None
            continue
        matched: dict[int, set[int]] = defaultdict(set)
        true_positive: list[float] = []
        false_positive: list[float] = []
        for _, image_id, prediction_box in sorted(candidates[label], key=lambda item: item[0], reverse=True):
            gt_boxes = ground_truth[image_id].get(label, [])
            if not gt_boxes:
                true_positive.append(0.0)
                false_positive.append(1.0)
                continue
            ious = box_iou_xyxy(prediction_box, np.asarray(gt_boxes, dtype=np.float32))
            best_index = int(ious.argmax())
            if float(ious[best_index]) >= 0.5 and best_index not in matched[image_id]:
                matched[image_id].add(best_index)
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)
        true_positive_array = np.cumsum(np.asarray(true_positive, dtype=np.float32))
        false_positive_array = np.cumsum(np.asarray(false_positive, dtype=np.float32))
        recalls = true_positive_array / total_gt
        precisions = np.divide(
            true_positive_array,
            true_positive_array + false_positive_array,
            out=np.zeros_like(true_positive_array),
            where=(true_positive_array + false_positive_array) > 0,
        )
        ap_by_class[label] = _average_precision(recalls, precisions)
        recall_by_class[label] = float(recalls[-1]) if len(recalls) else 0.0

    valid_aps = [value for value in ap_by_class.values() if value is not None]
    return {
        "map50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "ap50": ap_by_class,
        "recall50": recall_by_class,
    }
