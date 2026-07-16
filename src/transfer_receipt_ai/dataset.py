"""COCO dataset loader and photo-like detector augmentation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .labels import LABEL_TO_ID, validate_label


def _boxes_to_corners(boxes_xyxy: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            boxes_xyxy[:, [0, 1]],
            boxes_xyxy[:, [2, 1]],
            boxes_xyxy[:, [2, 3]],
            boxes_xyxy[:, [0, 3]],
        ],
        axis=1,
    ).astype(np.float32)


def _transform_boxes(boxes_xyxy: np.ndarray, homography: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform axis-aligned boxes via their four corners and clip to image."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, np.zeros((0,), dtype=bool)
    corners = _boxes_to_corners(boxes_xyxy).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, homography.astype(np.float32)).reshape(-1, 4, 2)
    x1 = np.clip(transformed[:, :, 0].min(axis=1), 0, width - 1)
    y1 = np.clip(transformed[:, :, 1].min(axis=1), 0, height - 1)
    x2 = np.clip(transformed[:, :, 0].max(axis=1), 0, width - 1)
    y2 = np.clip(transformed[:, :, 1].max(axis=1), 0, height - 1)
    result = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    keep = (x2 - x1 >= 2.0) & (y2 - y1 >= 2.0)
    return result[keep], keep


def augment_photo(image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply geometry/quality corruption typical of a screen photographed by phone.

    Chinese UI pages are never horizontally flipped: that would create invalid
    glyphs.  The raw preparation stage handles 90° orientation; this function
    adds small residual rotation, perspective, blur, illumination and JPEG noise.
    """
    image = image_rgb.copy()
    boxes = boxes_xyxy.copy().astype(np.float32)
    height, width = image.shape[:2]
    original_count = len(boxes)
    surviving_indices = np.arange(original_count, dtype=np.int64)

    if np.random.random() < 0.55:
        angle = float(np.random.uniform(-6.0, 6.0))
        affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        homography = np.vstack([affine, [0.0, 0.0, 1.0]])
        boxes, transformed_keep = _transform_boxes(boxes, homography, width, height)
        surviving_indices = surviving_indices[transformed_keep]

    if np.random.random() < 0.45:
        jitter_x, jitter_y = width * 0.035, height * 0.035
        source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        destination = source + np.array(
            [
                [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        image = cv2.warpPerspective(
            image,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        boxes, transformed_keep = _transform_boxes(boxes, homography, width, height)
        surviving_indices = surviving_indices[transformed_keep]

    if np.random.random() < 0.7:
        alpha = float(np.random.uniform(0.72, 1.28))
        beta = float(np.random.uniform(-24, 24))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.random() < 0.25:
        kernel_size = int(np.random.choice([3, 5]))
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if np.random.random() < 0.25:
        quality = int(np.random.randint(45, 90))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    keep = np.zeros((original_count,), dtype=bool)
    keep[surviving_indices] = True
    return image, boxes, keep


class ReceiptCocoDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Detection data with fixed labels independent of COCO category IDs."""

    def __init__(self, image_dir: str | Path, annotation_path: str | Path, training: bool = False) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        document = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{annotation_path} must be a COCO JSON object")
        categories = document.get("categories", [])
        self.category_to_model_label: dict[int, int] = {}
        for category in categories:
            if not isinstance(category, dict):
                continue
            name = category.get("name")
            if not isinstance(name, str):
                continue
            validate_label(name)
            self.category_to_model_label[int(category["id"])] = LABEL_TO_ID[name]
        if not self.category_to_model_label:
            raise ValueError("COCO annotations have no valid transfer-receipt categories")

        images = document.get("images", [])
        if not isinstance(images, list):
            raise ValueError("COCO images must be a list")
        self.images = sorted((image for image in images if isinstance(image, dict)), key=lambda image: int(image["id"]))
        if not self.images:
            raise ValueError("COCO annotations contain no images")
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in document.get("annotations", []):
            if isinstance(annotation, dict) and int(annotation.get("category_id", -1)) in self.category_to_model_label:
                self.annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.training = training

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.image_dir / str(image_info["file_name"])
        with Image.open(image_path) as pil_image:
            image_rgb = np.asarray(ImageOps.exif_transpose(pil_image).convert("RGB")).copy()
        annotations = self.annotations_by_image.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        iscrowd: list[int] = []
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                continue
            x, y, width, height = (float(value) for value in bbox)
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_to_model_label[int(annotation["category_id"])])
            iscrowd.append(int(annotation.get("iscrowd", 0)))
        boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels_array = np.asarray(labels, dtype=np.int64)
        iscrowd_array = np.asarray(iscrowd, dtype=np.int64)
        if self.training:
            image_rgb, boxes_array, keep = augment_photo(image_rgb, boxes_array)
            labels_array = labels_array[keep]
            iscrowd_array = iscrowd_array[keep]

        boxes_tensor = torch.as_tensor(boxes_array, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes_tensor,
            "labels": torch.as_tensor(labels_array, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
            "iscrowd": torch.as_tensor(iscrowd_array, dtype=torch.int64),
        }
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float().div(255.0)
        return image_tensor, target


def collate_detection_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
