import json

import pytest
from PIL import Image

from transfer_receipt_ai.labelme_to_coco import convert_labelme_to_coco
from transfer_receipt_ai.labels import DETECTION_CLASSES, LABEL_TO_ID


def test_fixed_five_field_label_contract() -> None:
    assert DETECTION_CLASSES == (
        "time",
        "amount",
        "transfer_status",
        "recipient_field",
        "payment_method_field",
    )
    assert LABEL_TO_ID == {
        "time": 1,
        "amount": 2,
        "transfer_status": 3,
        "recipient_field": 4,
        "payment_method_field": 5,
    }


def test_labelme_conversion_keeps_text_truth(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 60), "white").save(images / "sample.jpg")
    (labels / "sample.json").write_text(
        json.dumps(
            {
                "imagePath": "sample.jpg",
                "imageWidth": 100,
                "imageHeight": 60,
                "shapes": [
                    {
                        "label": "amount",
                        "points": [[10, 20], [70, 45]],
                        "description": "¥199.93",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "annotations.json"
    coco = convert_labelme_to_coco(labels, images, output)
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 60.0, 25.0]
    assert coco["annotations"][0]["text"] == "¥199.93"
    assert output.is_file()


def test_complete_validation_requires_each_label_exactly_once(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 100), "white").save(images / "sample.jpg")
    shapes = [
        {
            "label": label,
            "points": [[5, 5 + index * 10], [90, 12 + index * 10]],
            "shape_type": "rectangle",
        }
        for index, label in enumerate(DETECTION_CLASSES)
    ]
    label_path = labels / "sample.json"
    label_path.write_text(
        json.dumps({"imagePath": "sample.jpg", "imageWidth": 100, "imageHeight": 100, "shapes": shapes}),
        encoding="utf-8",
    )
    result = convert_labelme_to_coco(
        labels,
        images,
        tmp_path / "complete.json",
        require_complete=True,
    )
    assert len(result["annotations"]) == 5

    shapes.pop()
    label_path.write_text(
        json.dumps({"imagePath": "sample.jpg", "imageWidth": 100, "imageHeight": 100, "shapes": shapes}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"(?s)failed for 1 file\(s\).*missing=payment_method_field"):
        convert_labelme_to_coco(
            labels,
            images,
            tmp_path / "incomplete.json",
            require_complete=True,
        )

    filtered = convert_labelme_to_coco(
        labels,
        images,
        tmp_path / "complete-only.json",
        complete_only=True,
    )
    assert filtered["images"] == []
    assert filtered["annotations"] == []
    assert filtered["info"]["skipped_incomplete"] == ["sample.json: missing=payment_method_field"]
