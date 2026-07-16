import json

from PIL import Image

from transfer_receipt_ai.labelme_to_coco import convert_labelme_to_coco


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
