import pytest

from transfer_receipt_ai.labels import DETECTION_CLASSES, LABEL_TO_ID
from transfer_receipt_ai.merge_coco import merge_coco_documents


def _complete_document(file_name: str) -> dict:
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 100, "height": 200}],
        "annotations": [
            {
                "id": index,
                "image_id": 1,
                "category_id": LABEL_TO_ID[label],
                "bbox": [1, index * 10, 80, 8],
                "area": 640,
                "iscrowd": 0,
            }
            for index, label in enumerate(DETECTION_CLASSES, start=1)
        ],
        "categories": [
            {"id": LABEL_TO_ID[label], "name": label}
            for label in DETECTION_CLASSES
        ],
    }


def test_merge_reassigns_ids_and_keeps_all_five_classes() -> None:
    merged = merge_coco_documents(
        [_complete_document("manual.jpg"), _complete_document("auto.jpg")],
        source_names=["manual", "auto"],
    )
    assert [image["id"] for image in merged["images"]] == [1, 2]
    assert len(merged["annotations"]) == 10
    assert {annotation["image_id"] for annotation in merged["annotations"]} == {1, 2}
    assert {annotation["category_id"] for annotation in merged["annotations"]} == set(LABEL_TO_ID.values())


def test_merge_rejects_duplicate_file_names_case_insensitively() -> None:
    with pytest.raises(ValueError, match="Duplicate image"):
        merge_coco_documents([_complete_document("same.jpg"), _complete_document("SAME.JPG")])


def test_merge_rejects_incomplete_images() -> None:
    incomplete = _complete_document("broken.jpg")
    incomplete["annotations"].pop()
    with pytest.raises(ValueError, match="missing=payment_method_field"):
        merge_coco_documents([incomplete])


def test_merge_rejects_out_of_bounds_boxes() -> None:
    broken = _complete_document("broken.jpg")
    broken["annotations"][0]["bbox"] = [90, 10, 20, 10]
    with pytest.raises(ValueError, match="outside the image"):
        merge_coco_documents([broken])
