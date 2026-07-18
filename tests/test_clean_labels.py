import json

from transfer_receipt_ai.clean_labels import clean_incomplete_labels
from transfer_receipt_ai.labels import DETECTION_CLASSES


def _document(labels: list[str]) -> dict:
    return {
        "imagePath": "sample.jpg",
        "shapes": [
            {"label": label, "points": [[0, 0], [10, 10]], "shape_type": "rectangle"}
            for label in labels
        ],
    }


def test_quarantine_moves_only_incomplete_json(tmp_path) -> None:
    labels_dir = tmp_path / "images"
    labels_dir.mkdir()
    complete = labels_dir / "complete.json"
    incomplete = labels_dir / "incomplete.json"
    image = labels_dir / "incomplete.jpg"
    complete.write_text(json.dumps(_document(list(DETECTION_CLASSES))), encoding="utf-8")
    incomplete.write_text(json.dumps(_document(list(DETECTION_CLASSES[:-1]))), encoding="utf-8")
    image.write_bytes(b"image")
    rejected = tmp_path / "rejected"

    records = clean_incomplete_labels(
        labels_dir=labels_dir,
        action="quarantine",
        rejected_dir=rejected,
    )

    assert len(records) == 1
    assert complete.is_file()
    assert not incomplete.exists()
    assert (rejected / "incomplete.json").is_file()
    assert image.read_bytes() == b"image"


def test_dry_run_does_not_move_incomplete_json(tmp_path) -> None:
    labels_dir = tmp_path / "images"
    labels_dir.mkdir()
    incomplete = labels_dir / "incomplete.json"
    incomplete.write_text(json.dumps(_document(["time"])), encoding="utf-8")

    records = clean_incomplete_labels(labels_dir=labels_dir, action="dry-run")

    assert records[0]["status"] == "would_quarantine"
    assert incomplete.is_file()


def test_delete_removes_json_but_not_matching_image(tmp_path) -> None:
    labels_dir = tmp_path / "images"
    labels_dir.mkdir()
    incomplete = labels_dir / "incomplete.json"
    image = labels_dir / "incomplete.jpg"
    incomplete.write_text(json.dumps(_document(["time"])), encoding="utf-8")
    image.write_bytes(b"image")

    records = clean_incomplete_labels(labels_dir=labels_dir, action="delete")

    assert records[0]["status"] == "deleted"
    assert not incomplete.exists()
    assert image.is_file()


def test_non_labelme_json_is_ignored(tmp_path) -> None:
    labels_dir = tmp_path / "images"
    labels_dir.mkdir()
    config = labels_dir / "config.json"
    config.write_text('{"setting": true}', encoding="utf-8")

    assert clean_incomplete_labels(labels_dir=labels_dir, action="dry-run") == []
    assert config.is_file()
