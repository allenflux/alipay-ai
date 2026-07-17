import json

from PIL import Image

from transfer_receipt_ai.auto_label import generate_auto_labels
from transfer_receipt_ai.labelme_to_coco import convert_labelme_to_coco
from transfer_receipt_ai.labels import DETECTION_CLASSES
from transfer_receipt_ai.model import Detection


class _FakePredictor:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def predict(self, _image):
        self.calls += 1
        return self.detections


def _five_detections():
    return [
        Detection(label, 0.95 - index * 0.01, (-2, 5 + index * 10, 95, 13 + index * 10))
        for index, label in enumerate(DETECTION_CLASSES)
    ]


def test_auto_label_writes_fixed_order_labelme_rectangles(tmp_path) -> None:
    images = tmp_path / "images"
    output = tmp_path / "labels"
    images.mkdir()
    Image.new("RGB", (80, 70), "white").save(images / "sample.jpg")
    predictor = _FakePredictor(_five_detections())

    records = generate_auto_labels(
        predictor=predictor,
        images_path=images,
        output_dir=output,
        review_threshold=0.80,
    )

    document = json.loads((output / "sample.json").read_text(encoding="utf-8"))
    assert [shape["label"] for shape in document["shapes"]] == list(DETECTION_CLASSES)
    assert all(shape["shape_type"] == "rectangle" for shape in document["shapes"])
    assert document["shapes"][0]["points"][0][0] == 0.0
    assert document["shapes"][0]["points"][1][0] == 80.0
    assert document["_auto_label"]["reviewed"] is False
    assert records[0]["status"] == "written_complete"
    coco = convert_labelme_to_coco(output, images, tmp_path / "annotations.json", require_complete=True)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 5


def test_auto_label_never_overwrites_manual_json_by_default(tmp_path) -> None:
    images = tmp_path / "images"
    output = tmp_path / "labels"
    images.mkdir()
    output.mkdir()
    Image.new("RGB", (80, 70), "white").save(images / "sample.jpg")
    manual_path = output / "sample.json"
    manual_path.write_text('{"manual": true}\n', encoding="utf-8")
    predictor = _FakePredictor(_five_detections())

    records = generate_auto_labels(predictor=predictor, images_path=images, output_dir=output)

    assert predictor.calls == 0
    assert manual_path.read_text(encoding="utf-8") == '{"manual": true}\n'
    assert records[0]["status"] == "skipped_existing"


def test_auto_label_records_missing_fields_and_can_require_complete(tmp_path) -> None:
    images = tmp_path / "images"
    output = tmp_path / "labels"
    images.mkdir()
    Image.new("RGB", (80, 70), "white").save(images / "sample.jpg")
    predictor = _FakePredictor(_five_detections()[:-1])

    records = generate_auto_labels(
        predictor=predictor,
        images_path=images,
        output_dir=output,
        require_complete=True,
    )

    assert not (output / "sample.json").exists()
    assert records[0]["status"] == "skipped_incomplete"
    assert records[0]["missing"] == ["payment_method_field"]


def test_auto_label_mirrors_nested_directories_and_skips_external_manual_labels(tmp_path) -> None:
    images = tmp_path / "images"
    output = tmp_path / "auto"
    manual = tmp_path / "manual"
    (images / "batch").mkdir(parents=True)
    (manual / "batch").mkdir(parents=True)
    Image.new("RGB", (80, 70), "white").save(images / "batch" / "sample.jpg")
    (manual / "batch" / "sample.json").write_text('{"manual": true}\n', encoding="utf-8")
    predictor = _FakePredictor(_five_detections())

    records = generate_auto_labels(
        predictor=predictor,
        images_path=images,
        output_dir=output,
        existing_labels_dirs=[manual],
    )

    assert predictor.calls == 0
    assert records[0]["status"] == "skipped_existing"
    assert not (output / "batch" / "sample.json").exists()
