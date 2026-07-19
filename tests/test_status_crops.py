import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from transfer_receipt_ai.status_crops import (
    _key_action,
    _selection_key,
    crop_status_region,
    export_status_crops,
    reconstruct_rectified,
    review_status_crops,
)


def _write_bundle(results: Path, source: Path, relative: str, *, x: int = 3) -> Path:
    result_path = results / relative
    result_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "source": source.resolve().as_posix(),
        "geometry": {
            "source_size": {"width": 20, "height": 12},
            "rectified_size": {"width": 20, "height": 12},
            "H_original_to_rectified": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "detections": [
            {
                "label": "transfer_status",
                "score": 0.9,
                "bbox_rectified": [x, 2, x + 8, 8],
            }
        ],
    }
    result_path.write_text(json.dumps(document), encoding="utf-8")
    return result_path


def _source(path: Path, value: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((12, 20, 3), value, dtype=np.uint8)
    pixels[:, :, 1] = np.arange(20, dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def test_reconstruct_and_crop_status_region() -> None:
    source = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    payload = {
        "geometry": {
            "source_size": {"width": 20, "height": 12},
            "rectified_size": {"width": 20, "height": 12},
            "H_original_to_rectified": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    }
    rectified = reconstruct_rectified(payload, source)
    crop = crop_status_region(rectified, [3, 2, 11, 8], margin_ratio=0)

    np.testing.assert_array_equal(rectified, source)
    np.testing.assert_array_equal(crop, source[2:8, 3:11])


def test_export_preserves_paths_and_writes_clean_manifest(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    result = _write_bundle(results, source, "nested/receipt.json")
    # Batch metadata is JSON too, but is not a result bundle.
    (results / "inference_manifest.json").write_text("[]", encoding="utf-8")
    output = tmp_path / "crops"

    records = export_status_crops(results_dir=results, output_dir=output, margin_ratio=0)

    assert records == [
        {
            "crop": (output / "nested/receipt.jpg").resolve().as_posix(),
            "source": source.resolve().as_posix(),
            "result_json": result.resolve().as_posix(),
            "group_id": "nested/receipt",
            "label": None,
        }
    ]
    with Image.open(output / "nested/receipt.jpg") as crop:
        assert crop.size == (8, 6)
    manifest_records = [
        json.loads(line)
        for line in (output / "status_crops_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_records == records
    assert (output / "status_crops_errors.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads((output / "status_crops_config.json").read_text(encoding="utf-8")) == {
        "schema": 1,
        "results": results.resolve().as_posix(),
        "margin": 0.0,
    }
    assert not (output / "status_crops_config.json.tmp").exists()


def test_export_limit_is_deterministic_and_skip_is_idempotent(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    for name in ("c.json", "a.json", "b.json"):
        _write_bundle(results, source, name)
    output = tmp_path / "crops"

    first = export_status_crops(results_dir=results, output_dir=output, limit=2)
    crop_mtimes = {record["group_id"]: Path(str(record["crop"])).stat().st_mtime_ns for record in first}
    second = export_status_crops(results_dir=results, output_dir=output, limit=2, skip_existing=True)

    assert [record["group_id"] for record in second] == [record["group_id"] for record in first]
    assert {
        record["group_id"]: Path(str(record["crop"])).stat().st_mtime_ns for record in second
    } == crop_mtimes


def test_expanding_limit_keeps_existing_cohort_prefix_when_new_result_hashes_earlier(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    ordered_names = sorted(
        (f"candidate_{index}.json" for index in range(20)),
        key=lambda name: _selection_key(Path(name)),
    )
    new_earlier = ordered_names[0]
    initial_names = ordered_names[-2:]
    for name in initial_names:
        _write_bundle(results, source, name)
    output = tmp_path / "crops"

    first = export_status_crops(results_dir=results, output_dir=output, limit=2)
    first_result_jsons = [record["result_json"] for record in first]
    _write_bundle(results, source, new_earlier)
    expanded = export_status_crops(results_dir=results, output_dir=output, limit=3)

    assert [record["result_json"] for record in expanded[:2]] == first_result_jsons
    assert Path(str(expanded[2]["result_json"])).name == new_earlier


def test_limit_cannot_shrink_an_existing_cohort(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    _write_bundle(results, source, "one.json")
    _write_bundle(results, source, "two.json")
    output = tmp_path / "crops"
    original = export_status_crops(results_dir=results, output_dir=output, limit=2)

    with pytest.raises(ValueError, match="smaller than the existing cohort"):
        export_status_crops(results_dir=results, output_dir=output, limit=1)

    persisted = [
        json.loads(line)
        for line in (output / "status_crops_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert persisted == original


def test_export_groups_timestamped_captures_of_same_receipt(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    first_source = _source(tmp_path / "raw" / "voucher_123_20260701000133.png", 80)
    second_source = _source(tmp_path / "raw" / "voucher_123_20260701000134.png", 120)
    _write_bundle(results, first_source, "first.json")
    _write_bundle(results, second_source, "second.json")

    records = export_status_crops(results_dir=results, output_dir=tmp_path / "crops")

    assert [record["group_id"] for record in records] == ["voucher_123", "voucher_123"]


def test_export_can_continue_after_bad_bundle(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    _write_bundle(results, source, "good.json")
    bad = _write_bundle(results, source, "bad.json")
    document = json.loads(bad.read_text(encoding="utf-8"))
    document["detections"] = []
    bad.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "crops"

    records = export_status_crops(
        results_dir=results,
        output_dir=output,
        continue_on_error=True,
    )

    assert [record["group_id"] for record in records] == ["good"]
    errors = [json.loads(line) for line in (output / "status_crops_errors.jsonl").read_text().splitlines()]
    assert errors[0]["result_json"] == bad.resolve().as_posix()
    assert "exactly one transfer_status" in errors[0]["message"]


@pytest.mark.parametrize("output_location", ("child", "parent"))
def test_export_rejects_results_output_overlap_in_both_directions(tmp_path, output_location) -> None:
    if output_location == "child":
        results = tmp_path / "results"
        results.mkdir()
        output = results / "crops"
    else:
        output = tmp_path / "export_root"
        results = output / "results"
        results.mkdir(parents=True)

    with pytest.raises(ValueError, match="must not overlap in either direction"):
        export_status_crops(results_dir=results, output_dir=output)


@pytest.mark.parametrize("output_location", ("child", "parent"))
def test_export_rejects_source_output_overlap_without_changing_source(tmp_path, output_location) -> None:
    results = tmp_path / "results"
    results.mkdir()
    if output_location == "child":
        source_directory = tmp_path / "raw"
        output = source_directory / "crops"
    else:
        output = tmp_path / "source_export_root"
        source_directory = output / "raw"
    source = _source(source_directory / "source.png")
    original = source.read_bytes()
    _write_bundle(results, source, "receipt.json")

    with pytest.raises(ValueError, match="source image directory must not overlap"):
        export_status_crops(
            results_dir=results,
            output_dir=output,
            skip_existing=False,
            continue_on_error=True,
        )

    assert source.read_bytes() == original


def test_export_preflights_all_source_trees_before_any_write(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    first_source = _source(tmp_path / "raw-a" / "first.png")
    protected_source = _source(tmp_path / "raw-b" / "second.png")
    _write_bundle(results, first_source, "first.json")
    _write_bundle(results, protected_source, "second.json")
    output = protected_source.parent / "crops"

    with pytest.raises(ValueError, match="source image directory must not overlap"):
        export_status_crops(
            results_dir=results,
            output_dir=output,
            continue_on_error=True,
        )

    assert not output.exists()


def test_export_refuses_custom_errors_file_inside_source_tree(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    _write_bundle(results, source, "receipt.json")

    with pytest.raises(ValueError, match="source image directory must not overlap"):
        export_status_crops(
            results_dir=results,
            output_dir=tmp_path / "crops",
            errors_path=source.parent / "errors.jsonl",
            continue_on_error=True,
        )

    assert not (source.parent / "errors.jsonl").exists()


def test_export_refuses_custom_manifest_that_would_touch_source_tree(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    original = source.read_bytes()
    _write_bundle(results, source, "receipt.json")

    with pytest.raises(ValueError, match="source image directory must not overlap"):
        export_status_crops(
            results_dir=results,
            output_dir=tmp_path / "crops",
            manifest_path=source,
            continue_on_error=True,
        )

    assert source.read_bytes() == original


def test_changed_config_requires_overwrite_then_updates_and_recrops(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    _write_bundle(results, source, "receipt.json")
    output = tmp_path / "crops"
    export_status_crops(results_dir=results, output_dir=output, margin_ratio=0)
    crop = output / "receipt.jpg"
    first_size = Image.open(crop).size

    with pytest.raises(ValueError, match="configuration changed.*--overwrite"):
        export_status_crops(results_dir=results, output_dir=output, margin_ratio=0.5)

    before_overwrite = crop.stat().st_mtime_ns
    os.utime(crop, ns=(before_overwrite - 1_000_000, before_overwrite - 1_000_000))
    export_status_crops(
        results_dir=results,
        output_dir=output,
        margin_ratio=0.5,
        skip_existing=False,
    )

    assert Image.open(crop).size != first_size
    assert crop.stat().st_mtime_ns > before_overwrite - 1_000_000
    assert json.loads((output / "status_crops_config.json").read_text(encoding="utf-8")) == {
        "schema": 1,
        "results": results.resolve().as_posix(),
        "margin": 0.5,
    }


def test_existing_crop_without_config_requires_overwrite(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    source = _source(tmp_path / "raw" / "source.png")
    _write_bundle(results, source, "receipt.json")
    output = tmp_path / "crops"
    output.mkdir()
    (output / "receipt.jpg").write_bytes(b"legacy")

    with pytest.raises(ValueError, match="have no status_crops_config.json.*--overwrite"):
        export_status_crops(results_dir=results, output_dir=output)

    export_status_crops(results_dir=results, output_dir=output, skip_existing=False)
    assert (output / "status_crops_config.json").is_file()


def test_integer_uppercase_q_quits_while_back_keys_still_go_back() -> None:
    assert _key_action(81) == "quit"
    assert _key_action(8) == "back"
    assert _key_action(2424832) == "back"
    assert _key_action("B") == "back"


def test_review_saves_atomically_and_resumes_from_first_unlabelled(tmp_path) -> None:
    crop_one = _source(tmp_path / "one.png", 80)
    crop_two = _source(tmp_path / "two.png", 120)
    manifest = tmp_path / "manifest.jsonl"
    base = {"source": "/raw/source.jpg", "label": None}
    manifest.write_text(
        "\n".join(
            (
                    json.dumps(
                        {
                            **base,
                            "crop": crop_one.as_posix(),
                            "result_json": "/results/one.json",
                            "group_id": "one",
                        }
                    ),
                    json.dumps(
                        {
                            **base,
                            "crop": crop_two.as_posix(),
                            "result_json": "/results/two.json",
                            "group_id": "two",
                        }
                    ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.jsonl"
    seen: list[str] = []
    first_keys = iter(("1", "q"))

    review_status_crops(
        manifest_path=manifest,
        labels_path=labels,
        key_provider=lambda record, _crop, _index, _total: (seen.append(str(record["group_id"])), next(first_keys))[1],
    )
    assert seen == ["one", "two"]
    first_review = [json.loads(line) for line in labels.read_text().splitlines()]
    assert [record["label"] for record in first_review] == ["check_offset", None]

    resumed_seen: list[str] = []
    review_status_crops(
        manifest_path=manifest,
        labels_path=labels,
        key_provider=lambda record, _crop, _index, _total: (
            resumed_seen.append(str(record["group_id"])),
            "2",
        )[1],
    )
    final_review = [json.loads(line) for line in labels.read_text().splitlines()]
    assert resumed_seen == ["two"]
    assert [record["label"] for record in final_review] == ["check_offset", "check_aligned"]
    assert not labels.with_suffix(".jsonl.tmp").exists()
