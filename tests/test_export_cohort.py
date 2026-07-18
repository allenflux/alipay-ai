import json

import pytest

from transfer_receipt_ai.export_cohort import export_inference_cohort


def test_export_cohort_combines_success_and_error_records(tmp_path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    first = source_root / "first.jpg"
    second = source_root / "nested" / "second.jpg"
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    success = tmp_path / "success.jsonl"
    errors = tmp_path / "errors.jsonl"
    success.write_text(json.dumps({"source": first.as_posix()}) + "\n", encoding="utf-8")
    errors.write_text(json.dumps({"source": second.as_posix()}) + "\n", encoding="utf-8")

    output = tmp_path / "active" / "raw"
    records = export_inference_cohort(
        record_paths=[success, errors],
        source_root=source_root,
        output_dir=output,
        expected_count=2,
    )

    assert len(records) == 2
    assert (output / "first.jpg").read_bytes() == b"first"
    assert (output / "nested" / "second.jpg").read_bytes() == b"second"
    manifest = output.parent / "raw_cohort_manifest.jsonl"
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 2

    rerun = export_inference_cohort(
        record_paths=[success, errors],
        source_root=source_root,
        output_dir=output,
        expected_count=2,
    )
    assert {record["status"] for record in rerun} == {"skipped_identical"}


def test_export_cohort_rejects_wrong_expected_count(tmp_path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    source = source_root / "one.jpg"
    source.write_bytes(b"one")
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"source": source.as_posix()}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 2 unique sources, found 1"):
        export_inference_cohort(
            record_paths=[records],
            source_root=source_root,
            output_dir=tmp_path / "output",
            expected_count=2,
        )


def test_export_cohort_rejects_source_outside_root(tmp_path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"source": outside.as_posix()}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside source root"):
        export_inference_cohort(
            record_paths=[records],
            source_root=source_root,
            output_dir=tmp_path / "output",
        )


def test_export_cohort_rejects_output_inside_source_root(tmp_path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    source = source_root / "one.jpg"
    source.write_bytes(b"one")
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"source": source.as_posix()}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory must be outside source root"):
        export_inference_cohort(
            record_paths=[records],
            source_root=source_root,
            output_dir=source_root / "active_learning" / "raw",
        )


def test_export_cohort_can_keep_a_cumulative_manifest(tmp_path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    first = source_root / "first.jpg"
    second = source_root / "nested" / "second.jpg"
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_errors = tmp_path / "first_errors.jsonl"
    second_errors = tmp_path / "second_errors.jsonl"
    first_errors.write_text(json.dumps({"source": first.as_posix()}) + "\n", encoding="utf-8")
    second_errors.write_text(json.dumps({"source": second.as_posix()}) + "\n", encoding="utf-8")
    output = tmp_path / "active" / "raw"

    export_inference_cohort(
        record_paths=[first_errors],
        source_root=source_root,
        output_dir=output,
        cumulative_manifest=True,
    )
    current = export_inference_cohort(
        record_paths=[second_errors],
        source_root=source_root,
        output_dir=output,
        cumulative_manifest=True,
    )

    assert len(current) == 1
    manifest = output.parent / "raw_cohort_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert {record["relative_path"] for record in records} == {"first.jpg", "nested/second.jpg"}
