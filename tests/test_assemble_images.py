import pytest

from transfer_receipt_ai.assemble_images import assemble_image_roots


def test_assemble_image_roots_copies_without_changing_inputs(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "a.jpg"
    second = second_root / "nested" / "b.png"
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output = tmp_path / "combined"

    copied, skipped = assemble_image_roots([first_root, second_root], output)

    assert (copied, skipped) == (2, 0)
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert (output / "a.jpg").read_bytes() == b"first"
    assert (output / "nested" / "b.png").read_bytes() == b"second"

    assert assemble_image_roots([first_root, second_root], output) == (0, 2)


def test_assemble_image_roots_rejects_relative_path_collision(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "same.jpg").write_bytes(b"first")
    (second_root / "same.jpg").write_bytes(b"second")

    with pytest.raises(ValueError, match="path collision"):
        assemble_image_roots([first_root, second_root], tmp_path / "combined")
