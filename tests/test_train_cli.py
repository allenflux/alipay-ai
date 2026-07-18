from pathlib import Path

import pytest

from transfer_receipt_ai.train import train_detector


def test_resume_and_init_checkpoint_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_detector(
            train_images=tmp_path,
            train_annotations=tmp_path / "train.json",
            output_dir=tmp_path / "output",
            resume=tmp_path / "last.pt",
            init_checkpoint=tmp_path / "best.pt",
        )
