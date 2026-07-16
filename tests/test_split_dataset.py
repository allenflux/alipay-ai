from transfer_receipt_ai.split_dataset import split_coco


def test_variants_never_cross_split() -> None:
    images = [{"id": index, "file_name": f"r{index}.jpg", "width": 10, "height": 10} for index in range(1, 9)]
    document = {
        "images": images,
        "annotations": [],
        "categories": [{"id": 1, "name": "amount"}],
    }
    groups = {f"r{index}.jpg": f"transaction_{(index - 1) // 2}" for index in range(1, 9)}
    splits = split_coco(document, groups=groups, seed=7)
    memberships = {}
    for split_name, split in splits.items():
        for image in split["images"]:
            group = groups[image["file_name"]]
            assert group not in memberships or memberships[group] == split_name
            memberships[group] = split_name
    assert len(memberships) == 4
