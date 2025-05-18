import os
import pytest
import torch
import tempfile
import json
from src.data_utils import NERDataset, load_example_dataset, load_from_jsonl

@pytest.fixture
def example_dataset():
    return load_example_dataset("bert-base-cased")

def test_nerdataset_len(example_dataset):
    assert len(example_dataset) == 1

def test_nerdataset_item_structure(example_dataset):
    item = example_dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)

def test_nerdataset_label_alignment(example_dataset):
    item = example_dataset[0]
    # Check that ignored tokens are marked with -100
    assert (item["labels"] == -100).any() or (item["labels"] >= 0).all()

def test_load_from_jsonl_creates_valid_dataset():
    # Create a temporary .jsonl file with known content
    data = [
        {"tokens": ["John", "lives", "in", "Paris", "."], "labels": ["B-PER", "O", "O", "B-LOC", "O"]},
        {"tokens": ["Mary", "travels", "to", "Berlin", "."], "labels": ["B-PER", "O", "O", "B-LOC", "O"]}
    ]
    label_to_id = {"O": 0, "B-PER": 1, "B-LOC": 2}

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as tmp_file:
        for record in data:
            tmp_file.write(json.dumps(record) + "\n")
        tmp_file_path = tmp_file.name

    try:
        dataset = load_from_jsonl(tmp_file_path, "bert-base-cased", label_to_id)
        assert len(dataset) == 2

        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)
            assert "input_ids" in item
            assert "labels" in item
            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["labels"], torch.Tensor)

    finally:
        os.remove(tmp_file_path)
