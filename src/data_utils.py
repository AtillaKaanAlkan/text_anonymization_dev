import torch
from typing import List, Dict
from transformers import AutoTokenizer
import json

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[List[str]], labels: List[List[int]], tokenizer_name: str, label_to_id: Dict[str, int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx]
        word_labels = self.labels[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Align labels with tokenizer's word-to-token mapping
        labels_aligned = []
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels_aligned.append(-100)  # Ignore token
            elif word_idx != previous_word_idx:
                labels_aligned.append(word_labels[word_idx])
            else:
                labels_aligned.append(-100)  # Ignore sub-token

            previous_word_idx = word_idx

        # Flatten tensor dimensions
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["ner_tags"] = torch.tensor(labels_aligned)

        return item

def load_example_dataset(tokenizer_name: str) -> NERDataset:
    """
    Loads a small in-memory dataset for testing.

    Returns:
        NERDataset: PyTorch-compatible dataset.
    """
    # Example data: words and BIO labels
    example_texts = [["John", "lives", "in", "Paris", "."]]
    example_labels = [[1, 0, 0, 2, 0]]  # Example label ids (1=PER, 2=LOC, 0=O)

    label_to_id = {"O": 0, "B-PER": 1, "B-LOC": 2}
    return NERDataset(example_texts, example_labels, tokenizer_name, label_to_id)


def load_from_jsonl(file_path: str, tokenizer_name: str, label_to_id: Dict[str, int], max_length: int = 128) -> NERDataset:
    """
    Loads tokenized sentences and their labels from a JSONL file.

    Args:
        file_path (str): Path to the .jsonl file.
        tokenizer_name (str): Hugging Face model name.
        label_to_id (dict): Mapping of label names to IDs.
        max_length (int): Maximum token length.

    Returns:
        NERDataset: A PyTorch-compatible dataset.
    """
    texts: List[List[str]] = []
    labels: List[List[int]] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            tokens = record["tokens"]
            label_names = record["ner_tags"]
            label_ids = [label_to_id[label] for label in label_names]

            texts.append(tokens)
            labels.append(label_ids)

    return NERDataset(texts, labels, tokenizer_name, label_to_id, max_length)

