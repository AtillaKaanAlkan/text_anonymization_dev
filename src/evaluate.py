import os
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.model import BertForTokenClassification
from src.data_utils import load_from_jsonl


def collapse_bio_tags(tag: str) -> str:
    """Convert BIO tag to entity type, keeping 'O' unchanged."""
    if tag == "O":
        return "O"
    return tag.split("-")[-1]  # Keeps only "PER" from "B-PER", etc.


def evaluate_model(
    eval_file: str,
    model_dir: str,
    label_to_id: dict,
    batch_size: int = 8,
    max_length: int = 128
):
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BertForTokenClassification(model_name=model_dir, num_labels=len(label_to_id))
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_state_dict.pt")))
    model.to(device)
    model.eval()

    # Load Evaluation Dataset
    dataset = load_from_jsonl(eval_file, model_dir, label_to_id, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)

            for true, pred, mask in zip(batch["ner_tags"], predictions, batch["attention_mask"]):
                true = true[mask.bool()].cpu().tolist()
                pred = pred[mask.bool()].cpu().tolist()
                # Filter out -100 labels (ignored tokens)
                filtered = [(t, p) for t, p in zip(true, pred) if t != -100]
                if filtered:
                    t_labels, p_labels = zip(*filtered)
                    true_labels.extend(t_labels)
                    pred_labels.extend(p_labels)

    # Convert label IDs to names
    id_to_label = {v: k for k, v in label_to_id.items()}
    true_label_names = [collapse_bio_tags(id_to_label[i]) for i in true_labels]
    pred_label_names = [collapse_bio_tags(id_to_label[i]) for i in pred_labels]

    # Generate and print classification report
    report = classification_report(true_label_names, pred_label_names, digits=4)
    print("\nEvaluation Report (Entity-Level):\n")
    print(report)

    return report
