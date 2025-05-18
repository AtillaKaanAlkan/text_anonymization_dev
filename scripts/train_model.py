import argparse
import json
import sys
import os

sys.path.append(os.path.abspath("."))  # Ensure the package path is correct

from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description = "Train a BERT-based Token Classification Model")

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the training data file in JSONL format."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="adsabs/astroBERT",
        help="Hugging Face model name (default: astroBERT)."
    )
    parser.add_argument(
        "--label_map",
        type=str,
        required=True,
        help="Path to a JSON file mapping labels to IDs (e.g., {'O': 0, 'B-PER': 1, 'B-LOC': 2})."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_output",
        help="Directory to save the trained model and tokenizer."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length."
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Optional path to the validation data file in JSONL format."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load label_to_id mapping from JSON
    with open(args.label_map, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)

    train_model(
        train_file=args.train_file,
        model_name=args.model_name,
        label_to_id=label_to_id,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        eval_file=args.eval_file  # Pass evaluation file if provided
    )
