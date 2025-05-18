import argparse
import json
import sys
import os

# Ensure src is discoverable as a module
sys.path.append(os.path.abspath("."))

from src.inference import Anonymizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run anonymization on a text file")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory (e.g., models/ner_model)"
    )
    parser.add_argument(
        "--label_map",
        type=str,
        required=True,
        help="Path to the JSON file mapping labels to IDs (e.g., data/label_to_id.json)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file with one text per line"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path where the anonymized output will be saved"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load label-to-id mapping
    with open(args.label_map, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)

    # Initialize the anonymizer
    anonymizer = Anonymizer(model_dir=args.model_dir, label_to_id=label_to_id)

    # Process input file line by line
    with open(args.input_file, "r", encoding="utf-8") as infile, \
         open(args.output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            anonymized_line = anonymizer.anonymize(line)
            outfile.write(anonymized_line + "\n")

    print(f"Anonymization completed. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
