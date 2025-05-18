import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from src.model import BertForTokenClassification
from src.data_utils import load_from_jsonl
from src.evaluate import evaluate_model


def train_model(
    train_file: str,
    model_name: str,
    label_to_id: dict,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    eval_file: str = None,  # Optional validation file
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training dataset
    dataset = load_from_jsonl(train_file, model_name, label_to_id, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model and tokenizer
    model = BertForTokenClassification(model_name=model_name, num_labels=len(label_to_id))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    best_f1 = 0.0  # Track best F1-score

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["ner_tags"]
            )
            loss = outputs["loss"]
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Average Loss: {avg_epoch_loss:.4f}")

        # Optional Evaluation after each epoch
        if eval_file:
            print("\nRunning evaluation on validation set...")
            report_text = evaluate_model(
                eval_file=eval_file,
                model_dir=output_dir,
                label_to_id=label_to_id,
                batch_size=batch_size,
                max_length=max_length
            )

            # Extract F1-score from the evaluation report
            f1_line = [line for line in report_text.strip().split("\n") if "weighted avg" in line]
            if f1_line:
                try:
                    f1_score = float(f1_line[0].split()[-2])
                    print(f"F1-score: {f1_score}")

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        print("New best model found. Saving model...")
                        save_model(model, tokenizer, output_dir)
                except Exception as e:
                    print(f"Warning: Failed to parse F1-score. {e}")

    # Save final model state regardless of evaluation results
    print("Saving final model state...")
    save_model(model, tokenizer, output_dir)
    print(f"\nTraining completed. Best F1-score: {best_f1:.4f}")


def save_model(model, tokenizer, output_dir):
    """
    Save model and tokenizer to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the BERT backbone
    model.bert.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save classification head weights separately
    torch.save(model.state_dict(), os.path.join(output_dir, "model_state_dict.pt"))
