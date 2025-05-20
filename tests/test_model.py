import torch
from src.model import BertForTokenClassification, AutoTokenizer
import pytest
import json



with open("tests/models_to_test.json", "r", encoding="utf-8") as f:
    models_to_test = json.load(f)



def test_dummy():
    assert 1 == 1

@pytest.mark.parametrize("model_name", models_to_test['models_to_test'])
def test_model_forward_pass(model_name):
    # Config
    #model_name = "camembert-base"
    num_labels = 3
    batch_size = 2
    seq_length = 5

    # Instantiate model
    model = BertForTokenClassification(model_name=model_name, num_labels=num_labels)
    
    # Dummy inputs (batch_size x seq_length)
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Assertions
    assert "logits" in output
    logits = output["logits"]
    assert logits.shape == (batch_size, seq_length, num_labels)





@pytest.mark.parametrize("model_name", models_to_test['models_to_test'])
def test_model_can_predict_labels(model_name):
    num_labels = 5  # Example number of labels for token classification
    model = BertForTokenClassification(model_name=model_name, num_labels=num_labels)
    
    # Load corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample_text = "Albert Einstein est né en Allemagne."

    # Tokenize input
    tokens = tokenizer(sample_text, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Run forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]

    # Check output shape
    assert logits.shape[0] == 1  # Batch size
    assert logits.shape[1] == input_ids.shape[1]  # Sequence length
    assert logits.shape[2] == num_labels  # Number of labels

    # Convert logits to predictions
    predictions = torch.argmax(logits, dim=-1)

    # Ensure predictions are not empty
    assert predictions.numel() > 0

