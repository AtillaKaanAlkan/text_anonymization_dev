import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class BertForTokenClassification(nn.Module):
    def __init__(self, model_name: str = "camembert-base", num_labels: int = 2):
        """
        Initialize the model for token classification.
        
        Args:
            model_name (str): Hugging Face model identifier (e.g., "bert-base-cased").
            num_labels (int): Number of classification labels.
        """
        super(BertForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        #self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            labels (torch.Tensor, optional): Labels for supervised learning.

        Returns:
            dict: Outputs including loss (if labels are provided) and logits.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        #sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        output = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten predictions and labels for loss computation
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output["loss"] = loss

        return output

def load_model(model_path: str, num_labels: int):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model directory.
        num_labels (int): Number of classification labels.

    Returns:
        BertForTokenClassification: Loaded model instance.
    """
    model = BertForTokenClassification(model_name=model_path, num_labels=num_labels)
    
    return model
