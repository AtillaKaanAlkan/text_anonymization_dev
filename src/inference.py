import torch
from transformers import AutoTokenizer
from src.model import BertForTokenClassification


class Anonymizer:
    """
    Provides entity prediction and anonymization using a pre-trained BERT-based model.
    """

    def __init__(self, model_dir: str, label_to_id: dict):
        """
        Initialize the Anonymizer with a pre-trained model and tokenizer.

        Args:
            model_dir (str): Path to the saved model and tokenizer directory.
            label_to_id (dict): Mapping of entity labels to their IDs.
        """
        self.model_dir = model_dir
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = BertForTokenClassification(model_name=model_dir, num_labels=len(label_to_id))
        self.model.load_state_dict(torch.load(f"{model_dir}/model_state_dict.pt"))
        self.model.to(self.device)
        self.model.eval()

    def predict_entities(self, text: str) -> list:
        """
        Predicts token-level entity labels for the given input text.

        Args:
            text (str): Raw input sentence.

        Returns:
            List[Tuple[str, str]]: List of (word, predicted_label) pairs.
        """
        words = text.strip().split()
        tokens = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()

        predicted_entities = [
            (word, self.id_to_label.get(label_id, "O"))
            for word, label_id in zip(words, predictions[:len(words)])
        ]

        return predicted_entities

    def anonymize(self, text: str) -> str:
        """
        Anonymizes detected entities in the input text.

        Args:
            text (str): Raw input sentence.

        Returns:
            str: Anonymized sentence.
        """
        # Define replacement strategy
        entity_replacement_map = {
            "B-Person": "[NAME]",
            "I-Person": "[NAME]",
            "B-Location": "[LOCATION]",
            "I-Location": "[LOCATION]",
            "B-Organization": "[ORGANIZATION]",
            "I-Organization": "[ORGANIZATION]"
            # Add more mappings as needed
        }

        entities = self.predict_entities(text)
        anonymized_tokens = [
            entity_replacement_map.get(label, word) for word, label in entities
        ]

        return " ".join(anonymized_tokens)
