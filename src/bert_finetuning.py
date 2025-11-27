"""BERT fine-tuning utilities using Hugging Face Transformers."""
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def load_tokenizer(model_name: str = 'bert-base-uncased'):
    return AutoTokenizer.from_pretrained(model_name)


def load_tf_model(model_name: str = 'bert-base-uncased', num_labels: int = 2):
    return TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
