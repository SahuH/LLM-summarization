import re
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset


def get_embeddings(text_list, ner_tokenizer, ner_model):
  encoded_input = ner_tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)

  encoded_input = {k: v for k, v in encoded_input.items()}

  with torch.no_grad():
      text_outputs = ner_model(**encoded_input)

  return text_outputs.last_hidden_state.mean(dim=1)  # Mean pooling to get single embedding for tex


def semantic_search(input_text, input_question):
    split_pattern = r"\. |\.\n"     #split basis ". " and ".\n"
    sentences = re.split(split_pattern, input_text)

    # Remove empty strings from the list (if any)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Create a dictionary with a key "text" containing the text sentences
    data_dict = {"text": sentences}

    # Create a dataset from the dictionary
    sentences_dataset = Dataset.from_dict(data_dict)

    ner_model_name = "SahuH/distilbert-ner"
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    ner_model = AutoModel.from_pretrained(ner_model_name)

    embeddings_dataset = sentences_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"], ner_tokenizer, ner_model).detach().cpu().numpy()[0]}
    )

    embeddings_dataset.add_faiss_index(column="embeddings")

    question_embedding = get_embeddings([input_question], ner_tokenizer, ner_model).cpu().detach().numpy()

    scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=15
    )

    return '.\n'.join(samples['text'])