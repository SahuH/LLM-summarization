import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
from datasets import Dataset, ClassLabel, Value, DatasetDict

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)

    labels = []
    for i, label in enumerate(examples[f"numeric_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


task = "ner" 
model_checkpoint = "distilbert-base-cased"
batch_size = 8

datasets = load_dataset("ktgiahieu/maccrobat2018_2020")
datasets = datasets['train'].train_test_split(test_size=0.1)

# Define the ClassLabel feature with the label names
label_list = ['B-Activity', 'B-Administration', 'B-Age', 'B-Area', 'B-Biological_attribute', 'B-Biological_structure', 'B-Clinical_event', 'B-Color', 'B-Coreference', 'B-Date', 'B-Detailed_description', 'B-Diagnostic_procedure', 'B-Disease_disorder', 'B-Distance', 'B-Dosage', 'B-Duration', 'B-Family_history', 'B-Frequency', 'B-Height', 'B-History', 'B-Lab_value', 'B-Mass', 'B-Medication', 'B-Nonbiological_location', 'B-Occupation', 'B-Other_entity', 'B-Other_event', 'B-Outcome', 'B-Personal_background', 'B-Qualitative_concept', 'B-Quantitative_concept', 'B-Severity', 'B-Sex', 'B-Shape', 'B-Sign_symptom', 'B-Subject', 'B-Texture', 'B-Therapeutic_procedure', 'B-Time', 'B-Volume', 'B-Weight', 'I-Activity', 'I-Administration', 'I-Age', 'I-Area', 'I-Biological_attribute', 'I-Biological_structure', 'I-Clinical_event', 'I-Color', 'I-Coreference', 'I-Date', 'I-Detailed_description', 'I-Diagnostic_procedure', 'I-Disease_disorder', 'I-Distance', 'I-Dosage', 'I-Duration', 'I-Family_history', 'I-Frequency', 'I-Height', 'I-History', 'I-Lab_value', 'I-Mass', 'I-Medication', 'I-Nonbiological_location', 'I-Occupation', 'I-Other_entity', 'I-Other_event', 'I-Outcome', 'I-Personal_background', 'I-Qualitative_concept', 'I-Quantitative_concept', 'I-Severity', 'I-Shape', 'I-Sign_symptom', 'I-Subject', 'I-Texture', 'I-Therapeutic_procedure', 'I-Time', 'I-Volume', 'I-Weight', 'O']
label = ClassLabel(names=label_list)
# Define a new feature with the numeric labels
numeric_labels_feature = Value("int32")

datasets = datasets.map(lambda example: {"tokens": example['tokens'],
    "tags": example["tags"],
    "numeric_tags": [label.encode_example(x) for x in example["tags"]],
})
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

label_all_tokens = True
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    output_dir="distilbert-maccrobat",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    push_to_hub=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# trainer.push_to_hub()

file_path = "text.txt"
with open(file_path, 'r') as file:
    input_text = file.read()

encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
text_ner_outputs = model(**encoded_input)

# map tokens to original text
tokenized_text = tokenizer.tokenize(input_text)
tokenized_text.insert(0, '[CLS]')
tokenized_text.append('[SEP]')
tokenized_text = tokenized_text[:512]
tokenized_text = [token.replace('##', '') for token in tokenized_text]


# map labels to original text
labels = []
for i in range(len(text_ner_outputs.logits[0])):
    label = np.argmax(text_ner_outputs.logits[0][i].detach().numpy())
    labels.append(label_list[label])

# map tokens to labels
token_labels = []
for i in range(len(tokenized_text)):
    token_labels.append([tokenized_text[i], labels[i]])

