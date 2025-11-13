# ---------------------------------------------------- Imports und Installation
# (Installiere die Libraries vorher im Terminal, z.B. mit pip install datasets transformers torch evaluate seqeval)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, pipeline
import numpy as np
import evaluate

# ---------------------------------------------------- Schritt 1: CoNLLpp-Datensatz und Tokenizer laden
print('\n---- Schritt 1: Dataset und Tokenizer Laden ----')
# Lade den CoNLLpp-Datensatz
dataset = load_dataset("conllpp")
print(f'> Dataset geladen, mit Splits: {list(dataset.keys())}')
print(f'> Beispiel für einen Datensatz-Eintrag: {dataset["train"][0]}')

# Lade den BERT-Tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f'> Tokenizer "{model_checkpoint}" geladen.')

# Extrahiere NER-Tags und mache Zuordnung
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
print(f'> Extrahierte NER-Tags: {label_list}')
print(f'> Anzahl NER-Tags: {num_labels}')

# ------------------------------------------------------ Schritt 2: Modell mit NER-Konfiguration laden
print('\n---- Schritt 2: Modell Laden und Konfigurieren ----')
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
print(f'> Modell geladen: {model.__class__.__name__}')
print(f'> Modell-Konfiguration überprüft: num_labels = {model.config.num_labels}')
print(f'> Beispiel Zuordnung: 0 -> {model.config.id2label[0]}')

# ------------------------------------------------------ Schritt 3: Tokenisierung und Label-Ausrichtung
print('\n---- Schritt 3: Tokenisierung und Label Alignment ----')
def tokenize_and_align_labels(examples):
    # Tokenisiere Inputs, splitte nach Wörtern (is_split_into_words=True)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Spezielle Tokens werden ignoriert (-100)
            if word_idx is None:
                label_ids.append(-100)
            # Nur erstes Token eines Wortes bekommt das Label
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenisiere Datensatz und aligniere Labels
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
print('> Tokenisierung und Label-Alignment abgeschlossen.')
print(f'> Beispiel: tokenisierte Tokens (input_ids): {tokenized_datasets["train"][0]["input_ids"][:10]}')
print(f'> Zugeordnete Labels zu ersten Tokens: {tokenized_datasets["train"][0]["labels"][:10]}')

# ------------------------------------------------------ Schritt 4: Data Collator für Padding (Batch-Längen angleichen)
print('\n---- Schritt 4: Data Collator bereitstellen ----')
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
print('> Data Collator initialisiert (übernimmt dynamisches Padding im Batch).')

# ------------------------------------------------------ Schritt 5: Fehlerfunktion (NER-Metrik)
print('\n---- Schritt 5: Fehlerfunktion & Metriken ----')
metric = evaluate.load("seqeval")
print('> seqeval-Metrik für NER geladen.')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    # Entferne -100 Labels, damit nur Haupttokens bewertet werden
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(f'> Evaluations-Ergebnis: {results}')
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ------------------------------------------------------ Schritt 6: Training vorbereiten und durchführen
print('\n---- Schritt 6: Training starten ----')
training_args = TrainingArguments(
    output_dir="./bert-ner-conllpp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Passe ggf. an bei wenig RAM (z.B. 4)
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print('> Trainer initialisiert. Starte Training...')
trainer.train()
print('> Training abgeschlossen!')

# ------------------------------------------------------ Schritt 7: Testen mit Transformers-Pipeline
print('\n---- Schritt 7: Ergebnisse testen mit Pipeline ----')
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)
print('> NER-Pipeline zum Testen initialisiert.')
test_text = "Angela Merkel lebt in Berlin."
results = ner_pipeline(test_text)

print(f'> Test-Satz: {test_text}')
if not results:
    print('Keine Entities gefunden.')
else:
    print('Forecast Entities erkannt:')
    for entity in results:
        print(f"  Text: {entity['word']}\n  Typ (Label): {entity['entity_group']}\n  Score: {entity['score']:.3f}")
