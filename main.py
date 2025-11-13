# ---------------------------------------------------- Imports und Installation
# (Installiere die Libraries vorher im Terminal, z.B. mit pip install datasets transformers torch evaluate seqeval)

import datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, pipeline
import numpy as np
import evaluate
# ---------------------------------------------------- Schritt 1: CoNLLpp-Datensatz und Tokenizer laden
print('\n---- Schritt 1: Dataset und Tokenizer Laden ----')
# Lade den CoNLLpp-Datensatz ohne Dataset-Script (trust_remote_code wird nicht mehr unterstützt)

# CoNLLpp-Quellen (wie im ursprünglichen Script)
_URL = "https://github.com/ZihanWangKi/CrossWeigh/raw/master/data/"
_TRAINING_FILE = "conllpp_train.txt"
_DEV_FILE = "conllpp_dev.txt"
_TEST_FILE = "conllpp_test.txt"

# Definiere die möglichen Tag-Namen (wie im ursprünglichen Script), damit wir ClassLabel verwenden können
POS_TAGS = [
    '"', "''", "#", "$", "(", ")", ",", ".", ":", "``", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "NN|SYM", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"
]
CHUNK_TAGS = [
    "O", "B-ADJP", "I-ADJP", "B-ADVP", "I-ADVP", "B-CONJP", "I-CONJP", "B-INTJ", "I-INTJ", "B-LST", "I-LST", "B-NP", "I-NP", "B-PP", "I-PP", "B-PRT", "I-PRT", "B-SBAR", "I-SBAR", "B-UCP", "I-UCP", "B-VP", "I-VP"
]
NER_TAGS = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]

def _download_text(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def _parse_conll_sentences(text: str):
    """Parst CoNLL-ähnliche Dateien in satzweise Beispiele."""
    guid = 0
    tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
    for line in text.splitlines():
        if line.startswith("-DOCSTART-") or line.strip() == "":
            if tokens:
                yield {
                    "id": str(guid),
                    "tokens": tokens,
                    "pos_tags": pos_tags,
                    "chunk_tags": chunk_tags,
                    "ner_tags": ner_tags,
                }
                guid += 1
                tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
        else:
            # conll2003 tokens are space separated
            splits = line.split(" ")
            if len(splits) < 4:
                # defensiv: Zeile überspringen wenn unvollständig
                continue
            tokens.append(splits[0])
            pos_tags.append(splits[1])
            chunk_tags.append(splits[2])
            ner_tags.append(splits[3].strip())
    if tokens:
        yield {
            "id": str(guid),
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "ner_tags": ner_tags,
        }


def load_conllpp_as_datasets() -> datasets.DatasetDict:
    """Lädt CoNLLpp aus den Original-Quellen und baut ein DatasetDict ohne Dataset-Skript."""
    # Lade Texte
    train_text = _download_text(f"{_URL}{_TRAINING_FILE}")
    dev_text = _download_text(f"{_URL}{_DEV_FILE}")
    test_text = _download_text(f"{_URL}{_TEST_FILE}")

    # Definiere Features mit ClassLabel, sodass wir später label_list korrekt auslesen können
    features = datasets.Features({
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "pos_tags": datasets.Sequence(datasets.ClassLabel(names=POS_TAGS)),
        "chunk_tags": datasets.Sequence(datasets.ClassLabel(names=CHUNK_TAGS)),
        "ner_tags": datasets.Sequence(datasets.ClassLabel(names=NER_TAGS)),
    })

    # Erzeuge Datasets aus Generatoren
    train_ds = datasets.Dataset.from_generator(lambda: _parse_conll_sentences(train_text), features=features)
    val_ds = datasets.Dataset.from_generator(lambda: _parse_conll_sentences(dev_text), features=features)
    test_ds = datasets.Dataset.from_generator(lambda: _parse_conll_sentences(test_text), features=features)

    ds_dict = datasets.DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })
    return ds_dict

# tatsächliches Laden
dataset = load_conllpp_as_datasets()
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
    # Unterstütze ältere und neuere Transformers-Versionen
    try:
        # Newer: EvalPrediction with attributes
        predictions = eval_pred.predictions
        labels = getattr(eval_pred, 'label_ids', getattr(eval_pred, 'labels', None))
        if labels is None:
            # Fallback to tuple-like unpacking
            predictions, labels = eval_pred
    except AttributeError:
        # Older: tuple of (predictions, labels)
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
        "precision": results.get("overall_precision", results.get("precision", 0.0)),
        "recall": results.get("overall_recall", results.get("recall", 0.0)),
        "f1": results.get("overall_f1", results.get("f1", 0.0)),
        "accuracy": results.get("overall_accuracy", results.get("accuracy", 0.0)),
    }

# ------------------------------------------------------ Schritt 6: Training vorbereiten und durchführen
print('\n---- Schritt 6: Training starten ----')
# GPU-/Mixed-Precision-Optimierung: aktiviere fp16 automatisch bei CUDA
try:
    import torch
    use_fp16 = bool(getattr(torch.cuda, "is_available", lambda: False)())
except Exception:
    use_fp16 = False
print(f'> Hardware: {"GPU erkannt – aktiviere fp16 Mixed Precision" if use_fp16 else "keine GPU – Training auf CPU ohne Mixed Precision"}')
training_args = TrainingArguments(
    output_dir="./bert-ner-conllpp",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Passe ggf. an bei wenig RAM (z.B. 4)
    per_device_eval_batch_size=8,
    fp16=use_fp16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
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
# Abwärtskompatible Initialisierung: neuere Versionen nutzen aggregation_strategy, ältere grouped_entities
try:
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
except TypeError:
    try:
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            grouped_entities=True
        )
    except TypeError:
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer
        )
print('> NER-Pipeline zum Testen initialisiert.')
test_text = "Elon Musk is the CEO of SpaceX and Tesla."
results = ner_pipeline(test_text)

print(f'> Test-Satz: {test_text}')
if not results:
    print('Keine Entities gefunden.')
else:
    print('Forecast Entities erkannt:')
    for entity in results:
        word = entity.get('word', entity.get('text', ''))
        label = entity.get('entity_group', entity.get('entity', ''))
        score = entity.get('score', None)
        if score is not None:
            print(f"  Text: {word}\n  Typ (Label): {label}\n  Score: {score:.3f}")
        else:
            print(f"  Text: {word}\n  Typ (Label): {label}")
