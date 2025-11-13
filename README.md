# Mitschrift für das Modul NER (GK9102)
## Einführung in NER
Named Entity Recognition (NER) ist ein NLP-Verfahren zur Erkennung und Klassifizierung von sogenannten 
„benannten Entitäten“ wie Personen, Organisationen oder Ortsnamen im Text. 
Ein NER-Modell arbeitet typischerweise auf Token-Ebene und entscheidet für jedes Token, 
ob es zu einer Entität gehört, und zu welcher Kategorie.

## Tokenization
 Der Text wird in Token zerlegt, z.B. einzelne Wörter oder Zeichenfolgen. Tokenization ist die Voraussetzung, 
 da NER auf Token-Basis arbeitet und jedes Token separat klassifiziert werden muss.\
 *Beispiel für ein Token:* 
 Beispielsatz: "Chatbots sind toll."\
 **Tokens**: ["Chatbots", "sind", "toll", "."] (Wort-Tokenization)\
 **Tokens**: ["C", "h", "a", "t", "b", "o", ...] (Zeichen-Tokenization)
 
Gerade bei komplexeren oder seltenen Wörtern teilt ein sogenannter Subword-Tokenizer 
Wörter in häufige Wortteile auf. Beispiel:\
"Projektmanagement" → ["Projekt", "manage", "ment"]

## Datensatz das wir verwenden
Name: “CoNLLpp” (kurz für „CoNLL plus plus“ bzw. eine korrigierte Version von CoNLL2003).
Aufgabe: Token-Klassifikation / Named Entity Recognition (NER)
Sprache: Englisch.

Um den Datensatz zu laden, muss man das Repository klonen:
```bash
git clone https://huggingface.co/datasets/ZihanWangKi/conllpp
```
Danach kommt ein Ordner namens `./conllpp`, der die Daten enthält.

**Spalten/Felder (features):**
- **id** : String – eine Instanzkennung.
- **tokens** : Liste von Strings – die Token eines Satzes.
- **pos_tags** : Liste von Klassifikationslabels – POS (Part-of-Speech) Tags.
- **chunk_tags** : Liste – Chunking Tags (z. B. B-ADJP, I-ADJP etc.).
- **ner_tags** : Liste – NER Tags (z. B. B-PER, I-PER, B-ORG etc.).

## Erklärung des Codes



# Mitschrift für das Modul NER (GK9102)
## Einführung in NER
Named Entity Recognition (NER) ist ein NLP-Verfahren zur Erkennung und Klassifizierung von sogenannten 
„benannten Entitäten“ wie Personen, Organisationen oder Ortsnamen im Text. 
Ein NER-Modell arbeitet typischerweise auf Token-Ebene und entscheidet für jedes Token, 
ob es zu einer Entität gehört, und zu welcher Kategorie.

## Tokenization
 Der Text wird in Token zerlegt, z.B. einzelne Wörter oder Zeichenfolgen. Tokenization ist die Voraussetzung, 
 da NER auf Token-Basis arbeitet und jedes Token separat klassifiziert werden muss.\
 *Beispiel für ein Token:* 
 Beispielsatz: "Chatbots sind toll."\
 **Tokens**: ["Chatbots", "sind", "toll", "."] (Wort-Tokenization)\
 **Tokens**: ["C", "h", "a", "t", "b", "o", ...] (Zeichen-Tokenization)
 
Gerade bei komplexeren oder seltenen Wörtern teilt ein sogenannter Subword-Tokenizer 
Wörter in häufige Wortteile auf. Beispiel:\
"Projektmanagement" → ["Projekt", "manage", "ment"]

## Datensatz das wir verwenden
Name: “CoNLLpp” (kurz für „CoNLL plus plus“ bzw. eine korrigierte Version von CoNLL2003).
Aufgabe: Token-Klassifikation / Named Entity Recognition (NER)
Sprache: Englisch.

Um den Datensatz zu laden, muss man das Repository klonen:
```bash
git clone https://huggingface.co/datasets/ZihanWangKi/conllpp
```
Danach kommt ein Ordner namens `./conllpp`, der die Daten enthält.

**Spalten/Felder (features):**
- **id** : String – eine Instanzkennung.
- **tokens** : Liste von Strings – die Token eines Satzes.
- **pos_tags** : Liste von Klassifikationslabels – POS (Part-of-Speech) Tags.
- **chunk_tags** : Liste – Chunking Tags (z. B. B-ADJP, I-ADJP etc.).
- **ner_tags** : Liste – NER Tags (z. B. B-PER, I-PER, B-ORG etc.).

## Erklärung des Codes


## Inferenz (Text einlesen und Entities erkennen)
Mit dem Skript `infer.py` kannst du ein trainiertes Modell (Hugging Face Checkpoint-Ordner oder .pt/.bin Datei) laden und auf Text anwenden.

Beispiele:

1) Modell aus Checkpoint-Ordner verwenden (z. B. der Trainings-Output `bert-ner-conllpp/checkpoint-5268`):
```bash
python infer.py --model .\bert-ner-conllpp\checkpoint-5268 --text "Elon Musk is the CEO of SpaceX and Tesla."
```

2) Modell aus .pt-Datei laden (z. B. `my_model.pt`). Wenn die Labels nicht im Modell hinterlegt sind, werden Standard-CoNLL-Labels verwendet. Optional kannst du eine Labels-JSON angeben:
```bash
python infer.py --model .\my_model.pt --text "Barack Obama was born in Hawaii."
# Optional mit eigener Labels-Datei (Liste oder {"labels": [...]})
python infer.py --model .\my_model.pt --labels .\labels.json --text "Barack Obama was born in Hawaii."
```

3) Text aus Datei (eine Zeile pro Text) oder von STDIN:
```bash
# Datei
python infer.py --model .\bert-ner-conllpp\checkpoint-5268 --input .\texte.txt

# STDIN
type .\texte.txt | python infer.py --model .\bert-ner-conllpp\checkpoint-5268
```

Weitere Optionen:
- `--tokenizer`: eigener Tokenizer-Name/Pfad (Standard: versucht den Modellordner, sonst `bert-base-cased`).
- `--device`: `auto` (Standard), `-1` für CPU oder eine GPU-ID wie `0`.
- `--batch-size`: Batch-Größe für die Pipeline (Standard 8).
- `--no-aggregate`: deaktiviert die Entitäts-Zusammenfassung, falls du rohe Token-Labels sehen möchtest.
