import argparse
import json
import os
import sys
from typing import List, Optional, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    pipeline,
)

# Default labels for CoNLL NER if none are provided/found
DEFAULT_NER_TAGS = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]


def load_label_list(model_dir_or_file: str, labels_json: Optional[str]) -> List[str]:
    # 1) explicit labels json file
    if labels_json:
        with open(labels_json, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, dict) and "labels" in labels:
            labels = labels["labels"]
        if not isinstance(labels, list):
            raise ValueError("Labels JSON must be a list or an object with key 'labels'.")
        return labels

    # 2) try HF config in directory
    if os.path.isdir(model_dir_or_file):
        try:
            config = AutoConfig.from_pretrained(model_dir_or_file)
            if getattr(config, "id2label", None):
                # id2label is a dict mapping index(string/int) -> label
                # ensure order by index
                id2label = config.id2label
                # keys might be strings of ints
                labels = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
                return labels
        except Exception:
            pass

    # 3) fallback to default
    return DEFAULT_NER_TAGS


def build_model(model_path: str, label_list: List[str]):
    num_labels = len(label_list)
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    if os.path.isdir(model_path):
        # Standard HF directory with config and weights
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        return model

    # Otherwise, try to load a state_dict from a .pt/.bin file
    base_model_name = "bert-base-cased"
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    state = torch.load(model_path, map_location="cpu")
    # Some training scripts save {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # If keys are prefixed (e.g., "module."), try to clean them
    def _strip_prefix(sdict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sdict.items() }

    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError:
        # Try stripping common prefixes
        for p in ["module.", "model."]:
            try:
                model.load_state_dict(_strip_prefix(state, p), strict=False)
                break
            except RuntimeError:
                continue
    return model


def init_tokenizer(tokenizer_arg: Optional[str], model_path: str):
    if tokenizer_arg:
        return AutoTokenizer.from_pretrained(tokenizer_arg)
    if os.path.isdir(model_path):
        try:
            return AutoTokenizer.from_pretrained(model_path)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained("bert-base-cased")


def build_pipeline(model, tokenizer, device: int, no_aggregate: bool):
    # Newer API: aggregation_strategy, older: grouped_entities
    if no_aggregate:
        try:
            return pipeline("ner", model=model, tokenizer=tokenizer, device=device)
        except TypeError:
            return pipeline("ner", model=model, tokenizer=tokenizer)

    try:
        return pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device
        )
    except TypeError:
        try:
            return pipeline(
                "ner", model=model, tokenizer=tokenizer, grouped_entities=True, device=device
            )
        except TypeError:
            return pipeline("ner", model=model, tokenizer=tokenizer, device=device)


def read_inputs(args) -> List[str]:
    texts: List[str] = []
    if args.text:
        texts.append(args.text)
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    if not texts:
        # read from stdin until EOF
        buf = sys.stdin.read().strip()
        if buf:
            texts.append(buf)
    return texts


def format_result(item: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize across HF versions
    word = item.get("word", item.get("text", ""))
    label = item.get("entity_group", item.get("entity", ""))
    score = item.get("score", None)
    start = item.get("start", None)
    end = item.get("end", None)
    return {
        "text": word,
        "label": label,
        "score": float(score) if score is not None else None,
        "start": start,
        "end": end,
    }


def main():
    parser = argparse.ArgumentParser(description="Run NER inference on text using a trained model (.pt or HF dir)")
    parser.add_argument("--model", required=True, help="Path to model directory (HF) or .pt/.bin file")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name or path (default: try model dir, else bert-base-cased)")
    parser.add_argument("--labels", default=None, help="Optional path to JSON with a list of labels or {'labels': [...]} ")
    parser.add_argument("--text", default=None, help="Inline text to analyze")
    parser.add_argument("--input", default=None, help="Path to a text file with one document per line")
    parser.add_argument("--no-aggregate", action="store_true", help="Disable entity span aggregation")
    parser.add_argument("--device", default="auto", help="auto, -1 (CPU), or GPU index like 0")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for pipeline")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    else:
        try:
            device = int(args.device)
        except ValueError:
            device = -1

    # Load labels and model/tokenizer
    label_list = load_label_list(args.model, args.labels)
    tokenizer = init_tokenizer(args.tokenizer, args.model)
    model = build_model(args.model, label_list)

    nlp = build_pipeline(model, tokenizer, device, args.no_aggregate)

    texts = read_inputs(args)
    if not texts:
        print("Keine Eingabetexte gefunden. Verwende --text, --input oder STDIN.")
        sys.exit(1)

    # Run inference in batches
    outputs: List[Any] = []
    batch_size = max(1, args.batch_size)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = nlp(batch)
        # results is list[list[entities]] when batch input
        for text, ents in zip(batch, results):
            formatted = [format_result(e) for e in ents]
            print(f"\nText: {text}")
            if not formatted:
                print("Keine Entities gefunden.")
            else:
                print("Entities:")
                for e in formatted:
                    base = f"  [{e['label']}] {e['text']}"
                    if e["score"] is not None:
                        base += f"  (score={e['score']:.3f})"
                    if e["start"] is not None and e["end"] is not None:
                        base += f"  span=({e['start']},{e['end']})"
                    print(base)
        outputs.append(results)


if __name__ == "__main__":
    main()
