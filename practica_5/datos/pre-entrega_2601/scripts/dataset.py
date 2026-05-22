from __future__ import annotations

import json
from pathlib import Path


def word_labels_to_char_labels(tokens: list[str], labels: list[str]) -> tuple[str, list[str]]:
    if len(tokens) != len(labels):
        raise ValueError("tokens y labels deben tener la misma longitud.")

    text = "".join(tokens)
    char_labels: list[str] = []

    for token, label in zip(tokens, labels):
        char_labels.extend([label] * len(token))

    return text, char_labels


def slim_sentence(sentence: dict) -> dict:
    """Devuelve solo los campos necesarios del dataset fusionado."""
    out: dict = {
        "frase_id": sentence["frase_id"],
        "text": sentence["text"],
    }

    if "tokens" in sentence:
        out["tokens"] = sentence["tokens"]

    out["labels"] = sentence["labels"]
    return out


def save_merged_dataset(path: Path, sentences: list[dict]) -> None:
    """Guarda únicamente la lista de frases anotadas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [slim_sentence(sentence) for sentence in sentences]
    path.write_text(
        json.dumps(slim, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_merged_dataset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return payload

    return payload["sentences"]