from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from scripts.dataset import save_merged_dataset

VALID_LABELS = frozenset({"o", "pi", "pc", "li", "lc"})
LABEL_TYPOS = {"ps": "pi", "o ": "o", " o": "o"}


def normalize_merge_label(label: str | None) -> str:
    """Normaliza etiquetas a: o, pi, pc, li, lc. Cualquier otra etiqueta pasa a o."""
    if not label:
        return "o"

    cleaned = label.strip().lower()
    cleaned = LABEL_TYPOS.get(cleaned, cleaned)

    if cleaned not in VALID_LABELS:
        return "o"

    return cleaned


def _entity_prefix(label: str) -> str | None:
    if label == "o":
        return None

    return label[0]


def coerce_bio_label(label: str, prev_label: str) -> str:
    """Ajusta inicio/continuación según el contexto previo."""
    label = normalize_merge_label(label)

    if label == "o":
        return "o"

    prefix = _entity_prefix(label)
    prev_label = normalize_merge_label(prev_label)

    if prefix is None or prev_label == "o" or _entity_prefix(prev_label) != prefix:
        return f"{prefix}i"

    return f"{prefix}c"


def merge_disagreeing_labels(votes: list[str], prev_label: str) -> str:
    """Resuelve conflictos entre anotadores."""
    votes = [normalize_merge_label(vote) for vote in votes]

    if len(set(votes)) == 1:
        return coerce_bio_label(votes[0], prev_label)

    non_o = [vote for vote in votes if vote != "o"]

    if not non_o:
        return "o"

    if len(non_o) == 1:
        return coerce_bio_label(non_o[0], prev_label)

    prefixes = {_entity_prefix(vote) for vote in non_o}

    if len(prefixes) == 1:
        return coerce_bio_label(non_o[0], prev_label)

    return coerce_bio_label(Counter(non_o).most_common(1)[0][0], prev_label)


def records_to_text_and_labels(records: list[dict]) -> tuple[str, list[str]]:
    """Expande etiquetas carácter a carácter para informes."""
    text = "".join(item["clave"] for item in records)
    labels: list[str] = []

    for item in records:
        label = normalize_merge_label(item.get("valor"))
        labels.extend([label] * len(item["clave"]))

    return text, labels


def records_to_word_labels(records: list[dict]) -> tuple[str, list[str], list[str]]:
    """Devuelve texto, tokens y etiquetas a nivel de unidad anotada."""
    tokens = [item["clave"] for item in records]
    labels = [normalize_merge_label(item.get("valor")) for item in records]
    text = "".join(tokens)

    return text, tokens, labels


def word_labels_to_char_labels(tokens: list[str], labels: list[str]) -> tuple[str, list[str]]:
    if len(tokens) != len(labels):
        raise ValueError("tokens y labels deben tener la misma longitud.")

    text = "".join(tokens)
    char_labels: list[str] = []

    for token, label in zip(tokens, labels):
        char_labels.extend([label] * len(token))

    if len(char_labels) != len(text):
        raise ValueError("Longitud de etiquetas por carácter inconsistente.")

    return text, char_labels


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Las secuencias deben tener la misma longitud.")

    if not labels_a:
        return 1.0

    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)

    matrix: dict[tuple[str, str], int] = Counter()

    for left, right in zip(labels_a, labels_b):
        matrix[(left, right)] += 1

    observed = sum(matrix[(category, category)] for category in categories) / n

    marg_a = {
        category: sum(matrix[(category, other)] for other in categories) / n
        for category in categories
    }

    marg_b = {
        category: sum(matrix[(other, category)] for other in categories) / n
        for category in categories
    }

    expected = sum(marg_a[category] * marg_b[category] for category in categories)

    if expected == 1.0:
        return 1.0

    return (observed - expected) / (1.0 - expected)


def normalize_annotation_text(text: str) -> str:
    """Normaliza comillas y apóstrofos para comparar frases."""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def merge_sentence_labels(label_sets: list[list[str]]) -> tuple[list[str], float]:
    if not label_sets:
        return [], 1.0

    length = len(label_sets[0])

    if any(len(labels) != length for labels in label_sets):
        raise ValueError("Las anotaciones de una misma frase no coinciden en longitud.")

    normalized_sets = [
        [normalize_merge_label(label) for label in labels] for labels in label_sets
    ]

    merged: list[str] = []
    agreements: list[float] = []
    previous = "o"

    for idx in range(length):
        votes = [labels[idx] for labels in normalized_sets]

        if len(set(votes)) == 1:
            candidate = coerce_bio_label(votes[0], previous)
        else:
            candidate = merge_disagreeing_labels(votes, previous)

        merged.append(candidate)
        previous = candidate

        if len(label_sets) == 2:
            raw_a = normalize_merge_label(label_sets[0][idx])
            raw_b = normalize_merge_label(label_sets[1][idx])
            agreements.append(1.0 if raw_a == raw_b else 0.0)

    token_agreement = sum(agreements) / len(agreements) if agreements else 1.0
    return merged, token_agreement


def extract_frase_records(records: list[dict], frase_text: str) -> list[dict] | None:
    full_text, _ = records_to_text_and_labels(records)
    start = full_text.find(frase_text)

    if start < 0:
        norm_full = normalize_annotation_text(full_text)
        norm_frase = normalize_annotation_text(frase_text)
        start = norm_full.find(norm_frase)

        if start < 0:
            return None

    end = start + len(frase_text)
    cursor = 0
    chunk: list[dict] = []

    for record in records:
        piece = record["clave"]
        piece_start = cursor
        piece_end = cursor + len(piece)

        if piece_end <= start:
            cursor = piece_end
            continue

        if piece_start >= end:
            break

        chunk.append(record)
        cursor = piece_end

    rebuilt, _ = records_to_text_and_labels(chunk)

    if (
        rebuilt != frase_text
        and normalize_annotation_text(rebuilt) != normalize_annotation_text(frase_text)
    ):
        logger.warning("Segmento parcial no coincide exactamente con la frase objetivo.")

    return chunk


def load_assignments(path: Path) -> tuple[str, list[str], list[list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    granularidad = payload.get("granularidad", "palabra")

    return granularidad, payload["frases"], payload["asignaciones"]


def merge_annotations(
    json_dir: Path,
    assignments_path: Path,
    output_path: Path,
) -> dict:
    granularidad, frases, assignments = load_assignments(assignments_path)
    frase_texts = [frase.lower() for frase in frases]

    logger.info("Fusionando anotaciones (granularidad={})", granularidad)

    merged_sentences: list[dict] = []
    kappas: list[float] = []
    token_agreements: list[float] = []

    word_sets_by_frase: dict[int, list[tuple[list[str], list[str]]]] = {
        idx: [] for idx in range(len(frases))
    }

    for json_idx in range(len(assignments)):
        json_path = json_dir / f"json_{json_idx + 1:02d}.json"

        if not json_path.exists():
            continue

        records = json.loads(json_path.read_text(encoding="utf-8"))

        for frase_idx in assignments[json_idx]:
            chunk = extract_frase_records(records, frase_texts[frase_idx])

            if chunk is None:
                logger.warning("Frase {} no encontrada en {}", frase_idx, json_path.name)
                continue

            _, tokens, labels = records_to_word_labels(chunk)

            if any(label != "o" for label in labels):
                word_sets_by_frase[frase_idx].append((tokens, labels))

    for frase_idx, word_sets in word_sets_by_frase.items():
        if not word_sets:
            continue

        label_sets = [labels for _, labels in word_sets]
        tokens = word_sets[0][0]

        if len(word_sets) == 2 and word_sets[0][0] != word_sets[1][0]:
            logger.warning(
                "Frase {} tokens distintos entre anotadores; se usa el primero.",
                frase_idx,
            )

        if len(label_sets) == 2 and len(label_sets[0]) == len(label_sets[1]):
            kappas.append(
                cohen_kappa(
                    [normalize_merge_label(label) for label in label_sets[0]],
                    [normalize_merge_label(label) for label in label_sets[1]],
                )
            )

        merged_labels, agreement = merge_sentence_labels(label_sets)
        token_agreements.append(agreement)

        merged_sentences.append(
            {
                "frase_id": frase_idx,
                "text": frase_texts[frase_idx],
                "tokens": tokens,
                "labels": merged_labels,
            }
        )

    report = {
        "granularidad": granularidad,
        "n_frases": len(merged_sentences),
        "mean_token_agreement": sum(token_agreements) / max(len(token_agreements), 1),
        "mean_cohen_kappa": sum(kappas) / max(len(kappas), 1),
        "pairwise_kappas": kappas,
    }

    save_merged_dataset(output_path, merged_sentences)

    logger.info("Fusión guardada en {}", output_path)
    logger.info("Acuerdo medio por token: {:.3f}", report["mean_token_agreement"])
    logger.info("Kappa de Cohen medio: {:.3f}", report["mean_cohen_kappa"])

    return report