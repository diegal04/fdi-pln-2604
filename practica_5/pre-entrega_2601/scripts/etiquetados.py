from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from scripts.dataset import save_merged_dataset
from scripts.merge import (
    cohen_kappa,
    extract_frase_records,
    load_assignments,
    merge_sentence_labels,
    normalize_merge_label,
    records_to_word_labels,
)


def project_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parents[1] / relative_path


@dataclass
class FraseMergeResult:
    frase_id: int
    text: str
    lote: str
    json_indices: list[int]
    sources: list[str]
    label_sets: list[list[str]]
    merged_labels: list[str]
    token_agreement: float
    kappa: float | None
    disagreements: int


@dataclass
class MergeBundle:
    report: dict
    sentences: list[dict]
    frase_details: list[FraseMergeResult] = field(default_factory=list)


def _json_dir_candidates(json_idx: int) -> list[str]:
    n = json_idx + 1
    return [
        f"json_{n:02d}",
        f"json_{n}",
        f"json{n}",
        f"json{n:02d}",
    ]


def resolve_etiquetados_json_dir(etiquetados_root: Path, json_idx: int) -> Path | None:
    for name in _json_dir_candidates(json_idx):
        path = etiquetados_root / name
        if path.is_dir():
            return path

    return None


def resolve_annotation_file(json_dir: Path, parte_suffix: str) -> Path | None:
    """parte_suffix: 'p1' o 'p2'."""
    matches = sorted(json_dir.glob(f"*_{parte_suffix}.json"))

    if matches:
        return matches[0]

    return None


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_lote(
    etiquetados_root: Path,
    assignments_path: Path,
    lote: str,
    parte_suffix: str,
    frase_id_offset: int = 0,
) -> tuple[list[dict], list[FraseMergeResult], dict]:
    granularidad, frases, assignments = load_assignments(assignments_path)
    frase_texts = [frase.lower() for frase in frases]

    merged_sentences: list[dict] = []
    frase_details: list[FraseMergeResult] = []
    kappas: list[float] = []
    token_agreements: list[float] = []
    skipped_no_pair = 0
    skipped_unlabeled = 0

    json_indices_by_frase: dict[int, list[int]] = {i: [] for i in range(len(frases))}

    for json_idx, frase_indices in enumerate(assignments):
        for frase_idx in frase_indices:
            json_indices_by_frase[frase_idx].append(json_idx)

    for frase_idx, json_indices in json_indices_by_frase.items():
        if len(json_indices) != 2:
            logger.warning(
                "Frase %s en %s tiene %s JSON (se esperaban 2).",
                frase_idx,
                lote,
                len(json_indices),
            )
            skipped_no_pair += 1
            continue

        word_sets: list[tuple[list[str], list[str]]] = []
        sources: list[str] = []

        for json_idx in json_indices:
            json_dir = resolve_etiquetados_json_dir(etiquetados_root, json_idx)

            if json_dir is None:
                logger.warning("No existe carpeta para json index %s", json_idx)
                continue

            ann_path = resolve_annotation_file(json_dir, parte_suffix)

            if ann_path is None:
                logger.warning("Sin anotación %s en %s", parte_suffix, json_dir)
                continue

            records = load_records(ann_path)
            chunk = extract_frase_records(records, frase_texts[frase_idx])

            if chunk is None:
                logger.warning("Frase %s no hallada en %s", frase_idx, ann_path.name)
                continue

            _, tokens, labels = records_to_word_labels(chunk)
            word_sets.append((tokens, labels))
            sources.append(f"{json_dir.name}/{ann_path.name}")

        if len(word_sets) < 2:
            skipped_unlabeled += 1
            continue

        label_sets = [labels for _, labels in word_sets]
        tokens = word_sets[0][0]

        if word_sets[0][0] != word_sets[1][0]:
            logger.warning(
                "Frase %s en %s: tokens distintos entre anotadores.",
                frase_idx,
                lote,
            )

        if len(label_sets[0]) != len(label_sets[1]):
            logger.warning(
                "Frase %s longitudes distintas: %s vs %s palabras",
                frase_idx,
                len(label_sets[0]),
                len(label_sets[1]),
            )
            continue

        norm_a = [normalize_merge_label(label) for label in label_sets[0]]
        norm_b = [normalize_merge_label(label) for label in label_sets[1]]

        kappa = cohen_kappa(norm_a, norm_b)
        merged_labels, agreement = merge_sentence_labels(label_sets)
        disagreements = sum(1 for a, b in zip(norm_a, norm_b) if a != b)

        kappas.append(kappa)
        token_agreements.append(agreement)

        global_id = frase_id_offset + frase_idx

        frase_details.append(
            FraseMergeResult(
                frase_id=global_id,
                text=frase_texts[frase_idx],
                lote=lote,
                json_indices=json_indices,
                sources=sources,
                label_sets=label_sets,
                merged_labels=merged_labels,
                token_agreement=agreement,
                kappa=kappa,
                disagreements=disagreements,
            )
        )

        merged_sentences.append(
            {
                "frase_id": global_id,
                "lote": lote,
                "text": frase_texts[frase_idx],
                "tokens": tokens,
                "labels": merged_labels,
                "sources": sources,
                "kappa": kappa,
                "token_agreement": agreement,
            }
        )

    lote_report = {
        "lote": lote,
        "granularidad": granularidad,
        "n_frases_corpus": len(frases),
        "n_frases_fusionadas": len(merged_sentences),
        "skipped_no_pair": skipped_no_pair,
        "skipped_unlabeled": skipped_unlabeled,
        "mean_token_agreement": sum(token_agreements) / max(len(token_agreements), 1),
        "mean_cohen_kappa": sum(kappas) / max(len(kappas), 1),
        "pairwise_kappas": kappas,
    }

    return merged_sentences, frase_details, lote_report


def merge_etiquetados(
    etiquetados_root: Path | None = None,
    parte1_assignments: Path | None = None,
    parte2_assignments: Path | None = None,
    output_path: Path | None = None,
) -> MergeBundle:
    etiquetados_root = etiquetados_root or project_path("etiquetados")
    parte1_assignments = parte1_assignments or project_path(
        "asignaciones/asignaciones_parte1/asignaciones.json"
    )
    parte2_assignments = parte2_assignments or project_path(
        "asignaciones/asignaciones_parte2/asignaciones.json"
    )
    output_path = output_path or project_path("merged.json")

    s1, d1, r1 = merge_lote(
        etiquetados_root=etiquetados_root,
        assignments_path=parte1_assignments,
        lote="parte1",
        parte_suffix="p1",
        frase_id_offset=0,
    )

    offset = r1["n_frases_corpus"]

    s2, d2, r2 = merge_lote(
        etiquetados_root=etiquetados_root,
        assignments_path=parte2_assignments,
        lote="parte2",
        parte_suffix="p2",
        frase_id_offset=offset,
    )

    all_sentences = s1 + s2
    all_details = d1 + d2
    all_kappas = r1["pairwise_kappas"] + r2["pairwise_kappas"]
    all_agreements = [detail.token_agreement for detail in all_details]

    report = {
        "source": str(etiquetados_root),
        "n_frases": len(all_sentences),
        "mean_token_agreement": sum(all_agreements) / max(len(all_agreements), 1),
        "mean_cohen_kappa": sum(all_kappas) / max(len(all_kappas), 1),
        "pairwise_kappas": all_kappas,
        "lotes": [r1, r2],
    }

    save_merged_dataset(output_path, all_sentences)

    logger.info("Fusión etiquetados → %s", output_path)
    logger.info(
        "Frases fusionadas: %s (parte1=%s, parte2=%s)",
        len(all_sentences),
        r1["n_frases_fusionadas"],
        r2["n_frases_fusionadas"],
    )
    logger.info("κ medio global: {:.3f}", report["mean_cohen_kappa"])

    return MergeBundle(
        report=report,
        sentences=all_sentences,
        frase_details=all_details,
    )