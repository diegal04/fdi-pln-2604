"""Utilidades compartidas y constantes de la CLI."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------

PROJECT_DIR = Path.cwd()

DEFAULT_CORPUS = Path("datos") / "pre-entrega_2604" / "corpus_original"
DEFAULT_ANNOTATIONS = Path("datos") / "pre-entrega_2604" / "merged.json"
DEFAULT_TOKENIZER = Path("resultados_finales") / "tokenizer.json"
DEFAULT_LM_WEIGHTS = Path("resultados_finales") / "p5_causal_2604.pth"
DEFAULT_NER_WEIGHTS = Path("resultados_finales") / "p5_ner_2604.pth"
DEFAULT_EXPERIMENTS_DIR = Path("resultados_finales") / "experiments"
DEFAULT_LM_HISTORY = DEFAULT_EXPERIMENTS_DIR / "lm_exp2" / "history.json"
DEFAULT_NER_HISTORY = DEFAULT_EXPERIMENTS_DIR / "ner_final" / "history.json"
DEFAULT_NER_METRICS = DEFAULT_EXPERIMENTS_DIR / "ner_final" / "metrics"

# ---------------------------------------------------------------------------
# Configuraciones por defecto de grid search
# ---------------------------------------------------------------------------

GRID_TOKENIZER_DEFAULTS = {
    "vocab_sizes": "200,300,500",
}
GRID_LM_DEFAULTS = {
    "epochs": "3,5",
    "batch_sizes": "32,64",
    "lrs": "0.0003,0.0001",
    "d_models": "128",
    "n_heads": "4",
    "n_layers": "2,4",
    "dropouts": "0.1,0.2",
    "context_size": 128,
    "expansion": 4,
    "warmup_steps": 100,
    "weight_decay": 0.1,
}
GRID_NER_DEFAULTS = {
    "epochs": "50",
    "batch_sizes": "16,32,64",
    "lrs": "0.001,0.0005,0.0003,0.0001",
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "dropout": 0.3,
    "entity_loss_weights": "5,10,15,20",
    "continuation_weight_multiplier": 3.0,
    "selection_accuracy_floor": 0.8,
    "selection_non_o_weight": 1.5,
    "context_size": 128,
    "expansion": 4,
    "warmup_steps": 50,
    "weight_decay": 0.1,
}

# ---------------------------------------------------------------------------
# Funciones de utilidad
# ---------------------------------------------------------------------------


def _read_corpus(path: Path) -> str:
    """Lee un fichero o concatena todos los .txt de un directorio."""
    if path.is_dir():
        return "\n\n".join(
            p.read_text(encoding="utf-8") for p in sorted(path.glob("*.txt"))
        )
    return path.read_text(encoding="utf-8")


def _load_ner_data(path: Path):
    """Carga merged.json como lista de pares (tokens, labels)."""
    merged = json.loads(path.read_text(encoding="utf-8"))
    data = []
    for i, item in enumerate(merged):
        tokens = item.get("tokens", item.get("id"))
        labels = item.get("labels")
        if tokens is None or labels is None:
            raise click.ClickException(
                f"Entrada {i} de {path} sin campos tokens/id y labels."
            )
        if not isinstance(tokens, list) or not isinstance(labels, list):
            raise click.ClickException(
                f"Entrada {i} de {path} debe contener listas en tokens/id y labels."
            )
        if len(tokens) != len(labels):
            raise click.ClickException(
                f"Entrada {i} de {path} con {len(tokens)} tokens y "
                f"{len(labels)} labels."
            )
        data.append((tokens, labels))
    return data


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _device(torch) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_state_dict(torch, path: Path, device: str):
    """Carga checkpoints simples o diccionarios con clave model_state."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    return state


def _load_weights(model, state, strict: bool):
    """Carga pesos dando un error claro si la arquitectura no coincide."""
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise click.ClickException(
            "No se han podido cargar los pesos. Comprueba que el tokenizador, "
            "vocab_size, d_model, n_heads, n_layers y context_size coinciden "
            "con los usados al entrenar el checkpoint."
        ) from exc


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_file(path: Path, purpose: str):
    if not path.exists():
        raise click.ClickException(
            f"No se encuentra {purpose}: {path}. "
            "Entrenalo primero o pasa otra ruta con la opcion correspondiente."
        )


def _final_val_loss(history):
    if not history:
        return None
    return history[-1].get("val_loss")


def _final_val_accuracy(history):
    if not history:
        return None
    return history[-1].get("val_accuracy")


def _final_metric(history, key):
    if not history:
        return None
    return history[-1].get(key)


def _best_history_row(history):
    if not history:
        return None
    for row in history:
        if row.get("is_best"):
            return row
    return history[-1]


def _best_metric(history, key):
    row = _best_history_row(history)
    if row is None:
        return None
    return row.get(key)


def _ner_selection_score(
    history,
    accuracy_floor=0.8,
    non_o_weight=1.5,
):
    """Score para elegir el mejor NER en grid-search.

    Si el accuracy total cae por debajo del umbral, descartamos el run.
    Si no, combinamos accuracy total y accuracy de clases no-O.
    """
    row = _best_history_row(history)
    if row is not None and row.get("selection_score") is not None:
        return row["selection_score"]
    val_accuracy = _final_metric(history, "val_accuracy")
    val_non_o_accuracy = _final_metric(history, "val_non_o_accuracy")
    if val_accuracy is None or val_non_o_accuracy is None:
        return None
    if val_accuracy < accuracy_floor:
        return -1.0
    return val_accuracy + non_o_weight * val_non_o_accuracy


def _mark_best_result(results):
    best_idx, best_score = None, None
    for i, result in enumerate(results):
        result["is_best"] = False
        score = result.get("selection_score")
        if score is None or score < 0:
            continue
        if best_score is None or score > best_score:
            best_idx, best_score = i, score
    if best_idx is not None:
        results[best_idx]["is_best"] = True


def _validate_heads(d_model: int, n_heads: int):
    if d_model % n_heads != 0:
        raise click.ClickException("d_model debe ser divisible entre n_heads.")
