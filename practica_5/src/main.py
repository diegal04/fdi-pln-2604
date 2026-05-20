"""CLI de la practica 5.

Punto de entrada unico para entrenar el tokenizador, preentrenar el LLM,
hacer fine-tuning NER, generar texto y extraer entidades.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import click


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CORPUS = ROOT_DIR / "pre-entrega_2601" / "corpus_original"
DEFAULT_ANNOTATIONS = ROOT_DIR / "pre-entrega_2601" / "merged.json"
DEFAULT_TOKENIZER = ROOT_DIR / "tokenizer.json"
DEFAULT_LM_WEIGHTS = ROOT_DIR / "pesos_modelo.pth"

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
}
GRID_NER_DEFAULTS = {
    "epochs": "50",
    "batch_sizes": "8,16,32,64",
    "lrs": "0.001,0.0005,0.0003,0.0001,0.00005",
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.2,
    "entity_loss_weights": "3,5,7,10,15,20,25,30",
    "selection_accuracy_floor": 0.8,
    "selection_non_o_weight": 1.5,
    "context_size": 128,
    "expansion": 4,
}


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
    state = torch.load(path, map_location=device)
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


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Entrenamiento, generacion y NER."""


@cli.group()
def train():
    """Entrena partes del proyecto."""


@train.command("tokenizer")
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option("--vocab-size", default=300, show_default=True, type=int)
@click.option(
    "--out",
    default=ROOT_DIR / "tokenizer.json",
    show_default=True,
    type=click.Path(path_type=Path),
)
def train_tokenizer(corpus: Path, vocab_size: int, out: Path):
    """Entrena y guarda el tokenizador BPE."""
    from tokenizer import BPETokenizer

    tokenizer = BPETokenizer(_read_corpus(corpus), vocab_size=vocab_size)
    tokenizer.save(out)
    click.echo(f"Tokenizador guardado en {out}")
    click.echo(tokenizer)


@train.command("lm")
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Tokenizador BPE guardado con train tokenizer.",
)
@click.option(
    "--out",
    default=ROOT_DIR / "pesos_modelo.pth",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--context-size", default=128, show_default=True, type=int)
@click.option("--epochs", default=5, show_default=True, type=int)
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--lr", default=3e-4, show_default=True, type=float)
@click.option("--d-model", default=128, show_default=True, type=int)
@click.option("--n-heads", default=4, show_default=True, type=int)
@click.option("--n-layers", default=2, show_default=True, type=int)
@click.option("--expansion", default=4, show_default=True, type=int)
@click.option("--dropout", default=0.2, show_default=True, type=float)
def train_lm(
    corpus: Path,
    tokenizer_path: Path,
    out: Path,
    context_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    expansion: int,
    dropout: float,
):
    """Entrena el modelo causal con un tokenizador ya fijado."""
    import torch
    from causalLLM import CausalLLM
    from tokenizer import BPETokenizer
    from train import train as train_model

    _validate_heads(d_model, n_heads)
    device = _device(torch)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokens = tokenizer.encode(_read_corpus(corpus))
    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=context_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
    ).to(device)

    history = train_model(
        model,
        tokens,
        epochs=epochs,
        context_size=context_size,
        batch_size=batch_size,
        lr=lr,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    _save_json(history, out.with_suffix(".history.json"))
    click.echo(f"Modelo guardado en {out}")
    click.echo(f"Historial guardado en {out.with_suffix('.history.json')}")


@train.command("ner")
@click.argument("annotations", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--lm-weights",
    default=ROOT_DIR / "pesos_modelo.pth",
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--out",
    default=ROOT_DIR / "pesos_ner.pth",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--context-size", default=128, show_default=True, type=int)
@click.option("--epochs", default=5, show_default=True, type=int)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--lr", default=3e-4, show_default=True, type=float)
@click.option("--d-model", default=128, show_default=True, type=int)
@click.option("--n-heads", default=4, show_default=True, type=int)
@click.option("--n-layers", default=2, show_default=True, type=int)
@click.option("--expansion", default=4, show_default=True, type=int)
@click.option("--dropout", default=0.2, show_default=True, type=float)
@click.option(
    "--entity-loss-weight",
    default=10.0,
    show_default=True,
    type=float,
    help="Multiplicador de loss para pi, pc, li y lc. La clase o pesa 1.",
)
@click.option(
    "--metrics-dir",
    default=ROOT_DIR / "ner_metrics",
    show_default=True,
    type=click.Path(path_type=Path),
)
def train_ner(
    annotations: Path,
    tokenizer_path: Path,
    lm_weights: Path,
    out: Path,
    context_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    expansion: int,
    dropout: float,
    entity_loss_weight: float,
    metrics_dir: Path,
):
    """Fine-tuning NER desde pesos preentrenados."""
    import torch
    from ner import NERLLM, NUM_LABELS, train_ner as train_ner_model
    from tokenizer import BPETokenizer

    _validate_heads(d_model, n_heads)
    device = _device(torch)
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = NERLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=context_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
        num_labels=NUM_LABELS,
    ).to(device)

    _load_weights(model, _load_state_dict(torch, lm_weights, device), strict=False)
    history = train_ner_model(
        model,
        _load_ner_data(annotations),
        tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_len=context_size,
        add_spaces=False,
        entity_loss_weight=entity_loss_weight,
        metrics_dir=metrics_dir,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    _save_json(history, out.with_suffix(".history.json"))
    click.echo(f"Modelo NER guardado en {out}")
    click.echo(f"Historial guardado en {out.with_suffix('.history.json')}")
    click.echo(f"Metricas NER guardadas en {metrics_dir}")


@cli.group("grid-search")
def grid_search():
    """Explora hiperparametros de tokenizador, LM y NER."""


@grid_search.command("tokenizer")
@click.argument(
    "corpus",
    required=False,
    default=DEFAULT_CORPUS,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--vocab-sizes",
    default=GRID_TOKENIZER_DEFAULTS["vocab_sizes"],
    show_default=True,
)
@click.option(
    "--out-dir",
    default=ROOT_DIR / "grid_tokenizers",
    show_default=True,
    type=click.Path(path_type=Path),
)
def grid_search_tokenizer(corpus: Path, vocab_sizes: str, out_dir: Path):
    """Entrena varios tokenizadores BPE con distintos vocab_size."""
    from tokenizer import BPETokenizer

    text = _read_corpus(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for vocab_size in _parse_ints(vocab_sizes):
        tokenizer = BPETokenizer(text, vocab_size=vocab_size)
        out = out_dir / f"tokenizer_vocab_{vocab_size}.json"
        tokenizer.save(out)
        results.append(
            {
                "requested_vocab_size": vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "path": str(out),
            }
        )
        click.echo(f"Tokenizador vocab_size={vocab_size} guardado en {out}")
    _save_json(results, out_dir / "results.json")


@grid_search.command("lm")
@click.argument(
    "corpus",
    required=False,
    default=DEFAULT_CORPUS,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=DEFAULT_TOKENIZER,
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--out-dir",
    default=ROOT_DIR / "grid_lm",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--epochs", "epochs_raw", default=GRID_LM_DEFAULTS["epochs"], show_default=True
)
@click.option(
    "--batch-sizes",
    "batch_raw",
    default=GRID_LM_DEFAULTS["batch_sizes"],
    show_default=True,
)
@click.option("--lrs", "lr_raw", default=GRID_LM_DEFAULTS["lrs"], show_default=True)
@click.option(
    "--d-models",
    "d_model_raw",
    default=GRID_LM_DEFAULTS["d_models"],
    show_default=True,
)
@click.option(
    "--n-heads", "n_heads_raw", default=GRID_LM_DEFAULTS["n_heads"], show_default=True
)
@click.option(
    "--n-layers",
    "n_layers_raw",
    default=GRID_LM_DEFAULTS["n_layers"],
    show_default=True,
)
@click.option(
    "--dropouts",
    "dropout_raw",
    default=GRID_LM_DEFAULTS["dropouts"],
    show_default=True,
)
@click.option(
    "--context-size",
    default=GRID_LM_DEFAULTS["context_size"],
    show_default=True,
    type=int,
)
@click.option(
    "--expansion", default=GRID_LM_DEFAULTS["expansion"], show_default=True, type=int
)
@click.option("--max-runs", default=None, type=int)
def grid_search_lm(
    corpus: Path,
    tokenizer_path: Path,
    out_dir: Path,
    epochs_raw: str,
    batch_raw: str,
    lr_raw: str,
    d_model_raw: str,
    n_heads_raw: str,
    n_layers_raw: str,
    dropout_raw: str,
    context_size: int,
    expansion: int,
    max_runs: int | None,
):
    """Lanza varias configuraciones de pretraining LM y guarda resultados."""
    import torch
    from causalLLM import CausalLLM
    from tokenizer import BPETokenizer
    from train import train as train_model

    device = _device(torch)
    _require_file(tokenizer_path, "el tokenizador")
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokens = tokenizer.encode(_read_corpus(corpus))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    configs = product(
        _parse_ints(epochs_raw),
        _parse_ints(batch_raw),
        _parse_floats(lr_raw),
        _parse_ints(d_model_raw),
        _parse_ints(n_heads_raw),
        _parse_ints(n_layers_raw),
        _parse_floats(dropout_raw),
    )
    for run_id, (epochs, batch_size, lr, d_model, n_heads, n_layers, dropout) in enumerate(
        configs, start=1
    ):
        if max_runs is not None and run_id > max_runs:
            break

        config = {
            "run": run_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "context_size": context_size,
            "expansion": expansion,
        }
        if d_model % n_heads != 0:
            results.append({"config": config, "status": "skipped_incompatible_heads"})
            _save_json(results, out_dir / "results.json")
            continue

        click.echo(f"[LM {run_id}] {config}")
        model = CausalLLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=context_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            expansion=expansion,
            dropout=dropout,
        ).to(device)
        history = train_model(
            model,
            tokens,
            epochs=epochs,
            context_size=context_size,
            batch_size=batch_size,
            lr=lr,
        )
        model_path = out_dir / f"lm_run_{run_id:03d}.pth"
        torch.save(model.state_dict(), model_path)
        results.append(
            {
                "config": config,
                "model_path": str(model_path),
                "history": history,
                "final_val_loss": _final_val_loss(history),
            }
        )
        _save_json(results, out_dir / "results.json")

    click.echo(f"Resultados guardados en {out_dir / 'results.json'}")


@grid_search.command("ner")
@click.argument(
    "annotations",
    required=False,
    default=DEFAULT_ANNOTATIONS,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=DEFAULT_TOKENIZER,
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--lm-weights",
    default=DEFAULT_LM_WEIGHTS,
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--out-dir",
    default=ROOT_DIR / "grid_ner",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--epochs", "epochs_raw", default=GRID_NER_DEFAULTS["epochs"], show_default=True
)
@click.option(
    "--batch-sizes",
    "batch_raw",
    default=GRID_NER_DEFAULTS["batch_sizes"],
    show_default=True,
)
@click.option("--lrs", "lr_raw", default=GRID_NER_DEFAULTS["lrs"], show_default=True)
@click.option("--d-model", default=GRID_NER_DEFAULTS["d_model"], show_default=True, type=int)
@click.option("--n-heads", default=GRID_NER_DEFAULTS["n_heads"], show_default=True, type=int)
@click.option("--n-layers", default=GRID_NER_DEFAULTS["n_layers"], show_default=True, type=int)
@click.option("--dropout", default=GRID_NER_DEFAULTS["dropout"], show_default=True, type=float)
@click.option(
    "--entity-loss-weights",
    "entity_loss_weight_raw",
    default=GRID_NER_DEFAULTS["entity_loss_weights"],
    show_default=True,
)
@click.option(
    "--selection-accuracy-floor",
    default=GRID_NER_DEFAULTS["selection_accuracy_floor"],
    show_default=True,
    type=float,
    help="Descarta runs cuyo val_accuracy quede por debajo de este umbral.",
)
@click.option(
    "--selection-non-o-weight",
    default=GRID_NER_DEFAULTS["selection_non_o_weight"],
    show_default=True,
    type=float,
    help="Peso de val_non_o_accuracy en el score de seleccion.",
)
@click.option(
    "--context-size",
    default=GRID_NER_DEFAULTS["context_size"],
    show_default=True,
    type=int,
)
@click.option(
    "--expansion", default=GRID_NER_DEFAULTS["expansion"], show_default=True, type=int
)
@click.option("--max-runs", default=None, type=int)
def grid_search_ner(
    annotations: Path,
    tokenizer_path: Path,
    lm_weights: Path,
    out_dir: Path,
    epochs_raw: str,
    batch_raw: str,
    lr_raw: str,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    entity_loss_weight_raw: str,
    selection_accuracy_floor: float,
    selection_non_o_weight: float,
    context_size: int,
    expansion: int,
    max_runs: int | None,
):
    """Lanza varias configuraciones de fine-tuning NER y guarda resultados."""
    import torch
    from ner import (
        NERLLM,
        NUM_LABELS,
        save_ner_metrics,
        train_ner as train_ner_model,
    )
    from tokenizer import BPETokenizer

    device = _device(torch)
    _require_file(tokenizer_path, "el tokenizador")
    tokenizer = BPETokenizer.load(tokenizer_path)
    ner_data = _load_ner_data(annotations)
    lm_state = _load_state_dict(torch, lm_weights, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best_score = None
    best_model_state = None
    best_history = None
    best_config = None
    _validate_heads(d_model, n_heads)

    configs = product(
        _parse_ints(epochs_raw),
        _parse_ints(batch_raw),
        _parse_floats(lr_raw),
        _parse_floats(entity_loss_weight_raw),
    )
    for run_id, (
        epochs,
        batch_size,
        lr,
        entity_loss_weight,
    ) in enumerate(configs, start=1):
        if max_runs is not None and run_id > max_runs:
            break

        config = {
            "run": run_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "architecture_from_lm_checkpoint": str(lm_weights),
            "entity_loss_weight": entity_loss_weight,
            "selection_accuracy_floor": selection_accuracy_floor,
            "selection_non_o_weight": selection_non_o_weight,
            "context_size": context_size,
            "expansion": expansion,
        }

        click.echo(f"[NER {run_id}] {config}")
        model = NERLLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=context_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            expansion=expansion,
            dropout=dropout,
            num_labels=NUM_LABELS,
        ).to(device)
        _load_weights(model, lm_state, strict=False)
        history = train_ner_model(
            model,
            ner_data,
            tokenizer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_len=context_size,
            add_spaces=False,
            entity_loss_weight=entity_loss_weight,
            selection_accuracy_floor=selection_accuracy_floor,
            selection_non_o_weight=selection_non_o_weight,
            metrics_dir=None,
        )
        selection_score = _ner_selection_score(
            history,
            accuracy_floor=selection_accuracy_floor,
            non_o_weight=selection_non_o_weight,
        )
        result = {
            "config": config,
            "selection_score": selection_score,
            "selection_formula": (
                "val_accuracy + "
                f"{selection_non_o_weight} * val_non_o_accuracy"
            ),
            "selection_accuracy_floor": selection_accuracy_floor,
            "final_val_loss": _final_val_loss(history),
            "final_val_accuracy": _final_val_accuracy(history),
            "best_epoch": _best_metric(history, "epoch"),
            "best_val_loss": _best_metric(history, "val_loss"),
            "best_val_accuracy": _best_metric(history, "val_accuracy"),
            "best_val_non_o_accuracy": _best_metric(
                history,
                "val_non_o_accuracy",
            ),
            "best_val_macro_entity_f1": _best_metric(
                history,
                "val_macro_entity_f1",
            ),
            "final_train_non_o_accuracy": _final_metric(
                history,
                "train_non_o_accuracy",
            ),
            "final_val_non_o_accuracy": _final_metric(
                history,
                "val_non_o_accuracy",
            ),
            "final_val_entity_accuracy": _final_metric(
                history,
                "val_entity_accuracy",
            ),
            "final_val_macro_entity_f1": _final_metric(
                history,
                "val_macro_entity_f1",
            ),
        }
        results.append(result)
        if selection_score is not None and selection_score >= 0:
            if best_score is None or selection_score > best_score:
                best_score = selection_score
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_history = history
                best_config = config
        _mark_best_result(results)
        _save_json(results, out_dir / "results.json")

    if best_model_state is None:
        click.echo("No hay ningun modelo NER valido segun el score configurado.")
        click.echo(f"Resultados guardados en {out_dir / 'results.json'}")
        return

    best_model_path = out_dir / "best_ner.pth"
    best_metrics_dir = out_dir / "best_ner_metrics"
    torch.save(best_model_state, best_model_path)
    save_ner_metrics(
        best_history,
        entity_loss_weight=best_config["entity_loss_weight"],
        metrics_dir=best_metrics_dir,
    )
    for result in results:
        if result.get("is_best"):
            result["model_path"] = str(best_model_path)
            result["metrics_dir"] = str(best_metrics_dir)
            best_result = dict(result)
            best_result["history"] = best_history
            _save_json(best_result, out_dir / "best_result.json")
            break
    _save_json(results, out_dir / "results.json")
    click.echo(f"Mejor modelo NER guardado en {best_model_path}")
    click.echo(f"Metricas del mejor modelo guardadas en {best_metrics_dir}")
    click.echo(f"Resultados guardados en {out_dir / 'results.json'}")


@cli.command()
@click.argument("prompt")
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--weights",
    default=ROOT_DIR / "pesos_modelo.pth",
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--context-size", default=128, show_default=True, type=int)
@click.option("--d-model", default=128, show_default=True, type=int)
@click.option("--n-heads", default=4, show_default=True, type=int)
@click.option("--n-layers", default=2, show_default=True, type=int)
@click.option("--expansion", default=4, show_default=True, type=int)
@click.option("--dropout", default=0.2, show_default=True, type=float)
@click.option("--max-tokens", default=200, show_default=True, type=int)
@click.option("--temperature", default=0.8, show_default=True, type=float)
@click.option("--top-k", default=None, type=int)
def generate(
    prompt: str,
    tokenizer_path: Path,
    weights: Path,
    context_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    expansion: int,
    dropout: float,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
):
    """Genera texto desde un prompt."""
    import torch
    from causalLLM import CausalLLM
    from tokenizer import BPETokenizer

    _validate_heads(d_model, n_heads)
    device = _device(torch)
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=context_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
    ).to(device)
    _load_weights(model, _load_state_dict(torch, weights, device), strict=True)
    pred = model.generate(
        tokenizer.encode(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    click.echo(prompt + "".join(tokenizer.decode(pred)))


@cli.command()
@click.argument("text_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--weights",
    default=ROOT_DIR / "pesos_ner.pth",
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--context-size", default=128, show_default=True, type=int)
@click.option("--d-model", default=128, show_default=True, type=int)
@click.option("--n-heads", default=4, show_default=True, type=int)
@click.option("--n-layers", default=2, show_default=True, type=int)
@click.option("--expansion", default=4, show_default=True, type=int)
@click.option("--dropout", default=0.2, show_default=True, type=float)
@click.option("--json-output", is_flag=True, help="Imprime entidades en JSON.")
def entities(
    text_file: Path,
    tokenizer_path: Path,
    weights: Path,
    context_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    expansion: int,
    dropout: float,
    json_output: bool,
):
    """Lista entidades nombradas encontradas en un fichero de texto."""
    import torch
    from ner import NERLLM, NUM_LABELS, split_text_tokens
    from tokenizer import BPETokenizer

    _validate_heads(d_model, n_heads)
    device = _device(torch)
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = NERLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=context_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
        num_labels=NUM_LABELS,
    ).to(device)
    _load_weights(model, _load_state_dict(torch, weights, device), strict=True)
    words = split_text_tokens(text_file.read_text(encoding="utf-8"))
    found = model.predict_entities(words, tokenizer, add_spaces=False)

    if json_output:
        click.echo(json.dumps([{"text": text, "type": kind} for text, kind in found]))
    else:
        for text, kind in found:
            click.echo(f"{kind}\t{text}")


cli.add_command(grid_search, "grid_search")


if __name__ == "__main__":
    cli()
