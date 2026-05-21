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


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
}

PROJECT_DIR = Path.cwd()
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CORPUS = Path("pre-entrega_2601") / "corpus_original"
DEFAULT_ANNOTATIONS = Path("pre-entrega_2601") / "merged.json"
DEFAULT_TOKENIZER = Path("tokenizer.json")
DEFAULT_LM_WEIGHTS = Path("p5_causal_2604.pth")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")
DEFAULT_LM_HISTORY = DEFAULT_EXPERIMENTS_DIR / "lm_exp2" / "history.json"
DEFAULT_NER_HISTORY = DEFAULT_EXPERIMENTS_DIR / "ner_final" / "history.json"
DEFAULT_NER_METRICS = DEFAULT_EXPERIMENTS_DIR / "ner_final" / "metrics"

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


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
def cli():
    """CLI de la practica 5: LM causal y NER.

    \b
    Comandos principales:
      train tokenizer   Entrena el tokenizador BPE.
      train lm          Preentrena el modelo causal.
      train ner         Ajusta el modelo NER desde el LM.
      generate          Genera texto desde un prompt.
      entities          Extrae entidades de un fichero de texto.
      grid-search       Explora hiperparametros.

    \b
    Ejemplos rapidos:
      fdi-pln-2604-p5 generate "alice looked around" --max-tokens 40
      fdi-pln-2604-p5 entities fragmento.txt --json-output
      fdi-pln-2604-p5 train ner pre-entrega_2601/merged.json --tokenizer tokenizer.json

    Usa COMMAND --help para ver parametros de cada comando.
    """


@cli.group(no_args_is_help=True)
def train():
    """Entrena tokenizador, LM causal o NER.

    \b
    Subcomandos:
      tokenizer  Construye tokenizer.json desde un corpus.
      lm         Entrena p5_causal_2604.pth.
      ner        Entrena p5_ner_2604.pth usando pesos LM.
    """


@train.command("tokenizer", no_args_is_help=True)
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--vocab-size",
    default=500,
    show_default=True,
    type=int,
    help="Tamano maximo del vocabulario BPE.",
)
@click.option(
    "--out",
    default=Path("tokenizer.json"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el tokenizador entrenado.",
)
def train_tokenizer(corpus: Path, vocab_size: int, out: Path):
    """Entrena y guarda el tokenizador BPE.

    CORPUS puede ser un fichero de texto o un directorio con .txt.

    \b
    Ejemplo:
      fdi-pln-2604-p5 train tokenizer corpus_pretrain --vocab-size 500 --out tokenizer.json
    """
    from tokenizer import BPETokenizer

    tokenizer = BPETokenizer(_read_corpus(corpus), vocab_size=vocab_size)
    tokenizer.save(out)
    click.echo(f"Tokenizador guardado en {out}")
    click.echo(tokenizer)


@train.command("lm", no_args_is_help=True)
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
    default=Path("p5_causal_2604.pth"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el checkpoint LM.",
)
@click.option(
    "--history-out",
    default=DEFAULT_LM_HISTORY,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el historial de entrenamiento.",
)
@click.option(
    "--context-size",
    default=128,
    show_default=True,
    type=int,
    help="Longitud de contexto usada para entrenar el LM.",
)
@click.option("--epochs", default=5, show_default=True, type=int, help="Numero de epocas.")
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    type=int,
    help="Tamano de batch.",
)
@click.option("--lr", default=3e-4, show_default=True, type=float, help="Learning rate.")
@click.option("--d-model", default=128, show_default=True, type=int, help="Dimension interna.")
@click.option("--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas.")
@click.option("--n-layers", default=2, show_default=True, type=int, help="Numero de bloques.")
@click.option("--expansion", default=4, show_default=True, type=int, help="Factor FFN.")
@click.option("--dropout", default=0.2, show_default=True, type=float, help="Dropout.")
@click.option(
    "--warmup-steps",
    default=100,
    show_default=True,
    type=int,
    help="Pasos de warmup lineal del LR scheduler.",
)
@click.option(
    "--weight-decay",
    default=0.1,
    show_default=True,
    type=float,
    help="Regularizacion L2 del optimizador AdamW.",
)
def train_lm(
    corpus: Path,
    tokenizer_path: Path,
    out: Path,
    history_out: Path,
    context_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    expansion: int,
    dropout: float,
    warmup_steps: int,
    weight_decay: float,
):
    """Entrena el modelo causal con un tokenizador ya fijado.

    CORPUS puede ser un fichero o un directorio con .txt. El checkpoint se
    guarda por defecto como p5_causal_2604.pth.

    \b
    Ejemplo:
      fdi-pln-2604-p5 train lm corpus_pretrain --tokenizer tokenizer.json --epochs 15
    """
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
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    _save_json(history, history_out)
    click.echo(f"Modelo guardado en {out}")
    click.echo(f"Historial guardado en {history_out}")


@train.command("ner", no_args_is_help=True)
@click.argument("annotations", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tokenizer",
    "tokenizer_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Tokenizador BPE usado durante el pretraining.",
)
@click.option(
    "--lm-weights",
    default=Path("p5_causal_2604.pth"),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint LM usado para inicializar el backbone.",
)
@click.option(
    "--out",
    default=Path("p5_ner_2604.pth"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el checkpoint NER.",
)
@click.option("--context-size", default=128, show_default=True, type=int, help="Longitud maxima.")
@click.option("--epochs", default=5, show_default=True, type=int, help="Numero de epocas.")
@click.option("--batch-size", default=32, show_default=True, type=int, help="Tamano de batch.")
@click.option("--lr", default=3e-4, show_default=True, type=float, help="Learning rate.")
@click.option("--d-model", default=128, show_default=True, type=int, help="Dimension interna.")
@click.option("--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas.")
@click.option("--n-layers", default=2, show_default=True, type=int, help="Numero de bloques.")
@click.option("--expansion", default=4, show_default=True, type=int, help="Factor FFN.")
@click.option("--dropout", default=0.2, show_default=True, type=float, help="Dropout.")
@click.option(
    "--entity-loss-weight",
    default=10.0,
    show_default=True,
    type=float,
    help="Multiplicador de loss para pi, pc, li y lc. La clase o pesa 1.",
)
@click.option(
    "--metrics-dir",
    default=DEFAULT_NER_METRICS,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directorio donde guardar graficos y metricas NER.",
)
@click.option(
    "--history-out",
    default=DEFAULT_NER_HISTORY,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el historial de entrenamiento.",
)
@click.option(
    "--warmup-steps",
    default=50,
    show_default=True,
    type=int,
    help="Pasos de warmup lineal del LR scheduler NER.",
)
@click.option(
    "--weight-decay",
    default=0.1,
    show_default=True,
    type=float,
    help="Regularizacion L2 del optimizador AdamW.",
)
@click.option(
    "--freeze-epochs",
    default=0,
    show_default=True,
    type=int,
    help="Epochs con backbone congelado. 0 = sin congelar.",
)
@click.option(
    "--location-weight-multiplier",
    default=1.0,
    show_default=True,
    type=float,
    help="Multiplicador extra de loss para li/lc respecto a pi/pc.",
)
@click.option(
    "--continuation-weight-multiplier",
    default=1.0,
    show_default=True,
    type=float,
    help="Multiplicador extra de loss para pc/lc respecto a pi/li.",
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
    history_out: Path,
    warmup_steps: int,
    weight_decay: float,
    freeze_epochs: int,
    location_weight_multiplier: float,
    continuation_weight_multiplier: float,
):
    """Fine-tuning NER desde pesos preentrenados.

    ANNOTATIONS debe apuntar a un merged.json con tokens y labels BIO.

    \b
    Ejemplo:
      fdi-pln-2604-p5 train ner pre-entrega_2601/merged.json --tokenizer tokenizer.json
      fdi-pln-2604-p5 train ner pre-entrega_2601/merged.json --tokenizer tokenizer.json --epochs 50 --lr 0.001
    """
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
        location_weight_multiplier=location_weight_multiplier,
        continuation_weight_multiplier=continuation_weight_multiplier,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        freeze_epochs=freeze_epochs,
        metrics_dir=metrics_dir,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    _save_json(history, history_out)
    click.echo(f"Modelo NER guardado en {out}")
    click.echo(f"Historial guardado en {history_out}")
    click.echo(f"Metricas NER guardadas en {metrics_dir}")


@cli.group("grid-search", no_args_is_help=True)
def grid_search():
    """Explora hiperparametros de tokenizador, LM y NER.

    \b
    Subcomandos:
      tokenizer  Entrena varios tokenizadores con vocab_size distinto.
      lm         Lanza varias configuraciones de pretraining LM.
      ner        Lanza el grid search NER principal.
    """


@grid_search.command("tokenizer", no_args_is_help=True)
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
    help="Lista separada por comas de tamanos de vocabulario.",
)
@click.option(
    "--out-dir",
    default=PROJECT_DIR / DEFAULT_EXPERIMENTS_DIR / "grid_tokenizers",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directorio donde guardar tokenizadores y results.json.",
)
def grid_search_tokenizer(corpus: Path, vocab_sizes: str, out_dir: Path):
    """Entrena varios tokenizadores BPE con distintos vocab_size.

    \b
    Ejemplo:
      fdi-pln-2604-p5 grid-search tokenizer corpus_pretrain --vocab-sizes 200,300,500
    """
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


@grid_search.command("lm", no_args_is_help=True)
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
    help="Tokenizador BPE que usaran todos los runs.",
)
@click.option(
    "--out-dir",
    default=PROJECT_DIR / DEFAULT_EXPERIMENTS_DIR / "grid_lm",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directorio donde guardar checkpoints y results.json.",
)
@click.option(
    "--epochs",
    "epochs_raw",
    default=GRID_LM_DEFAULTS["epochs"],
    show_default=True,
    help="Lista separada por comas de epocas.",
)
@click.option(
    "--batch-sizes",
    "batch_raw",
    default=GRID_LM_DEFAULTS["batch_sizes"],
    show_default=True,
    help="Lista separada por comas de batch sizes.",
)
@click.option(
    "--lrs",
    "lr_raw",
    default=GRID_LM_DEFAULTS["lrs"],
    show_default=True,
    help="Lista separada por comas de learning rates.",
)
@click.option(
    "--d-models",
    "d_model_raw",
    default=GRID_LM_DEFAULTS["d_models"],
    show_default=True,
    help="Lista separada por comas de d_model.",
)
@click.option(
    "--n-heads", "n_heads_raw", default=GRID_LM_DEFAULTS["n_heads"], show_default=True
)
@click.option(
    "--n-layers",
    "n_layers_raw",
    default=GRID_LM_DEFAULTS["n_layers"],
    show_default=True,
    help="Lista separada por comas de n_layers.",
)
@click.option(
    "--dropouts",
    "dropout_raw",
    default=GRID_LM_DEFAULTS["dropouts"],
    show_default=True,
    help="Lista separada por comas de dropout.",
)
@click.option(
    "--context-size",
    default=GRID_LM_DEFAULTS["context_size"],
    show_default=True,
    type=int,
    help="Longitud de contexto comun a todos los runs.",
)
@click.option(
    "--expansion",
    default=GRID_LM_DEFAULTS["expansion"],
    show_default=True,
    type=int,
    help="Factor FFN comun a todos los runs.",
)
@click.option(
    "--warmup-steps",
    default=GRID_LM_DEFAULTS["warmup_steps"],
    show_default=True,
    type=int,
    help="Warmup comun a todos los runs.",
)
@click.option(
    "--weight-decay",
    default=GRID_LM_DEFAULTS["weight_decay"],
    show_default=True,
    type=float,
    help="Weight decay comun a todos los runs.",
)
@click.option("--max-runs", default=None, type=int, help="Corta el grid tras N runs.")
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
    warmup_steps: int,
    weight_decay: float,
    max_runs: int | None,
):
    """Lanza varias configuraciones de pretraining LM y guarda resultados.

    \b
    Ejemplo:
      fdi-pln-2604-p5 grid-search lm corpus_pretrain --tokenizer tokenizer.json
      fdi-pln-2604-p5 grid-search lm corpus_pretrain --epochs 3,5 --batch-sizes 32,64
    """
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
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
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


@grid_search.command("ner", no_args_is_help=True)
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
    help="Tokenizador BPE usado por todos los runs.",
)
@click.option(
    "--lm-weights",
    default=DEFAULT_LM_WEIGHTS,
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint LM para inicializar el backbone.",
)
@click.option(
    "--out-dir",
    default=PROJECT_DIR / DEFAULT_EXPERIMENTS_DIR / "grid_ner",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directorio donde guardar resultados, mejor modelo y metricas.",
)
@click.option(
    "--epochs",
    "epochs_raw",
    default=GRID_NER_DEFAULTS["epochs"],
    show_default=True,
    help="Lista separada por comas de epocas.",
)
@click.option(
    "--batch-sizes",
    "batch_raw",
    default=GRID_NER_DEFAULTS["batch_sizes"],
    show_default=True,
    help="Lista separada por comas de batch sizes.",
)
@click.option(
    "--lrs",
    "lr_raw",
    default=GRID_NER_DEFAULTS["lrs"],
    show_default=True,
    help="Lista separada por comas de learning rates.",
)
@click.option(
    "--d-model",
    default=GRID_NER_DEFAULTS["d_model"],
    show_default=True,
    type=int,
    help="Dimension interna fija; debe coincidir con el LM.",
)
@click.option(
    "--n-heads",
    default=GRID_NER_DEFAULTS["n_heads"],
    show_default=True,
    type=int,
    help="Numero de cabezas fijo; debe coincidir con el LM.",
)
@click.option(
    "--n-layers",
    default=GRID_NER_DEFAULTS["n_layers"],
    show_default=True,
    type=int,
    help="Numero de capas fijo; debe coincidir con el LM.",
)
@click.option("--dropout", default=GRID_NER_DEFAULTS["dropout"], show_default=True, type=float, help="Dropout fijo.")
@click.option(
    "--entity-loss-weights",
    "entity_loss_weight_raw",
    default=GRID_NER_DEFAULTS["entity_loss_weights"],
    show_default=True,
    help="Lista separada por comas de pesos para clases de entidad.",
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
    help="Longitud maxima de contexto.",
)
@click.option(
    "--expansion",
    default=GRID_NER_DEFAULTS["expansion"],
    show_default=True,
    type=int,
    help="Factor FFN.",
)
@click.option(
    "--warmup-steps",
    default=GRID_NER_DEFAULTS["warmup_steps"],
    show_default=True,
    type=int,
    help="Warmup del scheduler.",
)
@click.option(
    "--weight-decay",
    default=GRID_NER_DEFAULTS["weight_decay"],
    show_default=True,
    type=float,
    help="Weight decay del optimizador.",
)
@click.option(
    "--continuation-weight-multiplier",
    default=GRID_NER_DEFAULTS["continuation_weight_multiplier"],
    show_default=True,
    type=float,
    help="Multiplicador de loss para pc/lc.",
)
@click.option("--max-runs", default=None, type=int, help="Corta el grid tras N runs.")
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
    warmup_steps: int,
    weight_decay: float,
    continuation_weight_multiplier: float,
    max_runs: int | None,
):
    """Lanza varias configuraciones de fine-tuning NER y guarda resultados.

    \b
    Ejemplo:
      fdi-pln-2604-p5 grid-search ner pre-entrega_2601/merged.json
      fdi-pln-2604-p5 grid-search ner pre-entrega_2601/merged.json --max-runs 3
    """
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
            "continuation_weight_multiplier": continuation_weight_multiplier,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
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
            continuation_weight_multiplier=continuation_weight_multiplier,
            selection_accuracy_floor=selection_accuracy_floor,
            selection_non_o_weight=selection_non_o_weight,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
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
        continuation_weight_multiplier=best_config["continuation_weight_multiplier"],
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


@cli.command(no_args_is_help=True)
@click.argument("prompt")
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=Path("tokenizer.json"),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Tokenizador BPE guardado.",
)
@click.option(
    "--weights",
    default=Path("p5_causal_2604.pth"),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint LM a cargar.",
)
@click.option("--context-size", default=128, show_default=True, type=int, help="Contexto del modelo cargado.")
@click.option("--d-model", default=128, show_default=True, type=int, help="Dimension interna del modelo.")
@click.option("--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas.")
@click.option("--n-layers", default=4, show_default=True, type=int, help="Numero de capas.")
@click.option("--expansion", default=4, show_default=True, type=int, help="Factor FFN.")
@click.option("--dropout", default=0.2, show_default=True, type=float, help="Dropout de la arquitectura.")
@click.option("--max-tokens", default=200, show_default=True, type=int, help="Tokens nuevos a generar.")
@click.option("--temperature", default=0.8, show_default=True, type=float, help="Temperatura de muestreo.")
@click.option("--top-k", default=None, type=int, help="Limita el muestreo a los K tokens mas probables.")
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
    """Genera texto desde un prompt.

    \b
    Ejemplos:
      fdi-pln-2604-p5 generate "alice looked around" --max-tokens 40
      fdi-pln-2604-p5 generate "the queen said" --temperature 0.7 --top-k 20
    """
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


@cli.command(no_args_is_help=True)
@click.argument("text_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=Path("tokenizer.json"),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Tokenizador BPE guardado.",
)
@click.option(
    "--weights",
    default=Path("p5_ner_2604.pth"),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint NER a cargar.",
)
@click.option("--context-size", default=128, show_default=True, type=int, help="Contexto del modelo cargado.")
@click.option("--d-model", default=128, show_default=True, type=int, help="Dimension interna del modelo.")
@click.option("--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas.")
@click.option("--n-layers", default=4, show_default=True, type=int, help="Numero de capas.")
@click.option("--expansion", default=4, show_default=True, type=int, help="Factor FFN.")
@click.option("--dropout", default=0.3, show_default=True, type=float, help="Dropout de la arquitectura.")
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
    """Lista entidades nombradas encontradas en un fichero de texto.

    TEXT_FILE es obligatorio porque el enunciado pide entrada por fichero.
    Para probar una frase directa puedes usar /dev/stdin en Linux.

    \b
    Ejemplos:
      fdi-pln-2604-p5 entities fragmento.txt
      fdi-pln-2604-p5 entities fragmento.txt --json-output
      printf '%s\\n' "alice spoke to the queen" | fdi-pln-2604-p5 entities /dev/stdin
    """
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


if __name__ == "__main__":
    cli()
