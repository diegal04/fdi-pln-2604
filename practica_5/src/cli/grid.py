"""Comandos de grid search: tokenizer, lm, ner."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import click

from .utils import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_CORPUS,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_LM_WEIGHTS,
    DEFAULT_TOKENIZER,
    GRID_LM_DEFAULTS,
    GRID_NER_DEFAULTS,
    GRID_TOKENIZER_DEFAULTS,
    PROJECT_DIR,
    _best_metric,
    _device,
    _final_metric,
    _final_val_accuracy,
    _final_val_loss,
    _load_ner_data,
    _load_state_dict,
    _load_weights,
    _mark_best_result,
    _ner_selection_score,
    _parse_floats,
    _parse_ints,
    _read_corpus,
    _require_file,
    _save_json,
    _validate_heads,
)


@click.group("grid-search", no_args_is_help=True)
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
      fdi-pln-2604-p5 grid-search tokenizer datos/corpus_pretrain --vocab-sizes 200,300,500
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
      fdi-pln-2604-p5 grid-search lm datos/corpus_pretrain --tokenizer resultados_finales/tokenizer.json
      fdi-pln-2604-p5 grid-search lm datos/corpus_pretrain --epochs 3,5 --batch-sizes 32,64
    """
    import torch

    from models.causal_lm import CausalLLM
    from tokenizer import BPETokenizer
    from training.lm import train as train_model

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
    help="Dimension fija; debe coincidir con el LM.",
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
      fdi-pln-2604-p5 grid-search ner datos/pre-entrega_2601/merged.json
      fdi-pln-2604-p5 grid-search ner datos/pre-entrega_2601/merged.json --max-runs 3
    """
    import torch

    from ner import NERLLM, NUM_LABELS, save_ner_metrics, train_ner as train_ner_model
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
            "best_val_non_o_accuracy": _best_metric(history, "val_non_o_accuracy"),
            "best_val_macro_entity_f1": _best_metric(history, "val_macro_entity_f1"),
            "final_train_non_o_accuracy": _final_metric(history, "train_non_o_accuracy"),
            "final_val_non_o_accuracy": _final_metric(history, "val_non_o_accuracy"),
            "final_val_entity_accuracy": _final_metric(history, "val_entity_accuracy"),
            "final_val_macro_entity_f1": _final_metric(history, "val_macro_entity_f1"),
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
