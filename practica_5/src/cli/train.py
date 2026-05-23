"""Comandos de entrenamiento: tokenizer, lm, ner."""

from __future__ import annotations

from pathlib import Path

import click

from .utils import (
    DEFAULT_LM_HISTORY,
    DEFAULT_LM_WEIGHTS,
    DEFAULT_NER_HISTORY,
    DEFAULT_NER_METRICS,
    DEFAULT_NER_WEIGHTS,
    DEFAULT_TOKENIZER,
    _device,
    _load_ner_data,
    _load_state_dict,
    _load_weights,
    _read_corpus,
    _save_json,
    _validate_heads,
)


@click.group(no_args_is_help=True)
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
    default=DEFAULT_TOKENIZER,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el tokenizador entrenado.",
)
def train_tokenizer(corpus: Path, vocab_size: int, out: Path):
    """Entrena y guarda el tokenizador BPE.

    CORPUS puede ser un fichero de texto o un directorio con .txt.

    \b
    Ejemplo:
      fdi-pln-2604-p5 train tokenizer datos/corpus_pretrain --vocab-size 500
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
    default=DEFAULT_LM_WEIGHTS,
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
@click.option(
    "--epochs", default=5, show_default=True, type=int, help="Numero de epocas."
)
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    type=int,
    help="Tamano de batch.",
)
@click.option(
    "--lr", default=3e-4, show_default=True, type=float, help="Learning rate."
)
@click.option(
    "--d-model", default=128, show_default=True, type=int, help="Dimension interna."
)
@click.option(
    "--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas."
)
@click.option(
    "--n-layers", default=2, show_default=True, type=int, help="Numero de bloques."
)
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
    guarda por defecto como resultados_finales/p5_causal_2604.pth.

    \b
    Ejemplo:
      fdi-pln-2604-p5 train lm datos/corpus_pretrain --tokenizer resultados_finales/tokenizer.json --epochs 15
    """
    import torch

    from models.causal_lm import CausalLLM
    from tokenizer import BPETokenizer
    from training.lm import train as train_model

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
    default=DEFAULT_LM_WEIGHTS,
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint LM usado para inicializar el backbone.",
)
@click.option(
    "--out",
    default=DEFAULT_NER_WEIGHTS,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Ruta donde guardar el checkpoint NER.",
)
@click.option(
    "--context-size", default=128, show_default=True, type=int, help="Longitud maxima."
)
@click.option(
    "--epochs", default=5, show_default=True, type=int, help="Numero de epocas."
)
@click.option(
    "--batch-size", default=32, show_default=True, type=int, help="Tamano de batch."
)
@click.option(
    "--lr", default=3e-4, show_default=True, type=float, help="Learning rate."
)
@click.option(
    "--d-model", default=128, show_default=True, type=int, help="Dimension interna."
)
@click.option(
    "--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas."
)
@click.option(
    "--n-layers", default=2, show_default=True, type=int, help="Numero de bloques."
)
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
      fdi-pln-2604-p5 train ner datos/pre-entrega_2601/merged.json --tokenizer resultados_finales/tokenizer.json
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
