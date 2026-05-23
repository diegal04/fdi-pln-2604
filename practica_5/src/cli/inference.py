"""Comandos de inferencia: generate y entities."""

from __future__ import annotations

from pathlib import Path

import click

from .utils import (
    DEFAULT_LM_WEIGHTS,
    DEFAULT_NER_WEIGHTS,
    DEFAULT_TOKENIZER,
    _device,
    _load_state_dict,
    _load_weights,
    _require_file,
    _validate_heads,
)


@click.command("generate")
@click.argument("prompt", required=False, default="")
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=DEFAULT_TOKENIZER,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Tokenizador BPE guardado.",
)
@click.option(
    "--weights",
    "weights_path",
    default=DEFAULT_LM_WEIGHTS,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Checkpoint del modelo LM.",
)
@click.option(
    "--max-new-tokens",
    default=200,
    show_default=True,
    type=int,
    help="Numero maximo de tokens nuevos a generar.",
)
@click.option(
    "--temperature",
    default=1.0,
    show_default=True,
    type=float,
    help="Temperatura de sampleo. Valores bajos = mas determinista.",
)
@click.option(
    "--top-k",
    default=50,
    show_default=True,
    type=int,
    help="Top-k sampling. 0 = desactivado.",
)
@click.option(
    "--d-model", default=128, show_default=True, type=int, help="Dimension interna."
)
@click.option(
    "--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas."
)
@click.option(
    "--n-layers", default=4, show_default=True, type=int, help="Numero de bloques."
)
@click.option(
    "--context-size", default=128, show_default=True, type=int, help="Longitud maxima."
)
@click.option("--expansion", default=4, show_default=True, type=int, help="Factor FFN.")
@click.option(
    "--dropout",
    default=0.0,
    show_default=True,
    type=float,
    help="Dropout (0 para inferencia).",
)
def generate(
    prompt: str,
    tokenizer_path: Path,
    weights_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    context_size: int,
    expansion: int,
    dropout: float,
):
    """Genera texto a partir de un PROMPT con el LM causal.

    Si PROMPT esta vacio, genera desde el principio del contexto.

    \b
    Ejemplo:
      fdi-pln-2604-p5 generate "She walked into"
      fdi-pln-2604-p5 generate "Once upon" --max-new-tokens 200 --temperature 0.7
    """
    import torch

    from models.causal_lm import CausalLLM
    from tokenizer import BPETokenizer

    _require_file(tokenizer_path, "el tokenizador")
    _require_file(weights_path, "el checkpoint LM")
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
    _load_weights(model, _load_state_dict(torch, weights_path, device), strict=True)
    model.eval()

    prompt_ids = tokenizer.encode(prompt) if prompt else []
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
        )
    output_ids = generated_ids
    click.echo("".join(tokenizer.decode(prompt_ids + generated_ids)))


@click.command("entities")
@click.argument("text")  # cadena literal O ruta de fichero
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=DEFAULT_TOKENIZER,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Tokenizador BPE guardado.",
)
@click.option(
    "--weights",
    "weights_path",
    default=DEFAULT_NER_WEIGHTS,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Checkpoint del modelo NER.",
)
@click.option(
    "--d-model", default=128, show_default=True, type=int, help="Dimension interna."
)
@click.option(
    "--n-heads", default=4, show_default=True, type=int, help="Numero de cabezas."
)
@click.option(
    "--n-layers", default=4, show_default=True, type=int, help="Numero de bloques."
)
@click.option(
    "--context-size", default=128, show_default=True, type=int, help="Longitud maxima."
)
@click.option("--expansion", default=4, show_default=True, type=int, help="Factor FFN.")
@click.option(
    "--dropout",
    default=0.0,
    show_default=True,
    type=float,
    help="Dropout (0 para inferencia).",
)
@click.option(
    "--add-spaces/--no-add-spaces",
    default=False,
    show_default=True,
    help="Antepone espacio a cada word token antes de tokenizar.",
)
def entities(
    text: str,
    tokenizer_path: Path,
    weights_path: Path,
    d_model: int,
    n_heads: int,
    n_layers: int,
    context_size: int,
    expansion: int,
    dropout: float,
    add_spaces: bool,
):
    """Detecta entidades nombradas en TEXT o en un fichero.

    TEXT puede ser una cadena de texto directa o la ruta a un fichero .txt.
    Si es un fichero, se procesa parrafo a parrafo.

    \b
    Ejemplo:
      fdi-pln-2604-p5 entities "Alice met the White Rabbit in Wonderland."
      fdi-pln-2604-p5 entities datos/corpus_pretrain/alice_in_wonderland.txt
    """
    import torch

    from ner import NERLLM, NUM_LABELS, split_text_tokens
    from tokenizer import BPETokenizer

    _require_file(tokenizer_path, "el tokenizador")
    _require_file(weights_path, "el checkpoint NER")
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
    _load_weights(model, _load_state_dict(torch, weights_path, device), strict=True)

    # Soporte para ficheros de texto
    raw_path = Path(text)
    if raw_path.is_file():
        raw_text = raw_path.read_text(encoding="utf-8")
    else:
        raw_text = text

    # Dividir en parrafos para no exceder context_size en un solo pasada
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [raw_text]

    all_spans: list[tuple[str, str]] = []
    for para in paragraphs:
        words = split_text_tokens(para.lower())
        if not words:
            continue
        spans = model.predict_entities(words, tokenizer, add_spaces=add_spaces)
        all_spans.extend(spans)

    if not all_spans:
        click.echo("No se encontraron entidades.")
    else:
        for entity_text, entity_type in all_spans:
            click.echo(f"{entity_type:4s}  {entity_text}")
