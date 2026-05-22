"""Grupo principal de la CLI de practica 5."""

import click

from .train import train
from .grid import grid_search
from .inference import generate, entities
from .utils import _device  # noqa: F401  (accesible para tests)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
def cli():
    """CLI de la practica 5: LM causal + NER con Transformer desde cero.

    \b
    Comandos principales:
      train        Entrena tokenizador, LM o NER.
      grid-search  Lanza busqueda de hiperparametros.
      generate     Genera texto con el LM.
      entities     Detecta entidades con el modelo NER.

    Ejecuta un subcomando con --help para ver sus opciones.
    """


cli.add_command(train)
cli.add_command(grid_search)
cli.add_command(generate)
cli.add_command(entities)
