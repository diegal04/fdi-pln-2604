"""Punto de entrada CLI del agente autónomo de Butler.

Configura las opciones de línea de comandos mediante Click y lanza el agente.
"""

from __future__ import annotations

import sys

import urllib3
import click
from dynaconf import Dynaconf
from loguru import logger

from fdi_pln_p1 import (
    BUTLER_ADDRESS,
    DEFAULT_BUTLER_ADDRESS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PLAYER_NAME,
    ENV_BUTLER_ADDRESS,
    ENV_MODEL_NAME,
    ENV_PLAYER_NAME,
    MODO_MONOPUESTO,
    MODO_MULTIPUESTO,
    MODEL_NAME,
    PLAYER_NAME,
)
from fdi_pln_p1.agent import agente_autonomo, registrar_alias

# ---------------------------------------------------------------------------
# Configuración global de logging y warnings
# ---------------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger.remove()
logger.add(sys.stderr, level="INFO")


# ---------------------------------------------------------------------------
# CLI (Click)
# ---------------------------------------------------------------------------

@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Inicia el agente autonomo de Butler.\n\n"
        "\b\n"
        "Prioridad de configuracion:\n"
        "  1. Valor pasado por linea de comandos.\n"
        "  2. Variable de entorno asociada.\n"
        "  3. Valor por defecto del proyecto."
    ),
    epilog=(
        "\b\n"
        "Ejemplos:\n"
        "  uv run fdi-pln-entrega --help\n"
        "  uv run fdi-pln-entrega --name \"LOS ELEGIDOS\" --crear-alias\n"
        "  uv run fdi-pln-entrega --modo-puesto multipuesto\n"
        "  uv run fdi-pln-entrega --model qwen3-vl:4b --butler-address http://127.0.0.1:7719"
    ),
)
@click.option(
    "--name",
    "mi_nombre",
    default=DEFAULT_PLAYER_NAME,
    envvar=ENV_PLAYER_NAME,
    show_default=True,
    show_envvar=True,
    help="Alias con el que jugara el agente.",
)
@click.option(
    "--model",
    "modelo",
    default=DEFAULT_MODEL_NAME,
    envvar=ENV_MODEL_NAME,
    show_default=True,
    show_envvar=True,
    help="Modelo de Ollama que se usara para decidir las acciones.",
)
@click.option(
    "--butler-address",
    "url",
    default=DEFAULT_BUTLER_ADDRESS,
    envvar=ENV_BUTLER_ADDRESS,
    show_default=True,
    show_envvar=True,
    help="URL base del servidor Butler.",
)
@click.option(
    "--crear-alias/--no-crear-alias",
    default=False,
    help="Registra el alias antes de iniciar el agente.",
)
@click.option(
    "--modo-puesto",
    type=click.Choice([MODO_MONOPUESTO, MODO_MULTIPUESTO], case_sensitive=False),
    default=MODO_MONOPUESTO,
    show_default=True,
    help=(
        "Modo de uso de la API: monopuesto añade el parametro agente en cada peticion; "
        "multipuesto usa los endpoints globales sin ese parametro."
    ),
)
def main(mi_nombre, modelo, url, crear_alias, modo_puesto):
    """Configura y lanza el agente autónomo."""
    runtime_config = Dynaconf(environments=False)
    runtime_config.set("NAME", mi_nombre or PLAYER_NAME)
    runtime_config.set("MODEL", modelo or MODEL_NAME)
    runtime_config.set("BUTLER_ADDRESS", url or BUTLER_ADDRESS)

    mi_nombre = runtime_config.get("NAME")
    modelo = runtime_config.get("MODEL")
    url = runtime_config.get("BUTLER_ADDRESS")
    modo_puesto = (modo_puesto or MODO_MONOPUESTO).lower()

    if crear_alias:
        registrar_alias(mi_nombre=mi_nombre, url=url)
    agente_autonomo(
        mi_nombre=mi_nombre,
        url=url,
        modelo=modelo,
        modo_puesto=modo_puesto,
    )


if __name__ == "__main__":
    main()
