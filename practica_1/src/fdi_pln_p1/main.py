"""Punto de entrada CLI del agente autónomo de Butler.

Configura las opciones de línea de comandos mediante Click y lanza el agente.
"""

from __future__ import annotations

import sys

import urllib3
import click
from loguru import logger

from fdi_pln_p1 import (
    ENV_BUTLER_ADDRESS,
    ENV_MODEL_NAME,
    ENV_PLAYER_NAME,
    MODO_MONOPUESTO,
    MODO_MULTIPUESTO,
)
from fdi_pln_p1.agent_config.agent import agente_autonomo
from fdi_pln_p1.api_utils import registrar_alias

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
        "  2. Valor definido en el archivo .env.\n"
        "  3. Variable de entorno ya exportada si .env no lo define."
    ),
    epilog=(
        "\b\n"
        "Ejemplos:\n"
        "  uv run fdi-pln-2604-p1 --help\n"
        '  uv run fdi-pln-2604-p1 --name "LOS ELEGIDOS" --crear-alias\n'
        "  uv run fdi-pln-2604-p1 --modo-puesto multipuesto\n"
        "  uv run fdi-pln-2604-p1 --model qwen3-vl:4b --butler-address http://127.0.0.1:7719"
    ),
)
@click.option(
    "--name",
    "mi_nombre",
    default=None,
    envvar=ENV_PLAYER_NAME,
    show_envvar=True,
    help="Alias con el que jugara el agente.",
)
@click.option(
    "--model",
    "modelo",
    default=None,
    envvar=ENV_MODEL_NAME,
    show_envvar=True,
    help="Modelo de Ollama que se usara para decidir las acciones.",
)
@click.option(
    "--butler-address",
    "url",
    default=None,
    envvar=ENV_BUTLER_ADDRESS,
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
    missing = []
    if not mi_nombre:
        missing.append(ENV_PLAYER_NAME)
    if not modelo:
        missing.append(ENV_MODEL_NAME)
    if not url:
        missing.append(ENV_BUTLER_ADDRESS)
    if missing:
        missing_vars = ", ".join(missing)
        raise click.UsageError(
            f"Faltan valores de configuracion. Define {missing_vars} en .env "
            "o pasalos por linea de comandos."
        )

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
