"""Configuración compartida del paquete ``fdi_pln_p1``."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

# Carga .env como fuente principal de configuracion.
load_dotenv(find_dotenv(usecwd=True), override=True)

# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------
ENV_BUTLER_ADDRESS = "FDI_PLN__BUTLER_ADDRESS"
ENV_MODEL_NAME = "FDI_PLN__MODEL"
ENV_PLAYER_NAME = "FDI_PLN__NAME"

# ---------------------------------------------------------------------------
# Modos de operación de la API
# ---------------------------------------------------------------------------
MODO_MONOPUESTO = "monopuesto"
MODO_MULTIPUESTO = "multipuesto"

__all__ = [
    "ENV_BUTLER_ADDRESS",
    "ENV_MODEL_NAME",
    "ENV_PLAYER_NAME",
    "MODO_MONOPUESTO",
    "MODO_MULTIPUESTO",
]
