"""Configuracion compartida del paquete fdi_pln_p1."""

from __future__ import annotations

import os
from typing import Iterable

DEFAULT_BUTLER_ADDRESS = "http://147.96.81.252:7719"
DEFAULT_MODEL_NAME = "qwen3-vl:4b"
DEFAULT_PLAYER_NAME = "LOS ELEGIDOS"

ENV_BUTLER_ADDRESS = "FDI_PLN__BUTLER_ADDRESS"
ENV_MODEL_NAME = "FDI_PLN__MODEL"
ENV_PLAYER_NAME = "FDI_PLN__NAME"


def _first_non_empty_env(var_names: Iterable[str], default: str) -> str:
    for var_name in var_names:
        value = os.getenv(var_name, "").strip()
        if value:
            return value
    return default


BUTLER_ADDRESS = _first_non_empty_env(
    [ENV_BUTLER_ADDRESS], DEFAULT_BUTLER_ADDRESS
)
MODEL_NAME = _first_non_empty_env(
    [ENV_MODEL_NAME, "FDI_PLN__BUTLER_MODEL"], DEFAULT_MODEL_NAME
)
PLAYER_NAME = _first_non_empty_env(
    [ENV_PLAYER_NAME, "FDI_PLN__PLAYER_NAME"], DEFAULT_PLAYER_NAME
)

__all__ = [
    "BUTLER_ADDRESS",
    "MODEL_NAME",
    "PLAYER_NAME",
    "ENV_BUTLER_ADDRESS",
    "ENV_MODEL_NAME",
    "ENV_PLAYER_NAME",
]
