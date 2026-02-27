"""Utilidades pequeñas de normalización y parsing."""

from __future__ import annotations

from typing import Any


def to_int(valor: Any, default: int = 1) -> int:
    """Convierte *valor* a entero positivo (mínimo 1)."""
    try:
        return max(1, int(valor))
    except (TypeError, ValueError):
        return default


def extraer_destino(raw_dest: Any) -> str:
    """Extrae el alias del destinatario, aceptando ``str`` o ``dict``."""
    if isinstance(raw_dest, str):
        return raw_dest.strip()
    if isinstance(raw_dest, dict):
        alias = raw_dest.get("alias", "")
        return str(alias).strip()
    return ""
