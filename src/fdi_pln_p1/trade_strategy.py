from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class OfertaMemoria:
    recurso_que_busco: str | None = None
    recurso_que_doy: str | None = None


def es_oro(recurso: str) -> bool:
    return recurso.strip().lower() == "oro"


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """
    Ollama puede devolver arguments como dict o string JSON.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        raw_arguments = raw_arguments.strip()
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def normalizar_jugadores(gente_raw: Any, mi_nombre: str) -> list[str]:
    """
    Soporta /gente como list[str] o list[{"alias": "..."}].
    """
    if not isinstance(gente_raw, list):
        return []

    aliases: list[str] = []
    for entry in gente_raw:
        alias = ""
        if isinstance(entry, str):
            alias = entry
        elif isinstance(entry, dict):
            alias = str(entry.get("alias", "")).strip()

        alias = alias.strip()
        if alias and alias != mi_nombre and alias not in aliases:
            aliases.append(alias)
    return aliases


def _candidatos_oferta(
    faltan: dict[str, int],
    sobran: dict[str, int],
    mis_recursos: dict[str, int],
) -> tuple[list[str], list[str]]:
    candidatos_busco = [r for r, c in faltan.items() if c > 0]
    candidatos_doy = [r for r, c in sobran.items() if c > 0 and not es_oro(r)]

    if not candidatos_doy:
        candidatos_doy = [
            r for r, c in mis_recursos.items() if c > 0 and not es_oro(r) and r not in faltan
        ]

    return candidatos_busco, candidatos_doy


def ajustar_oferta_no_repetida(
    recurso_que_busco: str,
    recurso_que_doy: str,
    faltan: dict[str, int],
    sobran: dict[str, int],
    mis_recursos: dict[str, int],
    memoria: OfertaMemoria,
) -> tuple[str, str, bool]:
    """
    Evita repetir la misma pareja (busco, doy) cuando haya alternativa.
    Nunca devuelve oro como recurso ofrecido.
    """
    candidatos_busco, candidatos_doy = _candidatos_oferta(faltan, sobran, mis_recursos)
    if not candidatos_busco or not candidatos_doy:
        return "", "", False

    busco = recurso_que_busco if recurso_que_busco in candidatos_busco else candidatos_busco[0]
    doy = recurso_que_doy if recurso_que_doy in candidatos_doy else candidatos_doy[0]

    if busco == doy:
        alternativa_doy = next((r for r in candidatos_doy if r != busco), "")
        doy = alternativa_doy or doy

    oferta_actual = (busco, doy)
    oferta_anterior = (memoria.recurso_que_busco, memoria.recurso_que_doy)
    cambiado = False

    if oferta_actual == oferta_anterior:
        if len(candidatos_busco) > 1:
            busco = next(r for r in candidatos_busco if r != busco)
            cambiado = True
        elif len(candidatos_doy) > 1:
            doy = next(r for r in candidatos_doy if r != doy)
            cambiado = True

    if busco == doy:
        alternativa_doy = next((r for r in candidatos_doy if r != busco), "")
        if alternativa_doy:
            doy = alternativa_doy

    return busco, doy, cambiado

