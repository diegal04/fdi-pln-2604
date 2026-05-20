from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path


def extraer_frases(texto: str) -> list[str]:
    partes = texto.strip().split(".")
    return [frase.strip() for frase in partes if frase.strip()]


def contar_palabras(frase: str) -> int:
    return len(re.findall(r"\w+", frase, flags=re.UNICODE))


def filtrar_frases_por_longitud(
    frases: list[str],
    min_palabras: int,
) -> list[str]:
    return [frase for frase in frases if contar_palabras(frase) >= min_palabras]


def tokenizar_palabras(texto: str) -> list[str]:
    """Una unidad por palabra, espacio o signo de puntuación."""
    return re.findall(r"\s+|\w+|[^\w\s]", texto, flags=re.UNICODE)


def generar_asignaciones(
    n_frases: int,
    n_json: int,
    frases_por_json: int,
) -> list[list[int]]:
    total_slots = n_json * frases_por_json
    required_slots = n_frases * 2

    capacidades = [frases_por_json] * n_json

    if total_slots < required_slots:
        capacidades[-1] += required_slots - total_slots
    elif total_slots > required_slots:
        raise ValueError(
            "Hay más huecos que frases duplicadas. Ajusta los parámetros."
        )

    apariciones_restantes = {i: 2 for i in range(n_frases)}
    asignaciones = [[] for _ in range(n_json)]

    for idx_json in range(n_json):
        capacidad = capacidades[idx_json]

        for _ in range(capacidad):
            candidatas = [i for i, n in apariciones_restantes.items() if n > 0]

            if not candidatas:
                break

            elegida = max(candidatas, key=lambda i: apariciones_restantes[i])
            asignaciones[idx_json].append(elegida)
            apariciones_restantes[elegida] -= 1

    if any(value != 0 for value in apariciones_restantes.values()):
        raise RuntimeError(
            "No se pudo completar la asignación de frases en 2 JSON exactos."
        )

    return asignaciones


def records_from_units(units: list[str]) -> list[dict]:
    return [{"clave": unit, "valor": ""} for unit in units]


def resolver_n_frases(n_frases: int | None, n_json: int, frases_por_json: int) -> int:
    if n_frases is not None:
        return n_frases

    if (n_json * frases_por_json) % 2 != 0:
        raise ValueError(
            "n_json * frases_por_json debe ser par para que cada frase aparezca en 2 JSON."
        )

    return (n_json * frases_por_json) // 2


def crear_jsons_anotacion(
    archivo_entrada: Path,
    directorio_salida: Path,
    tokenizar: Callable[[str], list[str]],
    granularidad: str,
    n_frases: int | None = 50,
    n_json: int = 25,
    frases_por_json: int = 4,
    min_palabras: int = 0,
    seed: int = 44,
) -> dict:
    random.seed(seed)

    texto = archivo_entrada.read_text(encoding="utf-8")
    frases = extraer_frases(texto)

    if min_palabras > 0:
        frases = filtrar_frases_por_longitud(frases, min_palabras)

    n_frases = resolver_n_frases(n_frases, n_json, frases_por_json)

    if len(frases) < n_frases:
        raise ValueError(
            f"Solo hay {len(frases)} frases elegibles (min_palabras={min_palabras}); "
            f"se necesitan {n_frases}."
        )

    frases_seleccionadas = random.sample(frases, n_frases)
    asignaciones = generar_asignaciones(n_frases, n_json, frases_por_json)

    directorio_salida.mkdir(parents=True, exist_ok=True)

    for i, indices in enumerate(asignaciones, start=1):
        frases_json = [frases_seleccionadas[idx] for idx in indices]
        texto_json = " ".join(frases_json)
        units = tokenizar(texto_json)
        payload = records_from_units(units)
        salida = directorio_salida / f"json_{i:02d}.json"

        salida.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    directorio_salida.joinpath("frases_seleccionadas.json").write_text(
        json.dumps(frases_seleccionadas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    directorio_salida.joinpath("asignaciones.json").write_text(
        json.dumps(
            {
                "granularidad": granularidad,
                "min_palabras": min_palabras,
                "frases_por_json": frases_por_json,
                "n_json": n_json,
                "seed": seed,
                "frases": frases_seleccionadas,
                "asignaciones": asignaciones,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "directorio": str(directorio_salida),
        "n_frases": n_frases,
        "n_json": n_json,
        "frases_por_json": frases_por_json,
        "min_palabras": min_palabras,
    }