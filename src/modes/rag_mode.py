from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.modes.classic_mode import buscar as buscar_clasico
from src.preprocessing import ChunkRecord, QuijoteIndex, SearchResult, TextAnalysis
from src.modes.semantic_mode import buscar as buscar_semantico


MODE_RAG = "rag"


def recuperar_contexto(
    index: QuijoteIndex,
    consulta: str,
    retrieval_limit: int = 8,
    output_limit: int = 6,
) -> tuple[TextAnalysis, list[SearchResult], list[SearchResult], list[SearchResult]]:
    query_analysis, clasicos = buscar_clasico(index, consulta, retrieval_limit)
    _, semanticos = buscar_semantico(index, consulta, retrieval_limit)
    fusion = fusionar_resultados(clasicos, semanticos, output_limit)
    return query_analysis, fusion, clasicos, semanticos


def fusionar_resultados(
    clasicos: Iterable[SearchResult],
    semanticos: Iterable[SearchResult],
    output_limit: int,
) -> list[SearchResult]:
    fusion: dict[int, dict[str, object]] = defaultdict(
        lambda: {"chunk": None, "rrf": 0.0, "clasico": 0.0, "semantico": 0.0}
    )

    for rank, result in enumerate(clasicos, start=1):
        data = fusion[result.chunk.chunk_id]
        data["chunk"] = result.chunk
        data["rrf"] = float(data["rrf"]) + (1.0 / (60 + rank))
        data["clasico"] = result.score

    for rank, result in enumerate(semanticos, start=1):
        data = fusion[result.chunk.chunk_id]
        data["chunk"] = result.chunk
        data["rrf"] = float(data["rrf"]) + (1.0 / (60 + rank))
        data["semantico"] = result.score

    resultados: list[SearchResult] = []
    for data in fusion.values():
        chunk = data["chunk"]
        if not isinstance(chunk, ChunkRecord):
            continue

        resultados.append(
            SearchResult(
                chunk=chunk,
                score=float(data["rrf"]),
                modo=MODE_RAG,
                clasico_score=float(data["clasico"]),
                semantico_score=float(data["semantico"]),
            )
        )

    resultados.sort(key=lambda item: item.score, reverse=True)
    return resultados[:output_limit]


def generar_respuesta_ollama(
    consulta: str,
    contextos: list[SearchResult],
    modelo: str,
) -> str:
    try:
        from ollama import chat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Falta la librería `ollama`. Instálala con `uv add ollama`."
        ) from exc

    if not modelo.strip():
        raise RuntimeError("Debes indicar un modelo de Ollama antes de ejecutar RAG.")

    context_blocks = []
    for result in contextos:
        context_blocks.append(
            f"[C{result.chunk.chunk_id}] {result.chunk.titulo}\n"
            f"Sección: {result.chunk.seccion}\n"
            f"Texto:\n{result.chunk.texto}"
        )
    joined_context_blocks = "\n\n".join(context_blocks)

    response = chat(
        model=modelo,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responde en español usando solo los pasajes del Quijote proporcionados. "
                    "Si la información no basta, dilo. "
                    "Cita las referencias usando el formato [C12]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Consulta: {consulta}\n\n"
                    f"Pasajes recuperados:\n\n{joined_context_blocks}"
                ),
            },
        ],
    )

    if isinstance(response, dict):
        content = str(response.get("message", {}).get("content", "")).strip()
        if not content:
            raise RuntimeError("Ollama respondió sin contenido.")
        return content

    message = getattr(response, "message", None)
    content = str(getattr(message, "content", "")).strip()
    if not content:
        raise RuntimeError("Ollama respondió sin contenido.")
    return content
