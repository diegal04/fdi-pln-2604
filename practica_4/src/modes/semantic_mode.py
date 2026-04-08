from __future__ import annotations

from src.preprocessing import QuijoteIndex, SearchResult, TextAnalysis


MODE_SEMANTIC = "semantic"


def _cosine_similarity(
    left_vector: tuple[float, ...],
    left_norm: float,
    right_vector: tuple[float, ...],
    right_norm: float,
) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(left * right for left, right in zip(left_vector, right_vector))
    return dot_product / (left_norm * right_norm)


def buscar(
    index: QuijoteIndex,
    consulta: str,
    limit: int | None = None,
) -> tuple[TextAnalysis, list[SearchResult]]:
    query_analysis = index.analizar_texto(consulta)
    if query_analysis.embedding_norm == 0:
        return query_analysis, []

    resultados: list[SearchResult] = []
    for chunk in index.chunks:
        if chunk.analisis.embedding_norm == 0:
            continue

        score = _cosine_similarity(
            query_analysis.embedding,
            query_analysis.embedding_norm,
            chunk.analisis.embedding,
            chunk.analisis.embedding_norm,
        )
        if score <= 0:
            continue

        resultados.append(
            SearchResult(
                chunk=chunk,
                score=score,
                modo=MODE_SEMANTIC,
                semantico_score=score,
            )
        )

    resultados.sort(key=lambda item: item.score, reverse=True)
    return query_analysis, resultados if limit is None else resultados[:limit]
