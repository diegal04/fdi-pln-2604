from __future__ import annotations

from src.preprocessing import QuijoteIndex, SearchResult, TextAnalysis


MODE_CLASSIC = "classic"


def buscar(
    index: QuijoteIndex,
    consulta: str,
    limit: int | None = None,
) -> tuple[TextAnalysis, list[SearchResult]]:
    query_analysis = index.analizar_texto(consulta)
    if not query_analysis.lemma_set:
        return query_analysis, []

    resultados: list[SearchResult] = []
    for chunk in index.chunks:
        score = index.calcular_score_tfidf(query_analysis, chunk)
        if score <= 0:
            continue
        resultados.append(
            SearchResult(
                chunk=chunk,
                score=score,
                modo=MODE_CLASSIC,
                clasico_score=score,
            )
        )

    resultados.sort(key=lambda item: item.score, reverse=True)
    return query_analysis, resultados if limit is None else resultados[:limit]
