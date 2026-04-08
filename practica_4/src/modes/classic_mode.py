from __future__ import annotations

import math

from src.preprocessing import ChunkRecord, QuijoteIndex, SearchResult, TextAnalysis


MODE_CLASSIC = "classic"


def _calcular_score_tfidf(
    index: QuijoteIndex,
    query_analysis: TextAnalysis,
    chunk: ChunkRecord,
) -> float:
    if chunk.analisis.total_terminos == 0:
        return 0.0

    score_total = 0.0
    for lema, query_count in query_analysis.conteos.items():
        frequency = chunk.analisis.conteos.get(lema, 0)
        if frequency == 0:
            continue

        tf = frequency / chunk.analisis.total_terminos
        df = index.df_global.get(lema, 0)
        idf = math.log((1 + index.total_chunks) / (1 + df)) + 1.0
        query_weight = 1.0 + math.log(query_count)
        score_total += tf * idf * query_weight

    return score_total


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
        score = _calcular_score_tfidf(index, query_analysis, chunk)
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
