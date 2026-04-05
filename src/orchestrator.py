from __future__ import annotations

from dataclasses import dataclass

from src.modes.classic_mode import MODE_CLASSIC, buscar as buscar_clasico
from src.modes.rag_mode import MODE_RAG, recuperar_contexto
from src.modes.semantic_mode import MODE_SEMANTIC, buscar as buscar_semantico
from src.preprocessing import QuijoteIndex, SearchResult, TextAnalysis


@dataclass(slots=True)
class SearchExecution:
    mode: str
    query_analysis: TextAnalysis
    sidebar_results: list[SearchResult]
    mode_results: list[SearchResult]
    rag_classic_results: list[SearchResult]
    rag_semantic_results: list[SearchResult]


def orquestar_busqueda(
    index: QuijoteIndex,
    selected_mode: str,
    consulta: str,
    display_limit: int,
) -> SearchExecution:
    if selected_mode == MODE_CLASSIC:
        query_analysis, resultados = buscar_clasico(index, consulta)
        return SearchExecution(
            mode=MODE_CLASSIC,
            query_analysis=query_analysis,
            sidebar_results=resultados[:display_limit],
            mode_results=resultados,
            rag_classic_results=[],
            rag_semantic_results=[],
        )

    if selected_mode == MODE_SEMANTIC:
        query_analysis, resultados = buscar_semantico(index, consulta)
        return SearchExecution(
            mode=MODE_SEMANTIC,
            query_analysis=query_analysis,
            sidebar_results=resultados[:display_limit],
            mode_results=resultados,
            rag_classic_results=[],
            rag_semantic_results=[],
        )

    query_analysis, fusion, clasicos, semanticos = recuperar_contexto(index, consulta)
    return SearchExecution(
        mode=MODE_RAG,
        query_analysis=query_analysis,
        sidebar_results=fusion,
        mode_results=fusion,
        rag_classic_results=clasicos,
        rag_semantic_results=semanticos,
    )
