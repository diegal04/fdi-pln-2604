from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Group
from rich.markup import escape
from rich.text import Text

from src.modes.classic_mode import MODE_CLASSIC
from src.modes.rag_mode import MODE_RAG
from src.modes.semantic_mode import MODE_SEMANTIC
from src.preprocessing import SearchResult


MODE_BROWSE = "browse"


def render_initial_reader() -> str:
    return (
        "[b #8b0000]Quijote IR[/]\n\n"
        "Arranque inmediato activado.\n"
        "Pulsa Enter en la ruta para indexar o lanza una consulta para iniciar carga bajo demanda."
    )


def render_missing_default_corpus(path: Path) -> str:
    return (
        "[b red]No se encontro el corpus por defecto.[/b red]\n\n"
        f"Ruta esperada: {escape(str(path))}\n"
        "Puedes pegar otra ruta valida y pulsar Enter para indexar."
    )


def render_missing_file(path: Path) -> str:
    return f"[b red]Error:[/b red] No se encontro el archivo en {escape(str(path))}."


def render_index_ready(
    path: Path,
    stats: Mapping[str, int],
    chunk_size_words: int,
    chunk_overlap_words: int,
    model: str,
) -> str:
    return (
        "[b #8b0000]Indice listo[/]\n\n"
        f"Archivo indexado: {escape(str(path))}\n"
        f"Secciones detectadas: {stats['sections']}\n"
        f"Pasajes indexados: {stats['chunks']}\n"
        f"Tamano de chunk: {chunk_size_words} palabras\n"
        f"Overlap: {chunk_overlap_words} palabras\n"
        f"Modelo Ollama actual: {escape(model)}\n\n"
        "Ya puedes escribir una consulta para buscar."
    )


def render_index_cancelled(message: str) -> str:
    return f"[b #8b0000]Indexacion cancelada[/]\n\n{escape(message)}"


def render_index_error(reason: str) -> str:
    return (
        "[b red]Error durante la indexacion.[/b red]\n\n"
        f"{escape(reason)}\n\n"
        "Revisa la ruta y vuelve a pulsar Enter para reintentar."
    )


def render_loading_status(
    stage: str,
    completed: int | None,
    total: int | None,
    elapsed: float,
    notice: str | None = None,
) -> str:
    counts_text = "--"
    percent_text = "--"
    eta_text = "calculando ETA..."

    if total and completed is not None and total > 0:
        safe_completed = min(completed, total)
        percent = (safe_completed / total) * 100.0
        counts_text = f"{safe_completed}/{total}"
        percent_text = f"{percent:.1f}%"
        if safe_completed > 0:
            remaining_steps = max(0, total - safe_completed)
            eta_seconds = (elapsed / safe_completed) * remaining_steps
            eta_text = format_duration(eta_seconds)

    notice_block = ""
    if notice:
        notice_block = f"\n\n[dim]{escape(notice)}[/dim]"

    return (
        "[b #8b0000]Indexando corpus...[/]\n\n"
        f"Fase: {escape(stage)}\n"
        f"Progreso: {escape(counts_text)} ({escape(percent_text)})\n"
        f"Tiempo transcurrido: {format_duration(elapsed)}\n"
        f"ETA: {escape(eta_text)}"
        f"{notice_block}"
    )


def render_chunk_detail(
    chunk_title: str,
    chunk_text: str,
    query_lemmas: frozenset[str],
    metadata: str,
    nlp: Any | None,
) -> Group:
    highlighted_text = highlight_text(chunk_text, query_lemmas, nlp)
    chapter_title = Text(chunk_title, style="bold #8b0000", justify="center")
    content = Text.from_markup(f"{highlighted_text}\n\n{metadata}")
    return Group(
        Align.center(chapter_title, vertical="top"),
        content,
    )


def render_classic_summary(
    query_lemmas: frozenset[str],
    results_count: int,
    display_limit: int,
) -> str:
    if not query_lemmas:
        return (
            "La consulta no contiene terminos utiles tras eliminar stopwords. "
            "Prueba con nombres o conceptos mas informativos."
        )

    serialized_lemmas = ", ".join(sorted(query_lemmas))
    if results_count == 0:
        return (
            "[b red]Sin resultados clasicos.[/b red]\n\n"
            f"Consulta lematizada: {serialized_lemmas}"
        )

    shown = min(results_count, display_limit)
    return (
        "[b #8b0000]Busqueda clasica[/]\n\n"
        f"Consulta lematizada: {serialized_lemmas}\n"
        f"Resultados recuperados: {results_count}\n"
        f"Mostrando: {shown}\n\n"
        "Selecciona un pasaje en la barra lateral para ver el texto con los lemas resaltados."
    )


def render_semantic_summary(
    embedding_norm: float,
    results: list[SearchResult],
    display_limit: int,
) -> str:
    if embedding_norm == 0:
        return (
            "La consulta no genero un embedding util. "
            "Prueba con una frase con contenido lexico mas claro."
        )

    if not results:
        return "[b red]Sin resultados semanticos.[/b red]"

    shown = min(len(results), display_limit)
    return (
        "[b #8b0000]Busqueda semantica[/]\n\n"
        "Pasajes ordenados por similitud coseno con el embedding de la consulta.\n"
        f"Mostrando top: {shown}\n"
        f"Mejor score: {results[0].score:.4f}\n\n"
        "Selecciona un pasaje para inspeccionar el texto recuperado."
    )


def render_rag_error(
    reason: str,
    model: str,
    fusion_count: int,
    classic_count: int,
    semantic_count: int,
) -> str:
    return (
        "[b #8b0000]Contexto RAG recuperado[/]\n\n"
        "No se pudo generar la respuesta con Ollama.\n"
        f"Motivo: {escape(reason)}\n\n"
        f"Modelo seleccionado: {escape(model)}\n"
        f"Pasajes fusionados: {fusion_count}\n"
        f"Top clasicos usados: {classic_count}\n"
        f"Top semanticos usados: {semantic_count}\n\n"
        "Los pasajes de apoyo siguen disponibles en la barra lateral."
    )


def render_rag_success(answer: str, model: str, fusion: Iterable[SearchResult]) -> str:
    references = ", ".join(f"C{result.chunk.chunk_id}" for result in fusion)
    return (
        "[b #8b0000]Respuesta RAG[/]\n\n"
        f"Modelo: {escape(model)}\n\n"
        f"{escape(answer)}\n\n"
        f"[dim]Referencias disponibles en la barra lateral: {escape(references)}[/dim]"
    )


def render_model_updated(model: str) -> str:
    return (
        f"[b #8b0000]Modelo de Ollama actualizado[/]\n\nModelo actual: {escape(model)}"
    )


def format_result_metadata(result: SearchResult | None) -> str:
    if result is None or result.modo == MODE_BROWSE:
        return "[dim]Exploracion manual del corpus.[/dim]"
    if result.modo == MODE_CLASSIC:
        return f"[dim]TF-IDF: {result.score:.4f}[/dim]"
    if result.modo == MODE_SEMANTIC:
        return f"[dim]Similitud coseno: {result.score:.4f}[/dim]"
    return (
        "[dim]"
        f"RRF: {result.score:.4f} | "
        f"clasico: {result.clasico_score:.4f} | "
        f"semantico: {result.semantico_score:.4f}"
        "[/dim]"
    )


def format_sidebar_label(result: SearchResult) -> str:
    chapter_label = escape(format_sidebar_section(result.chunk.seccion))
    passage_label = escape(extract_sidebar_passage(result.chunk.titulo))
    head = f"[b]C{result.chunk.chunk_id:03d}[/b] · {chapter_label} · {passage_label}"
    return f"{head}\n[dim]score {result.score:.3f}[/dim]"


def highlight_text(text: str, query_lemmas: frozenset[str], nlp: Any | None) -> str:
    if not query_lemmas or nlp is None:
        return escape(text)

    doc = nlp(text)
    highlighted_parts: list[str] = []
    for token in doc:
        escaped_text = escape(token.text)
        lemma = token.lemma_.lower()
        if token.is_alpha and not token.is_stop and lemma in query_lemmas:
            highlighted_parts.append(
                f"[b #8b0000 on #d4af37]{escaped_text}[/]{token.whitespace_}"
            )
        else:
            highlighted_parts.append(f"{escaped_text}{token.whitespace_}")

    return "".join(highlighted_parts)


def format_sidebar_section(section: str) -> str:
    if not section:
        return "Seccion"

    section_head = section.split(".", 1)[0].strip()
    if section_head:
        return truncate(section_head, 32)
    return truncate(section.strip(), 32)


def extract_sidebar_passage(title: str) -> str:
    marker = " · pasaje "
    lowered = title.lower()
    marker_index = lowered.rfind(marker)
    if marker_index == -1:
        return "pasaje ?"

    passage_number = title[marker_index + len(marker) :].strip()
    if not passage_number:
        return "pasaje ?"
    return f"pasaje {passage_number}"


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}…"


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}:{secs:02d}"
