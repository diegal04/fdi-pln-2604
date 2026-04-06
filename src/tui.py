from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, Literal

from rich.align import Align
from rich.console import Group
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, ListItem, ListView, Select, Static
from textual.worker import Worker, WorkerState, get_current_worker

from src.modes.classic_mode import MODE_CLASSIC
from src.modes.rag_mode import MODE_RAG, generar_respuesta_ollama
from src.modes.semantic_mode import MODE_SEMANTIC
from src.orchestrator import orquestar_busqueda
from src.preprocessing import (
    IndexProgress,
    IndexingCancelled,
    QuijoteIndex,
    SearchResult,
    TextAnalysis,
)


MODE_BROWSE = "browse"


@dataclass(slots=True)
class IndexingWorkerResult:
    run_id: int
    path: Path
    stats: dict[str, int]
    index: QuijoteIndex
    nlp: Any


@dataclass(slots=True)
class ProgressSnapshot:
    run_id: int
    stage: str
    completed: int | None
    total: int | None
    updated_at: float


class QuijoteApp(App):
    BINDINGS = [
        Binding("ctrl+a", "focus_file", "Cargar Archivo"),
        Binding("ctrl+b", "focus_search", "Buscar"),
        Binding("ctrl+m", "focus_mode", "Modo"),
        Binding("ctrl+o", "focus_model", "Modelo"),
        Binding("ctrl+q", "quit", "Salir"),
    ]

    CSS = """
    Screen { background: #f4efe1; color: #1c1b19; }
    Header { background: #8b0000; color: #fef7e6; }
    Footer { background: #2b3a42; color: #fef7e6; }
    #main-body { height: 1fr; }
    #top-panel {
        height: 30%;
        min-height: 10;
        background: #eee8d5;
        border-bottom: solid #d4af37;
        padding: 0 1;
    }
    #top-left {
        width: 1fr;
        margin-right: 1;
        height: 1fr;
    }
    #top-right {
        width: 2fr;
        height: 1fr;
    }
    #mode-stack { height: auto; }
    .field-card {
        background: #f7f1e2;
        border: round #a33838;
        padding: 0;
        margin-bottom: 0;
    }
    .field-label {
        height: 1;
        margin-left: 0;
        color: #6d5a3d;
    }
    #mode-field, #model-field { width: 1fr; }
    #mode-field {
        height: auto;
        margin-top: 1;
        margin-bottom: 0;
        background: #7f1010;
        border: round #d4af37;
        padding: 0 1;
    }
    #mode-field .field-label {
        color: #f7e7bf;
        text-style: bold;
    }
    #model-field {
        height: auto;
        border: none;
        background: transparent;
        padding: 0;
    }
    #model-field .field-label { color: #6d5a3d; }
    #model-field #model-input {
        background: #fff8ec;
    }
    #file-field {
        height: auto;
        margin-bottom: 0;
    }
    #query-field {
        height: auto;
        min-height: 4;
        background: #f9f2df;
    }
    #query-field .field-label {
        color: #6d5a3d;
        text-style: none;
    }
    #file-input, #search-input, #mode-select, #model-input { width: 1fr; }
    Input {
        height: 1;
        margin: 0;
        padding: 0;
        background: #fffaf0;
        color: #1a1a1a;
        border: none;
    }
    Input:hover { border: none; background: #fff8ec; }
    Input:focus {
        border: none;
        background: #fffdf7;
    }
    #search-input {
        border: none;
        background: #fffdf4;
    }
    #search-input:focus {
        border: none;
        background: #fffef8;
    }
    Select {
        height: auto;
        margin: 0;
        background: transparent;
        color: #1a1a1a;
        border: none;
    }
    Select:hover, Select:focus {
        border: none;
        background: transparent;
    }
    #mode-select {
        height: 3;
        margin: 0;
        padding: 0;
        background: transparent;
        color: #1a1a1a;
        border: none;
    }
    #mode-select > SelectCurrent {
        height: 3;
        margin: 0;
        background: #fff6e2;
        color: #7f1010;
        border: round #d4af37;
        padding: 0 1;
    }
    #mode-select > SelectCurrent:ansi {
        height: 3;
        background: #fff6e2;
        color: #7f1010;
        border: round #d4af37;
    }
    #mode-select > SelectCurrent:hover {
        background: #fff9eb;
    }
    #mode-select:focus > SelectCurrent {
        background: #fffdf7;
        border: round #f0d68a;
    }
    #mode-select > SelectCurrent Static#label {
        color: #7f1010;
        background: transparent;
        text-style: bold;
    }
    #mode-select > SelectCurrent.-has-value Static#label {
        color: #7f1010;
        text-style: bold;
    }
    #mode-select > SelectCurrent .arrow {
        color: #8b5f1a;
        background: transparent;
    }
    #mode-select > SelectOverlay {
        background: #fffaf0;
        color: #1a1a1a;
        border: round #8f2626;
    }
    #mode-select > SelectOverlay:focus {
        border: round #d4af37;
        background: #fffdf7;
    }
    #mode-select > SelectOverlay > .option-list--option {
        color: #1a1a1a;
        background: #fffaf0;
    }
    #mode-select > SelectOverlay > .option-list--option-hover {
        color: #1a1a1a;
        background: #f5ebd3;
    }
    #mode-select > SelectOverlay > .option-list--option-highlighted {
        color: #7f1010;
        background: #f3e6c6;
    }
    #model-field.model-hidden {
        display: none;
    }
    #output-panel { height: 2fr; }
    #sidebar { width: 38%; background: #2b3a42; color: #eee8d5; border-right: solid #d4af37; }
    #sidebar, #reader-container {
        scrollbar-size-vertical: 1;
        scrollbar-color: #d8c8a3;
        scrollbar-color-hover: #cab588;
        scrollbar-color-active: #bba06e;
        scrollbar-background: #f6f1e4;
        scrollbar-background-hover: #f0e9d9;
        scrollbar-background-active: #e8ddc4;
        scrollbar-corner-color: #f6f1e4;
    }
    ListItem { padding: 0 1; }
    ListItem.--highlight, ListItem:hover { background: #6e1313; color: #fff8e7; }
    #reader-container { width: 62%; padding: 2 3; }
    #reader { height: auto; }
    """

    DISPLAY_LIMIT = 30

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp: Any | None = None
        self.index: QuijoteIndex | None = None
        self.selected_mode = MODE_CLASSIC
        self.current_query = ""
        self.current_query_analysis = TextAnalysis.empty()
        self.current_results: list[SearchResult] = []
        self.results_by_chunk_id: dict[int, SearchResult] = {}

        self.default_rag_model = os.getenv("P4_OLLAMA_MODEL", "qwen3:0.6b")
        self.default_corpus_path = Path(__file__).resolve().parent.parent / "2000-h.htm"

        self.index_state: Literal["idle", "loading", "ready", "error"] = "idle"
        self.indexed_path: Path | None = None
        self.active_worker: Worker[IndexingWorkerResult] | None = None
        self.active_stage = "En espera"
        self.index_start_time: float | None = None
        self.loading_notice: str | None = None
        self._index_run_id = 0
        self._progress_lock = Lock()
        self._progress_snapshot = ProgressSnapshot(
            run_id=0,
            stage="En espera",
            completed=None,
            total=None,
            updated_at=monotonic(),
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main-body"):
            with Horizontal(id="top-panel"):
                with Vertical(id="top-left"):
                    with Vertical(id="mode-stack"):
                        with Vertical(classes="field-card", id="mode-field"):
                            yield Static("MODO DE OPERACION (Ctrl+M)", classes="field-label")
                            yield Select(
                                [
                                    ("1. Clasica", MODE_CLASSIC),
                                    ("2. Semantica", MODE_SEMANTIC),
                                    ("3. RAG", MODE_RAG),
                                ],
                                value=MODE_CLASSIC,
                                allow_blank=False,
                                prompt="Modo",
                                id="mode-select",
                            )
                        with Vertical(classes="field-card model-hidden", id="model-field"):
                            yield Static("Modelo", classes="field-label")
                            yield Input(
                                value=self.default_rag_model,
                                placeholder="Modelo Ollama (ej. qwen3:0.6b)",
                                id="model-input",
                            )
                with Vertical(id="top-right"):
                    with Vertical(classes="field-card", id="file-field"):
                        yield Static("Archivo (Ctrl+A)", classes="field-label")
                        yield Input(
                            value=str(self.default_corpus_path),
                            placeholder="Pega la ruta del HTML del Quijote y pulsa Enter...",
                            id="file-input",
                        )
                    with Vertical(classes="field-card", id="query-field"):
                        yield Static("Consulta (Ctrl+B)", classes="field-label")
                        yield Input(
                            placeholder="Escribe tu consulta y pulsa Enter...",
                            id="search-input",
                        )
            with Horizontal(id="output-panel"):
                yield ListView(id="sidebar")
                with VerticalScroll(id="reader-container"):
                    yield Static(
                        "[b #8b0000]Quijote IR[/]\n\n"
                        "Arranque inmediato activado.\n"
                        "Pulsa Enter en la ruta para indexar o lanza una consulta para iniciar carga bajo demanda.",
                        id="reader",
                        expand=True,
                    )
        yield Footer()

    def on_mount(self) -> None:
        if not self.default_corpus_path.exists():
            self.index_state = "error"
            self.query_one("#reader", Static).update(
                "[b red]No se encontro el corpus por defecto.[/b red]\n\n"
                f"Ruta esperada: {escape(str(self.default_corpus_path))}\n"
                "Puedes pegar otra ruta valida y pulsar Enter para indexar."
            )

        self._sync_model_visibility()
        self.set_interval(0.5, self._on_progress_tick)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "file-input":
            self.cargar_archivo(event.value)
            return

        if event.input.id == "search-input":
            self.ejecutar_busqueda(event.value)
            return

        if event.input.id == "model-input":
            self.actualizar_modelo_ollama(event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "mode-select":
            return

        self.selected_mode = str(event.value)
        self._sync_model_visibility()
        if self.index_state != "ready" or self.index is None:
            return

        if self.current_query and self.index.total_chunks > 0:
            self.ejecutar_busqueda(self.current_query)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self.active_worker is None or event.worker is not self.active_worker:
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            if not isinstance(result, IndexingWorkerResult):
                self._finalizar_indexacion_error("La indexacion termino sin resultado util.")
                return
            if result.run_id != self._index_run_id:
                return

            self.nlp = result.nlp
            self.index = result.index
            self.indexed_path = result.path
            self.index_state = "ready"
            self.active_stage = "Indice listo"
            self.active_worker = None
            self.index_start_time = None
            self.loading_notice = None

            self.current_query = ""
            self.current_query_analysis = TextAnalysis.empty()
            self._mostrar_exploracion_inicial()
            self.query_one("#reader", Static).update(
                "[b #8b0000]Indice listo[/]\n\n"
                f"Archivo indexado: {escape(str(result.path))}\n"
                f"Secciones detectadas: {result.stats['sections']}\n"
                f"Pasajes indexados: {result.stats['chunks']}\n"
                f"Tamano de chunk: {result.index.chunk_size_words} palabras\n"
                f"Overlap: {result.index.chunk_overlap_words} palabras\n"
                f"Modelo Ollama actual: {escape(self._obtener_modelo_ollama())}\n\n"
                "Ya puedes escribir una consulta para buscar."
            )
            return

        if event.state not in {WorkerState.ERROR, WorkerState.CANCELLED}:
            return

        error = event.worker.error
        if isinstance(error, IndexingCancelled) or event.state == WorkerState.CANCELLED:
            self._finalizar_indexacion_cancelada(
                "Indexacion cancelada. Pulsa Enter en una ruta para iniciar de nuevo."
            )
            return

        self._finalizar_indexacion_error(str(error) if error else "Error desconocido en indexacion.")

    def cargar_archivo(self, ruta_str: str) -> None:
        normalized_path = ruta_str.strip() or str(self.default_corpus_path)
        path = Path(normalized_path)
        self.query_one("#file-input", Input).value = str(path)

        was_loading = self.index_state == "loading"
        if was_loading:
            self._cancel_active_worker()

        if not path.exists():
            if was_loading:
                self._index_run_id += 1
            self.active_worker = None
            self.index_state = "error"
            self.index = None
            self.indexed_path = None
            self.active_stage = "Error"
            self.index_start_time = None
            self.loading_notice = None
            self.current_query = ""
            self.current_query_analysis = TextAnalysis.empty()
            self._actualizar_sidebar([])
            self.query_one("#reader", Static).update(
                f"[b red]Error:[/b red] No se encontro el archivo en {escape(str(path))}."
            )
            return

        trigger = "manual_restart" if was_loading else "manual"
        self._start_indexing(path, trigger)

    @work(thread=True, group="indexing", exclusive=True, exit_on_error=False)
    def _indexar_en_background(self, path: Path, run_id: int) -> IndexingWorkerResult:
        worker = get_current_worker()

        self._set_progress(run_id, "Cargando modelo spaCy", None, None)
        if self._should_cancel(worker, run_id):
            raise IndexingCancelled("Indexacion cancelada.")

        nlp = self.nlp
        if nlp is None:
            import spacy

            nlp = spacy.load("es_core_news_lg")

        if self._should_cancel(worker, run_id):
            raise IndexingCancelled("Indexacion cancelada.")

        index = QuijoteIndex(nlp)

        def on_progress(progress: IndexProgress) -> None:
            self._set_progress(run_id, progress.stage, progress.completed, progress.total)

        stats = index.cargar_archivo(
            path,
            on_progress=on_progress,
            should_cancel=lambda: self._should_cancel(worker, run_id),
        )

        if self._should_cancel(worker, run_id):
            raise IndexingCancelled("Indexacion cancelada.")

        return IndexingWorkerResult(
            run_id=run_id,
            path=path,
            stats=stats,
            index=index,
            nlp=nlp,
        )

    def _start_indexing(self, path: Path, trigger: str) -> None:
        self._index_run_id += 1
        run_id = self._index_run_id

        self.index_state = "loading"
        self.index = None
        self.indexed_path = None
        self.active_stage = "Preparando indexacion"
        self.index_start_time = monotonic()
        self.current_query = ""
        self.current_query_analysis = TextAnalysis.empty()
        self._actualizar_sidebar([])

        if trigger == "search":
            self.loading_notice = (
                "No hay indice listo. Se inicia la indexacion; cuando termine, vuelve a pulsar Enter en la consulta."
            )
        elif trigger == "manual_restart":
            self.loading_notice = "Se cancelo la indexacion anterior y se inicio una nueva con la ruta actual."
        else:
            self.loading_notice = "Indexando el archivo seleccionado."

        self._set_progress(run_id, "Preparando indexacion", None, None)
        self.active_worker = self._indexar_en_background(path, run_id)
        self._render_loading_status()

    def _cancel_active_worker(self) -> None:
        if self.active_worker is None:
            return
        if not self.active_worker.is_finished:
            self.active_worker.cancel()

    def _finalizar_indexacion_cancelada(self, message: str) -> None:
        self.active_worker = None
        self.index_start_time = None
        self.index_state = "idle"
        self.index = None
        self.indexed_path = None
        self.loading_notice = None
        self.active_stage = "En espera"
        self._actualizar_sidebar([])
        self.query_one("#reader", Static).update(
            "[b #8b0000]Indexacion cancelada[/]\n\n"
            f"{escape(message)}"
        )

    def _finalizar_indexacion_error(self, reason: str) -> None:
        self.active_worker = None
        self.index_start_time = None
        self.index_state = "error"
        self.index = None
        self.indexed_path = None
        self.loading_notice = None
        self.active_stage = "Error"
        self._actualizar_sidebar([])
        self.query_one("#reader", Static).update(
            "[b red]Error durante la indexacion.[/b red]\n\n"
            f"{escape(reason)}\n\n"
            "Revisa la ruta y vuelve a pulsar Enter para reintentar."
        )

    def _on_progress_tick(self) -> None:
        if self.index_state != "loading":
            return
        self._render_loading_status()

    def _render_loading_status(self) -> None:
        if self.index_state != "loading":
            return

        snapshot = self._get_progress_snapshot()
        if snapshot.run_id != self._index_run_id:
            return

        elapsed = 0.0
        if self.index_start_time is not None:
            elapsed = max(0.0, monotonic() - self.index_start_time)

        counts_text = "--"
        percent_text = "--"
        eta_text = "calculando ETA..."

        if snapshot.total and snapshot.completed is not None and snapshot.total > 0:
            completed = min(snapshot.completed, snapshot.total)
            percent = (completed / snapshot.total) * 100.0
            counts_text = f"{completed}/{snapshot.total}"
            percent_text = f"{percent:.1f}%"
            if completed > 0:
                remaining_steps = max(0, snapshot.total - completed)
                eta_seconds = (elapsed / completed) * remaining_steps
                eta_text = self._formatear_duracion(eta_seconds)

        notice = ""
        if self.loading_notice:
            notice = f"\n\n[dim]{escape(self.loading_notice)}[/dim]"

        self.query_one("#reader", Static).update(
            "[b #8b0000]Indexando corpus...[/]\n\n"
            f"Fase: {escape(snapshot.stage)}\n"
            f"Progreso: {escape(counts_text)} ({escape(percent_text)})\n"
            f"Tiempo transcurrido: {self._formatear_duracion(elapsed)}\n"
            f"ETA: {escape(eta_text)}"
            f"{notice}"
        )

    def _set_progress(
        self,
        run_id: int,
        stage: str,
        completed: int | None,
        total: int | None,
    ) -> None:
        with self._progress_lock:
            if run_id != self._index_run_id:
                return
            self.active_stage = stage
            self._progress_snapshot = ProgressSnapshot(
                run_id=run_id,
                stage=stage,
                completed=completed,
                total=total,
                updated_at=monotonic(),
            )

    def _get_progress_snapshot(self) -> ProgressSnapshot:
        with self._progress_lock:
            snapshot = self._progress_snapshot
            return ProgressSnapshot(
                run_id=snapshot.run_id,
                stage=snapshot.stage,
                completed=snapshot.completed,
                total=snapshot.total,
                updated_at=snapshot.updated_at,
            )

    def _should_cancel(self, worker: Worker[IndexingWorkerResult], run_id: int) -> bool:
        return worker.is_cancelled or run_id != self._index_run_id

    def _formatear_duracion(self, seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        return f"{minutes:02d}:{secs:02d}"

    def ejecutar_busqueda(self, consulta: str) -> None:
        if self.index_state == "loading":
            self.loading_notice = (
                "Hay una indexacion en curso. Espera a que termine y vuelve a pulsar Enter para ejecutar la consulta."
            )
            self._render_loading_status()
            return

        index = self.index
        if self.index_state != "ready" or index is None:
            clean_query = consulta.strip()
            if not clean_query:
                self.current_query = ""
                self.current_query_analysis = TextAnalysis.empty()
                self.query_one("#reader", Static).update(
                    "Consulta vacia. Carga un archivo (Enter en ruta) o escribe una consulta con contenido."
                )
                return

            path_text = self.query_one("#file-input", Input).value.strip() or str(self.default_corpus_path)
            path = Path(path_text)
            if not path.exists():
                self.index_state = "error"
                self.query_one("#reader", Static).update(
                    f"[b red]Error:[/b red] No se encontro el archivo en {escape(str(path))}."
                )
                return

            self._start_indexing(path, "search")
            return

        if index.total_chunks == 0:
            self.query_one("#reader", Static).update(
                "[b red]Antes debes cargar e indexar el HTML del Quijote.[/b red]"
            )
            return

        self.current_query = consulta.strip()
        if not self.current_query:
            self.current_query_analysis = TextAnalysis.empty()
            self._mostrar_exploracion_inicial()
            self.query_one("#reader", Static).update(
                "Consulta vacia. Mostrando una seleccion inicial de pasajes."
            )
            return

        execution = orquestar_busqueda(
            index,
            self.selected_mode,
            self.current_query,
            self.DISPLAY_LIMIT,
        )
        self.current_query_analysis = execution.query_analysis
        self._actualizar_sidebar(execution.sidebar_results)

        if execution.mode == MODE_CLASSIC:
            self._mostrar_resumen_clasico(execution.mode_results)
            return

        if execution.mode == MODE_SEMANTIC:
            self._mostrar_resumen_semantico(execution.mode_results)
            return

        self._mostrar_respuesta_rag(
            execution.mode_results,
            execution.rag_classic_results,
            execution.rag_semantic_results,
        )

    def actualizar_modelo_ollama(self, modelo: str) -> None:
        normalized = modelo.strip() or self.default_rag_model
        model_input = self.query_one("#model-input", Input)
        model_input.value = normalized

        if self.index_state == "loading":
            self.loading_notice = (
                "Modelo actualizado. Se aplicara cuando termine la indexacion y ejecutes una consulta en modo RAG."
            )
            self._render_loading_status()
            return

        if (
            self.selected_mode == MODE_RAG
            and self.current_query
            and self.index_state == "ready"
            and self.index is not None
            and self.index.total_chunks > 0
        ):
            self.ejecutar_busqueda(self.current_query)
            return

        self.query_one("#reader", Static).update(
            "[b #8b0000]Modelo de Ollama actualizado[/]\n\n"
            f"Modelo actual: {escape(normalized)}"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.name is None:
            return

        index = self.index
        if index is None:
            return

        chunk_id = int(event.item.name)
        chunk = index.chunk_by_id.get(chunk_id)
        if chunk is None:
            return

        result = self.results_by_chunk_id.get(chunk_id)
        metadata = self._formatear_metadata_resultado(result)
        highlighted_text = self._resaltar_texto(chunk.texto, self.current_query_analysis.lemma_set)
        chapter_title = Text(chunk.titulo, style="bold #8b0000", justify="center")
        content = Text.from_markup(f"{highlighted_text}\n\n{metadata}")
        self.query_one("#reader", Static).update(
            Group(
                Align.center(chapter_title, vertical="top"),
                content,
            )
        )

    def _mostrar_exploracion_inicial(self) -> None:
        index = self.index
        if index is None:
            self._actualizar_sidebar([])
            return

        initial = [
            SearchResult(chunk=chunk, score=0.0, modo=MODE_BROWSE)
            for chunk in index.chunks[: self.DISPLAY_LIMIT]
        ]
        self._actualizar_sidebar(initial)

    def _actualizar_sidebar(self, resultados: list[SearchResult]) -> None:
        sidebar = self.query_one("#sidebar", ListView)
        sidebar.clear()

        self.current_results = resultados
        self.results_by_chunk_id = {result.chunk.chunk_id: result for result in resultados}

        for result in resultados:
            label = self._formatear_label_sidebar(result)
            sidebar.append(ListItem(Static(label), name=str(result.chunk.chunk_id)))

    def _mostrar_resumen_clasico(self, resultados: list[SearchResult]) -> None:
        reader = self.query_one("#reader", Static)
        if not self.current_query_analysis.lemma_set:
            reader.update(
                "La consulta no contiene terminos utiles tras eliminar stopwords. "
                "Prueba con nombres o conceptos mas informativos."
            )
            return

        if not resultados:
            reader.update(
                "[b red]Sin resultados clasicos.[/b red]\n\n"
                f"Consulta lematizada: {', '.join(sorted(self.current_query_analysis.lemma_set))}"
            )
            return

        shown = min(len(resultados), self.DISPLAY_LIMIT)
        reader.update(
            "[b #8b0000]Busqueda clasica[/]\n\n"
            f"Consulta lematizada: {', '.join(sorted(self.current_query_analysis.lemma_set))}\n"
            f"Resultados recuperados: {len(resultados)}\n"
            f"Mostrando: {shown}\n\n"
            "Selecciona un pasaje en la barra lateral para ver el texto con los lemas resaltados."
        )

    def _mostrar_resumen_semantico(self, resultados: list[SearchResult]) -> None:
        reader = self.query_one("#reader", Static)
        if self.current_query_analysis.embedding_norm == 0:
            reader.update(
                "La consulta no genero un embedding util. "
                "Prueba con una frase con contenido lexico mas claro."
            )
            return

        if not resultados:
            reader.update("[b red]Sin resultados semanticos.[/b red]")
            return

        shown = min(len(resultados), self.DISPLAY_LIMIT)
        reader.update(
            "[b #8b0000]Busqueda semantica[/]\n\n"
            "Pasajes ordenados por similitud coseno con el embedding de la consulta.\n"
            f"Mostrando top: {shown}\n"
            f"Mejor score: {resultados[0].score:.4f}\n\n"
            "Selecciona un pasaje para inspeccionar el texto recuperado."
        )

    def _mostrar_respuesta_rag(
        self,
        fusion: list[SearchResult],
        clasicos: list[SearchResult],
        semanticos: list[SearchResult],
    ) -> None:
        reader = self.query_one("#reader", Static)
        modelo = self._obtener_modelo_ollama()

        if not fusion:
            reader.update("[b red]RAG sin contexto suficiente.[/b red]")
            return

        try:
            answer = generar_respuesta_ollama(self.current_query, fusion, modelo)
        except Exception as exc:
            reader.update(
                "[b #8b0000]Contexto RAG recuperado[/]\n\n"
                "No se pudo generar la respuesta con Ollama.\n"
                f"Motivo: {escape(str(exc))}\n\n"
                f"Modelo seleccionado: {escape(modelo)}\n"
                f"Pasajes fusionados: {len(fusion)}\n"
                f"Top clasicos usados: {len(clasicos)}\n"
                f"Top semanticos usados: {len(semanticos)}\n\n"
                "Los pasajes de apoyo siguen disponibles en la barra lateral."
            )
            return

        referencias = ", ".join(f"C{result.chunk.chunk_id}" for result in fusion)
        reader.update(
            "[b #8b0000]Respuesta RAG[/]\n\n"
            f"Modelo: {escape(modelo)}\n\n"
            f"{escape(answer)}\n\n"
            f"[dim]Referencias disponibles en la barra lateral: {escape(referencias)}[/dim]"
        )

    def _obtener_modelo_ollama(self) -> str:
        model_input = self.query_one("#model-input", Input)
        modelo = model_input.value.strip()
        if not modelo:
            modelo = self.default_rag_model
            model_input.value = modelo
        return modelo

    def _resaltar_texto(self, texto: str, query_lemmas: frozenset[str]) -> str:
        if not query_lemmas or self.nlp is None:
            return escape(texto)

        doc = self.nlp(texto)
        highlighted_parts: list[str] = []
        for token in doc:
            escaped_text = escape(token.text)
            lemma = token.lemma_.lower()
            if token.is_alpha and not token.is_stop and lemma in query_lemmas:
                highlighted_parts.append(f"[b #8b0000 on #d4af37]{escaped_text}[/]{token.whitespace_}")
            else:
                highlighted_parts.append(f"{escaped_text}{token.whitespace_}")

        return "".join(highlighted_parts)

    def _formatear_label_sidebar(self, result: SearchResult) -> str:
        title = escape(self._truncar(result.chunk.titulo, 54))
        if result.modo == MODE_CLASSIC:
            return f"[{result.score:.4f}] C{result.chunk.chunk_id:03d} · {title}"
        if result.modo == MODE_SEMANTIC:
            return f"[cos {result.score:.4f}] C{result.chunk.chunk_id:03d} · {title}"
        if result.modo == MODE_RAG:
            return f"[rrf {result.score:.4f}] C{result.chunk.chunk_id:03d} · {title}"
        return f"C{result.chunk.chunk_id:03d} · {title}"

    def _formatear_metadata_resultado(self, result: SearchResult | None) -> str:
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

    def _truncar(self, texto: str, max_chars: int) -> str:
        return texto if len(texto) <= max_chars else f"{texto[: max_chars - 1]}…"

    def _sync_model_visibility(self) -> None:
        is_rag_mode = self.selected_mode == MODE_RAG
        model_field = self.query_one("#model-field", Vertical)
        model_input = self.query_one("#model-input", Input)

        model_field.set_class(not is_rag_mode, "model-hidden")
        model_input.disabled = not is_rag_mode

        if not is_rag_mode and model_input.has_focus:
            self.query_one("#mode-select", Select).focus()

    def action_focus_file(self) -> None:
        self.query_one("#file-input", Input).focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_focus_mode(self) -> None:
        self.query_one("#mode-select", Select).focus()

    def action_focus_model(self) -> None:
        if self.selected_mode != MODE_RAG:
            self.query_one("#mode-select", Select).focus()
            return
        self.query_one("#model-input", Input).focus()


def run() -> None:
    QuijoteApp().run()


if __name__ == "__main__":
    run()
