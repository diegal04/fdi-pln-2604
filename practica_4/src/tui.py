from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, Literal

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
from src.ui.indexing import IndexingWorkerResult, ProgressSnapshot
from src.ui.presenters import (
    MODE_BROWSE,
    format_result_metadata,
    format_sidebar_label,
    render_chunk_detail,
    render_classic_summary,
    render_index_cancelled,
    render_index_error,
    render_index_ready,
    render_initial_reader,
    render_loading_status,
    render_missing_default_corpus,
    render_missing_file,
    render_model_updated,
    render_rag_error,
    render_rag_success,
    render_semantic_summary,
)
from src.ui.styles import APP_CSS


class QuijoteApp(App):
    BINDINGS = [
        Binding("ctrl+a", "focus_file", "Cargar Archivo"),
        Binding("ctrl+b", "focus_search", "Buscar"),
        Binding("ctrl+m", "focus_mode", "Modo"),
        Binding("ctrl+o", "focus_model", "Modelo"),
        Binding("ctrl+q", "quit", "Salir"),
    ]

    CSS = APP_CSS
    DISPLAY_LIMIT = 20

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp: Any | None = None
        self.index: QuijoteIndex | None = None
        self.selected_mode = MODE_CLASSIC
        self.current_query = ""
        self.current_query_analysis = TextAnalysis.empty()
        self.current_results: list[SearchResult] = []
        self.results_by_chunk_id: dict[int, SearchResult] = {}

        self.default_rag_model = os.getenv("P4_OLLAMA_MODEL", "gemma4:e2b")
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
                            yield Static(
                                "MODO DE OPERACION (Ctrl+M)", classes="field-label"
                            )
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
                        with Vertical(
                            classes="field-card model-hidden", id="model-field"
                        ):
                            yield Static("Modelo", classes="field-label")
                            yield Input(
                                value=self.default_rag_model,
                                placeholder="Modelo Ollama (ej. gemma4:e2b)",
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
                    yield Static(render_initial_reader(), id="reader", expand=True)
        yield Footer()

    def on_mount(self) -> None:
        if not self.default_corpus_path.exists():
            self.index_state = "error"
            self._reader().update(
                render_missing_default_corpus(self.default_corpus_path)
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
                self._finalizar_indexacion_error(
                    "La indexacion termino sin resultado util."
                )
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

            self._reset_search_state()
            self._mostrar_exploracion_inicial()
            self._reader().update(
                render_index_ready(
                    result.path,
                    result.stats,
                    result.index.chunk_size_words,
                    result.index.chunk_overlap_words,
                    self._obtener_modelo_ollama(),
                )
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

        self._finalizar_indexacion_error(
            str(error) if error else "Error desconocido en indexacion."
        )

    def cargar_archivo(self, ruta_str: str) -> None:
        normalized_path = ruta_str.strip() or str(self.default_corpus_path)
        path = Path(normalized_path)
        self._file_input().value = str(path)

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
            self._reset_search_state()
            self._actualizar_sidebar([])
            self._reader().update(render_missing_file(path))
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
            self._set_progress(
                run_id, progress.stage, progress.completed, progress.total
            )

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
        self._reset_search_state()
        self._actualizar_sidebar([])

        if trigger == "search":
            self.loading_notice = "No hay indice listo. Se inicia la indexacion; cuando termine, vuelve a pulsar Enter en la consulta."
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
        self._reader().update(render_index_cancelled(message))

    def _finalizar_indexacion_error(self, reason: str) -> None:
        self.active_worker = None
        self.index_start_time = None
        self.index_state = "error"
        self.index = None
        self.indexed_path = None
        self.loading_notice = None
        self.active_stage = "Error"
        self._actualizar_sidebar([])
        self._reader().update(render_index_error(reason))

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

        self._reader().update(
            render_loading_status(
                snapshot.stage,
                snapshot.completed,
                snapshot.total,
                elapsed,
                self.loading_notice,
            )
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

    def ejecutar_busqueda(self, consulta: str) -> None:
        if self.index_state == "loading":
            self.loading_notice = "Hay una indexacion en curso. Espera a que termine y vuelve a pulsar Enter para ejecutar la consulta."
            self._render_loading_status()
            return

        index = self.index
        if self.index_state != "ready" or index is None:
            clean_query = consulta.strip()
            if not clean_query:
                self._reset_search_state()
                self._reader().update(
                    "Consulta vacia. Carga un archivo (Enter en ruta) o escribe una consulta con contenido."
                )
                return

            path_text = self._file_input().value.strip() or str(
                self.default_corpus_path
            )
            path = Path(path_text)
            if not path.exists():
                self.index_state = "error"
                self._reader().update(render_missing_file(path))
                return

            self._start_indexing(path, "search")
            return

        if index.total_chunks == 0:
            self._reader().update(
                "[b red]Antes debes cargar e indexar el HTML del Quijote.[/b red]"
            )
            return

        self.current_query = consulta.strip()
        if not self.current_query:
            self.current_query_analysis = TextAnalysis.empty()
            self._mostrar_exploracion_inicial()
            self._reader().update(
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
            self._reader().update(
                render_classic_summary(
                    self.current_query_analysis.lemma_set,
                    len(execution.mode_results),
                    self.DISPLAY_LIMIT,
                )
            )
            return

        if execution.mode == MODE_SEMANTIC:
            self._reader().update(
                render_semantic_summary(
                    self.current_query_analysis.embedding_norm,
                    execution.mode_results,
                    self.DISPLAY_LIMIT,
                )
            )
            return

        self._mostrar_respuesta_rag(
            execution.mode_results,
            execution.rag_classic_results,
            execution.rag_semantic_results,
        )

    def actualizar_modelo_ollama(self, modelo: str) -> None:
        normalized = modelo.strip() or self.default_rag_model
        model_input = self._model_input()
        model_input.value = normalized

        if self.index_state == "loading":
            self.loading_notice = "Modelo actualizado. Se aplicara cuando termine la indexacion y ejecutes una consulta en modo RAG."
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

        self._reader().update(render_model_updated(normalized))

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
        self._reader().update(
            render_chunk_detail(
                chunk.titulo,
                chunk.texto,
                self.current_query_analysis.lemma_set,
                format_result_metadata(result),
                self.nlp,
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
        sidebar = self._sidebar()
        sidebar.clear()

        self.current_results = resultados
        self.results_by_chunk_id = {
            result.chunk.chunk_id: result for result in resultados
        }

        for result in resultados:
            sidebar.append(
                ListItem(
                    Static(format_sidebar_label(result)),
                    name=str(result.chunk.chunk_id),
                )
            )

    def _mostrar_respuesta_rag(
        self,
        fusion: list[SearchResult],
        clasicos: list[SearchResult],
        semanticos: list[SearchResult],
    ) -> None:
        if not fusion:
            self._reader().update("[b red]RAG sin contexto suficiente.[/b red]")
            return

        modelo = self._obtener_modelo_ollama()
        try:
            answer = generar_respuesta_ollama(self.current_query, fusion, modelo)
        except Exception as exc:
            self._reader().update(
                render_rag_error(
                    str(exc),
                    modelo,
                    len(fusion),
                    len(clasicos),
                    len(semanticos),
                )
            )
            return

        self._reader().update(render_rag_success(answer, modelo, fusion))

    def _obtener_modelo_ollama(self) -> str:
        model_input = self._model_input()
        modelo = model_input.value.strip()
        if not modelo:
            modelo = self.default_rag_model
            model_input.value = modelo
        return modelo

    def _sync_model_visibility(self) -> None:
        is_rag_mode = self.selected_mode == MODE_RAG
        model_field = self.query_one("#model-field", Vertical)
        model_input = self._model_input()

        model_field.set_class(not is_rag_mode, "model-hidden")
        model_input.disabled = not is_rag_mode

        if not is_rag_mode and model_input.has_focus:
            self._mode_select().focus()

    def _reset_search_state(self) -> None:
        self.current_query = ""
        self.current_query_analysis = TextAnalysis.empty()

    def _file_input(self) -> Input:
        return self.query_one("#file-input", Input)

    def _model_input(self) -> Input:
        return self.query_one("#model-input", Input)

    def _mode_select(self) -> Select:
        return self.query_one("#mode-select", Select)

    def _reader(self) -> Static:
        return self.query_one("#reader", Static)

    def _sidebar(self) -> ListView:
        return self.query_one("#sidebar", ListView)

    def action_focus_file(self) -> None:
        self._file_input().focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_focus_mode(self) -> None:
        self._mode_select().focus()

    def action_focus_model(self) -> None:
        if self.selected_mode != MODE_RAG:
            self._mode_select().focus()
            return
        self._model_input().focus()


def run() -> None:
    QuijoteApp().run()


if __name__ == "__main__":
    run()
