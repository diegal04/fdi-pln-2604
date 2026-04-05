from __future__ import annotations

import os
from pathlib import Path

import spacy
from rich.align import Align
from rich.console import Group
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, ListItem, ListView, Select, Static

from src.modes.classic_mode import MODE_CLASSIC
from src.modes.rag_mode import MODE_RAG, generar_respuesta_ollama
from src.modes.semantic_mode import MODE_SEMANTIC
from src.orchestrator import orquestar_busqueda
from src.preprocessing import QuijoteIndex, SearchResult, TextAnalysis


MODE_BROWSE = "browse"


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
    #title-authors {
        height: 3;
        content-align: center middle;
        color: #8b0000;
        margin-bottom: 1;
    }
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
    .inputs-container { dock: top; height: auto; background: #eee8d5; padding: 1 2; border-bottom: solid #d4af37; }
    #search-row { height: auto; }
    #search-input { width: 1fr; margin-right: 1; }
    #mode-select { width: 21; margin-right: 1; }
    #model-input { width: 28; }
    Input, Select {
        margin-bottom: 1;
        background: #fdf9f0;
        color: #1a1a1a;
        border: round #8b0000;
    }
    Input:focus, Select:focus { border: round #d4af37; }
    #reader-container { width: 62%; padding: 2 3; }
    #reader { height: auto; }
    """

    DISPLAY_LIMIT = 30

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("es_core_news_lg")
        self.index = QuijoteIndex(self.nlp)
        self.selected_mode = MODE_CLASSIC
        self.current_query = ""
        self.current_query_analysis = TextAnalysis.empty()
        self.current_results: list[SearchResult] = []
        self.results_by_chunk_id: dict[int, SearchResult] = {}
        self.default_rag_model = os.getenv("P4_OLLAMA_MODEL", "qwen3:0.6b")
        self.default_corpus_path = Path(__file__).resolve().parent.parent / "2000-h.htm"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="inputs-container"):
            yield Static(
                "[b]PLN P4 - Motor de recuperacion y RAG sobre Don Quijote[/]\n"
                "Carlos Mantilla Mateos | Diego Alonso Arceiz",
                id="title-authors",
            )
            yield Input(
                value=str(self.default_corpus_path),
                placeholder="1. Pega la ruta del HTML del Quijote y pulsa Enter...",
                id="file-input",
            )
            with Horizontal(id="search-row"):
                yield Input(
                    placeholder="2. Escribe tu consulta y pulsa Enter...",
                    id="search-input",
                )
                yield Select(
                    [
                        ("1. Clásica", MODE_CLASSIC),
                        ("2. Semántica", MODE_SEMANTIC),
                        ("3. RAG", MODE_RAG),
                    ],
                    value=MODE_CLASSIC,
                    allow_blank=False,
                    prompt="Modo",
                    id="mode-select",
                )
                yield Input(
                    value=self.default_rag_model,
                    placeholder="Modelo Ollama",
                    id="model-input",
                )
        with Horizontal():
            yield ListView(id="sidebar")
            with VerticalScroll(id="reader-container"):
                yield Static(
                    "[b #8b0000]Quijote IR[/]\n\n"
                    "Carga el HTML, elige el modo y, si usas RAG, indica el modelo de Ollama.",
                    id="reader",
                    expand=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        if not self.default_corpus_path.exists():
            self.query_one("#reader", Static).update(
                "[b red]No se encontró el corpus por defecto.[/b red]\n\n"
                f"Ruta esperada: {escape(str(self.default_corpus_path))}"
            )
            return

        self.cargar_archivo(str(self.default_corpus_path))

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
        if self.current_query and self.index.total_chunks > 0:
            self.ejecutar_busqueda(self.current_query)

    def cargar_archivo(self, ruta_str: str) -> None:
        normalized_path = ruta_str.strip() or str(self.default_corpus_path)
        path = Path(normalized_path)
        reader = self.query_one("#reader", Static)
        self.query_one("#file-input", Input).value = str(path)

        if not path.exists():
            reader.update(f"[b red]Error:[/b red] No se encontró el archivo en {escape(str(path))}.")
            return

        reader.update("[i]Procesando HTML, construyendo chunks y calculando índices...[/i]")

        try:
            stats = self.index.cargar_archivo(path)
        except Exception as exc:
            reader.update(f"[b red]Error al procesar el archivo:[/b red] {escape(str(exc))}")
            return

        self.current_query = ""
        self.current_query_analysis = TextAnalysis.empty()
        self._mostrar_exploracion_inicial()
        reader.update(
            "[b #8b0000]Corpus indexado[/]\n\n"
            f"Secciones detectadas: {stats['sections']}\n"
            f"Pasajes indexados: {stats['chunks']}\n"
            f"Tamaño de chunk: {self.index.chunk_size_words} palabras\n"
            f"Overlap: {self.index.chunk_overlap_words} palabras\n"
            f"Modelo Ollama actual: {escape(self._obtener_modelo_ollama())}\n\n"
            "Escribe una consulta para buscar."
        )

    def ejecutar_busqueda(self, consulta: str) -> None:
        if self.index.total_chunks == 0:
            self.query_one("#reader", Static).update(
                "[b red]Antes debes cargar el HTML del Quijote.[/b red]"
            )
            return

        self.current_query = consulta.strip()
        if not self.current_query:
            self.current_query_analysis = TextAnalysis.empty()
            self._mostrar_exploracion_inicial()
            self.query_one("#reader", Static).update(
                "Consulta vacía. Mostrando una selección inicial de pasajes."
            )
            return

        execution = orquestar_busqueda(
            self.index,
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

        if self.selected_mode == MODE_RAG and self.current_query and self.index.total_chunks > 0:
            self.ejecutar_busqueda(self.current_query)
            return

        self.query_one("#reader", Static).update(
            "[b #8b0000]Modelo de Ollama actualizado[/]\n\n"
            f"Modelo actual: {escape(normalized)}"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.name is None:
            return

        chunk_id = int(event.item.name)
        chunk = self.index.chunk_by_id.get(chunk_id)
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
        initial = [
            SearchResult(chunk=chunk, score=0.0, modo=MODE_BROWSE)
            for chunk in self.index.chunks[: self.DISPLAY_LIMIT]
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
                "La consulta no contiene términos útiles tras eliminar stopwords. "
                "Prueba con nombres o conceptos más informativos."
            )
            return

        if not resultados:
            reader.update(
                "[b red]Sin resultados clásicos.[/b red]\n\n"
                f"Consulta lematizada: {', '.join(sorted(self.current_query_analysis.lemma_set))}"
            )
            return

        shown = min(len(resultados), self.DISPLAY_LIMIT)
        reader.update(
            "[b #8b0000]Búsqueda clásica[/]\n\n"
            f"Consulta lematizada: {', '.join(sorted(self.current_query_analysis.lemma_set))}\n"
            f"Resultados recuperados: {len(resultados)}\n"
            f"Mostrando: {shown}\n\n"
            "Selecciona un pasaje en la barra lateral para ver el texto con los lemas resaltados."
        )

    def _mostrar_resumen_semantico(self, resultados: list[SearchResult]) -> None:
        reader = self.query_one("#reader", Static)
        if self.current_query_analysis.embedding_norm == 0:
            reader.update(
                "La consulta no generó un embedding útil. "
                "Prueba con una frase con contenido léxico más claro."
            )
            return

        if not resultados:
            reader.update("[b red]Sin resultados semánticos.[/b red]")
            return

        shown = min(len(resultados), self.DISPLAY_LIMIT)
        reader.update(
            "[b #8b0000]Búsqueda semántica[/]\n\n"
            f"Pasajes ordenados por similitud coseno con el embedding de la consulta.\n"
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
                f"Top clásicos usados: {len(clasicos)}\n"
                f"Top semánticos usados: {len(semanticos)}\n\n"
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
        if not query_lemmas:
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
            return "[dim]Exploración manual del corpus.[/dim]"
        if result.modo == MODE_CLASSIC:
            return f"[dim]TF-IDF: {result.score:.4f}[/dim]"
        if result.modo == MODE_SEMANTIC:
            return f"[dim]Similitud coseno: {result.score:.4f}[/dim]"
        return (
            "[dim]"
            f"RRF: {result.score:.4f} | "
            f"clásico: {result.clasico_score:.4f} | "
            f"semántico: {result.semantico_score:.4f}"
            "[/dim]"
        )

    def _truncar(self, texto: str, max_chars: int) -> str:
        return texto if len(texto) <= max_chars else f"{texto[: max_chars - 1]}…"

    def action_focus_file(self) -> None:
        self.query_one("#file-input", Input).focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_focus_mode(self) -> None:
        self.query_one("#mode-select", Select).focus()

    def action_focus_model(self) -> None:
        self.query_one("#model-input", Input).focus()


def run() -> None:
    QuijoteApp().run()


if __name__ == "__main__":
    run()



