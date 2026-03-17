from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ListItem, ListView, Input
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from bs4 import BeautifulSoup
from pathlib import Path
from collections import Counter
import math
import spacy

class QuijoteApp(App):
    BINDINGS = [
        Binding("ctrl+a", "focus_file", "Cargar Archivo"),
        Binding("ctrl+b", "focus_search", "Buscar Palabra"),
        Binding("ctrl+q", "quit", "Salir")
    ]

    CSS = """
    Screen { background: #fdf6e3; color: #1a1a1a; }
    #sidebar { width: 35%; background: #2b3a42; color: #eee8d5; border-right: solid #d4af37; }
    ListItem { padding: 1; }
    ListItem:hover { background: #8b0000; }
    .inputs-container { dock: top; height: auto; background: #eee8d5; padding: 1 2; border-bottom: solid #d4af37; }
    Input { margin-bottom: 1; background: #fdf6e3; color: #1a1a1a; border: round #8b0000; }
    Input:focus { border: round #d4af37; }
    #reader-container { width: 65%; padding: 2 4; }
    #reader { height: auto; }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("es_core_news_sm")
        self.query_lemas = set()
        
        # Variables globales para TF-IDF
        self.df_global = Counter() # En cuántos capítulos aparece cada lema
        self.total_capitulos = 0

    def obtener_lemas_y_conteos(self, texto: str) -> Counter:
        """Devuelve un Counter con las frecuencias de cada lema en el texto."""
        if not texto:
            return Counter()
        
        doc = self.nlp(texto.lower())
        lemas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return Counter(lemas)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="inputs-container"):
            yield Input(placeholder="1. Pega la ruta del archivo HTML y pulsa Enter...", id="file-input")
            yield Input(placeholder="2. Escribe tu búsqueda (ej. 'gigantes' o 'Sancho')...", id="search-input")
        with Horizontal():
            yield ListView(id="sidebar")
            with VerticalScroll(id="reader-container"):
                yield Static("[b #8b0000]¡Bienvenido![/]\n[i]Carga tu archivo del Quijote.[/i]", id="reader", expand=True)
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "file-input":
            self.cargar_archivo(event.value)
        elif event.input.id == "search-input":
            self.ejecutar_busqueda(event.value)

    def cargar_archivo(self, ruta_str: str) -> None:
        path = Path(ruta_str.strip())
        reader = self.query_one("#reader", Static)
        
        if not path.exists():
            reader.update(f"[b red]Error:[/b red] No se encontró el archivo en: {path}")
            return

        reader.update("[i]Procesando libro y calculando frecuencias TF-IDF...[/i]")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            self.capitulos_datos = []
            self.df_global.clear()
            id_original = 0
            
            capitulo_actual_titulo = "Prólogo / Inicio"
            capitulo_actual_texto = []
            
            elementos = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p'])
            
            def guardar_capitulo(titulo, texto_lineas, id_cap):
                texto_completo = "\n\n".join(texto_lineas)
                conteos = self.obtener_lemas_y_conteos(texto_completo)
                
                # Actualizamos el Document Frequency (DF) global
                for lema in conteos.keys():
                    self.df_global[lema] += 1
                    
                self.capitulos_datos.append({
                    "id": id_cap,
                    "titulo": titulo,
                    "texto": texto_completo,
                    "conteos": conteos,
                    "total_palabras": sum(conteos.values())
                })
            
            for el in elementos:
                texto = el.get_text(strip=True)
                if not texto: continue
                
                texto_upper = texto.upper()
                es_titulo = (el.name in ['h1', 'h2', 'h3']) or texto_upper.startswith("CAPÍTULO") or texto_upper.startswith("CAPITULO")
                
                if es_titulo:
                    if capitulo_actual_texto:
                        guardar_capitulo(capitulo_actual_titulo, capitulo_actual_texto, id_original)
                        id_original += 1
                    capitulo_actual_titulo = texto[:45] + "..." if len(texto) > 45 else texto
                    capitulo_actual_texto = []
                else:
                    capitulo_actual_texto.append(texto)
            
            if capitulo_actual_texto:
                guardar_capitulo(capitulo_actual_titulo, capitulo_actual_texto, id_original)
                
            self.total_capitulos = len(self.capitulos_datos)
            self.query_lemas = set()
            self.actualizar_sidebar()
            reader.update(f"[b #8b0000]¡Archivo procesado![/]\nSe indexaron {self.total_capitulos} secciones para TF-IDF.")
            
        except Exception as e:
            reader.update(f"Error: {e}")

    def calcular_score_tfidf(self, capitulo) -> float:
        """Calcula la relevancia de un capítulo basado en los lemas buscados."""
        score_total = 0.0
        for lema in self.query_lemas:
            # TF: (frecuencia del término en el capítulo) / (total de palabras del capítulo)
            tf = capitulo["conteos"].get(lema, 0) / capitulo["total_palabras"] if capitulo["total_palabras"] > 0 else 0
            
            # IDF: log(N / DF). Sumamos 1 al DF para evitar divisiones por cero por si acaso
            df = self.df_global.get(lema, 0)
            idf = math.log(self.total_capitulos / (1 + df)) if self.total_capitulos > 0 else 0
            
            score_total += (tf * idf)
            
        return score_total

    def actualizar_sidebar(self, resultados_ordenados=None) -> int:
        sidebar = self.query_one("#sidebar", ListView)
        sidebar.clear()
        
        # Si no hay búsqueda, mostramos todos en orden original
        if resultados_ordenados is None:
            for cap in self.capitulos_datos:
                sidebar.append(ListItem(Static(cap["titulo"]), name=str(cap["id"])))
            return len(self.capitulos_datos)
        
        # Si hay búsqueda, mostramos los ordenados por TF-IDF
        for cap, score in resultados_ordenados:
            titulo_con_score = f"[{score:.4f}] {cap['titulo']}"
            sidebar.append(ListItem(Static(titulo_con_score), name=str(cap["id"])))
            
        return len(resultados_ordenados)

    def ejecutar_busqueda(self, palabra: str) -> None:
        if not hasattr(self, 'capitulos_datos'):
            return
            
        self.query_lemas = set(self.obtener_lemas_y_conteos(palabra).keys())
        reader = self.query_one("#reader", Static)
        
        if not self.query_lemas:
            self.actualizar_sidebar()
            reader.update("Mostrando todos los capítulos.")
            return

        # Calculamos TF-IDF para cada capítulo
        resultados = []
        for cap in self.capitulos_datos:
            score = self.calcular_score_tfidf(cap)
            if score > 0: # Solo guardamos si hay alguna coincidencia
                resultados.append((cap, score))
                
        # Ordenamos de mayor a menor score
        resultados.sort(key=lambda x: x[1], reverse=True)
        
        num_resultados = self.actualizar_sidebar(resultados)
        
        if num_resultados > 0:
            reader.update(f"[b green]Búsqueda inteligente aplicada.[/]\n\nSe encontraron {num_resultados} capítulos.\nLos resultados en la barra lateral están ordenados por relevancia (TF-IDF).")
        else:
            reader.update("No se encontraron coincidencias relevantes.")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Recuperamos el ID original que guardamos en la propiedad 'name'
        idx = int(event.item.name)
        capitulo = self.capitulos_datos[idx]
        
        texto_a_mostrar = capitulo["texto"]
        
        # Lógica de resaltado (igual que antes)
        if self.query_lemas:
            parrafos = texto_a_mostrar.split("\n\n")
            parrafos_filtrados = []
            
            for p in parrafos:
                doc = self.nlp(p)
                p_lemas = {t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop}
                if not self.query_lemas.isdisjoint(p_lemas):
                    p_resaltado = ""
                    for token in doc:
                        lemma = token.lemma_.lower()
                        if token.is_alpha and not token.is_stop and lemma in self.query_lemas:
                            p_resaltado += f"[b #8b0000 on #d4af37]{token.text}[/]{token.whitespace_}"
                        else:
                            p_resaltado += token.text_with_ws
                    parrafos_filtrados.append(p_resaltado)
            
            texto_a_mostrar = "\n\n[dim]...[/dim]\n\n".join(parrafos_filtrados)
            texto_a_mostrar = f"[italic #d4af37]Mostrando solo párrafos relevantes:[/]\n\n" + texto_a_mostrar

        reader = self.query_one("#reader", Static)
        reader.update(f"[b #8b0000]{capitulo['titulo']}[/]\n\n{texto_a_mostrar}")

    def action_focus_file(self) -> None:
        self.query_one("#file-input").focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input").focus()

if __name__ == "__main__":
    QuijoteApp().run()