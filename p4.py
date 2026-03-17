from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ListItem, ListView, Input
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from bs4 import BeautifulSoup
from pathlib import Path
import re # Importamos esto para resaltar palabras respetando mayúsculas

class QuijoteApp(App):
    BINDINGS = [
        Binding("ctrl+a", "focus_file", "Cargar Archivo"),
        Binding("ctrl+b", "focus_search", "Buscar Palabra"),
        Binding("ctrl+q", "quit", "Salir")
    ]

    CSS = """
    Screen {
        background: #fdf6e3;
        color: #1a1a1a;
    }
    
    #sidebar { 
        width: 30%; 
        background: #2b3a42;
        color: #eee8d5;
        border-right: solid #d4af37;
    }
    
    ListItem { padding: 1; }
    ListItem:hover { background: #8b0000; }
    
    .inputs-container {
        dock: top;
        height: auto;
        background: #eee8d5;
        padding: 1 2;
        border-bottom: solid #d4af37;
    }
    
    Input {
        margin-bottom: 1;
        background: #fdf6e3;
        color: #1a1a1a;
        border: round #8b0000;
    }
    Input:focus { border: round #d4af37; }
    
    #reader-container { width: 70%; padding: 2 4; }
    #reader { height: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Vertical(classes="inputs-container"):
            yield Input(placeholder="1. Pega aquí la ruta del archivo HTML y pulsa Enter...", id="file-input")
            yield Input(placeholder="2. Escribe una palabra para buscar (déjalo en blanco para ver todos)...", id="search-input")

        with Horizontal():
            yield ListView(id="sidebar")
            with VerticalScroll(id="reader-container"):
                mensaje_inicio = "[b #8b0000]¡Bienvenido a la aventura![/b #8b0000]\n[i]Carga tu archivo del Quijote arriba para comenzar a leer.[/i]"
                yield Static(mensaje_inicio, id="reader", expand=True)
        
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

        try:
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            self.capitulos_datos = []
            capitulo_actual_titulo = "Prólogo / Inicio"
            capitulo_actual_texto = []
            
            elementos = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p'])
            
            for el in elementos:
                texto = el.get_text(strip=True)
                if not texto: continue
                
                texto_upper = texto.upper()
                es_titulo = (el.name in ['h1', 'h2', 'h3']) or texto_upper.startswith("CAPÍTULO") or texto_upper.startswith("CAPITULO")
                
                if es_titulo:
                    if capitulo_actual_texto:
                        self.capitulos_datos.append({
                            "titulo": capitulo_actual_titulo,
                            "texto": "\n\n".join(capitulo_actual_texto)
                        })
                    capitulo_actual_titulo = texto[:50] + "..." if len(texto) > 50 else texto
                    capitulo_actual_texto = []
                else:
                    capitulo_actual_texto.append(texto)
            
            if capitulo_actual_texto:
                self.capitulos_datos.append({
                    "titulo": capitulo_actual_titulo,
                    "texto": "\n\n".join(capitulo_actual_texto)
                })

            # Reiniciamos la búsqueda actual al cargar un archivo nuevo
            self.palabra_busqueda = ""
            self.actualizar_sidebar()
            
            reader.update(f"[b #8b0000]¡Archivo analizado con éxito![/b #8b0000]\nSe procesaron {len(self.capitulos_datos)} secciones.\n\nSelecciona un capítulo en la barra lateral para leer.")
            
        except Exception as e:
            reader.update(f"Error al leer el archivo: {e}")

    def actualizar_sidebar(self, filtro: str = "") -> int:
        """Actualiza la barra lateral filtrando por palabra si es necesario."""
        sidebar = self.query_one("#sidebar", ListView)
        sidebar.clear()
        
        capitulos_encontrados = 0
        for i, cap in enumerate(self.capitulos_datos):
            # Si no hay filtro, o si la palabra está en el texto del capítulo
            if not filtro or filtro in cap["texto"].lower():
                # Guardamos el índice original 'i' en el name para saber qué capítulo es
                sidebar.append(ListItem(Static(cap["titulo"]), name=str(i)))
                capitulos_encontrados += 1
                
        return capitulos_encontrados

    def ejecutar_busqueda(self, palabra: str) -> None:
        if not hasattr(self, 'capitulos_datos'):
            self.query_one("#reader").update("[red]Primero debes cargar un archivo HTML válido.[/red]")
            return
            
        self.palabra_busqueda = palabra.lower()
        
        # Filtramos la barra lateral basándonos en la palabra
        num_resultados = self.actualizar_sidebar(self.palabra_busqueda)
        reader = self.query_one("#reader", Static)
        
        if self.palabra_busqueda:
            if num_resultados > 0:
                reader.update(f"[b green]Búsqueda aplicada.[/b green]\n\nSe ha encontrado la palabra '[b]{palabra}[/b]' en {num_resultados} capítulos.\n\n[i]La barra lateral izquierda se ha filtrado. Haz clic en un capítulo para leerlo (la palabra aparecerá resaltada).[/i]")
            else:
                reader.update(f"No se encontraron capítulos que contengan la palabra: [i]{palabra}[/i]\n\n[i]La barra lateral está vacía.[/i]")
        else:
            reader.update("Búsqueda borrada. Mostrando todos los capítulos en la barra lateral de nuevo.")
        
        self.query_one("#reader-container").scroll_to(0, 0)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = int(event.item.name)
        contenido_original = self.capitulos_datos[idx]["texto"]
        titulo = self.capitulos_datos[idx]["titulo"]
        
        texto_a_mostrar = contenido_original
        
        # Si hay una búsqueda activa, filtramos los párrafos
        if hasattr(self, 'palabra_busqueda') and self.palabra_busqueda:
            # Dividimos el capítulo entero en párrafos individuales
            parrafos = contenido_original.split("\n\n")
            parrafos_filtrados = []
            
            # Preparamos el buscador para resaltar respetando mayúsculas
            patron = re.compile(re.escape(self.palabra_busqueda), re.IGNORECASE)
            
            # Recorremos cada párrafo
            for p in parrafos:
                if self.palabra_busqueda in p.lower():
                    # Si la palabra está, la resaltamos y guardamos el párrafo
                    p_resaltado = patron.sub(lambda m: f"[b #8b0000 on #d4af37]{m.group(0)}[/]", p)
                    parrafos_filtrados.append(p_resaltado)
            
            # Unimos los párrafos filtrados, poniendo unos puntos suspensivos entre ellos
            texto_a_mostrar = "\n\n[dim]...[/dim]\n\n".join(parrafos_filtrados)
            
            # Añadimos un pequeño aviso arriba para recordar que es una vista filtrada
            aviso = f"[italic #d4af37]Mostrando solo los párrafos que contienen '{self.palabra_busqueda}':[/]\n\n"
            texto_a_mostrar = aviso + texto_a_mostrar

        reader = self.query_one("#reader", Static)
        reader.update(f"[b #8b0000]{titulo}[/]\n\n{texto_a_mostrar}")
        
        self.query_one("#reader-container").scroll_to(0, 0)

    def action_focus_file(self) -> None:
        self.query_one("#file-input").focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input").focus()

if __name__ == "__main__":
    QuijoteApp().run()