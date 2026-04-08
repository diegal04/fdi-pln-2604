# PLN P4 - Motor de recuperación y RAG sobre Don Quijote

Aplicación TUI (Textual) en Python para buscar información en *Don Quijote de la Mancha* a partir de consultas en lenguaje natural, con tres estrategias de recuperación:

- Búsqueda clásica léxica (lemas + TF-IDF).
- Búsqueda semántica (embeddings + coseno).
- Modo RAG (fusión de rankings + generación con Ollama).

## Participantes

- Carlos Mantilla Mateos - cmantill@ucm.es
- Diego Alonso Arceiz - diegal04@ucm.es

## Objetivo del proyecto

Construir un sistema de Recuperación de Información en español que permita:

1. Buscar pasajes relevantes del Quijote por coincidencia léxica.
2. Recuperar pasajes semánticamente similares aunque no compartan las mismas palabras.
3. Generar una respuesta final apoyada en contexto recuperado (RAG), citando pasajes.

## Funcionalidades implementadas

- Carga dinámica de corpus HTML desde la interfaz.
- Precarga de la ruta por defecto `2000-h.htm` en la TUI.
- Indexación manual o bajo demanda desde la TUI.
- Indexación en background con progreso por fases, ETA y reinicio cancelable.
- Preprocesado lingüístico con spaCy (`es_core_news_lg`).
- Chunking con solape configurable para preservar contexto.
- Búsqueda clásica con ranking TF-IDF propio.
- Búsqueda semántica con embedding denso por chunk y similitud coseno.
- Fusión híbrida de resultados con Reciprocal Rank Fusion (RRF).
- Generación de respuesta con Ollama usando solo el contexto recuperado.
- Visualización lateral de resultados y lectura detallada del pasaje seleccionado.
- Resaltado de lemas de la consulta dentro del texto del pasaje.
- Exploración inicial del corpus cuando el índice ya está listo y no hay consulta.
- Reejecución automática al cambiar de modo, y regeneración RAG al cambiar de modelo.
- Atajos de teclado para flujo rápido en la TUI.

## Requisitos

- Python `>=3.12`
- `uv`
- Modelo de spaCy en español (`es_core_news_lg`) (se instala con las dependencias del proyecto)
- Opcional para RAG: Ollama activo localmente + modelo descargado

## Ejecución

```bash
uv sync
uv run p4.py
```

Si vas a usar RAG, asegúrate de tener Ollama levantado y un modelo disponible. Por defecto la app usa `gemma4:e2b`, pero el usuario puede escribir cualquier otro modelo en la interfaz o sobrescribirlo con `P4_OLLAMA_MODEL`.

La app arranca con la ruta por defecto rellenada, pero no indexa el corpus hasta que pulses `Enter` en la ruta o lances una consulta no vacía.

## Flujo completo (end-to-end)

1. La app arranca con `2000-h.htm` rellenado en el campo de ruta y con modo `clásico` seleccionado.
2. La indexación comienza cuando el usuario pulsa `Enter` en la ruta o escribe una consulta no vacía sin índice disponible.
3. Un worker en background carga `es_core_news_lg` si todavía no estaba en memoria.
4. El HTML se parsea y se extraen secciones y párrafos.
5. Se crean chunks de texto con tamaño objetivo de 180 palabras y overlap de 45.
6. Cada chunk se analiza con spaCy para obtener lemas, conteos y vectores.
7. Se construyen en memoria los datos para ranking clásico y semántico.
8. La TUI muestra progreso por fase, porcentaje y ETA durante la indexación.
9. Cuando el índice está listo, se muestra una exploración inicial con los primeros pasajes del corpus.
10. En cada consulta se ejecuta el modo seleccionado (`clásico`, `semántico` o `rag`).
11. Se muestran resultados en sidebar y contenido/metadata en panel lector.

## Arquitectura y módulos

- `p4.py`
  - Punto de entrada de la app.
  - Arranca la TUI definida en `src/tui.py`.

- `src/tui.py`
  - Interfaz Textual principal y coordinación de eventos de usuario.
  - Gestiona carga de archivo, búsquedas y ciclo de indexación.

- `src/ui/`
  - `styles.py`: CSS de Textual extraído del archivo principal.
  - `presenters.py`: renderizadores y helpers de formateo para sidebar/panel lector.
  - `indexing.py`: dataclasses auxiliares del flujo de indexación en background.

- `src/orchestrator.py`
  - Orquesta la ejecución de los modos (`classic`, `semantic`, `rag`).
  - Centraliza la selección de resultados para sidebar y panel principal.

- `src/preprocessing.py`
  - Define los dataclasses base: `TextAnalysis`, `ChunkRecord`, `SearchResult`.
  - Implementa `QuijoteIndex`:
    - Extracción de secciones del HTML.
    - Chunking con solape.
    - Análisis lingüístico y construcción de índices.
    - Cálculo de score TF-IDF.
    - Cálculo de similitud coseno.

- `src/modes/classic_mode.py`
  - Ejecuta ranking clásico por TF-IDF sobre todos los chunks.

- `src/modes/semantic_mode.py`
  - Ejecuta ranking semántico por coseno entre embedding de consulta y de cada chunk.

- `src/modes/rag_mode.py`
  - Recupera top-k clásico + top-k semántico.
  - Fusiona rankings con RRF.
  - Llama a Ollama para respuesta final condicionada por contexto.

- `2000-h.htm`
  - Corpus base (Project Gutenberg).

## Implementación técnica en detalle

### 1) Extracción de secciones desde HTML

`QuijoteIndex._extraer_secciones` recorre etiquetas `h1`, `h2`, `h3`, `h4` y `p`.
Una etiqueta se trata como título si:

- Es `h1`, `h2` o `h3`, o
- Su texto empieza por `CAPÍTULO` (con o sin tilde en el texto fuente).

Esto permite estructurar el texto por bloques narrativos antes del chunking.

### 2) Chunking con overlap

Decisiones de chunking:

- `chunk_size_words = 180`
- `chunk_overlap_words = 45`

Estrategia:

- Si un párrafo supera 180 palabras, se segmenta internamente con solape.
- Luego se agrupan párrafos en chunks hasta aproximar 180 palabras.
- El siguiente chunk retrocede por párrafos hasta acumular ~45 palabras de overlap.

Motivo: balancear contexto suficiente para semántica/RAG sin perder granularidad para ranking.

### 3) Análisis lingüístico y representación

Para cada chunk y consulta:

- Se eliminan tokens no alfabéticos y stopwords.
- Se lematiza (`token.lemma_.lower()`).
- Se guardan:
  - `conteos` por lema.
  - `lemma_set`.
  - `total_terminos`.

Además, para la parte semántica:

- Se acumulan vectores de tokens con embedding (`token.has_vector`).
- Se pondera por IDF por lema para reducir peso de términos muy frecuentes.
- Se obtiene un embedding final promedio ponderado + su norma L2.

### 4) Ranking clásico (TF-IDF)

Score por lema:

- `tf = frecuencia_en_chunk / total_terminos_chunk`
- `idf = log((1 + N) / (1 + df)) + 1`
- `peso_query = 1 + log(frecuencia_en_query)`

Score final del chunk:

- `sum(tf * idf * peso_query)` para todos los lemas de la consulta.

### 5) Ranking semántico

Se calcula similitud coseno entre embedding de consulta y embedding de chunk:

- `cos = dot(q, d) / (||q|| * ||d||)`

Solo se mantienen scores positivos.

### 6) Fusión para RAG (híbrido)

`src/modes/rag_mode.py` recupera:

- Top `8` clásicos.
- Top `8` semánticos.

Luego aplica RRF:

- `score_rrf += 1 / (60 + rank)`

Se devuelven top `6` pasajes fusionados para construir el prompt de generación y poblar la barra lateral en modo RAG.

### 7) Generación con Ollama

`generar_respuesta_ollama`:

- Construye bloques de contexto etiquetados como `[C{id}]`.
- Usa un prompt de sistema con tres restricciones:
  - Responder en español.
  - Usar solo pasajes recuperados.
  - Citar referencias en formato `[C12]`.

Si falla Ollama, la app mantiene el contexto recuperado visible para inspección manual.

## Interfaz (TUI) y experiencia de uso

Entradas principales:

- Ruta de archivo HTML.
- Consulta.
- Selector de modo.
- Modelo de Ollama (solo visible y editable en modo RAG).

Atajos de teclado:

- `Ctrl+A`: foco en ruta de archivo.
- `Ctrl+B`: foco en consulta.
- `Ctrl+M`: foco en selector de modo.
- `Ctrl+O`: foco en modelo Ollama.
- `Ctrl+Q`: salir.

Comportamiento importante:

- Si no hay índice y escribes una consulta no vacía, la app inicia la indexación bajo demanda.
- Si la consulta dispara una indexación bajo demanda, cuando termine debes volver a pulsar `Enter` para ejecutar la búsqueda.
- Si cambias de archivo mientras una indexación sigue en curso, la indexación anterior se cancela y se reinicia con la nueva ruta.
- Si lanzas una búsqueda mientras se está indexando, la app no la encola; muestra el estado actual y pide reintentar al terminar.
- Si la consulta está vacía y el índice ya existe, la TUI vuelve al modo de exploración inicial del corpus.
- Si falta el corpus por defecto o la ruta indicada no existe, la TUI muestra error y permite reintentar con otra ruta.
- Si cambias de modo y hay consulta activa, la búsqueda se recalcula automáticamente.
- Si cambias modelo en modo RAG, la respuesta se regenera automáticamente.
- La barra lateral muestra score según modo (`TF-IDF`, `cos`, `rrf`).
- Al abrir un resultado clásico, se resaltan lemas de la consulta en el texto.

## Decisiones de diseño y por qué se tomaron

- Modelo único en memoria:
  - Simplifica el proyecto y evita complejidad de persistencia/vector DB.
  - Adecuado para corpus único y tamaño manejable.

- Lematización + stopwords para IR clásico:
  - Mejora recall en español frente a matching literal.

- Embeddings con ponderación IDF:
  - Reduce influencia de términos demasiado frecuentes.
  - Mejora discriminación semántica frente a promedio simple.

- RRF en vez de mezcla lineal de scores:
  - Robusto cuando las escalas de score clásico y semántico son distintas.
  - Fácil de interpretar y ajustar.


## Estructura del repositorio

```text
PLN_p4/
|- p4.py
|- src/
|  |- __init__.py
|  |- tui.py
|  |- orchestrator.py
|  |- preprocessing.py
|  |- ui/
|  |  |- __init__.py
|  |  |- indexing.py
|  |  |- presenters.py
|  |  `- styles.py
|  `- modes/
|     |- __init__.py
|     |- classic_mode.py
|     |- semantic_mode.py
|     `- rag_mode.py
|- 2000-h.htm
|- pyproject.toml
`- README.md
```

## Notas de reproducibilidad

- El corpus incluido (`2000-h.htm`) ocupa ~2.3 MB.
- Con el parser y chunking actuales, este corpus produce `137` secciones y `3632` chunks.
- Con parámetros por defecto (`180/45`), el chunking genera miles de pasajes para recuperar contexto fino.
- La app carga `spacy.load("es_core_news_lg")` en la primera indexación; si falta el modelo, la ejecución fallará en ese momento.
- El índice no se persiste en disco: se reconstruye en memoria cada vez que se indexa un HTML.
