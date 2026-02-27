# FDI PLN 2604 — Agente Autónomo Butler

Agente autónomo de intercambio de recursos para el juego **Butler**, implementado en Python con [Ollama](https://ollama.com) y *tool calling*.

En cada ronda el agente consulta su estado en el servidor (inventario, objetivo y buzón), construye un prompt, llama al modelo LLM y ejecuta la acción decidida: aceptar tratos, borrar cartas inútiles, enviar recursos o lanzar ofertas masivas.

---

## Integrantes

| Nombre | Email |
|---|---|
| Carlos Mantilla Mateos | cmantill@ucm.es |
| Diego Alonso Arceiz | diegal04@ucm.es |

**Grupo:** 4

---

## Requisitos previos

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (gestor de entorno y dependencias)
- [Ollama](https://ollama.com) corriendo localmente con el modelo deseado descargado

```bash
ollama pull qwen3-vl:4b
```

---

## Instalación

```bash
git clone <repo-url>
cd fdi-pln-2604
uv sync
```

---

## Ejecución

```bash
# Ver ayuda completa
uv run fdi-pln-2604-p1 --help

# Uso básico (leyendo la configuracion de .env)
uv run fdi-pln-2604-p1

# Sobrescribir algun valor por CLI
uv run fdi-pln-2604-p1 \
  --name "LOS ELEGIDOS" \
  --model "qwen3-vl:4b" \
  --butler-address "http://147.96.81.252:7719" \
  --crear-alias \
  --modo-puesto monopuesto
```

### Opciones CLI

| Opción | Variable de entorno | Origen por defecto | Descripción |
|---|---|---|---|
| `--name` | `FDI_PLN__NAME` | `.env` | Alias del jugador |
| `--model` | `FDI_PLN__MODEL` | `.env` | Modelo Ollama a utilizar |
| `--butler-address` | `FDI_PLN__BUTLER_ADDRESS` | `.env` | URL del servidor Butler |
| `--crear-alias` | — | `False` | Registra el alias antes de arrancar |
| `--modo-puesto` | — | `monopuesto` | Modo de la API (ver abajo) |

> **Prioridad de configuración:** argumento CLI › `.env` › variable de entorno ya exportada.

#### Modos de la API

- **`monopuesto`** — añade `?agente=<alias>` a cada petición. Necesario cuando varios agentes comparten el mismo servidor con sus propios estados.
- **`multipuesto`** — usa los endpoints globales sin el parámetro `agente`.

---

## Estrategia del agente

En cada iteración el agente sigue este orden de prioridad al decidir su acción:

1. **`caso_3_enviar`** — Si hay una carta que confirma un trato aceptado, envía el recurso prometido.
2. **`caso_1_aceptar`** — Si hay una carta que ofrece un recurso necesario a cambio de uno sobrante, se acepta el trato.
3. **`caso_2_borrar`** — Si hay una carta que no resulta útil, se elimina del buzón.
4. **`caso_4_ofertar_todos`** — Si el buzón está vacío o ya se procesaron todas las cartas, se envía una oferta masiva a todos los jugadores.

### Reglas de negociación

- Nunca se ofrece, envía ni intercambia **oro**.
- La memoria de oferta evita repetir la misma pareja `(recurso_busco, recurso_doy)` en rondas consecutivas si existe alternativa.
- Solo se intercambian recursos de los que se dispone stock real.
- Tras borrar una carta, con probabilidad 1/3 se lanza una difusión masiva adicional para mantener el flujo de ofertas.

---

## Problemas surgidos

- **Procesamiento secuencial del buzón.** El modelo solo puede leer las cartas y decidir una a una. Si hay muchos jugadores conectados enviando cartas, la cantidad de mensajes recibidos puede crecer mucho más rápido que la velocidad de procesamiento. Leer siempre la primera carta puede hacer que una carta recién recibida tarde demasiado en procesarse; leer solo la última puede hacer que se ignoren cartas importantes; y descartar cartas aleatoriamente reduce el atasco, pero también puede provocar pérdida de información relevante.

- **Ambigüedad en los tratos aceptados.** Un problema recurrente aparece cuando se acepta un trato sin indicar cantidades explícitas. Por ejemplo, si un jugador ofrece aceite por queso, pero no especifica cuántas unidades espera recibir, el receptor puede enviar 3 de aceite y descubrir después que el otro jugador no dispone de 3 de queso, perdiendo materiales en el proceso.

- **Elección del modelo.** Es necesario buscar un equilibrio entre calidad y latencia. Un modelo razonador puede tomar mejores decisiones, pero penaliza demasiado el tiempo de respuesta. Una alternativa puede ser usar un modelo más pesado, pero no razonador, como `ministral3.2` de 24B; aun así, existe el riesgo de que no llegue a ejecutarse completamente en GPU en los ordenadores del laboratorio.

- **Falta de consenso en la confirmación de tratos.** Otro problema ha sido la ausencia de un protocolo común entre jugadores para cerrar intercambios. Para compensar la falta de memoria del agente, se decidió que, al aceptar un trato, el mensaje de confirmación indique explícitamente el material que debe devolverse.

- **Ventaja proporcional de quien más cartas envía.** Existe una relación clara entre el número de cartas enviadas y el aumento de materiales obtenido. Como en las pruebas también se utilizará el agente del profesor, que envía cartas con mayor frecuencia, se añadió un factor de aleatoriedad: al descartar una carta, se envía una oferta a todos los jugadores 1 de cada 3 veces, o el 100% de las veces si el buzón está vacío. Con ello se intenta aumentar el ritmo de crecimiento sin caer en spam constante.

---

## Futuras mejoras

- **Ajuste fino de configuración de Ollama y de las peticiones.** En esta práctica se ha priorizado la funcionalidad básica y no se han explorado en profundidad parámetros de entorno ni de inferencia, como el número de tokens de contexto recibidos. Ajustar estos valores podría mejorar el resultado, aunque con modelos pequeños, por debajo de 16B parámetros, hay que vigilar que no olviden los primeros tokens del contexto.

- **Procesamiento paralelo de cartas.** No se ha podido comprobar hasta qué punto parámetros como `max_num_parallel` permiten seguir aprovechando el 100% de la GPU de los ordenadores del laboratorio. Si fuese viable, esto permitiría procesar varias cartas en paralelo y reducir el cuello de botella del buzón.

- **Ofertas más agresivas con recursos sobrantes.** También sería interesante ofrecer de una vez todos los recursos sobrantes para aumentar la probabilidad de que el destinatario acepte el trato. En la implementación actual las ofertas se hacen de una en una, aunque sí se guarda la oferta realizada para que la siguiente carta pueda modificar, si procede, los materiales ofrecidos, los materiales solicitados o ambos.

---

## Arquitectura del proyecto

```
src/fdi_pln_p1/
├── __init__.py          # Carga de .env y constantes de configuración compartidas
├── api_utils.py         # Cliente HTTP y helpers de acceso al servidor Butler
├── display_utils.py     # Renderizado en consola de tablas y vistas auxiliares
├── main.py              # Punto de entrada CLI y validación de configuración
└── agent_config/        # Subpaquete con toda la lógica del agente autónomo
    ├── __init__.py
    ├── agent.py             # Orquestación del bucle principal y despacho de acciones
    ├── agent_actions.py     # Ejecución de los cuatro casos de acción del agente
    ├── ollama_tools.py      # Definición de las tools expuestas al modelo Ollama
    ├── parsing_utils.py     # Normalización ligera de argumentos y campos heterogéneos
    ├── prompts.py           # Construcción del system prompt y user prompt de cada iteración
    └── trade_strategy.py    # Lógica de negociación: memoria de oferta, rotación y utilidades
```

### Responsabilidad de cada módulo

| Módulo | Responsabilidad |
|---|---|
| `__init__.py` | Carga de `.env` y definición de variables de entorno y modos |
| `api_utils.py` | Abstracción HTTP sobre `httpx`, adaptación por modo y registro de alias |
| `display_utils.py` | Presentación en consola de información auxiliar del agente |
| `main.py` | Definición del comando Click y validación de los parámetros obligatorios |
| **`agent_config/`** | **Subpaquete con la lógica completa del agente** |
| `agent_config/agent.py` | `agente_autonomo` (bucle), `_iterar_agente`, `_despachar_accion` y fallback si el modelo no usa tools |
| `agent_config/agent_actions.py` | Implementación de `caso_1_aceptar`, `caso_2_borrar`, `caso_3_enviar` y `caso_4_ofertar_todos` |
| `agent_config/ollama_tools.py` | Esquemas JSON de las cuatro tools que el modelo puede invocar |
| `agent_config/parsing_utils.py` | Conversión de enteros y extracción robusta de destinatarios |
| `agent_config/prompts.py` | `construir_system_prompt` y `construir_user_prompt` |
| `agent_config/trade_strategy.py` | `OfertaMemoria`, `ajustar_oferta_no_repetida`, `normalizar_jugadores`, `parse_tool_arguments` |

---

## Configuración con .env

La configuración principal del agente vive en el archivo `.env` situado en la raíz del proyecto:

```bash
FDI_PLN__NAME=LOS ELEGIDOS
FDI_PLN__MODEL=qwen3-vl:4b
FDI_PLN__BUTLER_ADDRESS=http://147.96.81.252:7719
```

Si hace falta, cualquier valor puede sobrescribirse por línea de comandos. Si `.env` no define alguno, también se puede proporcionar como variable de entorno exportada en la sesión.
