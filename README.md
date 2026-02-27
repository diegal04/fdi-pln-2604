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
uv run fdi-pln-entrega --help

# Uso básico (modo monopuesto, valores por defecto)
uv run fdi-pln-entrega

# Ejemplo completo
uv run fdi-pln-entrega \
  --name "LOS ELEGIDOS" \
  --model "qwen3-vl:4b" \
  --butler-address "http://147.96.81.252:7719" \
  --crear-alias \
  --modo-puesto monopuesto
```

### Opciones CLI

| Opción | Variable de entorno | Defecto | Descripción |
|---|---|---|---|
| `--name` | `FDI_PLN__NAME` | `LOS ELEGIDOS` | Alias del jugador |
| `--model` | `FDI_PLN__MODEL` | `qwen3-vl:4b` | Modelo Ollama a utilizar |
| `--butler-address` | `FDI_PLN__BUTLER_ADDRESS` | `http://147.96.81.252:7719` | URL del servidor Butler |
| `--crear-alias` | — | `False` | Registra el alias antes de arrancar |
| `--modo-puesto` | — | `monopuesto` | Modo de la API (ver abajo) |

> **Prioridad de configuración:** argumento CLI › variable de entorno › valor por defecto.

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

## Arquitectura del proyecto

```
src/fdi_pln_p1/
├── __init__.py          # Constantes, valores por defecto y resolución de variables de entorno
├── api_utils.py         # Cliente HTTP para el servidor Butler (GET / POST / DELETE)
├── trade_strategy.py    # Lógica de negociación: memoria de oferta, rotación y utilidades
├── ollama_tools.py      # Definición de las tools expuestas al modelo Ollama
├── prompts.py           # Construcción del system prompt y user prompt de cada iteración
├── agent.py             # Bucle principal del agente y manejadores de cada acción
└── main.py              # Punto de entrada CLI (Click + Dynaconf)
```

### Responsabilidad de cada módulo

| Módulo | Responsabilidad |
|---|---|
| `__init__.py` | Configuración global: URLs, modelos, modos y resolución desde el entorno |
| `api_utils.py` | Abstracción HTTP sobre `httpx` con manejo de errores y logging |
| `trade_strategy.py` | `OfertaMemoria`, `ajustar_oferta_no_repetida`, `normalizar_jugadores`, `parse_tool_arguments` |
| `ollama_tools.py` | Esquemas JSON de las cuatro tools que el modelo puede invocar |
| `prompts.py` | `construir_system_prompt` y `construir_user_prompt` |
| `agent.py` | `agente_autonomo` (bucle), `_iterar_agente`, `_despachar_accion` y los cuatro ejecutores |
| `main.py` | Definición del comando Click y bootstrap con Dynaconf |

---

## Variables de entorno

Las variables de entorno permiten configurar el agente sin tocar el código ni los argumentos CLI:

```bash
export FDI_PLN__NAME="MI EQUIPO"
export FDI_PLN__MODEL="llama3.2:3b"
export FDI_PLN__BUTLER_ADDRESS="http://127.0.0.1:7719"
```
