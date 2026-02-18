# FDI PLN 2604 - Agente Butler

Este proyecto implementa un agente autónomo para el juego Butler usando Ollama con `tool calling`.

## Integrantes

- Grupo: **4**
- Carlos Mantilla Mateos
- Diego Alonso Arceiz

## Objetivo del agente

El agente analiza su estado en cada ronda (`Recursos`, `Objetivo`, `Buzon`) y decide una acción entre cuatro casos:

- `caso_1_aceptar`: aceptar un trato recibido.
- `caso_2_borrar`: borrar una carta no útil.
- `caso_3_enviar`: enviar recurso en acuerdos ya cerrados.
- `caso_4_ofertar_todos`: enviar oferta masiva cuando no hay cartas útiles.

## Reglas clave de estrategia

- No ofrece ni envía `oro`.
- En ofertas masivas intenta no repetir la misma pareja `(recurso_que_busco, recurso_que_doy)` en rondas consecutivas si hay alternativa.
- Solo usa recursos que realmente tiene disponibles.

## Estructura del código

- `src/fdi_pln_p1/main.py`: loop principal del agente, construcción de prompt y ejecución de acciones.
- `src/fdi_pln_p1/api_utils.py`: cliente HTTP simple para Butler.
- `src/fdi_pln_p1/trade_strategy.py`: utilidades de negociación (memoria de oferta, rotación de oferta, parseo de argumentos de tools y normalización de jugadores).
- `src/fdi_pln_p1/__init__.py`: configuración por defecto y variables de entorno.

## Ejecución

```bash
uv run fdi-pln-entrega --help
uv run fdi-pln-entrega --name "LOS ELEGIDOS" --model "qwen3-vl:4b" --butler-address "http://127.0.0.1:7719" --crear-alias
```

Opciones CLI:

- `--name`: alias del jugador.
- `--model`: modelo de Ollama.
- `--butler-address`: dirección del servidor Butler.
- `--crear-alias`: registra el alias antes de arrancar.
