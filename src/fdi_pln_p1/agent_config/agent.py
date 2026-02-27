"""Bucle principal del agente autónomo de Butler.

Orquesta el ciclo de vida del agente: consulta el estado del juego,
solicita decisiones al modelo Ollama y despacha las acciones correspondientes.
"""

from __future__ import annotations

import time
from typing import Any

import ollama
from loguru import logger
from rich.console import Console

from fdi_pln_p1 import MODO_MONOPUESTO
from fdi_pln_p1.agent_config.agent_actions import (
    ejecutar_aceptar,
    ejecutar_borrar,
    ejecutar_enviar,
    ejecutar_oferta_masiva,
)
from fdi_pln_p1.api_utils import api_request_modo
from fdi_pln_p1.display_utils import mostrar_jugadores_partida
from fdi_pln_p1.agent_config.ollama_tools import OLLAMA_TOOLS
from fdi_pln_p1.agent_config.prompts import (
    construir_system_prompt,
    construir_user_prompt,
)
from fdi_pln_p1.agent_config.trade_strategy import (
    OfertaMemoria,
    es_oro,
    normalizar_jugadores,
    parse_tool_arguments,
)

console = Console()


def agente_autonomo(
    mi_nombre: str,
    url: str,
    modelo: str,
    modo_puesto: str,
) -> None:
    """Ejecuta el bucle principal del agente autónomo."""
    modo_api = "Monopuesto" if modo_puesto == MODO_MONOPUESTO else "Multipuesto"
    console.print(
        f"[bold green]🎮 JUGADOR ACTIVO:[/bold green] {mi_nombre} "
        f"[dim](Modo: Negociación Masiva | API: {modo_api})[/dim]"
    )
    logger.info(
        f"Agente iniciado | name={mi_nombre} | model={modelo} "
        f"| url={url} | modo={modo_puesto}"
    )

    memoria_oferta = OfertaMemoria()
    reintentos_sin_tool = [0]

    while True:
        try:
            _iterar_agente(
                mi_nombre,
                url,
                modelo,
                modo_puesto,
                memoria_oferta,
                reintentos_sin_tool,
            )
        except Exception as exc:
            logger.exception(f"Error en iteración del agente: {exc}")

        time.sleep(2)


def _iterar_agente(
    mi_nombre: str,
    url: str,
    modelo: str,
    modo_puesto: str,
    memoria_oferta: OfertaMemoria,
    reintentos_sin_tool: list[int],
) -> None:
    """Ejecuta una única iteración del bucle del agente."""
    console.print("\n[dim]" + "-" * 40 + "[/dim]")

    info = api_request_modo(url, "GET", "/info", modo_puesto, agente=mi_nombre)
    gente_raw = api_request_modo(url, "GET", "/gente", modo_puesto, agente=mi_nombre)

    if not isinstance(info, dict) or "Recursos" not in info:
        return

    mis_recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})
    buzon = {
        k: v for k, v in info.get("Buzon", {}).items() if v.get("dest") == mi_nombre
    }
    otros_jugadores = normalizar_jugadores(gente_raw, mi_nombre)
    mostrar_jugadores_partida(gente_raw, mi_nombre)

    faltan = {
        k: v - mis_recursos.get(k, 0)
        for k, v in objetivo.items()
        if mis_recursos.get(k, 0) < v
    }
    sobran = {
        k: v - objetivo.get(k, 0)
        for k, v in mis_recursos.items()
        if v > objetivo.get(k, 0) and not es_oro(k)
    }
    cartas_visibles = dict(list(buzon.items())[:1])

    console.print(f"🎒 TENGO: {mis_recursos}")
    console.print(f"🎯 FALTA: {faltan}")
    console.print(f"🔄 SOBRA: {sobran}")
    console.print(cartas_visibles)

    system_prompt = construir_system_prompt(mi_nombre)
    user_prompt = construir_user_prompt(
        faltan=faltan,
        sobran=sobran,
        otros_jugadores=otros_jugadores,
        cartas_visibles=cartas_visibles,
        memoria_oferta_busco=memoria_oferta.recurso_que_busco,
        memoria_oferta_doy=memoria_oferta.recurso_que_doy,
    )

    try:
        resp = ollama.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=OLLAMA_TOOLS,
        )
        logger.debug(f"Respuesta raw modelo: {resp}")
    except Exception as exc:
        logger.exception(f"Error en llamada a Ollama: {exc}")
        return

    tool_calls = resp["message"].get("tool_calls")
    if not tool_calls:
        _manejar_sin_tool_call(
            resp,
            cartas_visibles,
            mi_nombre,
            url,
            modo_puesto,
            reintentos_sin_tool,
        )
        return

    reintentos_sin_tool[0] = 0
    tool_call = tool_calls[0]
    accion = tool_call["function"]["name"]
    args = parse_tool_arguments(tool_call["function"].get("arguments"))
    console.print(f"🧠 IA DICE: [bold]{accion}[/bold] | args={args}")

    _despachar_accion(
        accion=accion,
        args=args,
        mi_nombre=mi_nombre,
        url=url,
        modo_puesto=modo_puesto,
        mis_recursos=mis_recursos,
        otros_jugadores=otros_jugadores,
        faltan=faltan,
        sobran=sobran,
        memoria_oferta=memoria_oferta,
    )


def _despachar_accion(
    accion: str,
    args: dict[str, Any],
    mi_nombre: str,
    url: str,
    modo_puesto: str,
    mis_recursos: dict[str, int],
    otros_jugadores: list[str],
    faltan: dict[str, int],
    sobran: dict[str, int],
    memoria_oferta: OfertaMemoria,
) -> None:
    """Despacha la acción decidida por el modelo al manejador correspondiente."""
    lanzar_oferta_masiva = False

    if accion == "caso_1_aceptar":
        ejecutar_aceptar(args, mi_nombre, url, modo_puesto, mis_recursos)
    elif accion == "caso_2_borrar":
        lanzar_oferta_masiva = ejecutar_borrar(args, mi_nombre, url, modo_puesto)
        if lanzar_oferta_masiva:
            args = {"recurso_que_busco": "", "recurso_que_doy": ""}
    elif accion == "caso_3_enviar":
        ejecutar_enviar(args, mi_nombre, url, modo_puesto, mis_recursos)
    elif accion == "caso_4_ofertar_todos":
        lanzar_oferta_masiva = True

    if lanzar_oferta_masiva:
        ejecutar_oferta_masiva(
            args,
            mi_nombre,
            url,
            modo_puesto,
            otros_jugadores,
            faltan,
            sobran,
            mis_recursos,
            memoria_oferta,
        )


def _manejar_sin_tool_call(
    resp: dict,
    cartas_visibles: dict,
    mi_nombre: str,
    url: str,
    modo_puesto: str,
    reintentos_sin_tool: list[int],
) -> None:
    """Gestiona el caso en que el modelo no invocó ninguna tool.

    Incrementa el contador de reintentos consecutivos. Solo borra la
    primera carta del buzón tras 3 fallos seguidos para evitar eliminar
    cartas que podrían procesarse en un reintento inmediato.
    """
    reintentos_sin_tool[0] += 1
    contenido_modelo = resp["message"].get("content", "").strip()
    if contenido_modelo:
        console.print(f"💬 [dim]Modelo generó texto:[/dim] {contenido_modelo[:200]}")
    console.print(
        f"[yellow]⚠️ El modelo no invocó ninguna tool "
        f"(intento {reintentos_sin_tool[0]}/3), reintentando...[/yellow]"
    )
    if cartas_visibles and reintentos_sin_tool[0] >= 3:
        mid_atascado = list(cartas_visibles.keys())[0]
        api_request_modo(
            url, "DELETE", f"/mail/{mid_atascado}", modo_puesto, agente=mi_nombre
        )
        console.print(f"🗑️ Carta {mid_atascado} descartada tras 3 fallos del modelo.")
        reintentos_sin_tool[0] = 0
