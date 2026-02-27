"""Bucle principal del agente autónomo de Butler.

Gestiona el ciclo de vida del agente: consulta el estado del juego,
solicita decisiones al modelo Ollama y ejecuta las acciones correspondientes.
"""

from __future__ import annotations

import random
import time
from typing import Any

import ollama
from loguru import logger
from rich.console import Console

from fdi_pln_p1 import MODO_MONOPUESTO
from fdi_pln_p1.api_utils import api_request
from fdi_pln_p1.ollama_tools import OLLAMA_TOOLS
from fdi_pln_p1.prompts import construir_system_prompt, construir_user_prompt
from fdi_pln_p1.trade_strategy import (
    OfertaMemoria,
    ajustar_oferta_no_repetida,
    es_oro,
    normalizar_jugadores,
    parse_tool_arguments,
)

console = Console()


# ---------------------------------------------------------------------------
# Utilidades de la API con soporte de modo (monopuesto / multipuesto)
# ---------------------------------------------------------------------------

def construir_params_api(
    modo_puesto: str,
    agente: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Construye los parámetros de query string según el modo de la API.

    En modo monopuesto añade automáticamente el parámetro ``agente``.
    """
    params_finales = dict(params or {})
    if modo_puesto == MODO_MONOPUESTO and agente:
        params_finales["agente"] = agente
    return params_finales or None


def api_request_modo(
    base_url: str,
    metodo: str,
    endpoint: str,
    modo_puesto: str,
    agente: str | None = None,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | bool:
    """Envoltorio de :func:`api_request` que inyecta el agente según el modo."""
    return api_request(
        base_url,
        metodo,
        endpoint,
        params=construir_params_api(
            modo_puesto=modo_puesto, agente=agente, params=params
        ),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Funciones auxiliares de conversión
# ---------------------------------------------------------------------------

def to_int(valor: Any, default: int = 1) -> int:
    """Convierte *valor* a entero positivo (mínimo 1)."""
    try:
        return max(1, int(valor))
    except (TypeError, ValueError):
        return default


def extraer_destino(raw_dest: Any) -> str:
    """Extrae el alias del destinatario, aceptando ``str`` o ``dict``."""
    if isinstance(raw_dest, str):
        return raw_dest.strip()
    if isinstance(raw_dest, dict):
        alias = raw_dest.get("alias", "")
        return str(alias).strip()
    return ""


# ---------------------------------------------------------------------------
# Registro de alias
# ---------------------------------------------------------------------------

def registrar_alias(mi_nombre: str, url: str) -> None:
    """Registra el alias del jugador en el servidor Butler."""
    resultado = api_request(url, "POST", f"/alias/{mi_nombre}")
    if isinstance(resultado, dict) and resultado.get("status") == "ok":
        console.print(f"✅ Alias '{mi_nombre}' registrado correctamente")
    else:
        logger.warning("Alias no registrado")


# ---------------------------------------------------------------------------
# Manejadores de acciones (caso_1 … caso_4)
# ---------------------------------------------------------------------------

def _ejecutar_aceptar(
    args: dict[str, Any],
    mi_nombre: str,
    url: str,
    modo_puesto: str,
    mis_recursos: dict[str, int],
) -> None:
    """Caso 1: acepta una carta del buzón y envía recursos."""
    dest = extraer_destino(args.get("dest"))
    item = str(args.get("item_enviar", "")).strip()
    cant = to_int(args.get("cant"), 1)
    item_esperado = str(args.get("item_esperado", "recursos")).strip()
    cant_esperada = to_int(args.get("cant_esperada", 1))
    mid = str(args.get("id_carta", "")).strip()

    if not dest or not item or es_oro(item):
        logger.warning("caso_1_aceptar descartado por datos inválidos")
        return
    if mis_recursos.get(item, 0) <= 0:
        logger.warning(f"caso_1_aceptar sin stock de '{item}'")
        return

    cant = min(cant, mis_recursos.get(item, 0))
    cuerpo_mensaje = (
        f"Acepto. Aquí tienes {cant} de {item}. "
        f"Nos tienes que enviar {cant_esperada} de {item_esperado}."
    )

    api_request_modo(
        url,
        "POST",
        "/carta",
        modo_puesto,
        agente=mi_nombre,
        payload={
            "remi": mi_nombre,
            "dest": dest,
            "asunto": "Trato aceptado",
            "cuerpo": cuerpo_mensaje,
        },
    )
    api_request_modo(
        url,
        "POST",
        f"/paquete/{dest}",
        modo_puesto,
        agente=mi_nombre,
        payload={item: cant},
    )

    console.print(
        f"✅ Trato cerrado con {dest}, enviamos {cant} de {item} "
        f"y pedimos {cant_esperada} de {item_esperado}."
    )
    if mid:
        api_request_modo(
            url, "DELETE", f"/mail/{mid}", modo_puesto, agente=mi_nombre
        )


def _ejecutar_borrar(
    args: dict[str, Any],
    mi_nombre: str,
    url: str,
    modo_puesto: str,
) -> bool:
    """Caso 2: borra una carta no útil.

    Returns:
        ``True`` si se debe lanzar una oferta masiva a continuación.
    """
    mid = str(args.get("id_carta", "")).strip()
    if not mid:
        return False

    api_request_modo(
        url, "DELETE", f"/mail/{mid}", modo_puesto, agente=mi_nombre
    )
    console.print("🗑️ Carta descartada.")

    # 1 de cada 3 veces tras descartar, se fuerza difusión masiva
    if random.randint(1, 3) == 1:
        console.print(
            "🎲 [1 de 3] Tras descartar, se activó la difusión masiva de ofertas."
        )
        return True
    return False


def _ejecutar_enviar(
    args: dict[str, Any],
    mi_nombre: str,
    url: str,
    modo_puesto: str,
    mis_recursos: dict[str, int],
) -> None:
    """Caso 3: envía recursos para cumplir un trato aceptado."""
    dest = extraer_destino(args.get("dest"))
    item = str(args.get("item_enviar", "")).strip()
    cant = to_int(args.get("cant"), 1)
    mid = str(args.get("id_carta", "")).strip()

    if not dest or not item or es_oro(item):
        logger.warning("caso_3_enviar descartado por datos inválidos")
        return
    if mis_recursos.get(item, 0) <= 0:
        logger.warning(f"caso_3_enviar sin stock de '{item}'")
        return

    cant = min(cant, mis_recursos.get(item, 0))
    api_request_modo(
        url,
        "POST",
        f"/paquete/{dest}",
        modo_puesto,
        agente=mi_nombre,
        payload={item: cant},
    )
    console.print(f"📦 Material enviado a {dest} (Cumpliendo trato).")
    if mid:
        api_request_modo(
            url, "DELETE", f"/mail/{mid}", modo_puesto, agente=mi_nombre
        )


def _ejecutar_oferta_masiva(
    args: dict[str, Any],
    mi_nombre: str,
    url: str,
    modo_puesto: str,
    otros_jugadores: list[str],
    faltan: dict[str, int],
    sobran: dict[str, int],
    mis_recursos: dict[str, int],
    memoria_oferta: OfertaMemoria,
) -> bool:
    """Caso 4: envía oferta de intercambio a todos los jugadores.

    Returns:
        ``True`` si al menos una carta se envió con éxito.
    """
    if not otros_jugadores:
        logger.warning("No hay otros jugadores para ofertar")
        return False

    busco, doy, cambio = ajustar_oferta_no_repetida(
        recurso_que_busco=str(args.get("recurso_que_busco", "")).strip(),
        recurso_que_doy=str(args.get("recurso_que_doy", "")).strip(),
        faltan=faltan,
        sobran=sobran,
        mis_recursos=mis_recursos,
        memoria=memoria_oferta,
    )

    if not busco or not doy:
        logger.warning("No hay combinación válida para oferta masiva")
        return False
    if es_oro(doy):
        logger.warning("Oferta bloqueada: no se puede ofrecer oro")
        return False

    if cambio:
        logger.info(f"Oferta rotada automáticamente: busco={busco}, doy={doy}")

    mensaje = f"Necesito {busco}. Te doy {doy}. ¿Hacemos trato?"
    console.print(
        f"📢 DIFUNDIENDO OFERTA A {len(otros_jugadores)} JUGADORES "
        f"(busco {busco}, doy {doy})..."
    )

    enviados_ok = 0
    for jugador in otros_jugadores:
        payload = {
            "remi": mi_nombre,
            "dest": jugador,
            "asunto": f"Busco {busco}",
            "cuerpo": mensaje,
        }
        respuesta = api_request_modo(
            url, "POST", "/carta", modo_puesto, agente=mi_nombre, payload=payload
        )
        if isinstance(respuesta, dict) and respuesta.get("status") == "ok":
            enviados_ok += 1

    if enviados_ok > 0:
        memoria_oferta.recurso_que_busco = busco
        memoria_oferta.recurso_que_doy = doy
        console.print("✅ Rueda de ofertas enviada.")
        time.sleep(5)
        return True

    logger.warning("No se pudo enviar ninguna carta de oferta")
    return False


# ---------------------------------------------------------------------------
# Despacho de acciones
# ---------------------------------------------------------------------------

def _manejar_sin_tool_call(
    resp: dict,
    cartas_visibles: dict,
    mi_nombre: str,
    url: str,
    modo_puesto: str,
) -> None:
    """Gestiona el caso en que el modelo no invocó ninguna tool."""
    contenido_modelo = resp["message"].get("content", "").strip()
    if contenido_modelo:
        console.print(
            f"💬 [dim]Modelo generó texto:[/dim] {contenido_modelo[:200]}"
        )
    console.print(
        "[yellow]⚠️ El modelo no invocó ninguna tool, reintentando...[/yellow]"
    )
    # Si había una carta visible, la borramos para no quedarnos atascados
    if cartas_visibles:
        mid_atascado = list(cartas_visibles.keys())[0]
        api_request_modo(
            url, "DELETE", f"/mail/{mid_atascado}", modo_puesto, agente=mi_nombre
        )
        console.print(f"🗑️ Carta {mid_atascado} descartada por fallo del modelo.")


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
        _ejecutar_aceptar(args, mi_nombre, url, modo_puesto, mis_recursos)

    elif accion == "caso_2_borrar":
        lanzar_oferta_masiva = _ejecutar_borrar(args, mi_nombre, url, modo_puesto)
        if lanzar_oferta_masiva:
            # Vaciamos los argumentos para que el ajustador elija los recursos
            args = {"recurso_que_busco": "", "recurso_que_doy": ""}

    elif accion == "caso_3_enviar":
        _ejecutar_enviar(args, mi_nombre, url, modo_puesto, mis_recursos)

    elif accion == "caso_4_ofertar_todos":
        lanzar_oferta_masiva = True

    # Oferta masiva (caso 4 directo o trigger aleatorio tras borrar)
    if lanzar_oferta_masiva:
        _ejecutar_oferta_masiva(
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


# ---------------------------------------------------------------------------
# Iteración individual del agente
# ---------------------------------------------------------------------------

def _iterar_agente(
    mi_nombre: str,
    url: str,
    modelo: str,
    modo_puesto: str,
    memoria_oferta: OfertaMemoria,
) -> None:
    """Ejecuta una única iteración del bucle del agente."""
    console.print("\n[dim]" + "-" * 40 + "[/dim]")

    # ---- Obtener estado del juego ----
    info = api_request_modo(url, "GET", "/info", modo_puesto, agente=mi_nombre)
    api_request_modo(url, "GET", "/info", modo_puesto, agente="PROFESOR")
    gente_raw = api_request_modo(url, "GET", "/gente", modo_puesto, agente=mi_nombre)
    logger.info(gente_raw)

    if not isinstance(info, dict) or "Recursos" not in info:
        return

    mis_recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})
    buzon = {
        k: v for k, v in info.get("Buzon", {}).items() if v.get("dest") == mi_nombre
    }
    otros_jugadores = normalizar_jugadores(gente_raw, mi_nombre)
    logger.info(otros_jugadores)

    # ---- Calcular necesidades ----
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

    # ---- Consultar al modelo ----
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

    # ---- Procesar respuesta ----
    tool_calls = resp["message"].get("tool_calls")
    if not tool_calls:
        _manejar_sin_tool_call(resp, cartas_visibles, mi_nombre, url, modo_puesto)
        return

    tool_call = tool_calls[0]
    accion = tool_call["function"]["name"]
    args = parse_tool_arguments(tool_call["function"].get("arguments"))
    console.print(f"🧠 IA DICE: [bold]{accion}[/bold] | args={args}")

    # ---- Ejecutar acción ----
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


# ---------------------------------------------------------------------------
# Bucle principal del agente
# ---------------------------------------------------------------------------

def agente_autonomo(
    mi_nombre: str,
    url: str,
    modelo: str,
    modo_puesto: str,
) -> None:
    """Ejecuta el bucle principal del agente autónomo.

    Args:
        mi_nombre: Alias del jugador.
        url: URL base del servidor Butler.
        modelo: Nombre del modelo Ollama a utilizar.
        modo_puesto: Modo de operación (``monopuesto`` o ``multipuesto``).
    """
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

    while True:
        try:
            _iterar_agente(mi_nombre, url, modelo, modo_puesto, memoria_oferta)
        except Exception as exc:
            logger.exception(f"Error en iteración del agente: {exc}")

        time.sleep(2)
