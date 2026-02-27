"""Acciones concretas que el agente puede ejecutar dentro de Butler."""

from __future__ import annotations

import random
import time
from typing import Any

from loguru import logger
from rich.console import Console

from fdi_pln_p1.api_utils import api_request_modo
from fdi_pln_p1.parsing_utils import extraer_destino, to_int
from fdi_pln_p1.trade_strategy import OfertaMemoria, ajustar_oferta_no_repetida, es_oro

console = Console()


def ejecutar_aceptar(
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
        api_request_modo(url, "DELETE", f"/mail/{mid}", modo_puesto, agente=mi_nombre)


def ejecutar_borrar(
    args: dict[str, Any],
    mi_nombre: str,
    url: str,
    modo_puesto: str,
) -> bool:
    """Caso 2: borra una carta no útil."""
    mid = str(args.get("id_carta", "")).strip()
    if not mid:
        return False

    api_request_modo(url, "DELETE", f"/mail/{mid}", modo_puesto, agente=mi_nombre)
    console.print("🗑️ Carta descartada.")

    if random.randint(1, 3) == 1:
        console.print(
            "🎲 [1 de 3] Tras descartar, se activó la difusión masiva de ofertas."
        )
        return True
    return False


def ejecutar_enviar(
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
        api_request_modo(url, "DELETE", f"/mail/{mid}", modo_puesto, agente=mi_nombre)


def ejecutar_oferta_masiva(
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
    """Caso 4: envía oferta de intercambio a todos los jugadores."""
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
