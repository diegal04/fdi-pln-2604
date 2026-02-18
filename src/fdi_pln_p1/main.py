import json
import sys
import time
import urllib3
from typing import Any

import click
import ollama
from dynaconf import Dynaconf
from loguru import logger
from rich.console import Console

from fdi_pln_p1 import (
    BUTLER_ADDRESS,
    ENV_BUTLER_ADDRESS,
    ENV_MODEL_NAME,
    ENV_PLAYER_NAME,
    MODEL_NAME,
    PLAYER_NAME,
)
from fdi_pln_p1.api_utils import api_request
from fdi_pln_p1.trade_strategy import (
    OfertaMemoria,
    ajustar_oferta_no_repetida,
    es_oro,
    normalizar_jugadores,
    parse_tool_arguments,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
console = Console()

logger.remove()
logger.add(sys.stderr, level="INFO")


def agente_autonomo(mi_nombre: str, url: str, modelo: str) -> None:
    """
    Función principal del agente.
    Se declara al inicio para que sea lo primero visible al abrir el archivo.
    """
    _agente_autonomo(mi_nombre=mi_nombre, url=url, modelo=modelo)


# --- Definición de tools para Ollama ---
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "caso_1_aceptar",
            "description": (
                "Aceptar un trato recibido. Usar cuando una carta ofrece algo que "
                "NECESITAS y pide algo que TIENES DE SOBRA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dest": {
                        "type": "string",
                        "description": "Nombre del jugador destinatario.",
                    },
                    "item_enviar": {
                        "type": "string",
                        "description": "Recurso que envías a cambio.",
                    },
                    "cant": {
                        "type": "integer",
                        "description": "Cantidad del recurso que envías.",
                    },
                    "id_carta": {
                        "type": "string",
                        "description": "ID de la carta que aceptas.",
                    },
                },
                "required": ["dest", "item_enviar", "cant", "id_carta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "caso_2_borrar",
            "description": (
                "Borrar una carta que no interesa o pide algo que no tienes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id_carta": {
                        "type": "string",
                        "description": "ID de la carta a borrar.",
                    },
                },
                "required": ["id_carta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "caso_3_enviar",
            "description": (
                "Enviar material para cumplir un acuerdo previo aceptado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dest": {
                        "type": "string",
                        "description": "Nombre del jugador destinatario.",
                    },
                    "item_enviar": {
                        "type": "string",
                        "description": "Recurso que envías.",
                    },
                    "cant": {
                        "type": "integer",
                        "description": "Cantidad del recurso que envías.",
                    },
                    "id_carta": {
                        "type": "string",
                        "description": "ID de la carta del acuerdo.",
                    },
                },
                "required": ["dest", "item_enviar", "cant", "id_carta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "caso_4_ofertar_todos",
            "description": (
                "Enviar oferta masiva a todos los jugadores cuando NO hay cartas "
                "útiles o el buzón está vacío."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recurso_que_busco": {
                        "type": "string",
                        "description": "Recurso que necesitas.",
                    },
                    "recurso_que_doy": {
                        "type": "string",
                        "description": "Recurso que ofreces a cambio.",
                    },
                },
                "required": ["recurso_que_busco", "recurso_que_doy"],
            },
        },
    },
]


def to_int(valor: Any, default: int = 1) -> int:
    try:
        return max(1, int(valor))
    except (TypeError, ValueError):
        return default


def extraer_destino(raw_dest: Any) -> str:
    if isinstance(raw_dest, str):
        return raw_dest.strip()
    if isinstance(raw_dest, dict):
        alias = raw_dest.get("alias", "")
        return str(alias).strip()
    return ""


def _agente_autonomo(mi_nombre: str, url: str, modelo: str) -> None:
    console.print(
        f"[bold green]🎮 JUGADOR ACTIVO:[/bold green] {mi_nombre} "
        f"[dim](Modo: Negociación Masiva)[/dim]"
    )
    logger.info(f"Agente iniciado | name={mi_nombre} | model={modelo} | url={url}")

    memoria_oferta = OfertaMemoria()

    while True:
        console.print("\n[dim]" + "-" * 40 + "[/dim]")

        info = api_request(url, "GET", "/info")
        gente_raw = api_request(url, "GET", "/gente")
        if not isinstance(info, dict) or "Recursos" not in info:
            time.sleep(2)
            continue

        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        buzon = {
            k: v for k, v in info.get("Buzon", {}).items() if v.get("dest") == mi_nombre
        }
        otros_jugadores = normalizar_jugadores(gente_raw, mi_nombre)

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
        cartas_visibles = dict(list(buzon.items())[-1:])

        console.print(f"🎒 TENGO: {mis_recursos}")
        console.print(f"🎯 FALTA: {faltan}")
        console.print(f"🔄 SOBRA: {sobran}")
        console.print(cartas_visibles)

        system_prompt = (
            f"Eres el jugador {mi_nombre} en un juego de intercambio de recursos. "
            "SIEMPRE debes invocar una de las funciones disponibles. "
            "NUNCA respondas con texto libre. SOLO llama a funciones."
        )

        ultima_oferta = {
            "recurso_que_busco": memoria_oferta.recurso_que_busco,
            "recurso_que_doy": memoria_oferta.recurso_que_doy,
        }
        prompt_usuario = (
            f"ESTADO ACTUAL:\n"
            f"- Necesito: {json.dumps(faltan)}\n"
            f"- Me sobra: {json.dumps(sobran)}\n"
            f"- Mensajes en buzón: {json.dumps(cartas_visibles)}\n"
            f"- Última oferta enviada: {json.dumps(ultima_oferta)}\n\n"
            "REGLAS DE DECISIÓN:\n"
            "- Si una carta ofrece algo que necesitas y pide algo que te sobra -> caso_1_aceptar\n"
            "- Si una carta no te interesa o pide algo que no tienes -> caso_2_borrar\n"
            "- Si una carta confirma un acuerdo previo -> caso_3_enviar\n"
            "- Si no hay cartas útiles o el buzón está vacío -> caso_4_ofertar_todos\n"
            "- Si vas a ofertar y hay alternativas, NO repitas la misma pareja recurso_que_busco/recurso_que_doy.\n"
            "- NUNCA ofrezcas ni envíes oro. El oro no se intercambia.\n\n"
            "DEBES invocar una de las 4 funciones. NO respondas con texto."
        )

        try:
            resp = ollama.chat(
                model=modelo,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_usuario},
                ],
                tools=OLLAMA_TOOLS,
            )
            logger.debug(f"Respuesta raw modelo: {resp}")
        except Exception as exc:
            logger.exception(f"Error en llamada a Ollama: {exc}")
            time.sleep(2)
            continue

        tool_calls = resp["message"].get("tool_calls")
        if not tool_calls:
            contenido_modelo = resp["message"].get("content", "").strip()
            if contenido_modelo:
                console.print(
                    f"💬 [dim]Modelo generó texto:[/dim] {contenido_modelo[:200]}"
                )
            console.print(
                "[yellow]⚠️ El modelo no invocó ninguna tool, reintentando...[/yellow]"
            )
            time.sleep(2)
            continue

        tool_call = tool_calls[0]
        accion = tool_call["function"]["name"]
        args = parse_tool_arguments(tool_call["function"].get("arguments"))
        console.print(f"🧠 IA DICE: [bold]{accion}[/bold] | args={args}")

        try:
            if accion == "caso_1_aceptar":
                dest = extraer_destino(args.get("dest"))
                item = str(args.get("item_enviar", "")).strip()
                cant = to_int(args.get("cant"), 1)
                mid = str(args.get("id_carta", "")).strip()

                if not dest or not item or es_oro(item):
                    logger.warning("caso_1_aceptar descartado por datos inválidos")
                elif mis_recursos.get(item, 0) <= 0:
                    logger.warning(f"caso_1_aceptar sin stock de '{item}'")
                else:
                    cant = min(cant, mis_recursos.get(item, 0))
                    api_request(
                        url,
                        "POST",
                        "/carta",
                        payload={
                            "remi": mi_nombre,
                            "dest": dest,
                            "asunto": "Trato",
                            "cuerpo": "Acepto. Aqui tienes.",
                        },
                    )
                    api_request(url, "POST", f"/paquete/{dest}", payload={item: cant})
                    console.print(f"✅ Trato cerrado con {dest}, por {cant} de {item}.")
                    if mid:
                        api_request(url, "DELETE", f"/mail/{mid}")

            elif accion == "caso_2_borrar":
                mid = str(args.get("id_carta", "")).strip()
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")
                    console.print("🗑️ Carta descartada.")

            elif accion == "caso_3_enviar":
                dest = extraer_destino(args.get("dest"))
                item = str(args.get("item_enviar", "")).strip()
                cant = to_int(args.get("cant"), 1)
                mid = str(args.get("id_carta", "")).strip()

                if not dest or not item or es_oro(item):
                    logger.warning("caso_3_enviar descartado por datos inválidos")
                elif mis_recursos.get(item, 0) <= 0:
                    logger.warning(f"caso_3_enviar sin stock de '{item}'")
                else:
                    cant = min(cant, mis_recursos.get(item, 0))
                    api_request(url, "POST", f"/paquete/{dest}", payload={item: cant})
                    console.print(f"📦 Material enviado a {dest}.")
                    if mid:
                        api_request(url, "DELETE", f"/mail/{mid}")

            elif accion == "caso_4_ofertar_todos":
                if not otros_jugadores:
                    logger.warning("No hay otros jugadores para ofertar")
                    time.sleep(2)
                    continue

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
                    time.sleep(2)
                    continue
                if es_oro(doy):
                    logger.warning("Oferta bloqueada: no se puede ofrecer oro")
                    time.sleep(2)
                    continue

                if cambio:
                    logger.info(
                        f"Oferta rotada automáticamente: busco={busco}, doy={doy}"
                    )

                mensaje = f"Necesito {busco}. Te doy {doy}. ¿Hacemos trato?"
                console.print(
                    f"📢 DIFUNDIENDO OFERTA A {len(otros_jugadores)} JUGADORES..."
                )

                enviados_ok = 0
                for jugador in otros_jugadores:
                    payload = {
                        "remi": mi_nombre,
                        "dest": jugador,
                        "asunto": f"Busco {busco}",
                        "cuerpo": mensaje,
                    }
                    respuesta = api_request(url, "POST", "/carta", payload=payload)
                    if isinstance(respuesta, dict) and respuesta.get("status") == "ok":
                        enviados_ok += 1

                if enviados_ok > 0:
                    memoria_oferta.recurso_que_busco = busco
                    memoria_oferta.recurso_que_doy = doy
                    console.print("✅ Rueda de ofertas enviada.")
                    time.sleep(5)
                else:
                    logger.warning("No se pudo enviar ninguna carta de oferta")

        except Exception as exc:
            logger.exception(f"Error en iteración del agente: {exc}")

        time.sleep(2)


def registrar_alias(mi_nombre: str, url: str) -> None:
    """Registra el alias del jugador en el servidor."""
    resultado = api_request(url, "POST", f"/alias/{mi_nombre}")
    if isinstance(resultado, dict) and resultado.get("status") == "ok":
        console.print(f"✅ Alias '{mi_nombre}' registrado correctamente")
    else:
        logger.warning("Alias no registrado")


@click.command()
@click.option(
    "--name",
    "mi_nombre",
    default=None,
    envvar=ENV_PLAYER_NAME,
    help="Alias del jugador (CLI > env > default).",
)
@click.option(
    "--model",
    "modelo",
    default=None,
    envvar=ENV_MODEL_NAME,
    help="Modelo de Ollama (CLI > env > default).",
)
@click.option(
    "--butler-address",
    "url",
    default=None,
    envvar=ENV_BUTLER_ADDRESS,
    help="Direccion Butler (CLI > env > default).",
)
@click.option(
    "--crear-alias/--no-crear-alias",
    default=False,
    help="Registra el alias antes de iniciar el agente.",
)
def main(mi_nombre, modelo, url, crear_alias):
    runtime_config = Dynaconf(environments=False)
    runtime_config.set("NAME", mi_nombre or PLAYER_NAME)
    runtime_config.set("MODEL", modelo or MODEL_NAME)
    runtime_config.set("BUTLER_ADDRESS", url or BUTLER_ADDRESS)

    mi_nombre = runtime_config.get("NAME")
    modelo = runtime_config.get("MODEL")
    url = runtime_config.get("BUTLER_ADDRESS")

    if crear_alias:
        registrar_alias(mi_nombre=mi_nombre, url=url)
    agente_autonomo(mi_nombre=mi_nombre, url=url, modelo=modelo)


if __name__ == "__main__":
    main()
