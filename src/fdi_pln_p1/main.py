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
                "Aceptar una carta del buzón que propone un intercambio favorable. "
                "Usar SOLO cuando la carta ofrece un recurso que aparece en tu lista "
                "de recursos que NECESITAS y a cambio pide un recurso que aparece en "
                "tu lista de recursos que te SOBRAN. Nunca envíes oro."
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
                        "description": (
                            "Nombre exacto del recurso que le envías como pago. "
                            "Debe ser uno de tus recursos sobrantes. "
                            "Nunca puede ser oro. Ejemplo: 'madera'."
                        ),
                    },
                    "cant": {
                        "type": "integer",
                        "description": "Cantidad del recurso que envías.",
                    },
                    "id_carta": {
                        "type": "string",
                        "description": (
                            "ID único de la carta que aceptas, tal como aparece "
                            "en las claves del buzón. Ejemplo: 'abc123'."
                        ),
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
                "Eliminar una carta del buzón que NO es útil. Usar cuando la carta "
                "pide un recurso que no te sobra, ofrece algo que no necesitas, "
                "o simplemente no te conviene el trato."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id_carta": {
                        "type": "string",
                        "description": (
                            "ID único de la carta a eliminar, tal como aparece "
                            "en las claves del buzón."
                        ),
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
                "Enviar recursos a un jugador para cumplir un acuerdo ya aceptado. "
                "Usar SOLO cuando una carta confirma que el otro jugador ya aceptó "
                "un trato previo y espera recibir material tuyo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dest": {
                        "type": "string",
                        "description": (
                            "Nombre (alias) del jugador al que envías el recurso. "
                        ),
                    },
                    "item_enviar": {
                        "type": "string",
                        "description": (
                            "Nombre exacto del recurso prometido en el acuerdo. "
                            "Nunca puede ser oro. Ejemplo: 'piedra'."
                        ),
                    },
                    "cant": {
                        "type": "integer",
                        "description": "Cantidad prometidadel recurso que envías.",
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
                "Enviar una oferta de intercambio a TODOS los jugadores. "
                "Usar cuando el buzón está vacío o ninguna carta del buzón es útil "
                "(después de borrar las inútiles). Los recursos deben ser strings "
                "simples. El recurso que buscas debe estar en tu lista de NECESITO "
                "y el que ofreces en tu lista de SOBRA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recurso_que_busco": {
                        "type": "string",
                        "description": (
                            "Nombre exacto de un recurso que necesitas conseguir "
                            "(debe aparecer en tu lista de recursos faltantes). "
                            "Ejemplo: 'trigo'."
                        ),
                    },
                    "recurso_que_doy": {
                        "type": "string",
                        "description": (
                            "Nombre exacto de un recurso que ofreces a cambio "
                            "(debe aparecer en tu lista de recursos sobrantes). "
                            "Nunca puede ser oro. Ejemplo: 'madera'."
                        ),
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
        cartas_visibles = dict(list(buzon.items())[-5:])

        console.print(f"🎒 TENGO: {mis_recursos}")
        console.print(f"🎯 FALTA: {faltan}")
        console.print(f"🔄 SOBRA: {sobran}")
        console.print(cartas_visibles)

        system_prompt = (
            f"Eres {mi_nombre}, un agente en un juego de intercambio de recursos. "
            "Tu ÚNICO objetivo es conseguir todos los recursos de tu lista NECESITO "
            "intercambiándolos por recursos de tu lista SOBRA.\n\n"
            "REGLAS ABSOLUTAS:\n"
            "- Responde SIEMPRE invocando exactamente UNA función. NUNCA generes texto.\n"
            "- NUNCA ofrezcas, envíes ni intercambies oro bajo ningún concepto.\n"
            "- Los parámetros 'dest' e 'item_enviar' deben ser strings simples, nunca objetos.\n"
            "- Si NECESITO está vacío, has ganado: usa caso_4_ofertar_todos solo si SOBRA no está vacío.\n"
            "- Si SOBRA está vacío, solo puedes aceptar cartas o borrarlas, no ofertar.\n\n"
            "PRIORIDAD DE ACCIONES (de mayor a menor):\n"
            "1. Si hay una carta que confirma un acuerdo previo → caso_3_enviar.\n"
            "2. Si hay una carta útil (ofrece lo que necesito a cambio de lo que me sobra) → caso_1_aceptar.\n"
            "3. Si hay una carta inútil → caso_2_borrar.\n"
            "4. Si el buzón está vacío o ya procesaste todas las cartas → caso_4_ofertar_todos."
        )

        ultima_oferta = {
            "recurso_que_busco": memoria_oferta.recurso_que_busco,
            "recurso_que_doy": memoria_oferta.recurso_que_doy,
        }
        hay_cartas = len(cartas_visibles) > 0
        prompt_usuario = (
            f"MI INVENTARIO:\n"
            f"- Recursos que NECESITO conseguir (me faltan): {json.dumps(faltan)}\n"
            f"- Recursos que me SOBRAN (puedo dar): {json.dumps(sobran)}\n"
            f"- Jugadores disponibles: {json.dumps(otros_jugadores)}\n\n"
            f"BUZÓN ({len(cartas_visibles)} carta(s)):\n"
            f"{json.dumps(cartas_visibles, indent=2) if hay_cartas else '(vacío)'}\n\n"
            f"Última oferta que ya envié: busco={ultima_oferta['recurso_que_busco']}, "
            f"doy={ultima_oferta['recurso_que_doy']}, no la repitas si hay alternativa\n\n"
            "ELIGE UNA ACCIÓN según la prioridad del system prompt:\n"
            "1) caso_1_aceptar — Hay una carta que OFRECE algo que necesito Y PIDE algo que me sobra.\n"
            "2) caso_2_borrar — Hay una carta que NO me interesa (pide algo que no tengo, "
            "ofrece algo que no necesito, o es irrelevante). Borra esa carta con su ID.\n"
            "3) caso_3_enviar — Hay una carta que CONFIRMA un acuerdo previo y debo enviar "
            "el material prometido. Usa 'id_carta' de esa carta.\n"
            "4) caso_4_ofertar_todos — El buzón está vacío o ninguna carta es útil. "
            "Envía una oferta masiva: 'recurso_que_busco' debe ser algo que NECESITO, "
            "'recurso_que_doy' debe ser algo que me SOBRA. "
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
