import json
import sys
import time
import urllib3

import click
import httpx
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

# Ignorar advertencias de seguridad SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
console = Console()

logger.remove()
logger.add(sys.stderr, level="INFO")

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


def api_request(base_url, metodo, endpoint, params=None, payload=None):
    """
    Función robusta para conectar con la API.
    """
    try:
        url_completa = base_url + endpoint
        if metodo in {"GET", "POST", "DELETE"}:
            r = httpx.request(
                metodo,
                url_completa,
                params=params,
                json=payload,
                verify=False,
                timeout=3.0,
            )
        else:
            return {}

        if metodo == "DELETE":
            return r.status_code < 400

        try:
            return r.json()
        except ValueError:
            return {}

    except Exception as e:
        logger.warning(f"Error de conexión HTTP: {e}")
        return {}


def agente_autonomo(mi_nombre, url, modelo):
    console.print(
        f"[bold green]🎮 JUGADOR ACTIVO:[/bold green] {mi_nombre} "
        f"[dim](Modo: Negociación Masiva)[/dim]"
    )
    logger.info(f"Agente iniciado | name={mi_nombre} | model={modelo} | url={url}")

    while True:
        console.print("\n[dim]" + "-" * 40 + "[/dim]")

        # 1. OBTENER DATOS
        info = api_request(url, "GET", "/info")
        gente_raw = api_request(url, "GET", "/gente")

        if not info or "Recursos" not in info:
            time.sleep(2)
            continue

        # 2. CALCULAR ESTADO
        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        buzon = {
            k: v for k, v in info.get("Buzon", {}).items() if v.get("dest") == mi_nombre
        }

        # Lista de jugadores (excluyéndome a mí)
        otros_jugadores = [
            p
            for p in (gente_raw if isinstance(gente_raw, list) else [])
            #if p != mi_nombre
        ]

        faltan = {
            k: v - mis_recursos.get(k, 0)
            for k, v in objetivo.items()
            if mis_recursos.get(k, 0) < v
        }
        sobran = {
            k: v - objetivo.get(k, 0)
            for k, v in mis_recursos.items()
            if v > objetivo.get(k, 0)
        }
        cartas_visibles = dict(list(buzon.items())[-1:])

        console.print(f"🎒 TENGO: {mis_recursos}")
        console.print(f"🎯 FALTA: {faltan}")
        console.print(f"🔄 SOBRA: {sobran}")
        console.print(cartas_visibles)

        # 3. PROMPT CON ESTADO ACTUAL
        system_prompt = (
            f"Eres el jugador {mi_nombre} en un juego de intercambio de recursos. "
            "SIEMPRE debes invocar una de las funciones disponibles. "
            "NUNCA respondas con texto libre. SOLO llama a funciones."
        )
        
        prompt_usuario = (
            f"ESTADO ACTUAL:\n"
            f"- Necesito: {json.dumps(faltan)}\n"
            f"- Me sobra: {json.dumps(sobran)}\n"
            f"- Mensajes en buzón: {json.dumps(cartas_visibles)}\n\n"
            "REGLAS DE DECISIÓN:\n"
            "- Si una carta ofrece algo que necesitas y pide algo que te sobra -> caso_1_aceptar\n"
            "- Si una carta no te interesa o pide algo que no tienes -> caso_2_borrar\n"
            "- Si una carta confirma un acuerdo previo -> caso_3_enviar\n"
            "- Si no hay cartas útiles o el buzón está vacío -> caso_4_ofertar_todos\n\n"
            "- NUNCA ofrezcas ni envíes oro. El oro no se intercambia en ningún caso.\n"
            "DEBES invocar una de las 4 funciones. NO respondas con texto."
        )

        try:
            # 4. CONSULTAR A LA IA (con tools para salida estructurada)
            resp = ollama.chat(
                model=modelo,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_usuario}
                ],
                tools=OLLAMA_TOOLS,
            )
            logger.debug(f"Respuesta raw modelo: {resp}")

            tool_calls = resp["message"].get("tool_calls")
            if not tool_calls:
                # Mostrar qué generó el modelo (texto libre)
                contenido_modelo = resp["message"].get("content", "").strip()
                if contenido_modelo:
                    console.print(f"💬 [dim]Modelo generó texto:[/dim] {contenido_modelo[:200]}")
                
                console.print("[yellow]⚠️ El modelo no invocó ninguna tool, reintentando...[/yellow]")
                time.sleep(2)
                continue
            
            print(tool_calls)
            tool_call = tool_calls[0]
            accion = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            console.print(f"🧠 IA DICE: [bold]{accion}[/bold] | args={args}")

            # 5. EJECUTAR ACCIONES

            if accion == "caso_1_aceptar":
                dest = args["dest"]
                item = args["item_enviar"]
                cant = args["cant"]
                mid = args["id_carta"]
                # Enviar carta
                api_request(
                    url,
                    "POST",
                    "/carta",
                    payload={
                        "remi": mi_nombre,
                        "dest": dest.get("alias"),
                        "asunto": "Trato",
                        "cuerpo": "Acepto. Aqui tienes.",
                    },
                )
                # Enviar paquete
                api_request(
                    url, "POST", "/paquete/", params={"dest": dest}, payload={item: cant}
                )
                #api_request(url, "POST", f"/paquete/{dest}", payload={item: cant})
                console.print(f"✅ Trato cerrado con {dest}, por {cant} de {item}.")
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")

            elif accion == "caso_2_borrar":
                mid = args["id_carta"]
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")
                    console.print("🗑️ Carta descartada.")

            elif accion == "caso_3_enviar":
                dest = args["dest"]
                item = args["item_enviar"]
                cant = args["cant"]
                mid = args["id_carta"]
                api_request(
                    url, "POST", "/paquete/", params={"dest": dest}, payload={item: cant}
                )
                console.print(f"📦 Material enviado a {dest}.")
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")

            elif accion == "caso_4_ofertar_todos":
                busco = args["recurso_que_busco"]
                doy = args["recurso_que_doy"]

                # Preparamos el mensaje de spam
                mensaje = f"Necesito {busco}. Te doy {doy}. ¿Hacemos trato?"
                console.print(f"📢 DIFUNDIENDO OFERTA A {len(otros_jugadores)} JUGADORES...")
                for jugador in otros_jugadores:
                    payload={
                            "remi": mi_nombre,
                            "dest": jugador.get("alias"),
                            "asunto": f"Busco {busco}",
                            "cuerpo": mensaje,
                        }
                    api_request(
                        url,
                        "POST",
                        "/carta",
                        payload=payload,
                    )
                    console.print(url , "POST", "/carta", payload)
                
                console.print("✅ Rueda de ofertas enviada.")
                # Pausa extra para no saturar si hay muchos jugadores
                time.sleep(5)

        except Exception as e:
            logger.exception(f"Error en iteración del agente: {e}")

        time.sleep(2)


def registrar_alias(mi_nombre, url):
    """Registra el alias del jugador en el servidor."""
    try:
        alias_url = f"{url}/alias/{mi_nombre}"
        response = httpx.post(alias_url, verify=False, timeout=5.0)

        if response.status_code == 200:
            console.print(f"✅ Alias '{mi_nombre}' registrado correctamente")
        else:
            logger.warning(f"Alias no registrado | status={response.status_code}")

    except Exception as e:
        logger.error(f"Error al crear alias: {e}")


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
