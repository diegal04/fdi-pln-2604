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
            if p != mi_nombre
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
        cartas_visibles = dict(list(buzon.items())[:3])

        console.print(f"🎒 TENGO: {mis_recursos}")
        console.print(f"🎯 FALTA: {faltan}")
        console.print(f"🔄 SOBRA: {sobran}")

        # 3. TU PROMPT (Modificado con CASO 4)
        prompt_usuario = f"""
        PERSONALIDAD
        Eres el jugador {mi_nombre}.
        ======================
        CONTEXTO
        Objetivo: Conseguir los recursos que faltan intercambiando los que sobran.
        
        ESTADO:
        - Necesito: {json.dumps(faltan)}
        - Me sobra: {json.dumps(sobran)}
        - Mensajes en buzón: {json.dumps(cartas_visibles)}

        CASO 1 (ACEPTAR TRATO):
        Si recibes una carta que ofrece algo que NECESITAS y pide algo que TIENES -> ACEPTA (Envía carta y paquete).
        Recibes una carta diciendo lo siguiente:
        Quiero 1 de madera y tengo para darte 3 de piedra, 2 de oro, y uno de queso, te interesa?

        En caso de necesitar alguno de los recursos que ofrece por ejemplo 1 de piedra y disponer de madera enviar una carta diciendo,
        acepto el trato y aparte enviar un paquete con 1 de madera.

        CASO 2 (BORRAR):
        Si la carta no te interesa o pide algo que no tienes -> BORRA LA CARTA.
        Recibes una carta diciendo lo siguiente:
        Quiero 1 de madera y tengo para darte 3 de piedra, 2 de oro, y uno de queso, te interesa?

        En caso de no necesitar alguno de los recursos que ofrece eliminar la carta.

        CASO 3 (CUMPLIR ACUERDO):
        Si la carta es una respuesta positiva a un trato previo -> ENVÍA EL MATERIAL (Paquete).

        CASO 4 (OFERTA MASIVA - IMPORTANTE):
        Si NO hay cartas útiles o el buzón está vacío -> ENVÍA CARTAS A TODO EL MUNDO.
        Debes decir qué necesitas y qué ofreces a cambio.
        
        ======================
        CAPACIDAD DE ACCION (Responde SOLO con el JSON correspondiente):

        1. Para CASO 1 (Trato Nuevo):
           {{ "accion": "CASO_1_ACEPTAR", "dest": "nombre", "item_enviar": "recurso", "cant": 1, "id_carta": "id" }}
           
        2. Para CASO 2 (Borrar):
           {{ "accion": "CASO_2_BORRAR", "id_carta": "id" }}
           
        3. Para CASO 3 (Enviar material):
           {{ "accion": "CASO_3_ENVIAR", "dest": "nombre", "item_enviar": "recurso", "cant": 1, "id_carta": "id" }}
           
        4. Para CASO 4 (SI NO HAY CARTAS ÚTILES):
           {{ "accion": "CASO_4_OFERTAR_TODOS", "recurso_que_busco": "item_buscado", "recurso_que_doy": "item_ofrecido" }}
           ¡¡¡¡IMPORTANTE SEGUIR LA ESTRUCTURA DEL JSON PARA CADA CASO!!
           tiene que empezar por action siempre
        """

        try:
            # 4. CONSULTAR A LA IA
            resp = ollama.chat(
                model=modelo, messages=[{"role": "user", "content": prompt_usuario}]
            )
            logger.debug(f"Respuesta raw modelo: {resp}")
            texto = resp["message"]["content"].strip()
            if "```" in texto:
                texto = texto.split("```")[1].replace("json", "").strip()

            decision = json.loads(texto)
            accion = decision.get("accion")
            if accion is None:
                accion = decision.get("action")
            console.print(f"🧠 IA DICE: [bold]{accion}[/bold]")

            # 5. EJECUTAR ACCIONES

            if accion == "CASO_1_ACEPTAR":
                dest, item, cant, mid = (
                    decision.get("dest"),
                    decision.get("item_enviar"),
                    decision.get("cant"),
                    decision.get("id_carta"),
                )
                # Enviar carta
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
                # Enviar paquete
                api_request(
                    url, "POST", "/paquete", params={"dest": dest}, payload={item: cant}
                )
                console.print(f"✅ Trato cerrado con {dest}, por {cant} de {item}.")
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")

            elif accion == "CASO_2_BORRAR":
                mid = decision.get("id_carta")
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")
                    console.print("🗑️ Carta descartada.")

            elif accion == "CASO_3_ENVIAR":
                dest, item, cant, mid = (
                    decision.get("dest"),
                    decision.get("item_enviar"),
                    decision.get("cant"),
                    decision.get("id_carta"),
                )
                api_request(
                    url, "POST", "/paquete", params={"dest": dest}, payload={item: cant}
                )
                console.print(f"📦 Material enviado a {dest}.")
                if mid:
                    api_request(url, "DELETE", f"/mail/{mid}")

            elif accion == "CASO_4_OFERTAR_TODOS":
                busco = decision.get("recurso_que_busco")
                doy = decision.get("recurso_que_doy")

                # Preparamos el mensaje de spam
                mensaje = f"Necesito {busco}. Te doy {doy}. ¿Hacemos trato?"
                console.print(f"📢 DIFUNDIENDO OFERTA A {len(otros_jugadores)} JUGADORES...")

                for jugador in otros_jugadores:
                    api_request(
                        url,
                        "POST",
                        "/carta",
                        payload={
                            "remi": mi_nombre,
                            "dest": jugador,
                            "asunto": f"Busco {busco}",
                            "cuerpo": mensaje,
                        },
                    )
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
