"""
Cliente asistido por IA para gestión de recursos en el juego.
Utiliza Ollama (modelo Qwen) para análisis estratégico.
"""

import sys
import json
import requests
import ollama
import time
import urllib3

# Ignorar advertencias de seguridad SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN ---
MI_NOMBRE = "LOS ELEGIDOS" 
URL = "http://147.96.81.252:7719"
MODELO = "qwen3-vl:4b"

def api_request(metodo, endpoint, params=None, payload=None):
    """
    Función robusta para conectar con la API.
    """
    try:
        url_completa = URL + endpoint
        if metodo == "GET":
            r = requests.get(url_completa, params=params, verify=False, timeout=3)
        elif metodo == "POST":
            r = requests.post(url_completa, params=params, json=payload, verify=False, timeout=3)
        elif metodo == "DELETE":
            r = requests.delete(url_completa, verify=False, timeout=3)
            return True 
        
        try:
            return r.json()
        except ValueError:
            return {} 
            
    except Exception as e:
        print(f"⚠️ Error de conexión: {e}")
        return {}

def agente_autonomo():
    print(f"🎮 JUGADOR ACTIVO: {MI_NOMBRE} (Modo: Negociación Masiva)")

    while True:
        print("\n" + "-"*40)
        
        # 1. OBTENER DATOS
        info = api_request("GET", "/info")
        gente_raw = api_request("GET", "/gente")
        
        if not info or "Recursos" not in info:
            time.sleep(2)
            continue

        # 2. CALCULAR ESTADO
        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        buzon = {k:v for k,v in info.get("Buzon", {}).items() if v.get("dest") == MI_NOMBRE}
        
        # Lista de jugadores (excluyéndome a mí)
        otros_jugadores = [p for p in (gente_raw if isinstance(gente_raw, list) else []) if p != MI_NOMBRE]

        faltan = {k: v - mis_recursos.get(k,0) for k,v in objetivo.items() if mis_recursos.get(k,0) < v}
        sobran = {k: v - objetivo.get(k,0) for k,v in mis_recursos.items() if v > objetivo.get(k,0)}
        cartas_visibles = dict(list(buzon.items())[:3])

        print(f"🎒 TENGO: {mis_recursos}")
        print(f"🎯 FALTA: {faltan}")
        print(f"🔄 SOBRA: {sobran}")

        # 3. TU PROMPT (Modificado con CASO 4)
        prompt_usuario = f"""
        PERSONALIDAD
        Eres el jugador {MI_NOMBRE}.
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
            resp = ollama.chat(model=MODELO, messages=[{"role": "user", "content": prompt_usuario}])
            print(resp)
            texto = resp['message']['content'].strip()
            if "```" in texto: texto = texto.split("```")[1].replace("json", "").strip()
            
            decision = json.loads(texto)
            accion = decision.get("acion")
            if accion is None:
                accion = decision.get("action")
            print(f"🧠 IA DICE: {accion}")

            # 5. EJECUTAR ACCIONES
            
            if accion == "CASO_1_ACEPTAR":
                dest, item, cant, mid = decision.get("dest"), decision.get("item_enviar"), decision.get("cant"), decision.get("id_carta")
                # Enviar carta
                api_request("POST", "/carta", payload={"remi": MI_NOMBRE, "dest": dest, "asunto": "Trato", "cuerpo": "Acepto. Aqui tienes."})
                # Enviar paquete
                api_request("POST", "/paquete", params={"dest": dest}, payload={item: cant})
                print(f"✅ Trato cerrado con {dest}, por {cant} de {item}.")
                if mid: api_request("DELETE", f"/mail/{mid}")

            elif accion == "CASO_2_BORRAR":
                mid = decision.get("id_carta")
                if mid:
                    api_request("DELETE", f"/mail/{mid}")
                    print("🗑️ Carta descartada.")

            elif accion == "CASO_3_ENVIAR":
                dest, item, cant, mid = decision.get("dest"), decision.get("item_enviar"), decision.get("cant"), decision.get("id_carta")
                api_request("POST", "/paquete", params={"dest": dest}, payload={item: cant})
                print(f"📦 Material enviado a {dest}.")
                if mid: api_request("DELETE", f"/mail/{mid}")

            elif accion == "CASO_4_OFERTAR_TODOS":
                busco = decision.get("recurso_que_busco")
                doy = decision.get("recurso_que_doy")
                
                # Preparamos el mensaje de spam
                mensaje = f"Necesito {busco}. Te doy {doy}. ¿Hacemos trato?"
                print(f"📢 DIFUNDIENDO OFERTA A {len(otros_jugadores)} JUGADORES...")
                
                for jugador in otros_jugadores:
                    api_request("POST", "/carta", payload={
                        "remi": MI_NOMBRE, 
                        "dest": jugador, 
                        "asunto": f"Busco {busco}", 
                        "cuerpo": mensaje
                    })
                print("✅ Rueda de ofertas enviada.")
                # Pausa extra para no saturar si hay muchos jugadores
                time.sleep(5)

        except Exception as e:
            print(f"⚠️ {e}")
        
        time.sleep(2)

def crear_alias():
    """Registra el alias del jugador en el servidor."""
    try:
        url = f"{URL}/alias/{MI_NOMBRE}"
        response = requests.post(url, verify=False, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Alias '{MI_NOMBRE}' registrado correctamente")
        else:
            print(f"⚠️ Código de respuesta: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error al crear alias: {e}")


if __name__ == "__main__":
    #crear_alias()
    agente_autonomo()