import requests
import json
import ollama
import time
import urllib3

# Ignorar advertencias de seguridad SSL
urllib3.disable_warnings()

# --- CONFIGURACIÓN ---
MI_NOMBRE = "LOS ELEGIDOS" 
URL = "http://147.96.81.252:8000"
MODELO = "qwen3-vl:8b"

def registrarse():
    """Registra el alias en el servidor antes de empezar."""
    print(f"🆔 Intentando registrar el alias: '{MI_NOMBRE}'...")
    # El nombre se pasa en la ruta: /alias/{nombre}
    url_registro = f"/alias/{MI_NOMBRE}"
    res = api_request("POST", url_registro)
    
    # Si la API devuelve un error porque ya existe, lo ignoramos y seguimos
    if isinstance(res, dict) and "error" not in res:
        print(f"✅ Registro verificado/completado.")
    else:
        print(f"ℹ️  Aviso en registro (puede que ya existas): {res}")


def api_request(metodo, endpoint, datos=None):
    """
    Función robusta para conectar con la API.
    Si el servidor devuelve vacío o falla, retorna {} para no romper el programa.
    """
    try:
        # Preparamos la URL completa
        url_completa = URL + endpoint
        
        # Hacemos la petición
        if metodo == "GET":
            r = requests.get(url_completa, params=datos, verify=False, timeout=3)
        elif metodo == "POST":
            # Para POST, usamos 'json' para el cuerpo (payload) y 'params' si fuera necesario
            r = requests.post(url_completa, json=datos, verify=False, timeout=3)
        elif metodo == "DELETE":
            r = requests.delete(url_completa, verify=False, timeout=3)
            return True # DELETE no suele devolver JSON útil

        # Intentamos leer el JSON. Si está vacío, devolvemos diccionario vacío.
        try:
            return r.json()
        except ValueError:
            return {} 
            
    except Exception as e:
        print(f"⚠️ Error de conexión: {e}")
        return {}

def agente_autonomo():
    print(f"🚀 Iniciando Agente: {MI_NOMBRE}")
    
    while True:
        print("\n" + "="*40)
        
        # 1. OBTENER INFORMACIÓN (Usando api_request)
        info = api_request("GET", "/info")
        gente = api_request("GET", "/gente")
        
        # Si la info está corrupta o vacía, esperamos y reintentamos
        if not info or "Recursos" not in info:
            print("⏳ Servidor ocupado o sin datos. Esperando...")
            time.sleep(2)
            continue

        # 2. PROCESAR DATOS
        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        
        # Filtramos solo MIS cartas
        buzon = {k:v for k,v in info.get("Buzon", {}).items() if v.get("dest") == MI_NOMBRE}

        # Calculamos lo que FALTA y lo que SOBRA
        faltan = {k: v - mis_recursos.get(k,0) for k,v in objetivo.items() if mis_recursos.get(k,0) < v}
        sobran = {k: v - objetivo.get(k,0) for k,v in mis_recursos.items() if v > objetivo.get(k,0)}
        
        print(f"📦 TENGO: {mis_recursos}")
        print(f"🎯 FALTA: {faltan}")
        print(f"📩 CARTAS: {len(buzon)} pendientes")

        # 3. PREPARAR PROMPT (Limitamos a 3 cartas para no liar a la IA)
        cartas_visibles = dict(list(buzon.items())[:3])
        
        prompt = f"""
        Eres el jugador {MI_NOMBRE}.
        
        ESTADO:
        - Necesito urgentemente: {json.dumps(faltan)}
        - Me sobra para cambiar: {json.dumps(sobran)}
        - Mensajes recibidos: {json.dumps(cartas_visibles)}

        OBJETIVO:
        1. Si un mensaje me ofrece lo que me falta -> ACEPTAR (Enviar lo que piden).
        2. Si el mensaje no sirve -> BORRAR.
        3. Si no tengo mensajes útiles -> PEDIR AYUDA a todos.

        RESPONDE SOLO EN FORMATO JSON:
        
        Opción A (Intercambiar):
        {{ "accion": "ENVIAR_PAQUETE", "destinatario": "nombre", "recurso": "item_que_envio", "cantidad": 1, "id_carta": "id_mensaje" }}
        
        Opción B (Limpiar):
        {{ "accion": "BORRAR_CARTA", "id_carta": "id_mensaje" }}
        
        Opción C (Pedir):
        {{ "accion": "PEDIR_AYUDA", "recurso_buscado": "item_que_necesito" }}
        """

        try:
            # 4. CONSULTAR A LA IA
            resp = ollama.chat(model=MODELO, messages=[{"role": "user", "content": prompt}])
            texto = resp['message']['content'].strip()
            
            # Limpieza básica del JSON por si la IA pone ```json ... ```
            if "```" in texto: texto = texto.split("```")[1].replace("json", "").strip()
            
            decision = json.loads(texto)
            accion = decision.get("accion")
            print(f"🧠 IA DECIDE: {accion}")

            # 5. EJECUTAR ACCIÓN
            if accion == "ENVIAR_PAQUETE":
                dest = decision.get("destinatario")
                item = decision.get("recurso")
                cant = int(decision.get("cantidad", 1))
                mid = decision.get("id_carta")
                
                # Verificamos si realmente tenemos el recurso antes de enviarlo
                if mis_recursos.get(item, 0) >= cant:
                    # Usamos 'params' para el destino y 'json' para el contenido del paquete
                    api_request("POST", "/paquete", {"dest": dest}, {item: cant})
                    print(f"✅ PAQUETE ENVIADO: {cant} de {item} a {dest}")
                    if mid: api_request("DELETE", f"/mail/{mid}")
                else:
                    print(f"🚫 ERROR: La IA quiso enviar {item} pero no tienes suficiente.")
                    # Si la IA se equivoca, borramos la carta para no entrar en bucle
                    if mid: api_request("DELETE", f"/mail/{mid}")

            elif accion == "BORRAR_CARTA":
                mid = decision.get("id_carta")
                if mid:
                    api_request("DELETE", f"/mail/{mid}")
                    print(f"🗑️ Mensaje {mid} borrado.")

            elif accion == "PEDIR_AYUDA":
                item = decision.get("recurso_buscado")
                if item in faltan:
                    jugadores = [j for j in gente if j != MI_NOMBRE]
                    print(f"📢 RADIO: Pidiendo {item} a todos...")
                    msg = f"Necesito {item}. Te doy ORO o {list(sobran.keys())}."
                    
                    for j in jugadores:
                        api_request("POST", "/carta", None, {"remi": MI_NOMBRE, "dest": j, "asunto": "Trato", "cuerpo": msg})
                    time.sleep(3) # Pausa para no hacer spam masivo

        except Exception as e:
            print(f"⚠️ Error procesando turno: {e}")
        
        # Pausa entre turnos
        time.sleep(2)

if __name__ == "__main__":
    agente_autonomo()

