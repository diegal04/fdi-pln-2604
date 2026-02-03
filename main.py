import requests
import urllib3
import ollama
import json
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN ---
BASE_URL = "http://147.96.81.252:8000"
MI_NOMBRE = "LOS ELEGIDOS"
MODELO = "qwen3-vl:8b"

def api_request(method, endpoint, params=None, payload=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            return requests.get(url, params=params, verify=False, timeout=5).json()
        elif method == "POST":
            return requests.post(url, params=params, json=payload, verify=False, timeout=5).json()
    except Exception as e:
        return {"error": str(e)}

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


def agente_autonomo():
    # --- CORRECCIÓN: Definimos la variable AQUÍ dentro para evitar el error ---
    ultimo_sondeo = 0 
    
    print(f"🚀 Agente '{MI_NOMBRE}' iniciado. Modo: NEGOCIADOR PRECISO.")

    while True:
        print("\n" + "="*50)
        
        # 1. OBTENCIÓN DE DATOS
        info = api_request("GET", "/info")
        gente_raw = api_request("GET", "/gente")

        otros_jugadores = [g for g in (gente_raw if isinstance(gente_raw, list) else []) if g != MI_NOMBRE]
        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        mi_buzon = {k: v for k, v in info.get("Buzon", {}).items() if v.get("dest") == MI_NOMBRE}

        # 2. CÁLCULO DE FALTANTES Y SOBRANTES
        faltantes = {}
        for res, nec in objetivo.items():
            tengo = mis_recursos.get(res, 0)
            if tengo < nec:
                faltantes[res] = nec - tengo
        
        sobrantes = {}
        for res, cant in mis_recursos.items():
            if cant > objetivo.get(res, 0):
                sobrantes[res] = cant - objetivo.get(res, 0)

        print(f"📦 Tengo: {mis_recursos}")
        print(f"🎯 ME FALTAN EXACTAMENTE: {faltantes}")
        print(f"🔄 Me sobran para cambiar: {sobrantes}")
        print(f"📩 Cartas pendientes: {len(mi_buzon)}")

        if not faltantes:
            print("🏆 ¡OBJETIVO COMPLETADO! Misión cumplida.")
        
        # 3. PROMPT DE INTELIGENCIA
        prompt = f"""
        ERES UN GESTOR DE RECURSOS. TU NOMBRE: {MI_NOMBRE}.
        
        TUS NECESIDADES EXACTAS: {json.dumps(faltantes)}
        TUS SOBRANTES PARA CAMBIAR: {json.dumps(sobrantes)}
        BUZÓN DE ENTRADA: {json.dumps(mi_buzon)}
        
        REGLAS:
        1. NO TE INVENTES INFORMACIÓN. No sabes qué tienen los demás.
        2. "NUNCA DES ORO" (a menos que sea emergencia, prefiere dar sobrantes).
        3. Prioridad: Responder cartas del buzón.
        4. Si el buzón está vacío, ordena un SONDEO_MASIVO pidiendo uno de los recursos que faltan.

        ACCIONES (Responde SOLO JSON):
        
        - OPCIÓN A (Preguntar a todos):
        {{ "accion": "SONDEO_MASIVO", "recurso_buscado": "nombre_recurso", "pensamiento": "..." }}

        - OPCIÓN B (Responder carta):
        {{ "accion": "RESPONDER_CARTA", "parametros": {{ "dest": "Nombre", "tipo_envio": "PAQUETE" o "CARTA", "recurso": "...", "cantidad": 1, "mensaje": "..." }} }}

        - OPCIÓN C: {{ "accion": "ESPERAR" }}
        """

        try:
            # Enviamos a Ollama
            response = ollama.chat(model=MODELO, messages=[{'role': 'user', 'content': prompt}])
            raw = response['message']['content'].strip()
            if "```" in raw: raw = raw.split("```")[1].replace("json", "").strip()
            
            decision = json.loads(raw)
            accion = decision.get("accion", "ESPERAR")
            pensamiento = decision.get("pensamiento", "")

            print(f"🧠 PENSAMIENTO: {pensamiento}")
            print(f"💡 ACCIÓN: {accion}")

            # --- EJECUCIÓN ---

            if accion == "SONDEO_MASIVO":
                # Verificamos si han pasado 60 segundos desde el último sondeo
                tiempo_actual = time.time()
                if tiempo_actual - ultimo_sondeo > 60:
                    recurso = decision.get("recurso_buscado")
                    cantidad_necesaria = faltantes.get(recurso, 1) # Por defecto 1 si falla
                    
                    print(f"📢 DIFUNDIENDO PETICIÓN A {len(otros_jugadores)} JUGADORES...")
                    
                    for jugador in otros_jugadores:
                        cuerpo_msg = f"Hola {jugador}, necesito urgentemente {cantidad_necesaria} de {recurso}. Tengo {sobrantes} para cambiar. ¿Hacemos trato?"
                        
                        api_request("POST", "/carta", payload={
                            "remi": MI_NOMBRE, "dest": jugador, 
                            "asunto": f"Busco {recurso}", "cuerpo": cuerpo_msg
                        })
                        print(f"   -> Carta enviada a {jugador}")
                    
                    # Actualizamos el contador de tiempo AQUÍ
                    ultimo_sondeo = tiempo_actual 
                    print("✅ Sondeo completado.")
                else:
                    segundos_restantes = int(60 - (tiempo_actual - ultimo_sondeo))
                    print(f"⏳ Esperando cooldown ({segundos_restantes}s) para no hacer spam.")

            elif accion == "RESPONDER_CARTA":
                p = decision.get("parametros", {})
                destino = p.get("dest")
                
                if destino and destino != MI_NOMBRE:
                    if p.get("tipo_envio") == "PAQUETE":
                        api_request("POST", "/paquete", params={"dest": destino}, 
                                    payload={p.get("recurso"): int(p.get("cantidad", 1))})
                        print(f"📦 Paquete enviado a {destino}")
                    else:
                        api_request("POST", "/carta", payload={
                            "remi": MI_NOMBRE, "dest": destino,
                            "asunto": "Respuesta", "cuerpo": p.get("mensaje", "Hola")
                        })
                        print(f"📩 Respuesta enviada a {destino}")

            # Limpieza de buzón
            if mi_buzon:
                print("🧹 Limpiando buzón...")
                for mid in mi_buzon.keys():
                    api_request("DELETE", f"/mail/{mid}")

        except Exception as e:
            print(f"❌ Error en el ciclo: {e}")


if __name__ == "__main__":
    registrarse()
    agente_autonomo()