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
URL = "http://147.96.81.252:8000"
MODELO = "qwen3-vl:8b"

def api_request(metodo, endpoint, params=None, datos=None):
    """
    Función robusta para conectar con la API.
    """
    try:
        url_completa = URL + endpoint
        if metodo == "GET":
            r = requests.get(url_completa, params=params, verify=False, timeout=3)
        elif metodo == "POST":
            r = requests.post(url_completa, params=params, json=datos, verify=False, timeout=3)
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
    print(f"🚀 Iniciando Agente: {MI_NOMBRE}")
    
    # Intentar registro inicial
    api_request("POST", f"/alias/{MI_NOMBRE}")
    
    while True:
        print("\n" + "="*40)
        
        # 1. OBTENER INFORMACIÓN
        info = api_request("GET", "/info")
        gente = api_request("GET", "/gente")
        
        if not info or "Recursos" not in info:
            print("⏳ Servidor ocupado o sin datos. Esperando...")
            time.sleep(2)
            continue

        # 2. PROCESAR DATOS
        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        buzon = {k:v for k,v in info.get("Buzon", {}).items() if v.get("dest") == MI_NOMBRE}

        faltan = {k: v - mis_recursos.get(k,0) for k,v in objetivo.items() if mis_recursos.get(k,0) < v}
        sobran = {k: v - objetivo.get(k,0) for k,v in mis_recursos.items() if v > objetivo.get(k,0)}
        
        print(f"📦 TENGO: {mis_recursos}")
        print(f"🎯 FALTA: {faltan}")
        print(f"📩 CARTAS: {len(buzon)} pendientes")

        # 3. PREPARAR PROMPT
        cartas_visibles = dict(list(buzon.items())[:3])
        
        prompt = f"""
        Eres el jugador {MI_NOMBRE}.
        ESTADO:
        - Necesito: {json.dumps(faltan)}
        - Me sobra: {json.dumps(sobran)}
        - Mensajes: {json.dumps(cartas_visibles)}

        RESPONDE SOLO JSON:
        Opción A (Intercambiar): {{ "accion": "ENVIAR_PAQUETE", "destinatario": "nombre", "recurso": "item", "cantidad": 1, "id_carta": "id" }}
        Opción B (Limpiar): {{ "accion": "BORRAR_CARTA", "id_carta": "id" }}
        Opción C (Pedir): {{ "accion": "PEDIR_AYUDA", "recurso_buscado": "item" }}
        """

        try:
            # 4. CONSULTAR A LA IA
            resp = ollama.chat(model=MODELO, messages=[{"role": "user", "content": prompt}])
            texto = resp['message']['content'].strip()
            
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
                
                if mis_recursos.get(item, 0) >= cant:
                    api_request("POST", "/paquete", {"dest": dest}, {item: cant})
                    print(f"✅ PAQUETE ENVIADO: {cant} de {item} a {dest}")
                    if mid: api_request("DELETE", f"/mail/{mid}")
                else:
                    if mid: api_request("DELETE", f"/mail/{mid}")

            elif accion == "BORRAR_CARTA":
                mid = decision.get("id_carta")
                if mid: api_request("DELETE", f"/mail/{mid}")

            elif accion == "PEDIR_AYUDA":
                item = decision.get("recurso_buscado")
                jugadores = [j for j in gente if j != MI_NOMBRE]
                for j in jugadores:
                    api_request("POST", "/carta", None, {"remi": MI_NOMBRE, "dest": j, "asunto": "Trato", "cuerpo": f"Necesito {item}"})

        except Exception as e:
            print(f"⚠️ Error en turno: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    agente_autonomo()