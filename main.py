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
    ultimo_sondeo = 0 
    recursos_anteriores = {}
    primera_vez = True
    
    print(f"[SISTEMA] Agente '{MI_NOMBRE}' iniciado. MODO: VELOCIDAD MAXIMA (SIN PAUSAS).")

    while True:
        print("-" * 60)
        
        # 1. OBTENCIÓN DE DATOS
        info = api_request("GET", "/info")
        gente_raw = api_request("GET", "/gente")

        otros_jugadores = [g for g in (gente_raw if isinstance(gente_raw, list) else []) if g != MI_NOMBRE]
        mis_recursos = info.get("Recursos", {})
        objetivo = info.get("Objetivo", {})
        
        mi_buzon_raw = info.get("Buzon", {})
        mi_buzon_items = [(k, v) for k, v in mi_buzon_raw.items() if v.get("dest") == MI_NOMBRE]
        
        # Limitamos visión a 5 mensajes para procesar rapido
        buzon_visible = dict(mi_buzon_items[:5]) 

        # --- MONITOR DE CAMBIOS ---
        if not primera_vez:
            todos_recursos = set(mis_recursos.keys()) | set(recursos_anteriores.keys())
            hay_cambios = False
            for res in todos_recursos:
                antes = recursos_anteriores.get(res, 0)
                ahora = mis_recursos.get(res, 0)
                diff = ahora - antes
                if diff > 0:
                    print(f"[ENTRADA] +{diff} de {res}")
                    hay_cambios = True
                elif diff < 0:
                    print(f"[SALIDA] -{abs(diff)} de {res}")
                    hay_cambios = True
            if not hay_cambios:
                print("[INFO] Inventario sin cambios.")
        else:
            primera_vez = False

        recursos_anteriores = mis_recursos.copy()

        # Cálculo de necesidades
        faltantes = {res: nec - mis_recursos.get(res, 0) for res, nec in objetivo.items() if mis_recursos.get(res, 0) < nec}
        sobrantes = {res: cant - objetivo.get(res, 0) for res, cant in mis_recursos.items() if cant > objetivo.get(res, 0)}

        print(f"[RECURSOS] Tienes: {mis_recursos}")
        print(f"[OBJETIVO] Faltan: {faltantes}")
        print(f"[BUZON] {len(mi_buzon_items)} mensajes pendientes.")

        # 2. PROMPT
        prompt = f"""
        ERES UN MERCADER RICO LLAMADO '{MI_NOMBRE}'.
        
        SITUACION:
        - Tienes ORO: {mis_recursos.get('oro', 0)} (USALO)
        - Te faltan: {json.dumps(faltantes)}
        - Te sobran: {json.dumps(sobrantes)}
        - Cartas recientes: {json.dumps(buzon_visible)}
        
        ESTRATEGIA:
        1. REVISA EL BUZON: 
           - Si ofrecen lo que falta y piden ORO -> ACEPTA (Accion: PAGAR_CARTA).
           - Si piden algo que sobra -> DASELO (Accion: PAGAR_CARTA).
           - Si no interesa -> BORRAR (Accion: DESCARTAR).
        2. SI EL BUZON ESTA VACIO:
           - Haz OFERTA_PUBLICA ofreciendo 1 de ORO por lo que falta.

        RESPONDE SOLO JSON:
        A) {{ "accion": "OFERTA_PUBLICA", "recurso_que_necesito": "queso" }}
        B) {{ "accion": "PAGAR_CARTA", "id_carta": "id_msg", "destinatario": "nombre", "recurso_a_enviar": "oro", "cantidad": 1 }}
        C) {{ "accion": "DESCARTAR", "id_carta": "id_msg" }}
        """

        try:
            response = ollama.chat(model=MODELO, messages=[{'role': 'user', 'content': prompt}])
            raw = response['message']['content'].strip()
            if "```" in raw: raw = raw.split("```")[1].replace("json", "").strip()
            
            decision = json.loads(raw)
            accion = decision.get("accion", "ESPERAR")
            
            print(f"[IA] Decide: {accion}")

            # --- EJECUCIÓN ---

            if accion == "OFERTA_PUBLICA":
                # Mantenemos control de tiempo solo para el broadcast (para no saturar a los otros jugadores)
                if time.time() - ultimo_sondeo > 30:
                    necesito = decision.get("recurso_que_necesito")
                    if necesito in faltantes:
                        print(f"[RADIO] Publicando oferta: Doy ORO por {necesito}")
                        for jugador in otros_jugadores:
                            api_request("POST", "/carta", payload={
                                "remi": MI_NOMBRE, "dest": jugador, 
                                "asunto": f"Compro {necesito}", 
                                "cuerpo": f"Necesito {necesito}. TE PAGO 1 DE ORO. Envia paquete."
                            })
                        ultimo_sondeo = time.time()
                else:
                    print("[INFO] Esperando cooldown de radio (broadcast).")

            elif accion == "PAGAR_CARTA":
                dest = decision.get("destinatario")
                rec = decision.get("recurso_a_enviar")
                cant = int(decision.get("cantidad", 1))
                mid = decision.get("id_carta")

                if mis_recursos.get(rec, 0) >= cant:
                    api_request("POST", "/paquete", params={"dest": dest}, payload={rec: cant})
                    print(f"[PAGO] Enviando {cant} de {rec} a {dest}")
                    if mid: api_request("DELETE", f"/mail/{mid}")
                else:
                    print(f"[ERROR] No tienes suficiente {rec} para pagar.")

            elif accion == "DESCARTAR":
                mid = decision.get("id_carta")
                if mid:
                    api_request("DELETE", f"/mail/{mid}")
                    print(f"[BORRADO] Carta {mid} eliminada.")

            # LIMPIEZA DE EMERGENCIA
            if len(mi_buzon_items) > 20:
                print("[LIMPIEZA] Borrando mensajes antiguos por saturacion...")
                for mid, _ in mi_buzon_items[:5]:
                    api_request("DELETE", f"/mail/{mid}")

        except Exception as e:
            print(f"[ERROR LOGICA] {e}")

if __name__ == "__main__":
    #registrarse()
    agente_autonomo()