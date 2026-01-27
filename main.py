import requests
import urllib3
import ollama  # Importamos la librería de IA
import json

# Desactivar advertencias SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN ---
BASE_URL = "http://147.96.81.252:8000"
MI_USUARIO = "LOS ELEGIDOS"
MODELO_OLLAMA = "qwen3-vl:8b"  # <--- Asegúrate que este es el nombre exacto en 'ollama list'

def obtener_info():
    print(f"📡 Conectando con la API del juego...")
    
    try:
        # 1. Obtenemos la info (donde vimos que están tus Recursos y Objetivo)
        # Usamos /info o el endpoint que te devolvió ese JSON en el log anterior
        resp = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        data = resp.json()
        
        # Extraemos solo lo útil para no marear a la IA
        #mis_recursos = data.get("Recursos", {})
        #objetivo = data.get("Objetivo", {})
        
        print("\n🤖 --- ANÁLISIS DEL AGENTE (Qwen) ---")

        # 2. Preparamos el Prompt para Qwen
        prompt = f"""
        Actúa como un asistente estratégico de un juego de gestión de recursos.
        
        DATOS ACTUALES:{json.dumps(data)}
        
        TAREA:
        hazme un resumen claro de los datos actuales
        """

        # 3. Enviamos a Ollama
        response = ollama.chat(model=MODELO_OLLAMA, messages=[
            {'role': 'user', 'content': prompt},
        ])

        # 4. Imprimimos la respuesta de la IA
        print(response['message']['content'])
        print("--------------------------------------")

    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API del juego.")
    except ollama.ResponseError as e:
        print(f"❌ Error de Ollama: {e}")
        print("💡 Pista: ¿Está corriendo 'ollama serve'? ¿El modelo se llama 'qwen3'?")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def obtener_estado():
    print(f"📡 Conectando con la API del juego...")
    
    try:
        # 1. Obtenemos la info (donde vimos que están tus Recursos y Objetivo)
        # Usamos /info o el endpoint que te devolvió ese JSON en el log anterior
        resp = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        data = resp.json()
        
        # Extraemos solo lo útil para no marear a la IA
        mis_recursos = data.get("Recursos", {})
        objetivo = data.get("Objetivo", {})
        
        print("\n🤖 --- ANÁLISIS DEL AGENTE (Qwen) ---")

        # 2. Preparamos el Prompt para Qwen
        prompt = f"""
        Actúa como un asistente estratégico de un juego de gestión de recursos.
        
        DATOS ACTUALES:
        - Mis Recursos: {json.dumps(mis_recursos)}
        - Objetivo para ganar: {json.dumps(objetivo)}
        
        TAREA:
        Compara mis recursos con el objetivo.
        1. Dime claramente qué recursos me faltan y cuántos de cada uno.
        2. Si ya tengo suficiente de todo, felicítame.
        3. Sé breve y directo. No uses markdown complejo.
        4. Dime que recursos tengo ya
        """

        # 3. Enviamos a Ollama
        response = ollama.chat(model=MODELO_OLLAMA, messages=[
            {'role': 'user', 'content': prompt},
        ])

        # 4. Imprimimos la respuesta de la IA
        print(response['message']['content'])
        print("--------------------------------------")

    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API del juego.")
    except ollama.ResponseError as e:
        print(f"❌ Error de Ollama: {e}")
        print("💡 Pista: ¿Está corriendo 'ollama serve'? ¿El modelo se llama 'qwen3'?")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def crear_alias():
    url_post = f"{BASE_URL}/alias/LOS ELEGIDOS"
    resp_post = requests.post(url_post, verify=False)


import requests
import urllib3
import sys

# Desactivar alertas SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN ---
BASE_URL = "http://147.96.81.252:8000"
MI_NOMBRE = "LOS ELEGIDOS"

def enviar_paquete():
    # 1. Tomar destinatario de la línea de comandos o preguntar
    destinatario = sys.argv[1] if len(sys.argv) > 1 else input("👤 ¿A quién se lo envías?: ")

    # 2. Pedir qué enviar
    recurso = input("🧱 Tipo de recurso (trigo, madera, piedra, tela...): ").lower().strip()
    try:
        cantidad = int(input(f"🔢 Cantidad de {recurso}: "))
    except ValueError:
        print("❌ Error: La cantidad debe ser un número entero.")
        return

    # 3. Configurar la petición según la documentación OAS
    url = f"{BASE_URL}/paquete"
    
    # El destinatario va como parámetro de consulta (?dest=NOMBRE)
    params = {"dest": destinatario}
    
    # El cuerpo es un diccionario de recursos: cantidad
    payload = {
        recurso: cantidad
    }

    try:
        print(f"🚀 Enviando {cantidad} de '{recurso}' a '{destinatario}'...")
        
        # Enviamos params (query) y json (body)
        response = requests.post(
            url, 
            params=params, 
            json=payload, 
            verify=False, 
            timeout=5
        )

        if response.status_code == 200:
            print("✅ ¡Paquete entregado!")
            print("Respuesta:", response.json())
        else:
            print(f"❌ Fallo en el envío (Código {response.status_code}):")
            print(response.text)

    except Exception as e:
        print(f"❌ Error de conexión: {e}")

if __name__ == "__main__":
    # Opcional: Registrarse primero si hace falta, o ir directo al grano
    #crear_alias()
    #obtener_estado()
    #obtener_info()
    enviar_paquete()

