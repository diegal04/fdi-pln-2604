"""
Cliente asistido por IA para gestión de recursos en el juego.
Utiliza Ollama (modelo Qwen) para análisis estratégico.
"""

import sys
import json
import requests
import urllib3
import ollama

# Desactivar advertencias SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== CONFIGURACIÓN ====================
BASE_URL = "http://147.96.81.252:8000"
MI_USUARIO = "LOS ELEGIDOS"
MODELO_OLLAMA = "qwen3-vl:8b"


# ==================== FUNCIONES DE CONSULTA ====================

def consultar_ia(prompt):
    """
    Envía un prompt al modelo de Ollama y devuelve la respuesta.
    
    Args:
        prompt: El texto del prompt a enviar a la IA
        
    Returns:
        str: La respuesta de la IA, o None si hay un error
    """
    try:
        response = ollama.chat(model=MODELO_OLLAMA, messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
        
    except ollama.ResponseError as e:
        print(f"❌ Error de Ollama: {e}")
        print("💡 Pista: ¿Está corriendo 'ollama serve'? ¿El modelo se llama correctamente?")
        return None
    except Exception as e:
        print(f"❌ Error inesperado al consultar IA: {e}")
        return None


def crear_alias():
    """Registra el alias del jugador en el servidor."""
    try:
        url = f"{BASE_URL}/alias/{MI_USUARIO}"
        response = requests.post(url, verify=False, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Alias '{MI_USUARIO}' registrado correctamente")
        else:
            print(f"⚠️ Código de respuesta: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error al crear alias: {e}")


def obtener_info():
    """Obtiene toda la información del juego y solicita resumen a la IA."""
    print("📡 Conectando con la API del juego...")
    
    try:
        response = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        data = response.json()
        
        print("\n🤖 --- ANÁLISIS DEL AGENTE (Qwen) ---")

        # Preparar prompt para la IA
        prompt = f"""
        Actúa como un asistente estratégico de un juego de gestión de recursos.
        
        DATOS ACTUALES:{json.dumps(data)}
        
        TAREA:
        hazme un resumen claro de los datos actuales
        """

        # Enviar a Ollama
        respuesta_ia = consultar_ia(prompt)
        
        if respuesta_ia:
            print(respuesta_ia)
            print("--------------------------------------")

    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API del juego.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


def obtener_estado():
    """Compara los recursos actuales con el objetivo usando IA."""
    print("📡 Conectando con la API del juego...")
    
    try:
        response = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        data = response.json()
        
        mis_recursos = data.get("Recursos", {})
        objetivo = data.get("Objetivo", {})
        
        print("\n🤖 --- ANÁLISIS DEL AGENTE (Qwen) ---")

        # Preparar prompt para la IA
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

        # Enviar a Ollama
        respuesta_ia = consultar_ia(prompt)
        
        if respuesta_ia:
            print(respuesta_ia)
            print("--------------------------------------")

    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API del juego.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


# ==================== FUNCIONES DE ACCIÓN ====================

def enviar_paquete():
    """Envía recursos a otro jugador."""
    # Obtener destinatario
    destinatario = input("👤 ¿A quién se lo envías?: ")

    # Pedir qué enviar
    recurso = input("🧱 Tipo de recurso (trigo, madera, piedra, tela...): ").lower().strip()
    
    try:
        cantidad = int(input(f"🔢 Cantidad de {recurso}: "))
    except ValueError:
        print("❌ Error: La cantidad debe ser un número entero.")
        return

    # Configurar petición
    url = f"{BASE_URL}/paquete"
    params = {"dest": destinatario}
    payload = {recurso: cantidad}

    try:
        print(f"🚀 Enviando {cantidad} de '{recurso}' a '{destinatario}'...")
        
        response = requests.post(
            url, 
            params=params, 
            json=payload, 
            verify=False, 
            timeout=5
        )

        if response.status_code == 200:
            print("✅ ¡Paquete entregado!")
            print(f"Respuesta: {response.json()}")
        else:
            print(f"❌ Fallo en el envío (Código {response.status_code}):")
            print(response.text)

    except Exception as e:
        print(f"❌ Error de conexión: {e}")


# ==================== MENÚ PRINCIPAL ====================

def mostrar_menu():
    """Muestra el menú principal y ejecuta la opción seleccionada."""
    while True:
        print("\n" + "="*50)
        print("🎮 MENÚ PRINCIPAL - Asistente con IA")
        print("="*50)
        print("1. Registrar alias")
        print("2. Ver información completa (con análisis IA)")
        print("3. Ver estado y recursos (con análisis IA)")
        print("4. Enviar paquete")
        print("5. Salir")
        print("="*50)
        
        opcion = input("\nSelecciona una opción (1-5): ").strip()
        
        if opcion == "1":
            crear_alias()
        elif opcion == "2":
            obtener_info()
        elif opcion == "3":
            obtener_estado()
        elif opcion == "4":
            enviar_paquete()
        elif opcion == "5":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


# ==================== PUNTO DE ENTRADA ====================

if __name__ == "__main__":
    mostrar_menu()

