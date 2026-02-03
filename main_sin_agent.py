"""
Cliente para gestión de recursos en el juego.
Permite consultar información, comparar estado actual con objetivos y enviar paquetes a otros jugadores.
"""

import sys
import json
import requests
import urllib3

# Desactivar advertencias SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== CONFIGURACIÓN ====================
BASE_URL = "http://147.96.81.252:8000"
MI_USUARIO = "LOS ELEGIDOS"


# ==================== FUNCIONES DE CONSULTA ====================

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
    """Obtiene toda la información del juego y muestra un resumen."""
    print("📡 Conectando con la API del juego...")
    
    try:
        response = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        data = response.json()
        
        print("\n" + "="*50)
        print("📊 INFORMACIÓN COMPLETA DEL JUEGO")
        print("="*50)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("="*50 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API del juego.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


def obtener_estado():
    """Compara los recursos actuales con el objetivo y muestra qué falta."""
    print("📡 Consultando estado del juego...")
    
    try:
        response = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        data = response.json()
        
        mis_recursos = data.get("Recursos", {})
        objetivo = data.get("Objetivo", {})
        
        print("\n" + "="*50)
        print("📦 MIS RECURSOS ACTUALES")
        print("="*50)
        for recurso, cantidad in mis_recursos.items():
            print(f"  • {recurso}: {cantidad}")
        
        print("\n" + "="*50)
        print("🎯 OBJETIVO PARA GANAR")
        print("="*50)
        for recurso, cantidad in objetivo.items():
            print(f"  • {recurso}: {cantidad}")
        
        print("\n" + "="*50)
        print("📊 ANÁLISIS DE RECURSOS")
        print("="*50)
        
        falta_algo = False
        for recurso, necesario in objetivo.items():
            tengo = mis_recursos.get(recurso, 0)
            diferencia = necesario - tengo
            
            if diferencia > 0:
                print(f"  ❌ {recurso}: Faltan {diferencia} (tienes {tengo}/{necesario})")
                falta_algo = True
            else:
                print(f"  ✅ {recurso}: Completado (tienes {tengo}/{necesario})")
        
        if not falta_algo:
            print("\n🎉 ¡FELICIDADES! Tienes todos los recursos necesarios para ganar.")
        
        print("="*50 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API del juego.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


# ==================== FUNCIONES DE ACCIÓN ====================

def enviar_paquete(destinatario=None):
    """
    Envía recursos a otro jugador.
    
    Args:
        destinatario: Nombre del jugador destino (opcional, se pedirá si no se proporciona)
    """
    # Obtener destinatario
    if destinatario is None:
        destinatario = input("👤 ¿A quién se lo envías?: ")
    
    # Obtener recurso y cantidad
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


def mostrar_menu():
    """Muestra el menú principal y ejecuta la opción seleccionada."""
    # Verificar conexión con el servidor antes de mostrar el menú
    print("📡 Verificando conexión con el servidor...")
    try:
        response = requests.get(f"{BASE_URL}/info", verify=False, timeout=5)
        if response.status_code == 200:
            print("✅ Conexión establecida correctamente\n")
        else:
            print(f"⚠️ Servidor respondió con código: {response.status_code}\n")
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar con el servidor.")
        print("💡 Verifica que el servidor esté activo y la URL sea correcta.\n")
        return
    except Exception as e:
        print(f"❌ Error al conectar: {e}\n")
        return
    
    while True:
        print("\n" + "="*50)
        print("🎮 MENÚ PRINCIPAL")
        print("="*50)
        print("1. Registrar alias")
        print("2. Ver información completa")
        print("3. Ver estado y recursos")
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

