import requests
import urllib3

# Omitimos las advertencias de seguridad por usar HTTPS sobre una IP directa
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN ---
BASE_URL = "http://147.96.81.252:8000"

# Intentamos conectar a la raíz (/) que suele dar un mensaje de bienvenida o estado
# Si en /docs ves que hay una ruta específica para probar (ej: /health), cámbiala aquí.
ENDPOINT = "/" 

url_completa = f"{BASE_URL}{ENDPOINT}docs"

print(f"📡 Conectando a {url_completa} ...")
response = requests.get(url_completa, verify=False, timeout=5)

if response.status_code == 200:
    print("✅ ¡Conexión Exitosa!")