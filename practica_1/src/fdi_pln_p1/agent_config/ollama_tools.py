"""Definición de herramientas (tools) expuestas al modelo Ollama.

Cada entrada describe una función que el modelo puede invocar para ejecutar
una acción dentro del juego de intercambio de recursos de Butler.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Caso 1 – Aceptar una carta del buzón
# ---------------------------------------------------------------------------
_CASO_1_ACEPTAR = {
    "type": "function",
    "function": {
        "name": "caso_1_aceptar",
        "description": (
            "Aceptar una carta del buzón que propone un intercambio favorable. "
            "Usar SOLO cuando la carta ofrece un recurso que aparece en tu lista "
            "de recursos que NECESITAS y a cambio pide un recurso que aparece en "
            "tu lista de recursos que te SOBRAN. Nunca envíes oro."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dest": {
                    "type": "string",
                    "description": "Nombre del jugador destinatario.",
                },
                "item_enviar": {
                    "type": "string",
                    "description": (
                        "Nombre exacto del recurso que le envías como pago. "
                        "Debe ser uno de tus recursos sobrantes. "
                        "Nunca puede ser oro. Ejemplo: 'madera'."
                    ),
                },
                "cant": {
                    "type": "integer",
                    "description": "Cantidad del recurso que envías.",
                },
                "item_esperado": {
                    "type": "string",
                    "description": "Nombre exacto del recurso que esperas que te envíen a cambio.",
                },
                "cant_esperada": {
                    "type": "integer",
                    "description": "Cantidad del recurso que esperas recibir a cambio.",
                },
                "id_carta": {
                    "type": "string",
                    "description": (
                        "ID único de la carta que aceptas, tal como aparece "
                        "en las claves del buzón. Ejemplo: 'abc123'."
                    ),
                },
            },
            "required": [
                "dest",
                "item_enviar",
                "cant",
                "item_esperado",
                "cant_esperada",
                "id_carta",
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# Caso 2 – Borrar una carta inútil
# ---------------------------------------------------------------------------
_CASO_2_BORRAR = {
    "type": "function",
    "function": {
        "name": "caso_2_borrar",
        "description": (
            "Eliminar una carta del buzón que NO es útil. Usar cuando la carta "
            "pide un recurso que no te sobra, ofrece algo que no necesitas, "
            "o simplemente no te conviene el trato."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id_carta": {
                    "type": "string",
                    "description": (
                        "ID único de la carta a eliminar, tal como aparece "
                        "en las claves del buzón."
                    ),
                },
            },
            "required": ["id_carta"],
        },
    },
}

# ---------------------------------------------------------------------------
# Caso 3 – Enviar recursos por un trato aceptado
# ---------------------------------------------------------------------------
_CASO_3_ENVIAR = {
    "type": "function",
    "function": {
        "name": "caso_3_enviar",
        "description": (
            "Enviar recursos a un jugador para cumplir un acuerdo ya aceptado. "
            "Usar SOLO cuando una carta dice 'acepto trato' o confirma que el "
            "otro jugador ya aceptó un trato previo y espera recibir material tuyo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dest": {
                    "type": "string",
                    "description": "Nombre (alias) del jugador al que envías el recurso.",
                },
                "item_enviar": {
                    "type": "string",
                    "description": (
                        "Nombre exacto del recurso prometido en el acuerdo. "
                        "Nunca puede ser oro. Ejemplo: 'piedra'."
                    ),
                },
                "cant": {
                    "type": "integer",
                    "description": "Cantidad prometida del recurso que envías.",
                },
                "id_carta": {
                    "type": "string",
                    "description": "ID de la carta del acuerdo.",
                },
            },
            "required": ["dest", "item_enviar", "cant", "id_carta"],
        },
    },
}

# ---------------------------------------------------------------------------
# Caso 4 – Enviar oferta masiva a todos los jugadores
# ---------------------------------------------------------------------------
_CASO_4_OFERTAR_TODOS = {
    "type": "function",
    "function": {
        "name": "caso_4_ofertar_todos",
        "description": (
            "Enviar una oferta de intercambio a TODOS los jugadores. "
            "Usar cuando el buzón está vacío o ninguna carta del buzón es útil "
            "(después de borrar las inútiles). Los recursos deben ser strings "
            "simples. El recurso que buscas debe estar en tu lista de NECESITO "
            "y el que ofreces en tu lista de SOBRA."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recurso_que_busco": {
                    "type": "string",
                    "description": (
                        "Nombre exacto de un recurso que necesitas conseguir "
                        "(debe aparecer en tu lista de recursos faltantes)."
                    ),
                },
                "recurso_que_doy": {
                    "type": "string",
                    "description": (
                        "Nombre exacto de un recurso que ofreces a cambio "
                        "(debe aparecer en tu lista de recursos sobrantes)."
                    ),
                },
            },
            "required": ["recurso_que_busco", "recurso_que_doy"],
        },
    },
}

# ---------------------------------------------------------------------------
# Lista completa de tools exportada al modelo
# ---------------------------------------------------------------------------
OLLAMA_TOOLS: list[dict] = [
    _CASO_1_ACEPTAR,
    _CASO_2_BORRAR,
    _CASO_3_ENVIAR,
    _CASO_4_OFERTAR_TODOS,
]
