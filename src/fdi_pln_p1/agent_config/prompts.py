"""Construcción de prompts del sistema y usuario para el modelo Ollama.

Centraliza la generación del texto que se envía al modelo en cada
iteración del bucle del agente.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Prompt de sistema (reglas y prioridades del agente)
# ---------------------------------------------------------------------------
def construir_system_prompt(mi_nombre: str) -> str:
    """Genera el prompt de sistema que define las reglas del agente.

    Args:
        mi_nombre: Alias del jugador controlado por el agente.
    """
    return (
        f"Eres {mi_nombre}, un agente en un juego de intercambio de recursos. "
        "Tu ÚNICO objetivo es conseguir todos los recursos de tu lista NECESITO "
        "intercambiándolos por recursos de tu lista SOBRA.\n\n"
        "REGLAS ABSOLUTAS:\n"
        "- Responde SIEMPRE invocando exactamente UNA función. NUNCA generes texto.\n"
        "- NUNCA ofrezcas, envíes ni intercambies oro bajo ningún concepto.\n"
        "- Los parámetros 'dest' e 'item_enviar' deben ser strings simples, nunca objetos.\n"
        "- Si NECESITO está vacío, has ganado: usa caso_4_ofertar_todos solo si SOBRA no está vacío.\n"
        "- Si SOBRA está vacío, solo puedes aceptar cartas o borrarlas, no ofertar.\n\n"
        "PRIORIDAD DE ACCIONES (de mayor a menor):\n"
        "1. Si hay una carta que dice 'acepto trato' o confirma un acuerdo previo, DEBES enviar lo necesario → caso_3_enviar.\n"
        "2. Si hay una carta útil (ofrece lo que necesito a cambio de lo que me sobra) → caso_1_aceptar.\n"
        "3. Si hay una carta inútil → caso_2_borrar.\n"
        "4. Si el buzón está vacío o ya procesaste todas las cartas → caso_4_ofertar_todos."
    )


# ---------------------------------------------------------------------------
# Prompt de usuario (estado actual del juego)
# ---------------------------------------------------------------------------
def construir_user_prompt(
    faltan: dict[str, int],
    sobran: dict[str, int],
    otros_jugadores: list[str],
    cartas_visibles: dict,
    memoria_oferta_busco: str | None,
    memoria_oferta_doy: str | None,
) -> str:
    """Genera el prompt de usuario con el estado actual del juego.

    Args:
        faltan: Recursos que faltan para cumplir el objetivo.
        sobran: Recursos sobrantes disponibles para intercambiar.
        otros_jugadores: Lista de alias de los demás jugadores.
        cartas_visibles: Subconjunto del buzón que se muestra al modelo.
        memoria_oferta_busco: Último recurso solicitado en oferta previa.
        memoria_oferta_doy: Último recurso ofrecido en oferta previa.
    """
    hay_cartas = len(cartas_visibles) > 0
    return (
        f"MI INVENTARIO:\n"
        f"- Recursos que NECESITO conseguir (me faltan): {json.dumps(faltan)}\n"
        f"- Recursos que me SOBRAN (puedo dar): {json.dumps(sobran)}\n"
        f"- Jugadores disponibles: {json.dumps(otros_jugadores)}\n\n"
        f"BUZÓN ({len(cartas_visibles)} carta(s)):\n"
        f"{json.dumps(cartas_visibles, indent=2) if hay_cartas else '(vacío)'}\n\n"
        f"Última oferta que ya envié: busco={memoria_oferta_busco}, "
        f"doy={memoria_oferta_doy}, no la repitas si hay alternativa\n\n"
        "ELIGE UNA ACCIÓN según la prioridad del system prompt:\n"
        "1) caso_1_aceptar — Hay una carta que OFRECE algo que necesito Y PIDE algo que me sobra.\n"
        "2) caso_2_borrar — Hay una carta que NO me interesa. Borra esa carta con su ID.\n"
        "3) caso_3_enviar — Hay una carta que dice 'acepto trato' y debo enviar el material prometido.\n"
        "4) caso_4_ofertar_todos — El buzón está vacío o ninguna carta es útil. "
        "Envía una oferta masiva."
    )
