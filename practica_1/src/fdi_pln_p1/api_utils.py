"""Utilidades HTTP para comunicarse con el servidor Butler.

Encapsula las peticiones GET, POST y DELETE de forma centralizada
con manejo de errores y logging.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from rich.console import Console

from fdi_pln_p1 import MODO_MONOPUESTO

console = Console()


def api_request(
    base_url: str,
    metodo: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | bool:
    """Realiza una petición HTTP al servidor Butler.

    Args:
        base_url: URL base del servidor (p.ej. ``http://host:7719``).
        metodo: Verbo HTTP (``GET``, ``POST`` o ``DELETE``).
        endpoint: Ruta del recurso (p.ej. ``/info``).
        params: Parámetros de query string opcionales.
        payload: Cuerpo JSON opcional (solo para ``POST``).

    Returns:
        - ``dict``: respuesta JSON para ``GET`` / ``POST``.
        - ``bool``: indica éxito para ``DELETE``.
        - ``{}``: en caso de error o verbo no soportado.
    """
    try:
        endpoint_normalizado = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url_completa = f"{base_url.rstrip('/')}{endpoint_normalizado}"

        if metodo not in {"GET", "POST", "DELETE"}:
            return {}

        response = httpx.request(
            metodo,
            url_completa,
            params=params,
            json=payload,
            verify=False,
            timeout=3.0,
        )

        if metodo == "DELETE":
            return response.status_code < 400

        try:
            data = response.json()
        except ValueError:
            data = {}

        if response.status_code >= 400:
            logger.warning(
                f"HTTP {response.status_code} en {metodo} {endpoint_normalizado}"
            )
        return data

    except Exception as exc:
        logger.warning(f"Error de conexión HTTP ({metodo} {endpoint}): {exc}")
        return {}


def construir_params_api(
    modo_puesto: str,
    agente: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Construye los parámetros de query string según el modo de la API."""
    params_finales = dict(params or {})
    if modo_puesto == MODO_MONOPUESTO and agente:
        params_finales["agente"] = agente
    return params_finales or None


def api_request_modo(
    base_url: str,
    metodo: str,
    endpoint: str,
    modo_puesto: str,
    agente: str | None = None,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | bool:
    """Envoltorio de :func:`api_request` que inyecta el agente según el modo."""
    return api_request(
        base_url,
        metodo,
        endpoint,
        params=construir_params_api(
            modo_puesto=modo_puesto, agente=agente, params=params
        ),
        payload=payload,
    )


def registrar_alias(mi_nombre: str, url: str) -> None:
    """Registra el alias del jugador en el servidor Butler."""
    resultado = api_request(url, "POST", f"/alias/{mi_nombre}")
    if isinstance(resultado, dict) and resultado.get("status") == "ok":
        console.print(f"✅ Alias '{mi_nombre}' registrado correctamente")
    else:
        logger.warning("Alias no registrado")
