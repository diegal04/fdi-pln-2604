from __future__ import annotations

from typing import Any

import httpx
from loguru import logger


def api_request(
    base_url: str,
    metodo: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | bool:
    """
    Petición HTTP simple para Butler.
    - GET/POST: devuelve JSON (dict) o {} si falla.
    - DELETE: devuelve bool de éxito.
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
        return data if isinstance(data, dict) else {}

    except Exception as exc:
        logger.warning(f"Error de conexión HTTP ({metodo} {endpoint}): {exc}")
        return {}

