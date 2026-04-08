"""Utilidades de presentacion en consola para el agente Butler."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def mostrar_jugadores_partida(gente_raw: Any, mi_nombre: str) -> None:
    """Muestra en consola una tabla con todos los jugadores visibles."""
    if not isinstance(gente_raw, list):
        console.print("[yellow]👥 No se pudo obtener la lista de jugadores.[/yellow]")
        return

    jugadores: list[str] = []
    for entry in gente_raw:
        alias = ""
        if isinstance(entry, str):
            alias = entry.strip()
        elif isinstance(entry, dict):
            alias = str(entry.get("alias", "")).strip()

        if alias and alias not in jugadores:
            jugadores.append(alias)

    if mi_nombre and mi_nombre not in jugadores:
        jugadores.append(mi_nombre)

    if not jugadores:
        console.print("[yellow]👥 No hay jugadores visibles en la partida.[/yellow]")
        return

    tabla = Table(title="Jugadores de la partida", header_style="bold cyan")
    tabla.add_column("#", justify="right", style="dim", width=3)
    tabla.add_column("Jugador", style="bold")
    tabla.add_column("Rol", justify="center")

    for indice, jugador in enumerate(jugadores, start=1):
        rol = "[bold green]Tu[/bold green]" if jugador == mi_nombre else "Rival"
        tabla.add_row(str(indice), jugador, rol)

    console.print(tabla)
