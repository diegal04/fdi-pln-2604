"""Wrapper del CLI principal de practica_5."""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from cli import cli  # noqa: E402


if __name__ == "__main__":
    cli()
