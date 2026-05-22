"""Plantillas nuevas: 6 frases por JSON, mínimo 20 palabras (no sobrescribe alice_jsons/)."""

from pathlib import Path

from fdi_pln_2611_p5.annotations.templates import (
    crear_jsons_anotacion,
    tokenizar_palabras,
)

# 13 anotadores × 6 frases = 78 huecos → 39 frases distintas (cada una en 2 JSON)
N_JSON = 13
FRASES_POR_JSON = 6
MIN_PALABRAS = 20
SEED = 46

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    info = crear_jsons_anotacion(
        archivo_entrada=base / "data" / "alice_in_wonderland.txt",
        directorio_salida=base / "data" / "alice_jsons_6frases",
        tokenizar=tokenizar_palabras,
        granularidad="palabra",
        n_frases=None,
        n_json=N_JSON,
        frases_por_json=FRASES_POR_JSON,
        min_palabras=MIN_PALABRAS,
        seed=SEED,
    )
    print(
        f"Generados {info['n_json']} JSON en {info['directorio']} "
        f"({info['n_frases']} frases, {info['frases_por_json']}/JSON, "
        f"min {info['min_palabras']} palabras)"
    )
