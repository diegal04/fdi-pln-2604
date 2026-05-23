# Practica 5 - Transformer desde cero para LM y NER

Implementacion desde cero de un Transformer para modelado de lenguaje causal y
reconocimiento de entidades nombradas (NER). El proyecto incluye tokenizador BPE
propio, preentrenamiento de un LM sobre literatura inglesa, fine-tuning NER con
transferencia desde el LM, busqueda de hiperparametros y artefactos finales de
entrega.

## Integrantes

- Diego Alonso Arceiz
- Carlos Mantilla Mateos

## Estructura del repositorio

```text
practica_5/
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py                         # Wrapper de compatibilidad para ejecutar la CLI
├── informe_2604.html               # Informe de hiperparametros y resultados
├── datos/
│   ├── corpus_pretrain/            # Corpus ampliado para pretraining LM
│   │   ├── alice_in_wonderland.txt
│   │   ├── looking_glass.txt
│   │   ├── treasure_island.txt
│   │   ├── pride_and_prejudice.txt
│   │   ├── oliver_twist.txt
│   │   ├── great_expectations.txt
│   │   └── jane_eyre.txt
│   └── pre-entrega_2604/           # Corpus NER anotado y documentacion
│       ├── merged.json             # Dataset final BIO: 68 frases
│       ├── README.md
│       ├── METADATOS.md
│       ├── informe_etiquetado.html
│       ├── corpus_original/
│       ├── asignaciones/
│       ├── etiquetados/
│       └── scripts/
├── resultados_finales/
│   ├── tokenizer.json              # Tokenizador BPE final
│   ├── p5_causal_2604.pth          # Pesos preentrenados del LM causal
│   ├── p5_ner_2604.pth             # Pesos fine-tuned del modelo NER
│   ├── experiments/
│   │   ├── lm_exp2/history.json
│   │   ├── grid_ner/
│   │   └── ner_final/
│   └── grid_ner/
└── src/
    ├── cli/                        # CLI Click: train, generate, entities, grid-search
    ├── models/                     # Transformer, CausalLLM, atencion y RoPE
    ├── ner/                        # Modelo, entrenamiento, metricas y visualizacion NER
    ├── training/                   # Entrenamiento del LM causal
    └── tokenizer.py                # Tokenizador BPE propio
```

## Requisitos

El proyecto se gestiona con `uv`. En el entorno de laboratorio Linux:

```bash
uv lock
uv sync
uv format --check
```

El paquete expone el ejecutable:

```bash
uv run fdi-pln-2604-p5 --help
```

Dependencias de terceros usadas por el codigo entregable:

- `torch`
- `click`
- `loguru`
- `numpy`

Todas estan permitidas por el enunciado.

## Arquitectura

### Tokenizador BPE

`src/tokenizer.py` implementa un tokenizador Byte Pair Encoding propio. El
tokenizador final se guarda en `resultados_finales/tokenizer.json` y usa un
vocabulario de 500 tokens.

### Transformer base

La arquitectura comun esta en `src/models/transformer.py` y
`src/models/attention.py`. Cada bloque usa:

- Multi-head self-attention.
- Mascarado causal opcional para LM.
- Feed-forward con GELU y expansion 4.
- LayerNorm en esquema pre-norm.
- Dropout en atencion y FFN.
- Validacion de compatibilidad entre `d_model` y `n_heads`.

Configuracion final:

| parametro | valor |
| --- | ---: |
| `vocab_size` | 500 |
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `expansion` | 4 |
| `context_size` | 128 |

### RoPE

El modelo usa Rotary Position Embeddings (RoPE) en la atencion:

- Las frecuencias se calculan como `theta_i = 1 / 10000^(2i / head_dim)`.
- `cos` y `sin` se precalculan hasta `max_seq_len` y quedan registrados como
  buffers del modulo.
- La rotacion se aplica a queries y keys antes del producto escalar.
- No se usan embeddings posicionales aprendidos.

Esto permite codificar relaciones posicionales de forma relativa y reduce la
dependencia de posiciones absolutas aprendidas en un corpus pequeno.

### LM causal

`src/models/causal_lm.py` define `CausalLLM`, que anade una cabeza lineal sobre
el Transformer para predecir el siguiente token. El entrenamiento se ejecuta con
atencion causal (`causal=True`) y el checkpoint final es:

```text
resultados_finales/p5_causal_2604.pth
```

### NER

`src/ner/model.py` define `NERLLM`, que reutiliza el backbone Transformer con
atencion bidireccional (`causal=False`) y una cabeza de clasificacion por token.
El backbone se inicializa desde `resultados_finales/p5_causal_2604.pth` y la
cabeza NER se entrena con etiquetas BIO simplificadas:

| etiqueta | significado |
| --- | --- |
| `o` | no entidad |
| `pi` | B-PER |
| `pc` | I-PER |
| `li` | B-LOC |
| `lc` | I-LOC |

Para compensar el desbalance del dataset, la loss pondera las clases de entidad
y separa los pesos de inicio y continuacion:

```text
o  = 1
pi = entity_loss_weight
pc = entity_loss_weight * continuation_weight_multiplier
li = entity_loss_weight * location_weight_multiplier
lc = entity_loss_weight * location_weight_multiplier * continuation_weight_multiplier
```

## Corpus

### Pretraining LM

El corpus de preentrenamiento esta en `datos/corpus_pretrain/`. Se amplio desde
los dos libros originales de Lewis Carroll hasta siete libros clasicos:

| fichero | tamano aproximado |
| --- | ---: |
| `alice_in_wonderland.txt` | 150 KB |
| `looking_glass.txt` | 169 KB |
| `treasure_island.txt` | 380 KB |
| `pride_and_prejudice.txt` | 738 KB |
| `oliver_twist.txt` | 917 KB |
| `great_expectations.txt` | 1038 KB |
| `jane_eyre.txt` | 1044 KB |
| **total** | **~4.4 MB** |

La ampliacion del corpus mejora la diversidad lexica y hace que el LM vea mas
nombres propios y estructuras narrativas antes del fine-tuning NER.

### Dataset NER

El dataset final esta en `datos/pre-entrega_2604/merged.json`:

- 68 frases.
- 7219 tokens.
- 227 tokens de entidad.
- Anotacion BIO para `PER` y `LOC`.
- Solape de anotadores y documentacion en `datos/pre-entrega_2604/`.

## Comandos principales

La CLI tiene ayuda integrada:

```bash
uv run fdi-pln-2604-p5 --help
uv run fdi-pln-2604-p5 train --help
uv run fdi-pln-2604-p5 train tokenizer --help
uv run fdi-pln-2604-p5 train lm --help
uv run fdi-pln-2604-p5 train ner --help
uv run fdi-pln-2604-p5 generate --help
uv run fdi-pln-2604-p5 entities --help
uv run fdi-pln-2604-p5 grid-search --help
uv run fdi-pln-2604-p5 grid-search ner --help
```

### 1. Entrenar tokenizador

```bash
uv run fdi-pln-2604-p5 train tokenizer datos/corpus_pretrain \
  --vocab-size 500 \
  --out resultados_finales/tokenizer.json
```

Parametros principales:

- `CORPUS`: fichero `.txt` o directorio con `.txt`.
- `--vocab-size`: tamano maximo del vocabulario BPE.
- `--out`: ruta de salida del tokenizador.

### 2. Preentrenar LM causal

```bash
uv run fdi-pln-2604-p5 train lm datos/corpus_pretrain \
  --tokenizer resultados_finales/tokenizer.json \
  --out resultados_finales/p5_causal_2604.pth \
  --history-out resultados_finales/experiments/lm_exp2/history.json \
  --context-size 128 \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.0003 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 4 \
  --expansion 4 \
  --dropout 0.2 \
  --warmup-steps 100 \
  --weight-decay 0.1
```

Parametros principales:

- `CORPUS`: fichero o directorio de pretraining.
- `--tokenizer`: tokenizador BPE ya entrenado.
- `--out`: pesos LM de salida.
- `--history-out`: historial JSON.
- `--context-size`: longitud maxima de contexto.
- `--epochs`, `--batch-size`, `--lr`: parametros de entrenamiento.
- `--d-model`, `--n-heads`, `--n-layers`, `--expansion`, `--dropout`:
  arquitectura del Transformer.
- `--warmup-steps`, `--weight-decay`: scheduler y regularizacion.

### 3. Entrenar NER

```bash
uv run fdi-pln-2604-p5 train ner datos/pre-entrega_2604/merged.json \
  --tokenizer resultados_finales/tokenizer.json \
  --lm-weights resultados_finales/p5_causal_2604.pth \
  --out resultados_finales/p5_ner_2604.pth \
  --history-out resultados_finales/experiments/ner_final/history.json \
  --metrics-dir resultados_finales/experiments/ner_final/metrics \
  --context-size 128 \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 4 \
  --expansion 4 \
  --dropout 0.3 \
  --entity-loss-weight 10 \
  --location-weight-multiplier 1.0 \
  --continuation-weight-multiplier 3.0 \
  --warmup-steps 50 \
  --weight-decay 0.15 \
  --freeze-epochs 0
```

Parametros principales:

- `ANNOTATIONS`: `merged.json` con tokens y etiquetas.
- `--tokenizer`: tokenizador usado en el LM.
- `--lm-weights`: checkpoint del LM para inicializar el backbone.
- `--out`: pesos NER de salida.
- `--metrics-dir`: directorio de graficos y metricas.
- `--entity-loss-weight`, `--location-weight-multiplier`,
  `--continuation-weight-multiplier`: ponderacion de loss por clase.
- `--freeze-epochs`: epocas con backbone congelado. El modelo final usa `0`.

### 4. Generar texto

```bash
uv run fdi-pln-2604-p5 generate "Alice looked around" \
  --tokenizer resultados_finales/tokenizer.json \
  --weights resultados_finales/p5_causal_2604.pth \
  --max-new-tokens 80 \
  --temperature 1.0 \
  --top-k 50 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 4 \
  --context-size 128 \
  --expansion 4 \
  --dropout 0.0
```

Parametros principales:

- `PROMPT`: texto inicial. Puede omitirse para generar desde contexto vacio.
- `--weights`: ruta al checkpoint LM que se quiere cargar.
- `--max-new-tokens`: numero maximo de tokens nuevos.
- `--temperature`: aleatoriedad del muestreo.
- `--top-k`: muestreo top-k. Usar `0` para desactivarlo.

### 5. Extraer entidades

Con texto directo:

```bash
uv run fdi-pln-2604-p5 entities "Alice met the White Rabbit in Wonderland." \
  --tokenizer resultados_finales/tokenizer.json \
  --weights resultados_finales/p5_ner_2604.pth \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 4 \
  --context-size 128 \
  --expansion 4 \
  --dropout 0.0
```

Con fichero:

```bash
uv run fdi-pln-2604-p5 entities fragmento.txt \
  --tokenizer resultados_finales/tokenizer.json \
  --weights resultados_finales/p5_ner_2604.pth
```

Parametros principales:

- `TEXT`: cadena literal o ruta de fichero.
- `--weights`: ruta al checkpoint NER que se quiere cargar.
- `--add-spaces/--no-add-spaces`: controla si se antepone espacio antes de
  tokenizar cada palabra. El modelo final usa el valor por defecto,
  `--no-add-spaces`.

Ejemplo esperado:

```text
PER   alice
PER   the white rabbit
LOC   wonderland
```

### 6. Grid search

Tokenizadores:

```bash
uv run fdi-pln-2604-p5 grid-search tokenizer datos/corpus_pretrain \
  --vocab-sizes 200,300,500 \
  --out-dir resultados_finales/experiments/grid_tokenizers
```

LM:

```bash
uv run fdi-pln-2604-p5 grid-search lm datos/corpus_pretrain \
  --tokenizer resultados_finales/tokenizer.json \
  --out-dir resultados_finales/experiments/grid_lm \
  --epochs 3,5 \
  --batch-sizes 32,64 \
  --lrs 0.0003,0.0001 \
  --d-models 128 \
  --n-heads 4 \
  --n-layers 2,4 \
  --dropouts 0.1,0.2 \
  --context-size 128 \
  --expansion 4 \
  --warmup-steps 100 \
  --weight-decay 0.1
```

NER:

```bash
uv run fdi-pln-2604-p5 grid-search ner datos/pre-entrega_2604/merged.json \
  --tokenizer resultados_finales/tokenizer.json \
  --lm-weights resultados_finales/p5_causal_2604.pth \
  --out-dir resultados_finales/experiments/grid_ner \
  --epochs 50 \
  --batch-sizes 16,32,64 \
  --lrs 0.001,0.0005,0.0003,0.0001 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 4 \
  --dropout 0.3 \
  --entity-loss-weights 5,10,15,20 \
  --selection-accuracy-floor 0.8 \
  --selection-non-o-weight 1.5 \
  --context-size 128 \
  --expansion 4 \
  --warmup-steps 50 \
  --weight-decay 0.1 \
  --continuation-weight-multiplier 3.0
```

Para pruebas cortas puede anadirse `--max-runs N`.

## Resultados documentados

El informe `informe_2604.html` resume la exploracion. Resultados principales:

- LM final: `train_loss=2.6794`, `val_loss=3.0874` en epoch 15.
- Grid NER: 48 runs.
- Mejor configuracion NER: `lr=0.001`, `batch_size=16`,
  `entity_loss_weight=10`, `continuation_weight_multiplier=3.0`.
- Mejor checkpoint NER: epoch 43, `selection_score=2.1301`.
- Recall final: `pi=85.0%`, `pc=75.4%`, `li=87.5%`, `lc=73.3%`.

## Construccion del wheel

En Linux con `uv`:

```bash
rm -rf dist build *.egg-info
uv build --wheel
```

El resultado esperado es:

```text
dist/fdi_pln_2604_p5-1.0-py3-none-any.whl
```

El wheel debe proporcionar el ejecutable:

```bash
uv run fdi-pln-2604-p5 --help
```

## Checklist de entrega

Contenido del zip del campus virtual:

```text
fdi_pln_2604_p5-1.0-py3-none-any.whl
p5_causal_2604.pth
p5_ner_2604.pth
informe_2604.html
```

Comando recomendado para crear el zip desde la raiz del repo:

```bash
zip -j entrega_p5_2604.zip \
  dist/fdi_pln_2604_p5-1.0-py3-none-any.whl \
  resultados_finales/p5_causal_2604.pth \
  resultados_finales/p5_ner_2604.pth \
  informe_2604.html
```

Validacion rapida antes de entregar:

```bash
uv format --check
uv run fdi-pln-2604-p5 --help
uv run fdi-pln-2604-p5 generate "Alice looked around" \
  --weights resultados_finales/p5_causal_2604.pth \
  --tokenizer resultados_finales/tokenizer.json \
  --max-new-tokens 20
uv run fdi-pln-2604-p5 entities "Alice met the White Rabbit in Wonderland." \
  --weights resultados_finales/p5_ner_2604.pth \
  --tokenizer resultados_finales/tokenizer.json
uv build --wheel
unzip -l entrega_p5_2604.zip
```

## Release GitHub

La entrega GitHub debe publicarse como release `p5v1.0`:

```bash
git add -A
git commit -m "Entrega practica 5 v1.0"
git push
git tag -a p5v1.0 -m "Practica 5 version 1.0"
git push origin p5v1.0
```

Despues, crear la release en GitHub asociada a la tag `p5v1.0` y adjuntar los
artefactos finales.
