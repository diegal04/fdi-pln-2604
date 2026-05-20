# Practica 5

## Exploracion de hiperparametros

La exploracion de hiperparametros se centro en el ajuste fino del modelo NER. La arquitectura del Transformer no se incluyo en el grid porque los pesos de partida proceden de `practica_5/pesos_modelo.pth`; cambiar `d_model`, `n_heads`, `n_layers`, `dropout`, `context_size` o `expansion` impediria cargar coherentemente el checkpoint preentrenado. Por tanto, se mantuvo fija la arquitectura y se exploraron los hiperparametros propios del fine-tuning.

### Problema detectado

La primera prueba mostro que la accuracy total no era suficiente para evaluar NER. El corpus esta muy desbalanceado: tras alinear las etiquetas a BPE, la mayoria de tokens pertenecen a la clase `o`, mientras que las entidades son minoritarias:

| etiqueta | conteo aproximado en train |
| --- | ---: |
| `o` | 7441 |
| `pi` | 86 |
| `pc` | 278 |
| `li` | 20 |
| `lc` | 61 |

Con este reparto, un modelo que predice casi todo como `o` puede obtener una accuracy total cercana a 0.96 y, aun asi, no reconocer entidades. Por eso se monitorizo tambien `non_o_accuracy`, que mide los aciertos sobre las posiciones cuya etiqueta real no es `o`. En la practica, esta metrica se interpreta como una recuperacion micro de tokens de entidad: responde a la pregunta "de los tokens que realmente eran entidad, cuantos he clasificado como su etiqueta correcta".

### Espacio de busqueda

Se mantuvo fija la arquitectura compatible con el checkpoint preentrenado:

| parametro | valor |
| --- | ---: |
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `dropout` | 0.2 |
| `context_size` | 128 |
| `expansion` | 4 |

El grid se aplico sobre los parametros de fine-tuning:

| hiperparametro | valores probados |
| --- | --- |
| `batch_size` | 8, 16, 32, 64 |
| `lr` | 0.001, 0.0005, 0.0003, 0.0001, 0.00005 |
| `entity_loss_weight` | 3, 5, 7, 10, 15, 20, 25, 30 |
| `epochs` | 50 |

Cada configuracion se entreno hasta 50 epocas, pero no se eligio necesariamente la ultima epoca. Durante el entrenamiento se guardo internamente el mejor checkpoint segun la metrica de seleccion, porque se observo que algunas configuraciones empeoraban en validacion al final del entrenamiento.

### Funcion de seleccion

Se uso una metrica compuesta:

```text
score = val_accuracy + 1.5 * val_non_o_accuracy
```

con filtro:

```text
si val_accuracy < 0.8, score = -1
```

La razon del filtro es evitar modelos que detecten muchas entidades a costa de romper completamente la clase mayoritaria `o`. La ponderacion 1.5 da mas importancia a las entidades, que son la parte dificil de la tarea, pero mantiene la accuracy total como mecanismo de control. Ademas, se reviso `val_macro_entity_f1` para detectar configuraciones con muchos falsos positivos, ya que `non_o_accuracy` por si sola no penaliza suficientemente predecir entidades de mas sobre tokens `o`.

### Mejor configuracion encontrada

La mejor configuracion fue el `run 087`:

| parametro | valor |
| --- | ---: |
| `batch_size` | 32 |
| `lr` | 0.001 |
| `entity_loss_weight` | 25 |
| mejor epoca | 33 |
| `val_accuracy` | 0.8674 |
| `val_non_o_accuracy` | 0.7273 |
| `val_macro_entity_f1` | 0.3093 |
| `selection_score` | 1.9583 |

El calculo del score fue:

```text
0.8674 + 1.5 * 0.7273 = 1.9583
```

El valor `val_non_o_accuracy = 0.7273` sale de la matriz de confusion de la mejor epoca. En validacion habia 44 tokens de entidad reales:

| etiqueta | soporte | aciertos | recall |
| --- | ---: | ---: | ---: |
| `pi` | 13 | 10 | 0.7692 |
| `pc` | 28 | 20 | 0.7143 |
| `li` | 1 | 0 | 0.0000 |
| `lc` | 2 | 2 | 1.0000 |
| **total no-`o`** | **44** | **32** | **0.7273** |

Por tanto:

```text
32 / 44 = 0.7273
```

Este resultado debe interpretarse con cuidado. Aunque el modelo recupera una parte razonable de las entidades reales, `val_macro_entity_f1 = 0.3093` muestra que sigue habiendo bastantes falsos positivos y errores de tipo. Esto revela una tension propia del corpus: al subir el peso de las entidades en la loss, el modelo deja de ignorarlas, pero tambien aumenta el riesgo de etiquetar como entidad tokens que realmente son `o`.

La mejor epoca fue la 33, no la 50. Esto confirma que entrenar mas tiempo no siempre mejora el modelo: hacia el final el modelo mantenia una accuracy total alta, pero perdia rendimiento sobre las clases de entidad. Por ejemplo, en la epoca 50 del mismo run la accuracy total era 0.9063, pero `val_non_o_accuracy` habia bajado a 0.5455.

### Conclusiones

La exploracion mostro tres realidades importantes del corpus y del modelo:

1. La accuracy total es enganosa en NER con corpus desbalanceado. Es imprescindible medir por separado las clases que no son `o`.
2. Penalizar mas las entidades en la loss mejora claramente la recuperacion de `pi`, `pc`, `li` y `lc`, pero pesos demasiado agresivos pueden aumentar falsos positivos y bajar la precision/F1.
3. El mejor modelo no coincide necesariamente con la ultima epoca. Por eso se adopto seleccion de mejor checkpoint interno segun una metrica compuesta.

En conjunto, la busqueda de hiperparametros no se limito a probar valores arbitrarios, sino que se adapto a una propiedad concreta del corpus: el fuerte desbalance entre `o` y entidades. La decision final prioriza un equilibrio entre no destruir la clasificacion general y mejorar el reconocimiento real de entidades nombradas.
