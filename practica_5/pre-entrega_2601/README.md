# PLN · Preentrega P5 · Named Entity Recognition (NER) · Grupo 1 de etiquetado

Proyecto desarrollado para la asignatura de Procesamiento del Lenguaje Natural (PLN).

Esta pre-entrega consiste en la construcción de un corpus anotado manualmente para tareas de reconocimiento de entidades nombradas (NER), junto con herramientas automáticas para gestionar anotaciones, fusionar resultados y analizar la calidad del corpus generado.

---

# Descripción general

El proyecto implementa un flujo completo de trabajo para anotación colaborativa de entidades nombradas utilizando fragmentos literarios de *Alice in Wonderland*.

A partir de estos textos se generaron automáticamente conjuntos de frases distribuidos entre distintos anotadores. Posteriormente, las anotaciones fueron fusionadas automáticamente y utilizadas para calcular métricas de calidad y acuerdo inter-anotador.

El sistema incluye:

- generación automática de plantillas de anotación;
- distribución de frases entre anotadores;
- doble anotación por fragmento;
- resolución automática de conflictos;
- cálculo de métricas de acuerdo;
- y generación automática de informes visuales.

---

# Tipos de entidades

La práctica se centra en dos tipos principales de entidades:

- personas y personajes (`PER`);
- localizaciones (`LOC`).

Para ello se utilizó un esquema de etiquetas simplificado inspirado en BIO.

---

# Corpus utilizado

Los textos utilizados se encuentran en:

```text
corpus_original/alice_in_wonderland.txt
corpus_original/looking_glass.txt
```

La mayoría de frases utilizadas para anotación se extrajeron automáticamente de *Alice in Wonderland*.

---

# Estructura del proyecto

La práctica se organizó separando claramente los datos originales, las anotaciones manuales, los scripts de procesamiento y los resultados generados automáticamente. Esto permite mantener un flujo de trabajo más limpio y facilita tanto la reutilización del corpus como el entrenamiento posterior del modelo NER.

```text
pre-entrega_2601/
├── asignaciones/              # Distribución de frases entre anotadores
├── corpus_original/           # Textos originales utilizados como corpus
├── etiquetados/               # JSON anotados manualmente
├── imagenes/                  # Capturas e imágenes utilizadas en el informe
├── scripts/                   # Scripts principales del proyecto
├── merged.json                # Corpus fusionado final
├── informe_etiquetado.html    # Informe visual generado automáticamente
├── METADATOS.md               # Informe técnico de la preentrega
└── README.md                  # Descripción general del proyecto
```

Los scripts principales del proyecto se encuentran dentro de scripts/. Desde ahí se implementa toda la lógica de generación de plantillas, fusión de anotaciones, cálculo de métricas y construcción automática del informe HTML.

---

# Funcionamiento general

El pipeline desarrollado sigue las siguientes fases:

1. Extracción automática de frases desde el corpus.
2. Generación de JSON de anotación.
3. Distribución de frases entre anotadores.
4. Anotación manual colaborativa.
5. Doble anotación de cada frase.
6. Fusión automática de resultados.
7. Resolución de conflictos entre etiquetas.
8. Cálculo de métricas de calidad.
9. Generación automática del informe HTML.

---

# Ejecución del informe

Para generar automáticamente el corpus fusionado y el informe visual HTML:

```bash
python -m scripts.report
```

El informe generado se almacena en:

```text
informe_etiquetado.html
```

En Windows puede abrirse automáticamente desde terminal mediante:

```powershell
start informe_etiquetado.html
```

En Linux:

```bash
xdg-open informe_etiquetado.html
```

Y en macOS:

```bash
open informe_etiquetado.html
```

---

## Generación de plantillas de anotación

```bash
python -m scripts.generar_jsons_6frases
```

Genera automáticamente los ficheros JSON de anotación y las asignaciones de frases.

---

# Resultados generales

El corpus generado presenta un acuerdo elevado entre anotadores.

Principales métricas obtenidas:

- κ de Cohen medio: **0.835**
- acuerdo medio por token: **98%**
- frases fusionadas: **59**
- pares anotados: **59**

Estos resultados indican una anotación consistente y suficientemente estable para utilizar el corpus en entrenamiento y evaluación de modelos NER.

---

# Informe visual

El proyecto genera automáticamente un informe HTML interactivo con:

- métricas globales;
- distribución de etiquetas;
- histogramas de κ;
- matrices de confusión;
- ranking de frases;
- cobertura de entidades;
- y análisis de concordancia entre anotadores.

---

## Resumen general

![Resumen general](imagenes/resumen_general.png)

---

## Distribución de entidades

![Distribución de etiquetas](imagenes/distribucion_etiquetas.png)

---

## Acuerdo inter-anotador

![Acuerdo entre anotadores](imagenes/acuerdo_anotadores.png)

---

# Tecnologías utilizadas

- Python 3
- JSON
- HTML
- CSS
- JavaScript
- Chart.js

---

# Participantes

- Pablo Alonso Romero
- Rodrigo Jesús-Portanet Martínez
- Carlota Salazar Martín
- Bautista Pelossi Schweizer
- Ignacio Ramírez Suárez
- Bryan Xavier Quilumba Farinango
- Carlos Mantilla Mateos
- Javier Martín Fuentes
- João Francisco Sampaio Pereira
- Yushan Yang Xu
- María Romero Huertas
- Marina Triviño de las Heras
- Carmen Fernández González

---

# Documentación adicional

La explicación técnica detallada del proceso de anotación, fusión y evaluación puede consultarse en:

```text
METADATOS.md
```

---

# Estado del proyecto

La preentrega queda completamente preparada para las siguientes fases de la práctica:

- entrenamiento del modelo NER;
- evaluación automática;
- experimentación con arquitecturas propias;
- y análisis posterior de resultados.