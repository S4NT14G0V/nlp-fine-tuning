---
language: en
license: apache-2.0
base_model: google-bert/bert-base-cased
datasets:
- eriktks/conll2003
pipeline_tag: token-classification
metrics:
- precision
- recall
- f1
tags:
- named-entity-recognition
- ner
- token-classification
- bert
---

# bert-ner-conll2003

`bert-base-cased` ajustado (*fine-tuned*) para **reconocimiento de entidades nombradas** en inglés. Las entidades se etiquetan en formato BIO con cuatro tipos:

- **PER:** personas.
- **ORG:** organizaciones.
- **LOC:** lugares.
- **MISC:** otras entidades, como nacionalidades, eventos y torneos.

Es el problema 2 (etiquetado de secuencias) del Proyecto 2 del curso *Fundamentos en Procesamiento del Lenguaje Natural* de la Universidad de Antioquia. Todo el desarrollo está en el repositorio [nlp-fine-tuning-transformers](https://github.com/s4nt14g0v/nlp-fine-tuning).

## Uso

```python
from transformers import pipeline

ner = pipeline("token-classification", model="s4nt14g0v/bert-ner-conll2003", aggregation_strategy="first")
ner("Santiago studies at the University of Antioquia in Medellin")
```

Recomiendo `aggregation_strategy="first"`: toma la etiqueta del primer sub-token de cada palabra, que es la misma estrategia usada en el entrenamiento.

## Datos

[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) (Tjong Kim Sang y De Meulder, 2003) contiene noticias de Reuters de 1996-1997. Tiene 14.041 oraciones de entrenamiento, 3.250 de validación y 3.453 de prueba. Se usó la partición oficial.

## Entrenamiento

- **Búsqueda de hiperparámetros:**
  - Muestra aleatoria de 4.000 oraciones.
  - Learning rate: 1e-5, 2e-5, 3e-5, 5e-5 y 1e-4.
  - Batch size: 16 y 32.
  - 3 épocas por configuración.
- **Configuración final:** learning rate 1e-4, batch size 32.
- **Alineación de etiquetas:** cada palabra conserva su etiqueta solo en el primer sub-token. Los demás sub-tokens reciben -100 y no cuentan en la pérdida.
- **Entrenamiento final:**
  - Hasta 5 épocas con early stopping (paciencia 2) según el F1 por entidad en validación.
  - fp16.
  - Duró 5,0 minutos en una GPU T4.

## Resultados (test, métricas por entidad con seqeval)

| Modelo | Precision | Recall | F1 |
|---|---|---|---|
| Etiqueta más frecuente por palabra (línea base) | 0,594 | 0,510 | 0,549 |
| **bert-ner-conll2003** | **0,894** | **0,914** | **0,904** |

| PER | LOC | ORG | MISC |
|---|---|---|---|
| 0,951 | 0,928 | 0,883 | 0,795 |

El recall en entidades que nunca aparecieron en el entrenamiento es 86,6 %, frente a 9,6 % de la línea base. Para las que sí aparecieron es 95,9 %, frente a 88,6 %.

## Limitaciones

- La confusión más común es entre **ORG** y **LOC**. Ocurre sobre todo con ciudades que también son nombres de equipos deportivos, como *CHICAGO* o *NEW YORK*.
- **MISC** es el tipo más difícil: agrupa categorías muy distintas y tiene menos ejemplos.
- Los datos son noticias de los años noventa, así que el rendimiento puede bajar con texto informal o de otros dominios.