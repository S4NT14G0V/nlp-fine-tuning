---
language: en
license: apache-2.0
base_model: google-bert/bert-base-uncased
datasets:
  - dair-ai/emotion
pipeline_tag: text-classification
metrics:
  - accuracy
  - f1
tags:
  - emotion-detection
  - text-classification
  - bert
---

# bert-emotion-tweets

`bert-base-uncased` ajustado (_fine-tuned_) para detectar **6 emociones en tweets en inglés**: sadness, joy, love, anger, fear y surprise.

Es el problema 1 (clasificación de secuencias) del Proyecto 2 del curso _Fundamentos en Procesamiento del Lenguaje Natural_ de la Universidad de Antioquia. Todo el desarrollo está en el repositorio [nlp-fine-tuning-transformers](https://github.com/s4nt14g0v/nlp-fine-tuning).

## Uso

```python
from transformers import pipeline

clasificador = pipeline("text-classification", model="s4nt14g0v/bert-emotion-tweets")
clasificador("i feel so excited about this project")
```

El modelo espera texto en inglés y en minúsculas, igual que los tweets del dataset.

## Datos

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (CARER, Saravia et al., 2018) trae 16.000 tweets de entrenamiento, 2.000 de validación y 2.000 de prueba. Se usó la partición oficial.

## Entrenamiento

- **Búsqueda de hiperparámetros:**
  - Muestra estratificada de 4.000 tweets.
  - Learning rate: 1e-5, 2e-5, 3e-5, 5e-5 y 1e-4.
  - Batch size: 16 y 32.
  - 3 épocas por configuración.
- **Configuración final:** learning rate 5e-5, batch size 16, `max_length` 64.
- **Entrenamiento final:**
  - Hasta 5 épocas con early stopping (paciencia 2) según el F1-macro en validación.
  - fp16.
  - Duró 7,2 minutos en una GPU T4.

## Resultados (test, 2.000 tweets)

| Modelo                                    | Accuracy   | F1-macro   |
| ----------------------------------------- | ---------- | ---------- |
| TF-IDF + Regresión Logística (línea base) | 0,8305     | 0,7255     |
| **bert-emotion-tweets**                   | **0,9325** | **0,8883** |

F1 por clase:

| sadness | joy   | love  | anger | fear  | surprise |
| ------- | ----- | ----- | ----- | ----- | -------- |
| 0,975   | 0,951 | 0,837 | 0,935 | 0,894 | 0,739    |

## Limitaciones

- La mayoría de los errores ocurre entre pares de emociones parecidas: _love_ con _joy_ y _surprise_ con _fear_.
- **Negaciones:** el modelo falla con algunas, por ejemplo _"i don't feel happy at all"_ se clasifica como _joy_. En el dataset la emoción casi siempre la marca una palabra clave, y hay pocos ejemplos con negación.
- **Ruido en las etiquetas:** el dataset se etiquetó por supervisión distante (con hashtags), y algunos tweets aparecen repetidos con emociones distintas.
- **Dominio:** solo funciona con texto en inglés, y su rendimiento puede bajar fuera del estilo "i feel ..." típico del dataset.
