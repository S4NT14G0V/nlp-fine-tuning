---
language:
  - en
  - es
license: apache-2.0
base_model: Helsinki-NLP/opus-mt-en-es
datasets:
  - Helsinki-NLP/opus_books
pipeline_tag: translation
metrics:
  - bleu
  - chrf
tags:
  - translation
  - marian
  - seq2seq
  - literature
---

# opus-mt-en-es-literatura

`Helsinki-NLP/opus-mt-en-es` (MarianMT, un encoder-decoder) ajustado (_fine-tuned_) para **traducir literatura clásica del inglés al español**. El modelo original ya traducía; el fine-tuning lo adapta al estilo de las traducciones publicadas de novelas: la raya de diálogo, el registro y el vocabulario de época.

Es el problema 3 (secuencia a secuencia) del Proyecto 2 del curso _Fundamentos en Procesamiento del Lenguaje Natural_ de la Universidad de Antioquia. Todo el desarrollo está en el repositorio [nlp-fine-tuning-transformers](https://github.com/s4nt14g0v/nlp-fine-tuning-transformers).

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

repo = "s4nt14g0v/opus-mt-en-es-literatura"
tokenizer = AutoTokenizer.from_pretrained(repo)
modelo = AutoModelForSeq2SeqLM.from_pretrained(repo)

entrada = tokenizer(["The old captain looked at the sea for a long time without saying a word."], return_tensors="pt", padding=True)
salida = modelo.generate(**entrada, num_beams=4, max_new_tokens=128)
print(tokenizer.batch_decode(salida, skip_special_tokens=True))
```

## Datos

[OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books), par en-es (Tiedemann, 2012). Son oraciones de novelas clásicas alineadas automáticamente con sus traducciones publicadas.

Antes de entrenar se limpió el corpus:

- Se eliminaron las filas de metadatos.
- Se eliminaron los pares con una relación de longitud extrema y los de más de 128 sub-tokens.
- Se eliminaron las oraciones en inglés repetidas.

Quedó el 96 % de los pares: 85.689 para entrenamiento, 2.000 para validación y 2.000 para prueba.

## Entrenamiento

- **Búsqueda de hiperparámetros:**
  - Muestra de 4.000 pares, evaluada sobre 500 pares de validación.
  - Learning rate: 1e-5, 2e-5, 3e-5, 5e-5 y 1e-4.
  - Batch size: 16 y 32.
- **Configuración final:** learning rate 5e-5, batch size 16, `max_length` 128.
- **Entrenamiento final:**
  - 3 épocas con early stopping según el BLEU en validación.
  - fp16.
  - Duró 32,5 minutos en una GPU T4.

## Resultados (test, 2.000 pares, beam search con 4 haces)

| Modelo                                       | BLEU      | chrF      |
| -------------------------------------------- | --------- | --------- |
| `Helsinki-NLP/opus-mt-en-es` sin fine-tuning | 16,93     | 44,56     |
| **opus-mt-en-es-literatura**                 | **23,40** | **48,46** |

El fine-tuning mejoró el BLEU en todos los libros, entre +3,1 y +9,2 puntos. Por oración, el chrF mejoró en el 66,6 % de los casos y empeoró en el 29,3 %.

## Limitaciones

- **Especialización:** fuera de la literatura, el modelo puede traducir peor que el original. Por ejemplo, _security updates_ pasa a "noticias de seguridad" y _natural language processing_ a "tratamiento de las lenguas naturales". Para un uso general es mejor el modelo base.
- **Posible contaminación:** OPUS Books forma parte de la colección con la que se preentrenó el modelo base, así que es posible que ya hubiera visto algunas oraciones del conjunto de test.
- **Ruido en el corpus:** al estar alineado automáticamente, contiene algunos pares que no son traducción uno del otro.
- **Métrica:** el BLEU compara contra una sola traducción publicada, así que penaliza traducciones correctas que usan otras palabras.
