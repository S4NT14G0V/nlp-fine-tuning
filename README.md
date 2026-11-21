# Fine-tuning de Transformers para tres tareas de NLP

**Clasificación de emociones · Reconocimiento de entidades nombradas · Traducción inglés-español**

Proyecto 2 del curso *Fundamentos en Procesamiento del Lenguaje Natural*, Universidad de Antioquia.

Autor: **Santiago Trespalacios Bolívar** · [GitHub](https://github.com/s4nt14g0v) · [LinkedIn](https://www.linkedin.com/in/santiago-trespalacios-bolivar/)

---

## Descripción

En este proyecto apliqué *fine-tuning* a modelos preentrenados de Hugging Face para resolver tres tipos de problemas de NLP:

| # | Tipo de problema | Tarea | Dataset | Modelo base |
|---|---|---|---|---|
| 1 | Clasificación de secuencias | Detectar 6 emociones en tweets en inglés | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) |
| 2 | Etiquetado de secuencias | Reconocimiento de entidades nombradas (PER, ORG, LOC, MISC) en noticias | [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003) | [`bert-base-cased`](https://huggingface.co/google-bert/bert-base-cased) |
| 3 | Secuencia a secuencia | Traducción inglés → español de literatura clásica | [`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (en-es) | [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) (MarianMT) |

Para el problema 3 usé MarianMT, un Transformer encoder-decoder con la misma arquitectura base que BART. Lo elegí porque ya viene preentrenado para el par inglés-español. Así el fine-tuning se convierte en una **adaptación de dominio** (de un traductor general a uno literario) que cabe en una GPU T4 gratuita.

Todo el desarrollo está en el notebook [`NLP_Proyecto_2.ipynb`](NLP_Proyecto_2.ipynb).

## Metodología

Seguí el mismo flujo en los tres problemas:

1. **Descripción del dataset:** dominio, utilidad, entrada, salida, tipo de problema y tamaño.
2. **Análisis exploratorio:**
   - Ejemplos reales de cada dataset.
   - Distribución de clases o entidades y longitud de los textos.
   - Calidad de los datos: duplicados, fugas entre conjuntos y pares mal alineados.
3. **Línea base** para medir cuánto aporta el fine-tuning:
   - P1: TF-IDF con regresión logística.
   - P2: la etiqueta más frecuente de cada palabra.
   - P3: el mismo modelo sin fine-tuning.
4. **Tokenización:**
   - El `max_length` se elige con el percentil 99 de la longitud en sub-tokens.
   - Uso padding dinámico.
   - En NER, la etiqueta de cada palabra se alinea con su primer sub-token.
5. **Búsqueda de hiperparámetros** sobre una muestra de 4.000 ejemplos:
   - Learning rate: `1e-5`, `2e-5`, `3e-5`, `5e-5` y `1e-4`.
   - Batch size: `16` y `32`.
   - 3 épocas por configuración.
6. **Entrenamiento final** con todos los datos:
   - Uso la mejor configuración de la búsqueda.
   - Early stopping y restauración del mejor checkpoint.
   - Precisión mixta (fp16).
7. **Evaluación en test**, que no se usó para ninguna decisión:
   - Métricas propias de cada tarea.
   - Matriz de confusión y análisis de errores.
   - Demostración con frases nuevas.

## Resultados (conjunto de test)

| Problema | Métrica | Línea base | Fine-tuned | Mejor configuración | Tiempo de entrenamiento |
|---|---|---|---|---|---|
| 1. Emociones en tweets | F1-macro | 0,726 | **0,888** | lr = 5e-5, batch = 16 | 7,2 min |
| 2. Entidades nombradas | F1 por entidad (seqeval) | 0,549 | **0,904** | lr = 1e-4, batch = 32 | 5,0 min |
| 3. Traducción en → es | BLEU (sacrebleu) | 16,93 | **23,40** | lr = 5e-5, batch = 16 | 32,5 min |

Métricas complementarias:

- **P1:** accuracy 0,9325 (línea base 0,8305).
- **P2:** precision 0,894 y recall 0,914. En entidades que nunca aparecieron en train, el recall es 86,6 %, frente a 9,6 % de la línea base.
- **P3:** chrF 48,46 (sin fine-tuning 44,56).

Los tres entrenamientos finales tomaron **44,7 minutos** en total en una GPU T4 de Google Colab.

## Modelos publicados

Los tres modelos finales están publicados en Hugging Face y se pueden usar directamente:

| Problema | Modelo |
|---|---|
| 1. Emociones en tweets | [`s4nt14g0v/bert-emotion-tweets`](https://huggingface.co/s4nt14g0v/bert-emotion-tweets) |
| 2. Entidades nombradas | [`s4nt14g0v/bert-ner-conll2003`](https://huggingface.co/s4nt14g0v/bert-ner-conll2003) |
| 3. Traducción en → es | [`s4nt14g0v/opus-mt-en-es-literatura`](https://huggingface.co/s4nt14g0v/opus-mt-en-es-literatura) |

```python
from transformers import pipeline

emociones = pipeline("text-classification", model="s4nt14g0v/bert-emotion-tweets")
print(emociones("i feel so excited about this project"))

ner = pipeline("token-classification", model="s4nt14g0v/bert-ner-conll2003", aggregation_strategy="first")
print(ner("Santiago studies at the University of Antioquia in Medellin"))
```

El traductor se usa con `AutoTokenizer` y `AutoModelForSeq2SeqLM`, porque las versiones recientes de `transformers` ya no incluyen el `pipeline` de traducción. El ejemplo está en la [ficha del modelo](https://huggingface.co/s4nt14g0v/opus-mt-en-es-literatura).

Dentro de [model_cards](https://github.com/S4NT14G0V/nlp-fine-tuning/tree/main/model_cards) se encuentra además un resume de cada uno de los problemas abordados.

## Estructura del repositorio

```
├── NLP_Proyecto_2.ipynb          # notebook con todo el desarrollo
├── README.md
└── resultados/
    ├── resultados_finales.json   # métricas de test de los tres problemas
    ├── p1_emotion/               # búsqueda de hiperparámetros del problema 1
    ├── p2_ner/                   # búsqueda de hiperparámetros del problema 2
    └── p3_traduccion/            # búsqueda de hiperparámetros del problema 3
└── model_cards/                  # resumen de los tres problemas
```

## Hallazgos principales

- **El learning rate fue el hiperparámetro más importante.** Con `1e-5` los modelos aprendieron demasiado lento en 3 épocas. Entre `3e-5` y `1e-4` los resultados fueron parecidos. El batch size influyó poco.
- **La mejora fue mayor cuando el modelo aprendía una tarea nueva.** En P1 y P2 la cabeza de clasificación empieza desde cero, y los errores bajaron cerca de 60 % y 80 % frente a la línea base. En P3 el modelo ya traducía, así que el fine-tuning le enseñó el estilo literario: la raya de diálogo, el registro y el vocabulario de época.
- **Cada tarea necesita su propia métrica:**
  - F1-macro en P1, porque las clases están desbalanceadas (*surprise* es solo el 3,6 % de los ejemplos).
  - F1 por entidad completa en P2, porque el 83 % de las palabras tiene la etiqueta `O`.
  - BLEU junto con chrF en P3.
- **El contexto importa.** BERT resuelve ambigüedades que un diccionario no puede resolver, como *Jordan* (persona o país) o *CHICAGO* (ciudad o equipo deportivo).

## Limitaciones

- **P1:** el modelo falla con algunas negaciones. Por ejemplo, *"i don't feel happy at all"* se clasifica como *joy*. En el dataset la emoción casi siempre la marca una palabra clave, y hay pocas negaciones.
- **P1 y P2:** las etiquetas tienen ruido. Hay tweets repetidos con emociones distintas y algunas anotaciones discutibles en CoNLL-2003, lo que pone un techo al desempeño.
- **P3, especialización:** el modelo ajustado traduce peor algunos términos modernos y técnicos. Por ejemplo, *security updates* pasa a "noticias de seguridad".
- **P3, posible contaminación:** OPUS Books forma parte de los datos con los que se preentrenó MarianMT.

## Cómo ejecutar el notebook

1. Abrir `NLP_Proyecto_2.ipynb` en Google Colab.
2. Seleccionar el entorno de ejecución con **GPU T4** (*Entorno de ejecución → Cambiar tipo de entorno de ejecución*).
3. Configurar las dos variables de la sección 1.2:
   - `USE_DRIVE = True`: guarda los resultados y los modelos en Google Drive, y permite reanudar la búsqueda de hiperparámetros si la sesión se corta.
   - `MODO_PRUEBA = True`: ejecuta todo con muestras pequeñas en pocos minutos, para verificar que el notebook funciona de principio a fin. Para reproducir los resultados reales debe estar en `False`.
4. Ejecutar todas las celdas. La sección 5 lee los modelos y resultados guardados, sin volver a entrenar.

## Tecnologías

Python · PyTorch · Hugging Face `transformers`, `datasets` y `accelerate` · scikit-learn · seqeval · sacrebleu · pandas · matplotlib · seaborn · Google Colab

## Referencias

- Curso de Hugging Face, capítulo de fine-tuning: https://huggingface.co/learn/llm-course/chapter3
- Saravia et al. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition*. EMNLP.
- Tjong Kim Sang y De Meulder (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition*.
- Tiedemann (2012). *Parallel Data, Tools and Interfaces in OPUS*. LREC.
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL.
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT.
