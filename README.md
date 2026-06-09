
# pengui-redn

Generador de texto con LSTM hecho en **Rust y Candle**. Ahora también responde preguntas.

Le pasas un archivo de texto, aprende solo, y genera texto nuevo. Se puede entrenar con formato `Pregunta: ... Respuesta: ...` para que conteste preguntas.

## Cómo se usa

```bash
# entrenar con un txt
cargo run --release libro.txt

# entrenar y generar desde un prompt
cargo run --release libro.txt "había una vez"

# cargar modelo guardado y generar
cargo run --release --load "en un lugar de la mancha"

# responder una pregunta (modo Q&A)
cargo run --release --ask "¿Qué es la literatura?"

# construir dataset Q&A desde un libro de texto
cargo run --release --build-qa libro.txt

# reanudar entrenamiento desde checkpoint
cargo run --release --resume datos.txt
```

## Modo Q&A

El modelo aprende el formato `Pregunta: ... Respuesta: ...` durante el entrenamiento.
Cuando se usa `--ask`, envuelve la pregunta en ese formato y genera la respuesta.
El dataset de entrenamiento incluye preguntas curadas sobre lengua, literatura y el Quijote.

El dataset se construye con `--build-qa` que extrae preguntas del texto fuente,
o se pueden agregar pares manualmente en `contenidosTEST/preguntas_respuestas.txt`.

## Pipeline

```
texto → Tokenizer → IDs → Embedding → LSTM → Linear → Logits → Sample → texto generado
```

## Componentes

| Componente | Qué hace | Dimensiones |
|---|---|---|
| **Tokenizer** | pasa texto a números y viceversa | char ↔ u32 |
| **Embedding** | convierte cada ID en un vector aprendido | (vocab_size, 256) |
| **LSTM** | procesa secuencias con memoria interna | 256 → 512 (hidden) |
| **Head (Linear)** | transforma la memoria en puntajes por caracter | (512, vocab_size) |
| **Sample** | elige el siguiente carácter según probabilidad | logits → token ID |

## Entrenamiento

1. agarra pedazos random de 300 caracteres del texto (`SEQ_LEN`)
2. procesa 64 pedazos a la vez (`BATCH_SIZE`)
3. calcula cross-entropy entre lo que predijo y lo real
4. ajusta los pesos con **AdamW** (learning rate = 0.001)
5. repite 60 épocas, guardando checkpoint cada 5

## Archivos

| Archivo | Qué tiene |
|---|---|
| `modelo.safetensors` | los pesos del modelo entrenado |
| `modelo.vocab` | el vocabulario (un carácter por línea) |
| `contenidosTEST/train_qa_v3.txt` | dataset combinado para Q&A |
| `contenidosTEST/preguntas_respuestas.txt` | pares Pregunta/Respuesta curados |

## Requisitos

- Rust edition 2024
- GPU NVIDIA con CUDA (si no tiene usa CPU nomas)
- `candle-core`, `candle-nn`, `rand`

## Estructura del código

```
src/main.rs
├── Tokenizer       - pasa texto a IDs y viceversa
├── CharRNN         - el modelo (embedding + lstm + head)
├── get_batch       - agarra pedazos random para entrenar
├── sample          - elige el siguiente carácter según probabilidad
├── train           - loop de entrenamiento con checkpoints
├── build_qa_dataset - construye dataset Q&A desde texto fuente
├── generate        - genera texto nuevo desde un prompt
└── main            - maneja argumentos (--load, --ask, --build-qa, --resume)
```

## Disclaimer

Proyecto didáctico con LSTM a nivel de caracteres. No esperes respuestas perfectas.
