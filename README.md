
# pengui-redn

Generador de texto con LSTM hecho en **Rust + Candle**. Ahora con tokenización **word-level** y modo **Q&A**.

El modelo aprende de textos en español y puede responder preguntas entrenado con pares `Pregunta: ... Respuesta: ...`.

## Cómo se usa

```bash
# entrenar con un archivo de texto
cargo run --release -- libro.txt

# cargar modelo guardado y generar texto
cargo run --release -- --load "en un lugar de la mancha"

# responder una pregunta (modo Q&A)
cargo run --release -- --ask "¿Qué es la literatura?"
```

Nota: los `--` antes de `--load` / `--ask` son para que cargo no interprete las flags.

## Modo Q&A

El modelo se entrena con pares en formato `Pregunta: ... Respuesta: ...`. Durante inferencia con `--ask`, envuelve la pregunta en ese formato y genera la respuesta palabra por palabra.

### Ejemplos de respuesta

```
$ cargo run --release -- --ask "¿Qué es la literatura?"
→ La literatura es el arte de la expresión escrita u oral. Comprende obras como novelas,
  cuentos, poesías y obras teatrales.

$ cargo run --release -- --ask "¿Quién es Don Quijote?"
→ Don Quijote es el protagonista de la novela de Cervantes. Es un hidalgo de unos
  cincuenta años que se vuelve loco por leer libros de caballerías.

$ cargo run --release -- --ask "¿Qué es un dialecto?"
→ El dialecto es la variedad del lenguaje determinada por la ubicación geográfica del
  hablante. Existen diferencias entre países, y también entre el dialecto rural y el urbano.
```

## Pipeline

```
texto → Tokenizer (word) → IDs → Embedding → LSTM → Linear → Logits → Sample → texto generado
```

## Componentes

| Componente | Qué hace | Dimensiones |
|---|---|---|
| **Tokenizer** | pasa texto a IDs palabra por palabra | palabra ↔ u32 |
| **Embedding** | convierte cada ID en un vector aprendido | (vocab_size, 512) |
| **LSTM** | procesa secuencias con memoria interna | 512 → 1024 (hidden) |
| **Head (Linear)** | transforma la memoria en puntajes por palabra | (1024, vocab_size) |
| **Sample** | elige la siguiente palabra según probabilidad (top-40 + temperatura) | logits → token ID |

## Datos de entrenamiento

El dataset principal (`contenidosTEST/train_word.txt`) combina:
- **15 repeticiones** de 50 pares Pregunta/Respuesta curados (130 KB)
- **Libro de texto** "Lengua y Literatura" (246 KB)
- **Don Quijote de la Mancha** (primeros ~100 KB)

Los pares Q&A están en `contenidosTEST/preguntas_respuestas.txt` y cubren:
- Conceptos de lengua y literatura (dialecto, sociolecto, narrador, etc.)
- Personajes y datos del Quijote (Don Quijote, Sancho, Rocinante, Cervantes)
- Géneros literarios y figuras retóricas

## Entrenamiento

1. tokeniza el texto en palabras (split por espacios)
2. construye vocabulario con las 8192 palabras más frecuentes
3. agarra bloques de 64 palabras (`SEQ_LEN`)
4. procesa 32 bloques por lote (`BATCH_SIZE`)
5. calcula cross-entropy entre predicción y palabra real
6. optimiza con **AdamW** (lr=0.001)
7. repite 80 épocas, guardando checkpoint cada 5

## Archivos

| Archivo | Qué tiene |
|---|---|
| `modelo.safetensors` | pesos del modelo entrenado (word-level) |
| `modelo.vocab` | vocabulario (una palabra por línea) |
| `contenidosTEST/preguntas_respuestas.txt` | 50 pares Pregunta/Respuesta curados |
| `contenidosTEST/train_word.txt` | dataset combinado para entrenar |
| `contenidosTEST/EL004338.txt` | libro de texto fuente |
| `contenidosTEST/el_quijote.txt` | Quijote fuente |

## Requisitos

- Rust edition 2024
- GPU NVIDIA con CUDA (si no tiene usa CPU)
- `candle-core`, `candle-nn`, `rand`

## Estructura del código

```
src/main.rs
├── Tokenizer    - tokenización word-level con vocabulario limitado
├── WordRNN      - modelo (embedding 512 + LSTM 1024 + linear head)
├── get_batch    - muestrea bloques aleatorios para entrenar
├── sample       - sampleo con temperatura + top-40
├── train        - loop de entrenamiento con checkpoints
├── generate     - genera texto autoregresivamente
└── main         - CLI: --load, --ask
```

## Disclaimer

Proyecto didáctico. El modelo responde bien preguntas que ya vio en entrenamiento (porque las memoriza), pero no entiende semántica. Para preguntas nuevas puede alucinar o dar respuestas incorrectas.
