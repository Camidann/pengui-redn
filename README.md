
# pengui-redn

Generador de texto con LSTM hecho en **Rust + Candle**. Tokenización **word-level** con fallback a caracteres, modo **Q&A**, detección de ignorancia y aprendizaje interactivo.

## Cómo se usa

```bash
# entrenar con un archivo de texto
cargo run --release -- libro.txt

# generar texto desde un prompt
cargo run --release -- --load "en un lugar de la mancha"

# responder una pregunta
cargo run --release -- --ask "¿Qué es la literatura?"

# interfaz TUI interactiva
cargo run --release -- --tui
```

Nota: los `--` antes de `--load` / `--ask` / son para que cargo no interprete las flags.

### Detección de ignorancia

Usa superposición de palabras clave (≥40%) contra las preguntas conocidas. Si el solapamiento es menor, responde "No sé sobre {tema}" en vez de alucinar.

## Modo TUI (`--tui`)

El modo TUI abre una interfaz de terminal interactiva con:
- historial de preguntas y respuestas
- campo de entrada en tiempo real
- ayuda en pantalla
- salida con `Esc`, `q` o `Ctrl+C`

Requiere un modelo ya entrenado guardado en `modelo.safetensors` y `modelo.vocab`.

## Modo consulta (`--ask`)

Responde una pregunta puntual. Si el tema es desconocido, avisa sin alucinar:

```
$ cargo run --release -- --ask "¿Qué es la literatura?"
→ La literatura es el arte de la expresión escrita u oral. Comprende obras como novelas,
  cuentos, poesías y obras teatrales.

$ cargo run --release -- --ask "¿Qué es la química cuántica?"
→ No sé sobre química. Pasá --enseñame para enseñarme.
```

## Tokenizer híbrido

El vocabulario se divide en dos zonas:
- **Índices 2..char_boundary**: caracteres individuales (para palabras desconocidas)
- **Índices char_boundary..vocab_size**: palabras completas más frecuentes (hasta 8192)

Al codificar, las palabras conocidas van como un solo token. Las desconocidas se descomponen en sus caracteres. Al decodificar, caracteres consecutivos se reagrupan en una sola palabra.

## Pipeline

```
texto → Tokenizer (word + char) → IDs → Embedding(512) → LSTM(1024) → Linear → Logits → Sample → texto
```

## Componentes

| Componente | Qué hace | Dimensiones |
|---|---|---|
| **Tokenizer** | pasa texto a IDs: palabras completas o caracteres | palabra/carácter ↔ u32 |
| **Embedding** | convierte cada ID en un vector aprendido | (vocab_size, 512) |
| **LSTM** | procesa secuencias con memoria interna | 512 → 1024 (hidden) |
| **Head (Linear)** | transforma la memoria en puntajes por palabra | (1024, vocab_size) |
| **Sample** | elige la siguiente palabra (top-40 + temperatura) | logits → token ID |
| **knows_topic** | detecta si el tema es conocido por keywords | solapamiento ≥40% |

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
2. construye vocabulario híbrido: caracteres únicos + 8192 palabras más frecuentes
3. agarra bloques de 64 tokens (`SEQ_LEN`)
4. procesa 32 bloques por lote (`BATCH_SIZE`)
5. calcula cross-entropy entre predicción y token real
6. optimiza con **AdamW** (lr=0.001)
7. repite 80 épocas, guardando checkpoint cada 5

## Aumentar dataset y scripts

Para mejorar la cobertura de preguntas conviene generar un dataset aumentado que combine textos grandes y repita los pares Q&A. Hay dos utilidades incluidas:

- Binario en Rust: `src/bin/augment_train.rs` (misma funcionalidad)
  Uso:
  ```bash
  cargo run --bin augment_train -- --repeats 30
  ```

Ambos generan el archivo `contenidosTEST/train_augmented.txt`. Luego entrená con ese archivo:

```bash
cargo run --release -- contenidosTEST/train_augmented.txt
```

Recomendaciones:
- Repetir cada par Q&A entre 20–50 veces para favorecer la memorización del modelo.
- Añadí más fuentes (FAQs, artículos, Wikipedia en español) para ampliar cobertura.
- Generá parafraseos de las preguntas cuando sea posible (sinónimos, orden distinto).
- Si tenés GPU, entrená en `--release` y confirmá que `Device::cuda_if_available` detecte CUDA.
- Si el dataset aumenta mucho, incrementá `EPOCHS` en `src/main.rs` o entrená más épocas.


## Archivos

| Archivo | Qué tiene |
|---|---|
| `modelo.safetensors` | pesos del modelo entrenado (word-level) |
| `modelo.vocab` | vocabulario híbrido (caracteres + palabras, uno por línea) |
| `contenidosTEST/preguntas_respuestas.txt` | pares Pregunta/Respuesta (se actualiza al enseñar) |
| `contenidosTEST/train_word.txt` | dataset combinado para entrenar |
| `contenidosTEST/EL004338.txt` | libro de texto fuente |
| `contenidosTEST/el_quijote.txt` | Quijote fuente |

## Requisitos

- Rust edition 2024
- GPU NVIDIA con CUDA (si no tiene elimina el ["cuda"] de cargo.toml CPU)
- `candle-core`, `candle-nn`, `rand`

## Estructura del código

```
src/main.rs
├── Tokenizer        - tokenización híbrida (word + char)
├── WordRNN          - modelo (embedding 512 + LSTM 1024 + linear head)
├── get_batch        - muestrea bloques aleatorios para entrenar
├── sample           - sampleo con temperatura + top-40
├── train            - loop de entrenamiento con checkpoints
├── generate         - genera texto autoregresivamente
├── finetune_on_text - fine-tuning rápido sobre texto nuevo
├── knows_topic      - detección de temas conocidos por keywords
├── chat_loop        - modo interactivo con enseñanza
├── extract_keywords - extrae palabras clave (filtra stopwords)
└── main             - CLI: --load, --ask, --chat, --tui
```

## Disclaimer

Proyecto didáctico. El modelo responde bien preguntas que ya vio en entrenamiento (las memoriza), pero no entiende semántica. Para temas nuevos dice "no sé" en vez de alucinar.
