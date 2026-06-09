
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

# chat interactivo con enseñanza automática
cargo run --release -- --chat
```

Nota: los `--` antes de `--load` / `--ask` / `--chat` son para que cargo no interprete las flags.

## Modo chat (`--chat`)

El modo interactivo detecta si conoce el tema de la pregunta:

- **Tema conocido** → responde directamente
- **Tema desconocido** → dice "No sé sobre {tema}", pide una explicación, la aprende con fine-tuning y responde

Ejemplo interactivo:

```
> ¿Qué son los neutrinos?

  No sé sobre neutrinos. Si me explicás, aprendo.
  (Escribí lo que sepas o dejá vacío para saltar)
  > Son partículas subatómicas muy ligeras.
  Aprendiendo...
  Ahora sé: Son partículas subatómicas muy ligeras.
> salir
```

### Enseñanza automática

Cuando el usuario enseña algo nuevo, el modelo:
1. Guarda el par Pregunta/Respuesta en `preguntas_respuestas.txt`
2. Lo agrega al dataset de entrenamiento `train_word.txt`
3. Hace fine-tuning (15 épocas, lr reducido 0.0005)
4. Guarda los pesos actualizados en `modelo.safetensors`
5. Re-responde usando el modelo actualizado

### Detección de ignorancia

Usa superposición de palabras clave (≥40%) contra las preguntas conocidas. Si el solapamiento es menor, responde "No sé sobre {tema}" en vez de alucinar.

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
- GPU NVIDIA con CUDA (si no tiene usa CPU)
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
└── main             - CLI: --load, --ask, --chat
```

## Disclaimer

Proyecto didáctico. El modelo responde bien preguntas que ya vio en entrenamiento (las memoriza), pero no entiende semántica. Para temas nuevos dice "no sé" en vez de alucinar.
