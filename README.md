![pengui-redn banner](pengui-redn-banner.svg)

# pengui-redn

Generador de texto con LSTM hecho en **Rust + Candle**.

Le pasas un archivo de texto, aprende solo, y despues genera texto nuevo que se parece al original.

## Como se usa

```bash
# entrenar con un txt
cargo run -- <file.txt>

# entrenar y generar desde un prompt
cargo run -- <file.txt> "prompt"

# cargar modelo guardado
cargo run -- --load

# cargar modelo guardado y pasarle prompt
cargo run -- --load "prompt"
```

## Que hace cada parte

### Pipeline completo

```
texto → Tokenizer → IDs → Embedding → LSTM → Linear → Logits → Sample → texto generado
```

### Componentes

| Componente | Que hace | Dimensiones |
|---|---|---|
| **Tokenizer** | pasa texto a numeros y viceversa | palabra ↔ u32 |
| **Embedding** | convierte cada ID en un vector de numeros que se aprende | (vocab_size, 256) |
| **LSTM** | procesa secuencias, tiene memoria interna | 256 → 512 (hidden) |
| **Head (Linear)** | transforma la memoria en puntajes para cada palabra | (512, vocab_size) |
| **Sample** | elige una palabra segun los puntajes y un poco de azar | logits → token ID |

### Entrenamiento

1. agarra pedazos random de 32 tokens del texto (`SEQ_LEN`)
2. procesa 32 pedazos a la vez (`BATCH_SIZE`)
3. calcula el error entre lo que predijo y lo que venia realmente (cross-entropy)
4. ajusta los pesos con **AdamW** (learning rate = 0.001)
5. repite todo 100 veces

### Generacion

Va generando de a un token: el token que genera lo vuelve a meter como entrada, y asi sucesivamente. Usa **temperatura 0.8** para que no sea ni muy robot ni muy delirio.

## Estructura del modelo

```rust
struct CharRNN {
    embedding: Embedding,  // vocab_size → 256
    lstm: LSTM,            // 256 → 512 (hidden)
    head: Linear,          // 512 → vocab_size (logits)
}
```

## Archivos que crea

| Archivo | Que tiene |
|---|---|
| `modelo.safetensors` | los pesos del modelo ya entrenado |
| `modelo.vocab` | el vocabulario (una palabra por linea) |

## Requisitos

- Rust edition 2024
- GPU NVIDIA con CUDA (si no tiene usa CPU nomas)
- `candle-core`, `candle-nn`, `rand`

## Conceptos basicos por si no sabes

### Tensor

Es la estructura principal de todo. Es como un array de N dimensiones:
- 0D: un numero (escalar)
- 1D: una lista (vector)
- 2D: una tabla (matriz)
- 3D+: imaginate un cubo de numeros, y de ahi para arriba

Todo en la red son tensores: los datos, los pesos, las predicciones, todo.

### Embedding

Cada palabra del vocabulario se convierte en un vector de 256 numeros. Al entrenar, la red aprende a que palabras parecidas tengan vectores parecidos. Es como un mapa de significados en numeros.

### LSTM

Es un tipo de red recurrente que no se olvida las cosas importantes. Usa compuertas (forget, input, output) para decidir que recordar y que olvidar. Tiene dos estados internos: el oculto (corto plazo) y el de celda (largo plazo).

### Cross-entropy

Mide que tan errada esta la prediccion. Si la red predice "gato" con alta probabilidad pero la respuesta era "perro", el error es grande. Si duda entre varias, el error es menor.

### Temperatura

Controla que tan "creativo" es el texto:
- baja (~0.1): siempre elige la palabra mas probable, texto repetitivo
- alta (~1.5): mas aleatorio, a veces dice cualquier cosa
- 0.8: el punto medio que use aca

## Archivos del codigo

```
src/main.rs
├── Tokenizer - pasa texto a IDs y viceversa
├── CharRNN   - el modelo con embedding, lstm y head
├── get_batch - agarra pedazos random para entrenar
├── sample    - elige el siguiente token segun probabilidad
├── train     - el loop de entrenamiento
├── generate  - genera texto nuevo
└── main      - maneja los argumentos y orquesta todo
```

## Disclaimer

Es un proyecto para aprender, no esperes un GPT. Genera texto que se parece al original pero no tiene sentido profundo. Para eso necesitarias transformers y billones de parametros, esto es mas bien una demo didactica.
