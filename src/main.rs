use std::collections::HashMap;
use std::fs;
use std::io::Write;

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{
    loss, AdamW, Embedding, Linear, LSTM, LSTMConfig, Module, Optimizer, RNN, VarBuilder, VarMap,
};
use rand::Rng;

// hiperparametros
const EMBED_SIZE: usize = 128;    // dimension del vector de cada caracter
const HIDDEN_SIZE: usize = 256;   // neuronas del LSTM
const SEQ_LEN: usize = 64;        // cuantos caracteres ve por tanda
const LEARNING_RATE: f64 = 0.003; // tasa de aprendizaje
const EPOCHS: usize = 20;         // vueltas al texto
const GEN_LEN: usize = 500;       // caracteres a generar

// traduce caracteres a ids y viceversa
struct Tokenizer {
    stoi: HashMap<char, u32>, // caracter → id
    itos: Vec<char>,          // id → caracter
}

impl Tokenizer {
    // construye el vocabulario con los caracteres unicos del texto
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let itos = chars;
        let stoi: HashMap<char, u32> = itos
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i as u32))
            .collect();
        Self { stoi, itos }
    }

    fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    // texto → ids
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }

    // ids → texto
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.itos[i as usize]).collect()
    }
}

// red neuronal: embedding → lstm → linear
struct CharRNN {
    embedding: Embedding, // vocabulario → vector de 128
    lstm: LSTM,           // 128 → 256 con memoria interna
    head: Linear,         // 256 → vocabulario (predice el proximo caracter)
}

impl CharRNN {
    // crea las 3 capas con los pesos del VarBuilder
    fn new(vocab_size: usize, vb: &VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, EMBED_SIZE, vb.pp("embed"))?;
        let lstm = LSTM::new(EMBED_SIZE, HIDDEN_SIZE, LSTMConfig::default(), vb.pp("lstm"))?;
        let head = candle_nn::linear(HIDDEN_SIZE, vocab_size, vb.pp("head"))?;
        Ok(Self { embedding, lstm, head })
    }

    // procesa una secuencia completa de ids y devuelve logits para cada posicion
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(x)?;      // (1, 64) → (1, 64, 128)
        let states = self.lstm.seq(&x)?;          // LSTM paso a paso, devuelve estados
        let x = self.lstm.states_to_tensor(&states)?; // estados → (1, 64, 256)
        self.head.forward(&x)                     // (1, 64, 256) → (1, 64, vocabulario)
    }
}

// elige un pedazo random del texto: input = chars[i..i+64], target = chars[i+1..i+65]
fn get_batch(data: &[u32], rng: &mut impl Rng) -> (Vec<u32>, Vec<u32>) {
    let max_start = data.len().saturating_sub(SEQ_LEN + 1);
    if max_start == 0 {
        return (vec![0u32; SEQ_LEN], vec![0u32; SEQ_LEN]);
    }
    let start = rng.gen_range(0..max_start);
    (
        data[start..start + SEQ_LEN].to_vec(),
        data[start + 1..start + SEQ_LEN + 1].to_vec(),
    )
}

// softmax con temperatura + sampleo probabilistico
fn sample(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&l| (l / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    let mut lum = 0.0;
    let r: f32 = rng.r#gen();
    for (i, &p) in scaled.iter().enumerate() {
        lum += p / sum;
        if r < lum {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

// loop de entrenamiento
fn train(device: &Device, tokenizer: &Tokenizer, data: &[u32]) -> Result<VarMap> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = CharRNN::new(tokenizer.vocab_size(), &vb)?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;
    let mut rng = rand::thread_rng();
    let max_steps = 200;
    let steps = (data.len() / SEQ_LEN).min(max_steps).max(1);
    if data.len() < SEQ_LEN * 2 {
        eprintln!("warn: texto muy corto ({})", data.len());
    }

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        for i in 0..steps {
            let (xs, ys) = get_batch(data, &mut rng);
            let xs = Tensor::from_slice(&xs, (1, SEQ_LEN), device)?;
            let ys = Tensor::from_slice(&ys, (1, SEQ_LEN), device)?;

            // forward: predice el siguiente caracter para cada posicion
            let logits = model.forward(&xs)?;
            // cross-entropy: que tan lejos esta la prediccion del real
            let loss = loss::cross_entropy(
                &logits.reshape((SEQ_LEN, tokenizer.vocab_size()))?,
                &ys.reshape((SEQ_LEN,))?,
            )?;
            // backward + adamw: ajusta pesos segun el error
            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;

            if i > 0 && i % 50 == 0 {
                print!(".");
                std::io::stdout().flush().ok();
            }
        }
        println!(
            " epoch {:>3} loss: {:.6}",
            epoch,
            total_loss / steps as f32
        );
    }
    Ok(varmap)
}

// genera texto autoregresivamente desde un prompt
fn generate(
    device: &Device,
    tokenizer: &Tokenizer,
    varmap: &VarMap,
    prompt: &str,
    gen_len: usize,
) -> Result<String> {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
    let model = CharRNN::new(tokenizer.vocab_size(), &vb)?;
    let mut rng = rand::thread_rng();
    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.is_empty() {
        return Ok(String::new());
    }
    let mut generated = prompt_ids.clone();

    // feedea el prompt para inicializar el estado del lstm
    let mut state = model.lstm.zero_state(1)?;
    for &id in &prompt_ids {
        let emb = model
            .embedding
            .forward(&Tensor::from_slice(&[id], (1, 1), device)?)?;
        let emb = emb.squeeze(1)?;
        state = model.lstm.step(&emb, &state)?;
    }

    // genera autoregresivamente: el output de un paso es el input del siguiente
    let mut last_id = *prompt_ids.last().unwrap();
    for _ in 0..gen_len {
        let emb = model
            .embedding
            .forward(&Tensor::from_slice(&[last_id], (1, 1), device)?)?;
        let emb = emb.squeeze(1)?;
        state = model.lstm.step(&emb, &state)?;
        // el hidden state h se proyecta a logits del vocabulario
        let logits = model
            .head
            .forward(&state.h.unsqueeze(0)?)?;
        let logits_vec = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        // samplea el proximo caracter segun la probabilidad
        last_id = sample(&logits_vec, 0.8, &mut rng);
        generated.push(last_id);
    }

    Ok(tokenizer.decode(&generated))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("uso: {} <archivo.txt> [prompt]", args[0]);
        std::process::exit(1);
    }

    // lee el archivo de texto
    let text = fs::read_to_string(&args[1])?;
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("usando: {device:?}");
    println!("caracteres: {}", text.chars().count());

    // prepara el tokenizer y codifica el texto a ids
    let tokenizer = Tokenizer::new(&text);
    println!("vocabulario: {} caracteres", tokenizer.vocab_size());

    let data = tokenizer.encode(&text);
    println!("entrenando...");
    let varmap = train(&device, &tokenizer, &data)?;

    let prompt = if args.len() > 2 {
        args[2].clone()
    } else {
        String::new()
    };

    println!("\n--- generacion ---\n");
    let output = generate(&device, &tokenizer, &varmap, &prompt, GEN_LEN)?;
    println!("{output}");

    Ok(())
}
