use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{
    loss, AdamW, Embedding, Linear, LSTM, LSTMConfig, Module, Optimizer, RNN, VarBuilder, VarMap,
};
use rand::Rng;

// hiperparametros
const EMBED_SIZE: usize = 256;    // dimension del vector de cada caracter
const HIDDEN_SIZE: usize = 512;   // neuronas del LSTM
const SEQ_LEN: usize = 128;       // cuantos caracteres ve por secuencia
const BATCH_SIZE: usize = 32;     // cuantas secuencias en paralelo
const LEARNING_RATE: f64 = 0.001; // tasa de aprendizaje
const EPOCHS: usize = 50;         // vueltas al texto
const GEN_LEN: usize = 500;      // caracteres a generar

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



    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let s: String = self.itos.iter().collect();
        fs::write(path, s)?;
        Ok(())
    }


    fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let s = fs::read_to_string(path)?;
        let itos: Vec<char> = s.chars().collect();
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i as u32))
            .collect();

        Ok(Self {stoi, itos})
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

    // procesa un batch completo de secuencias y devuelve logits para cada posicion
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(x)?;      
        let states = self.lstm.seq(&x)?;        
        let x = self.lstm.states_to_tensor(&states)?; 
        self.head.forward(&x)                 
    }
}

// elige BATCH_SIZE pedazos random del texto
// cada batch tiene input = chars[i..i+SEQ_LEN], target = chars[i+1..i+SEQ_LEN+1]
fn get_batch(data: &[u32], seq_len: usize, batch_size: usize, rng: &mut impl Rng) -> (Vec<u32>, Vec<u32>) {
    let max_start = data.len().saturating_sub(seq_len + 1);
    let mut xs = Vec::with_capacity(batch_size * seq_len);
    let mut ys = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        let start = if max_start > 0 {
            rng.gen_range(0..max_start)
        } else {
            0
        };
        xs.extend_from_slice(&data[start..start + seq_len]);
        ys.extend_from_slice(&data[start + 1..start + seq_len + 1]);
    }
    (xs, ys)
}

// softmax con temperatura ysampleo probabilistico
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
    let seq_len = SEQ_LEN.min(data.len() / 2);
    let batch_size = BATCH_SIZE.min(data.len() / seq_len.max(1));
    if seq_len < 8 {
        eprintln!("error: texto demasiado corto ({})", data.len());
        std::process::exit(1);
    }
    let max_steps = 500;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        for i in 0..max_steps {
            let (xs, ys) = get_batch(data, seq_len, batch_size, &mut rng);
            let xs = Tensor::from_slice(&xs, (batch_size, seq_len), device)?;
            let ys = Tensor::from_slice(&ys, (batch_size, seq_len), device)?;

            let logits = model.forward(&xs)?;// forward: predice el siguiente caracter para cada posicion
            let loss = loss::cross_entropy(  // cross-entropy: que tan lejos esta la prediccion del real
                &logits.reshape((batch_size * seq_len, tokenizer.vocab_size()))?,
                &ys.reshape((batch_size * seq_len,))?,
            )?;
            // backward y adamw ajustab pesos segun el error
            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;

            if i > 0 && i % 100 == 0 {
                print!(".");
                std::io::stdout().flush().ok();
            }
        }
        println!(
            " epoch {:>3} loss: {:.6}",
            epoch,
            total_loss / max_steps as f32
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
    let load = args.iter().any(|a| a == "--load");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("usando: {device:?}");

    let model_path = "modelo.safetensors";
    let vocab_path = "modelo.vocab";
    let saved_exists = Path::new(model_path).exists() && Path::new(vocab_path).exists();

    let (tokenizer, varmap) = if load || saved_exists {
        if !saved_exists {
            eprintln!("error: no hay modelo guardado. entrená uno primero.");
            std::process::exit(1);
        }
        println!("cargando modelo guardado...");
        let tokenizer = Tokenizer::load(vocab_path)?;
        println!("vocabulario: {} caracteres", tokenizer.vocab_size());
        let mut varmap = VarMap::new();
        let _model = CharRNN::new(
            tokenizer.vocab_size(),
            &VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        varmap.load(model_path)?;
        (tokenizer, varmap)
    } else {
        if args.len() < 2 || args[1].starts_with('-') {
            eprintln!("uso: {} <archivo.txt> [prompt]", args[0]);
            eprintln!("      {} --load [prompt]", args[0]);
            std::process::exit(1);
        }
        let text = fs::read_to_string(&args[1])?;
        println!("caracteres: {}", text.chars().count());
        let tokenizer = Tokenizer::new(&text);
        println!("vocabulario: {} caracteres", tokenizer.vocab_size());
        let data = tokenizer.encode(&text);
        println!("entrenando...");
        let start = std::time::Instant::now();
        let varmap = train(&device, &tokenizer, &data)?;
        let elapsed = start.elapsed();
        println!("entrenamiento completo en {:.2?}", elapsed); //guarda todo para poder cargarlo despues sin entrenar de nuevo
        varmap.save(model_path)?;
        tokenizer.save(vocab_path)?;
        println!("modelo guardado en {model_path}");
        (tokenizer, varmap)
    };

    let prompt = if load {  //si se cargo un modelo, el prompt va despues de --load
        let pos = args.iter().position(|a| a == "--load").unwrap();
        args.get(pos + 1).cloned().unwrap_or_default()
    } else {
        args.get(2).cloned().unwrap_or_default()
    };

    println!("\n--- generacion ---\n"); // genera texto desde el prompt usando el modelo entrenado o cargado
    let output = generate(&device, &tokenizer, &varmap, &prompt, GEN_LEN)?;
    println!("{output}");

    Ok(())
}
