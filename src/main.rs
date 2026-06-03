use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{
    loss, AdamW, Embedding, Linear, LSTM, LSTMConfig, Module, Optimizer, RNN, VarBuilder, VarMap,
};
use rand::Rng;

const EMBED_SIZE: usize = 256;
const HIDDEN_SIZE: usize = 512;
const SEQ_LEN: usize = 32;
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f64 = 0.001;
const EPOCHS: usize = 150;
const GEN_LEN: usize = 50;
const UNK_TOKEN: &str = "<UNK>";

struct Tokenizer {
    stoi: HashMap<String, u32>,
    itos: Vec<String>,
}

fn split_off_punct(word: &str) -> Vec<String> { //separo palabras de signos de puntuación así no se pegan
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut kind = None;
    for c in word.chars() {
        let k = c.is_alphabetic();
        match kind {
            Some(k2) if k2 != k => {
                pieces.push(current);
                current = String::new();
            }
            _ => {}
        }
        current.push(c);
        kind = Some(k);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

fn tokenize(text: &str) -> Vec<String> { //convierto el texto a tokens: minúsculas, separado por espacios y puntuación
    text.split_whitespace()
        .flat_map(|w| split_off_punct(w))
        .map(|t| t.to_lowercase())
        .collect()
}

impl Tokenizer { //construyo el vocabulario: mapeo palabra -> id y id -> palabra
    fn new(text: &str) -> Self {
        let mut words: Vec<String> = tokenize(text);
        words.sort();
        words.dedup();
        let mut itos = vec![UNK_TOKEN.to_string()];
        itos.extend(words);
        let stoi: HashMap<String, u32> = itos
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        Self { stoi, itos }
    }

    fn vocab_size(&self) -> usize { //cuántas palabras distintas conoce el modelo
        self.itos.len()
    }

    fn encode(&self, text: &str) -> Vec<u32> { //texto -> lista de ids
        tokenize(text)
            .iter()
            .map(|w| *self.stoi.get(w).unwrap_or(&0))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String { //ids -> texto de vuelta
        ids.iter()
            .map(|&i| self.itos[i as usize].as_str())
            .collect::<Vec<&str>>()
            .join(" ")
    }

    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let s = self.itos.join("\n");
        fs::write(path, s)?;
        Ok(())
    }

    fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = fs::read_to_string(path)?;
        let itos: Vec<String> = s.lines().map(|l| l.to_string()).collect();
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        Ok(Self { stoi, itos })
    }
}

struct CharRNN {
    embedding: Embedding,
    lstm: LSTM,
    head: Linear,
}

impl CharRNN {
    fn new(vocab_size: usize, vb: &VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, EMBED_SIZE, vb.pp("embed"))?;
        let lstm = LSTM::new(EMBED_SIZE, HIDDEN_SIZE, LSTMConfig::default(), vb.pp("lstm"))?;
        let head = candle_nn::linear(HIDDEN_SIZE, vocab_size, vb.pp("head"))?;
        Ok(Self { embedding, lstm, head })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(x)?;
        let states = self.lstm.seq(&x)?;
        let x = self.lstm.states_to_tensor(&states)?;
        self.head.forward(&x)
    }
}

fn get_batch(data: &[u32], seq_len: usize, batch_size: usize, rng: &mut impl Rng) -> (Vec<u32>, Vec<u32>) {
    let max_start = data.len().saturating_sub(seq_len + 1);
    let mut xs = Vec::with_capacity(batch_size * seq_len);
    let mut ys = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        let start = if max_start > 0 { rng.gen_range(0..max_start) } else { 0 };
        xs.extend_from_slice(&data[start..start + seq_len]);
        ys.extend_from_slice(&data[start + 1..start + seq_len + 1]);
    }
    (xs, ys)
}

fn sample(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .map(|&l| (l / temperature).exp())
        .enumerate()
        .collect();
    let sum: f32 = scaled.iter().map(|(_, p)| p).sum();
    for (_, p) in scaled.iter_mut() {
        *p /= sum;
    }
    scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k = 10.min(scaled.len());
    let mut top = scaled[..top_k].to_vec();
    let norm: f32 = top.iter().map(|(_, p)| p).sum();
    for (_, p) in top.iter_mut() {
        *p /= norm;
    }
    let mut cum = 0.0;
    let r: f32 = rng.r#gen();
    for (i, p) in top.iter() {
        cum += p;
        if r < cum {
            return *i as u32;
        }
    }
    top.last().unwrap().0 as u32
}

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

    let steps_per_epoch = (data.len() / (seq_len * batch_size)).max(20);

    for epoch in 0..EPOCHS {
        let mut total_loss = Tensor::new(0.0f32, device)?;
        for i in 0..steps_per_epoch {
            let (xs, ys) = get_batch(data, seq_len, batch_size, &mut rng);
            let xs = Tensor::from_slice(&xs, (batch_size, seq_len), device)?;
            let ys = Tensor::from_slice(&ys, (batch_size, seq_len), device)?;
            let logits = model.forward(&xs)?;
            let loss = loss::cross_entropy(
                &logits.reshape((batch_size * seq_len, tokenizer.vocab_size()))?,
                &ys.reshape((batch_size * seq_len,))?,
            )?;
            opt.backward_step(&loss)?;
            total_loss = total_loss.add(&loss.detach())?;

            if i > 0 && i % 100 == 0 {
                print!(".");
                std::io::stdout().flush().ok();
            }
        }
        println!(
            " epoch {:>3} loss: {:.6}",
            epoch,
            total_loss.to_scalar::<f32>()? / steps_per_epoch as f32
        );
    }
    Ok(varmap)
}

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

    let mut state = model.lstm.zero_state(1)?;
    for &id in &prompt_ids {
        let emb = model
            .embedding
            .forward(&Tensor::from_slice(&[id], (1, 1), device)?)?;
        let emb = emb.squeeze(1)?;
        state = model.lstm.step(&emb, &state)?;
    }

    let mut last_id = *prompt_ids.last().unwrap();
    for _ in 0..gen_len {
        let emb = model
            .embedding
            .forward(&Tensor::from_slice(&[last_id], (1, 1), device)?)?;
        let emb = emb.squeeze(1)?;
        state = model.lstm.step(&emb, &state)?;
        let logits = model
            .head
            .forward(&state.h.unsqueeze(0)?)?;
        let logits_vec = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        last_id = sample(&logits_vec, 0.6, &mut rng);
        generated.push(last_id);
    }

    Ok(tokenizer.decode(&generated))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let load = args.iter().any(|a| a == "--load");
    let has_file = args.len() >= 2 && !args[1].starts_with('-');

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("usando: {device:?}");

    let model_path = "modelo.safetensors";
    let vocab_path = "modelo.vocab";
    let saved_exists = Path::new(model_path).exists() && Path::new(vocab_path).exists();

    let (tokenizer, varmap) = if load {
        if !saved_exists {
            eprintln!("error: no hay modelo guardado. entrená uno primero.");
            std::process::exit(1);
        }
        println!("cargando modelo guardado...");
        let tokenizer = Tokenizer::load(vocab_path)?;
        println!("vocabulario: {} palabras", tokenizer.vocab_size());
        let mut varmap = VarMap::new();
        let _model = CharRNN::new(
            tokenizer.vocab_size(),
            &VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        varmap.load(model_path)?;
        (tokenizer, varmap)
    } else if has_file {
        let text = fs::read_to_string(&args[1])?;
        println!("caracteres: {}", text.chars().count());
        let tokenizer = Tokenizer::new(&text);
        println!("vocabulario: {} palabras", tokenizer.vocab_size());
        let data = tokenizer.encode(&text);
        println!("tokens totales: {}", data.len());
        println!("entrenando...");
        let start = std::time::Instant::now();
        let varmap = train(&device, &tokenizer, &data)?;
        let elapsed = start.elapsed();
        println!("entrenamiento completo en {:.2?}", elapsed);
        varmap.save(model_path)?;
        tokenizer.save(vocab_path)?;
        println!("modelo guardado en {model_path}");
        (tokenizer, varmap)
    } else if saved_exists {
        println!("cargando modelo guardado...");
        let tokenizer = Tokenizer::load(vocab_path)?;
        println!("vocabulario: {} palabras", tokenizer.vocab_size());
        let mut varmap = VarMap::new();
        let _model = CharRNN::new(
            tokenizer.vocab_size(),
            &VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        varmap.load(model_path)?;
        (tokenizer, varmap)
    } else {
        eprintln!("uso: {} <archivo.txt> [prompt]", args[0]);
        eprintln!("      {} --load [prompt]", args[0]);
        std::process::exit(1);
    };

    let prompt = if load {
        let pos = args.iter().position(|a| a == "--load").unwrap();
        args.get(pos + 1).cloned().unwrap_or_default()
    } else {
        args.get(2).cloned().unwrap_or_default()
    };

    println!("\n--- generacion ---\n");
    let output = generate(&device, &tokenizer, &varmap, &prompt, GEN_LEN)?;
    println!("{output}");

    Ok(())
}
