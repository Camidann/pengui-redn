use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{
    loss, AdamW, Embedding, Linear, LSTM, LSTMConfig, Module, Optimizer, RNN, VarBuilder, VarMap,
};
use rand::Rng;

pub const EMBED_SIZE: usize = 512;
pub const HIDDEN_SIZE: usize = 1024;
const SEQ_LEN: usize = 64;
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f64 = 0.001;
const EPOCHS: usize = 80;
pub const GEN_LEN: usize = 40;
pub const QA_LEN: usize = 30;
pub const VOCAB_SIZE: usize = 8192;

pub struct Tokenizer {
    stoi: HashMap<String, u32>,
    itos: Vec<String>,
    unk_id: u32,
    char_boundary: u32,
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

impl Tokenizer {
    pub fn new(text: &str, vocab_size: usize) -> Self {
        let words = tokenize(text);
        let mut char_set: Vec<String> = Vec::new();
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        for w in &words {
            *word_freq.entry(w.clone()).or_insert(0) += 1;
            for c in w.chars() {
                let c_str = c.to_string();
                if !char_set.contains(&c_str) {
                    char_set.push(c_str.clone());
                }
            }
        }
        char_set.sort();

        let mut sorted_words: Vec<(String, usize)> = word_freq.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        let mut itos: Vec<String> = vec!["<PAD>".to_string(), "<UNK>".to_string()];
        let char_boundary = (2 + char_set.len()) as u32;
        for c in &char_set {
            itos.push(c.clone());
        }
        for (w, _) in sorted_words.iter() {
            if itos.len() >= vocab_size {
                break;
            }
            if !itos.contains(w) {
                itos.push(w.clone());
            }
        }

        let stoi: HashMap<String, u32> = itos
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        Self { stoi, itos, unk_id: 1, char_boundary }
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let words = tokenize(text);
        let mut ids = Vec::new();
        for w in words {
            if let Some(&id) = self.stoi.get(&w) {
                ids.push(id);
            } else {
                for c in w.chars() {
                    let c_str = c.to_string();
                    ids.push(*self.stoi.get(&c_str).unwrap_or(&self.unk_id));
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            let id = ids[i];
            if id < self.char_boundary && id > 1 {
                let mut chars = String::new();
                while i < ids.len() && ids[i] < self.char_boundary && ids[i] > 1 {
                    chars.push_str(&self.itos[ids[i] as usize]);
                    i += 1;
                }
                parts.push(chars);
            } else {
                parts.push(self.itos[id as usize].clone());
                i += 1;
            }
        }
        parts.join(" ")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let s = self.itos.join("\n");
        fs::write(path, s)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = fs::read_to_string(path)?;
        let itos: Vec<String> = s.lines().map(|l| l.to_string()).collect();
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        let char_boundary = {
            let mut boundary = 2u32;
            for (i, w) in itos.iter().enumerate() {
                if i >= 2 && w.chars().count() == 1 && *w != "<PAD>" && *w != "<UNK>" {
                    boundary = i as u32 + 1;
                } else if i as u32 >= boundary {
                    break;
                }
            }
            boundary
        };
        Ok(Self { stoi, itos, unk_id: 1, char_boundary })
    }
}

pub struct WordRNN {
    pub embedding: Embedding,
    pub lstm: LSTM,
    pub head: Linear,
}

impl WordRNN {
    pub fn new(vocab_size: usize, vb: &VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, EMBED_SIZE, vb.pp("embed"))?;
        let lstm = LSTM::new(EMBED_SIZE, HIDDEN_SIZE, LSTMConfig::default(), vb.pp("lstm"))?;
        let head = candle_nn::linear(HIDDEN_SIZE, vocab_size, vb.pp("head"))?;
        Ok(Self { embedding, lstm, head })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(x)?;
        let states = self.lstm.seq(&x)?;
        let x = self.lstm.states_to_tensor(&states)?;
        self.head.forward(&x)
    }
}

pub fn get_batch(data: &[u32], seq_len: usize, batch_size: usize, rng: &mut impl Rng) -> (Vec<u32>, Vec<u32>) {
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

pub fn sample(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .map(|&l| (l / temperature).exp().max(1e-30))
        .enumerate()
        .collect();
    let sum: f32 = scaled.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return rng.gen_range(0..logits.len()) as u32;
    }
    for (_, p) in scaled.iter_mut() {
        *p /= sum;
    }
    scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = (40.min(scaled.len())).max(1);
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
    top[0].0 as u32
}

pub fn train(device: &Device, tokenizer: &Tokenizer, data: &[u32]) -> Result<VarMap> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = WordRNN::new(tokenizer.vocab_size(), &vb)?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;
    let mut rng = rand::thread_rng();
    let seq_len = SEQ_LEN.min(data.len() / 2);
    let batch_size = BATCH_SIZE.min(data.len() / seq_len.max(1));
    if seq_len < 4 {
        eprintln!("error: texto demasiado corto ({})", data.len());
        std::process::exit(1);
    }

    let steps_per_epoch = (data.len() / (seq_len * batch_size)).max(40);

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
        if epoch % 5 == 0 {
            varmap.save("modelo.safetensors").ok();
        }
    }
    Ok(varmap)
}

pub fn generate(
    device: &Device,
    tokenizer: &Tokenizer,
    varmap: &VarMap,
    prompt: &str,
    gen_len: usize,
    temperature: f32,
) -> Result<String> {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
    let model = WordRNN::new(tokenizer.vocab_size(), &vb)?;
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
        last_id = sample(&logits_vec, temperature, &mut rng);
        generated.push(last_id);
    }

    Ok(tokenizer.decode(&generated))
}

fn strip_prefix(text: &str, prefix: &str) -> String {
    if let Some(rest) = text.strip_prefix(prefix) {
        rest.trim().to_string()
    } else {
        text.trim().to_string()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let load = args.iter().any(|a| a == "--load");
    let ask = args.iter().any(|a| a == "--ask");
    let has_file = args.len() >= 2 && !args[1].starts_with('-');

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("usando: {device:?}");

    let model_path = "modelo.safetensors";
    let vocab_path = "modelo.vocab";
    let saved_exists = Path::new(model_path).exists() && Path::new(vocab_path).exists();

    let (tokenizer, varmap) = if load || ask {
        if !saved_exists {
            eprintln!("error: no hay modelo guardado. entrená uno primero.");
            std::process::exit(1);
        }
        println!("cargando modelo guardado...");
        let tokenizer = Tokenizer::load(vocab_path)?;
        println!("vocabulario: {} palabras", tokenizer.vocab_size());
        let mut varmap = VarMap::new();
        let _model = WordRNN::new(
            tokenizer.vocab_size(),
            &VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        varmap.load(model_path)?;
        (tokenizer, varmap)
    } else if has_file {
        let text = fs::read_to_string(&args[1])?;
        println!("caracteres: {}", text.chars().count());
        let tokenizer = Tokenizer::new(&text, VOCAB_SIZE);
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
        let _model = WordRNN::new(
            tokenizer.vocab_size(),
            &VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        varmap.load(model_path)?;
        (tokenizer, varmap)
    } else {
        eprintln!("uso:");
        eprintln!("  {} <archivo.txt>       entrenar modelo", args[0]);
        eprintln!("  {} --load [prompt]     cargar y generar", args[0]);
        eprintln!("  {} --ask <pregunta>    responder pregunta", args[0]);
        std::process::exit(1);
    };

    if ask {
        let pos = args.iter().position(|a| a == "--ask").unwrap();
        let question = args.get(pos + 1).cloned().unwrap_or_default();
        if question.is_empty() {
            eprintln!("error: pasá una pregunta. Ej: --ask \"¿Qué es la literatura?\"");
            std::process::exit(1);
        }
        let prompt = format!("Pregunta: {} Respuesta:", question);
        println!("--- respuesta ---\n");
        let output = generate(&device, &tokenizer, &varmap, &prompt, QA_LEN, 0.4)?;
        let answer = strip_prefix(&output, &prompt);
        println!("{}", answer);
        return Ok(());
    }

    let prompt = if load {
        let pos = args.iter().position(|a| a == "--load").unwrap();
        args.get(pos + 1).cloned().unwrap_or_default()
    } else {
        args.get(2).cloned().unwrap_or_default()
    };

    if !prompt.is_empty() {
        println!("\n--- generacion ---\n");
        let output = generate(&device, &tokenizer, &varmap, &prompt, GEN_LEN, 0.6)?;
        println!("{output}");
    }

    Ok(())
}
