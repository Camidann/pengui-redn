use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn read_file(p: &PathBuf) -> String {
    match fs::read_to_string(p) {
        Ok(s) => s,
        Err(_) => String::new(),
    }
}

fn extract_pairs(qtext: &str) -> Vec<String> {
    let mut pairs = Vec::new();
    let mut cur_q: Option<String> = None;
    let mut cur_a: Option<String> = None;
    let mut in_a = false;

    for line in qtext.lines() {
        let s = line.trim();
        if s.starts_with("Pregunta:") {
            if let (Some(q), Some(a)) = (cur_q.take(), cur_a.take()) {
                pairs.push(format!("Pregunta: {} Respuesta: {}", q, a));
            }
            cur_q = Some(s["Pregunta:".len()..].trim().to_string());
            cur_a = Some(String::new());
            in_a = false;
        } else if s.starts_with("Respuesta:") {
            cur_a = Some(s["Respuesta:".len()..].trim().to_string());
            in_a = true;
        } else if in_a && !s.is_empty() {
            if let Some(ref mut a) = cur_a {
                if !a.is_empty() {
                    a.push(' ');
                }
                a.push_str(s);
            }
        }
    }
    if let (Some(q), Some(a)) = (cur_q, cur_a) {
        pairs.push(format!("Pregunta: {} Respuesta: {}", q, a));
    }
    pairs
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut repeats: usize = 20;
    if args.len() > 1 {
        for i in 1..args.len() {
            if args[i] == "--repeats" || args[i] == "-r" {
                if let Some(v) = args.get(i + 1) {
                    repeats = v.parse().unwrap_or(repeats);
                }
            }
        }
    }

    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
    let cont = root.join("contenidosTEST");

    let qa_path = cont.join("preguntas_respuestas.txt");
    let out_path = cont.join("train_augmented.txt");
    let other_files = vec![
        cont.join("train_total.txt"),
        cont.join("el_quijote.txt"),
        cont.join("EL004338.txt"),
    ];

    let qtext = read_file(&qa_path);
    let pairs = extract_pairs(&qtext);
    if pairs.is_empty() {
        eprintln!("No se encontraron pares Q/A en {:?}", qa_path);
    }

    let mut out_parts: Vec<String> = Vec::new();

    for f in other_files.iter() {
        let txt = read_file(f);
        if !txt.is_empty() {
            println!("Incluyendo: {}", f.display());
            out_parts.push(txt);
        }
    }

    if !pairs.is_empty() {
        println!("Añadiendo {} pares repetidos {} veces", pairs.len(), repeats);
        let join_pairs = pairs.join("\n");
        for _ in 0..repeats {
            out_parts.push(join_pairs.clone());
        }
    }

    let train_word = cont.join("train_word.txt");
    if out_parts.is_empty() && train_word.exists() {
        println!("Incluyendo train_word.txt como fallback");
        out_parts.push(read_file(&train_word));
    }

    let final_text = out_parts.join("\n\n");
    match fs::File::create(&out_path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(final_text.as_bytes()) {
                eprintln!("Error escribiendo {}: {}", out_path.display(), e);
            } else {
                println!("Escrito: {}", out_path.display());
            }
        }
        Err(e) => eprintln!("No pude crear {}: {}", out_path.display(), e),
    }
}
