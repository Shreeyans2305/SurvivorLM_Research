# SurvivorLM

SurvivorLM is a **research repository** for building survival-focused language models through two parallel experiments:

1. **Instruction fine-tuning in Colab (recommended track)** using files in `local_fine_tuning/`
2. **Training a compact LLM locally from scratch (experimental track)** in `scratch_llm/`

This repo is intentionally hands-on and iterative: dataset extraction, cleaning, upsampling, tokenizer training, model training, and lightweight inference demos all live together so experiments are easy to reproduce and compare.

---

## 🔬 What this repository is for

This project is not a polished production package yet.
It is a **research playground** for answering questions like:

- How far can a small local transformer go on a survival domain?
- How much does curated instruction data help compared with raw corpus pretraining?
- What is the practical trade-off between local training and Colab-assisted adapter tuning?

---

## 🧭 Two training paths in this repo

### A) Colab-based instruction fine-tuning (`local_fine_tuning/`)

This is the main practical path for model quality.

- Notebook workflow lives in `local_fine_tuning/colab.ipynb` and `local_fine_tuning/qlora-gemma4survival.ipynb`
- Config is in `local_fine_tuning/qlora_config.yaml`
- Training data files are in `local_fine_tuning/data/`
- Adapters are saved under `local_fine_tuning/adapters/`

Most importantly, this path is built around the **upsampled instruction corpus**:

- `local_fine_tuning/instructions_upsampled.txt`
- mirrored dataset source in `data/dataset/instructions_upsampled.txt`

In short: **Colab + instruction upsampling** is the core fine-tuning strategy in this repository.

---

### B) Local from-scratch LLM experiment (`scratch_llm/`)

This folder contains your attempt to build and train a language model end-to-end locally:

- corpus extraction and cleaning scripts (`extractor*.py`, `clean_instructions.py`)
- staged dataset builders (`build_stage2*.py`, `build_stage3*.py`)
- tokenizer pipeline (`building_tokenizer_corpus.py`, `main.py`)
- local training and generation (`train.py`, `generate.py`, `test_model.py`)

This track is valuable as a research baseline and learning system, even when adapter fine-tuning on a larger base model outperforms it.

---

## 🧠 Model code at a glance

Core model implementation is in `model/`:

- `model/config.py` defines the compact transformer config (`vocab_size=8000`, `max_seq_len=512`, 4 layers, 4 heads, `d_model=256`)
- `model/gpt.py` implements causal self-attention + transformer blocks + LM head
- `model/instruction_dataset.py` builds Q/A-style masked training examples for supervised instruction tuning

This makes the research structure clear:

- **Architecture research:** `model/`
- **Data research:** `data/`, `scratch_llm/`
- **Fine-tuning research:** `local_fine_tuning/`, `fine-tuning/`

---

## 📦 Repository map

- `app.py` – lightweight Gradio interface for survival Q&A testing
- `data/` – raw, extracted, and staged corpora
- `model/` – custom small transformer implementation
- `scratch_llm/` – local training pipeline from tokenizer to inference
- `local_fine_tuning/` – Colab/QLoRA workflow and adapter outputs
- `fine-tuning/` – additional notebook-based adapter experiments
- `more/` – extra instruction and warmup corpora

---

## 🚀 Typical workflow used here

1. Gather and clean survival corpora
2. Build staged datasets and instruction variants
3. Upsample instruction data (`instructions_upsampled`)
4. Run Colab-based LoRA/QLoRA fine-tuning in `local_fine_tuning/`
5. Compare against local-from-scratch baseline in `scratch_llm/`
6. Test prompts through generation scripts or Gradio app

---

## ⚠️ Research status

This repository is an active experiment log.

- Expect rapid iteration and occasional broken runs
- Keep notes on dataset versioning and prompt formats
- Treat results as research artifacts, not safety guarantees

For real-world emergency situations, always defer to qualified local emergency services and official survival guidance.

---

## 🤝 Why this project is exciting

SurvivorLM combines classic ML craftsmanship with practical constraints:

- **resource-aware modeling** (small local transformer)
- **modern adaptation methods** (LoRA/QLoRA)
- **domain focus** (survival + first-aid style instruction data)

It is both a build log and a benchmark arena for answering one core question:

**How useful can a compact, specialized survival assistant become with careful data curation and tuning?**

