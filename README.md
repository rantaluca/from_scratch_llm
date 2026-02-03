# Tiny LLM 

<div align="center">
  <h2>Training a Small GPT-Style Language Model from Scratch with PyTorch</h2>
  <p align="center">
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-2.0.1-red.svg?logo=pytorch&logoColor=white" alt="PyTorch">
    </a>
    <a href="https://huggingface.co/datasets">
      <img src="https://img.shields.io/badge/Datasets-Hugging%20Face-yellow.svg?logo=huggingface&logoColor=white" alt="Hugging Face Datasets">
    </a>
    <a href="https://github.com/openai/tiktoken">
      <img src="https://img.shields.io/badge/TikToken-OpenAI-blue.svg?logo=openai&logoColor=white" alt="TikToken">
    </a>
  </p>
</div>

---

## Overview

**Tiny LLM** is a from-scratch implementation of a small GPT-style language model, trained using PyTorch. The project explores the process of training a transformer-based language model on diverse datasets, including Wikipedia, stories, Q&A, and instruct datasets. The goal is to understand the mechanics of modern language models, from data preprocessing to model architecture and training dynamics.

The model is trained on an **RTX 3060 (12GB VRAM)** with a **batch size of 16**, using mixed precision and gradient accumulation for efficiency.

---

## How It Works

### Data Pipeline
1. **Datasets**: The model is trained on a mix of:
   - English Wikipedia
   - Simple Stories
   - FineWeb-Edu
   - OpenWebText2
   - Q&A datasets (e.g., Alpaca, Reddit Instruct)
2. **Tokenizer**: A custom BPE tokenizer based on GPT-2, extended with special tokens (`<|im_start|>`, `<|im_end|>`, `<|pad|>`).
3. **Preprocessing**: Data is shuffled, tokenized, and padded dynamically to a context length of **512 tokens**.

### Model Architecture
- **Embedding Dimension**: 512
- **Attention Heads**: 8
- **Layers**: 8
- **Vocabulary Size**: 50,260
- **Optimizer**: AdamW with cosine annealing (lr=3e-4)
- **Precision**: Mixed (bfloat16)

### Training Loop
- **Gradient Accumulation**: 4 steps
- **Checkpoints**: Saved every 5,000 steps
- **Inference**: Real-time text generation during training to monitor progress

---

## Trial training on 25k steps

After the first **25,000 steps**, the model shows early signs of learning and loss is decreasing:

| Step  | Example Output                                                                                     |
|-------|----------------------------------------------------------------------------------------------------|
| 500   | Repetitive, nonsensical output: `"The on� to� to� to� ..."`                                         |
| 10,000| Begins forming coherent phrases: `"The first thing you can do you?"`                               |
| 20,000| Generates contextually relevant (but repetitive) responses: `"What is the difference between X and Y?"` |
| 25,000| Improved coherence, though still prone to loops: `"How do you think about the world?"`              |

> *Full training logs and generated samples are saved in `train_output.txt` and `second_train_output.txt`.*

## Training on the complete dataset 

To be done ...

---
## Installation

```sh
git clone https://github.com/rantaluca/tiny_llm.git
cd tiny_llm
pip install torch tiktoken datasets tqdm
