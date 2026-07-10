# next-gen-ime
Building the Next Generation of Bangla IME!

## Architecture

GRU-based next-word prediction with SentencePiece BPE tokenizer, optimized for
low-latency IME inference on Apple Silicon.

| Component | Details |
|-----------|---------|
| Model | GRU (2 layers, weight-tied output) |
| Tokenizer | SentencePiece BPE |
| Vocab | 8,000 tokens |
| Context | 16 subword tokens |
| Params | ~3.2M (~12 MB) |
| Inference | PyTorch, ONNX, Rust |

## Quick Start

### Training

```bash
# Full training (recommended: 25 epochs)
uv run python train_bangla_gru_sp.py --epochs 25

# Resume interrupted training
uv run python train_bangla_gru_sp.py --epochs 25 --resume bangla_gru_sp

# Custom config
uv run python train_bangla_gru_sp.py --epochs 25 --vocab_size 8000 --emb_dim 192 --hid_dim 384
```

Checkpoints are saved every epoch to `bangla_gru_sp/checkpoints/`. Training can be
paused (Ctrl+C) and resumed at any time with `--resume`.

### Inference

```bash
# Interactive prediction
uv run python train_bangla_gru_sp.py --load bangla_gru_sp --interactive

# Single prediction
uv run python train_bangla_gru_sp.py --load bangla_gru_sp --text "আমি বাংলায়"

# GUI
uv run python gui.py

# Prefix-aware prediction (for IME)
uv run python use_prefix_prediction.py
```

### ONNX Export

```bash
uv run --with onnx --with onnxruntime --with onnxscript python export_onnx.py
```

Uses the legacy TorchScript exporter (`dynamo=False`) since the dynamo exporter
mis-converts GRU layers. Verifies output matches PyTorch (typical diff < 1e-5).

### Rust Inference

```bash
cd inference && cargo run
```

Requires `brew install sentencepiece` so the Rust crate links against the system
library (avoids protobuf symbol conflicts with ort's bundled ONNX Runtime).

The Rust `Predictor` loads the ONNX model via `ort::Session`, tokenizes with
SentencePiece, and supports prefix-filtered top-k prediction (masks non-matching
logits to `-inf` before softmax).

## Training Details

### Data Sources

- **wikimedia/wikipedia** (Bengali, 20231101) — up to 50k articles (streaming)
- **shawon95/Bengali-Fake-Review-Dataset** — ~9k reviews

Texts are split into sentences at Bengali danda `।`, double danda `॥`, and
standard punctuation before tokenization, preventing cross-topic context leaking
and ensuring full data utilization by the SentencePiece trainer.

### Model

- **Embedding**: 192-dim with weight tying (shared with output projection)
- **GRU**: 384 hidden, 2 layers, dropout 0.3
- **LayerNorm** before output projection
- **Output**: projection from hidden to embedding space, then `F.linear` with
  tied embedding weights

Weight tying cuts the output layer parameters entirely, reducing model size by
~5x compared to a standard untied architecture.

### Training Config

- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Schedule**: Linear warmup (10% of epochs) + cosine annealing to 1e-6
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Batch**: 128 x 4 gradient accumulation = effective 512
- **Mixed precision**: float16 autocast on MPS (no GradScaler needed)
- **Early stopping**: patience 4 epochs

### M4/Apple Silicon Optimizations

- MPS float16 autocast in train and eval loops
- `PYTORCH_MPS_FAST_MATH=1` for faster Metal kernels
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for memory management
- `num_workers=0`, `pin_memory=False` (MPS best practice)

## File Structure

```
train_bangla_gru_sp.py   # Training script (model, tokenizer, training loop)
prefix_predictor.py      # Prefix-aware prediction for IME
export_onnx.py           # ONNX export with verification
gui.py                   # Tkinter evaluation GUI
use_prefix_prediction.py # Prefix prediction examples
bangla_gru_sp/           # Model artifacts
  model.pt               # PyTorch checkpoint
  model.onnx             # ONNX model
  sp_bangla.model        # SentencePiece model
  sp_bangla.vocab        # Vocabulary (TSV)
  sequences.pt           # Cached training sequences (for resume)
  checkpoints/           # Per-epoch checkpoints
inference/               # Rust ONNX inference
  src/predictor.rs       # Predictor with prefix filtering
  src/main.rs            # Demo CLI
```

## CLI Reference

```
--load DIR          Load model for inference (skip training)
--resume DIR        Resume training from checkpoint
--text TEXT         Predict next word for given text
--interactive       Interactive prediction mode
--epochs N          Number of epochs (default: 15)
--max_samples N     Max samples per dataset (default: 50000)
--context_len N     Context length in tokens (default: 16)
--vocab_size N      SentencePiece vocab size (default: 8000)
--emb_dim N         Embedding dimension (default: 192)
--hid_dim N         GRU hidden dimension (default: 384)
--weight_tying      Tie embedding/output weights (default: on)
--no-weight_tying   Disable weight tying
--label_smoothing F Label smoothing factor (default: 0.1)
--grad_accum_steps N  Gradient accumulation (default: 4)
--batch_size N      Batch size (default: 128)
--lr F              Learning rate (default: 0.001)
--patience N        Early stopping patience (default: 4)
```
