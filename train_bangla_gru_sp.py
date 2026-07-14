"""
বাংলা Next Word Prediction Model - GRU + SentencePiece (Mac M1 MPS)

Changes from original:
- GRU instead of LSTM (faster, fewer parameters, similar performance)
- SentencePiece tokenizer (subword tokenization, handles OOV better)
- MPS-optimized settings

Usage:
    python train_bangla_gru_sp.py
    python train_bangla_gru_sp.py --epochs 20 --batch_size 128
    python train_bangla_gru_sp.py --max_samples 50000 --vocab_size 16000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
import re
import random
import numpy as np
import time
import argparse
import os
import tempfile
from typing import List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# Device Setup for Mac M1 (MPS Best Practices)
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device with MPS-specific setup"""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✓ Using MPS (Metal Performance Shaders) on Apple Silicon")
            # Set MPS-specific optimizations
            os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
            os.environ.setdefault('PYTORCH_MPS_FAST_MATH', '1')
            return torch.device("mps")
    
    if torch.cuda.is_available():
        print("✓ Using CUDA")
        return torch.device("cuda")
    
    print("⚠ Using CPU (MPS not available)")
    return torch.device("cpu")


def sync_mps(device: torch.device):
    """Synchronize MPS device (useful for accurate timing and memory management)"""
    if device.type == "mps":
        torch.mps.synchronize()


def empty_mps_cache(device: torch.device):
    """Empty MPS cache to free memory"""
    if device.type == "mps":
        torch.mps.empty_cache()


# =============================================================================
# SentencePiece Tokenizer Wrapper
# =============================================================================

class BanglaSentencePieceTokenizer:
    """SentencePiece-based tokenizer for Bangla text"""

    BANGLA_PATTERN = re.compile(r'[\u0980-\u09FF]+')
    # Bengali sentence boundaries: danda, double danda, and common punctuation
    SENTENCE_SPLIT = re.compile(r'[।\u0965.!?\n]+')
    
    def __init__(self, vocab_size: int = 16000, model_type: str = "bpe"):
        """
        Args:
            vocab_size: Target vocabulary size (SentencePiece default: 8000-32000)
            model_type: 'bpe' (Byte-Pair Encoding) or 'unigram'
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp: Optional[spm.SentencePieceProcessor] = None
        self.model_prefix: Optional[str] = None
    
    def _preprocess_text(self, text: str) -> str:
        """Extract only Bangla text for training"""
        words = self.BANGLA_PATTERN.findall(str(text))
        return ' '.join(words)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences at Bengali danda, double danda, and punctuation."""
        parts = self.SENTENCE_SPLIT.split(str(text))
        sentences = []
        for part in parts:
            cleaned = self._preprocess_text(part)
            if cleaned.strip() and len(cleaned.strip()) > 5:
                sentences.append(cleaned.strip())
        return sentences
    
    def train(self, texts: List[str], model_dir: str = ".") -> 'BanglaSentencePieceTokenizer':
        """Train SentencePiece model on texts"""
        print("Training SentencePiece tokenizer...")
        
        # Create temporary file with training data (one sentence per line)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as f:
            train_file = f.name
            processed_count = 0
            for i, text in enumerate(texts):
                sentences = self._split_into_sentences(text)
                for sent in sentences:
                    f.write(sent + '\n')
                    processed_count += 1
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1:,} texts for tokenizer training...")
        
        print(f"  Total texts with Bangla content: {processed_count:,}")
        
        # Set model path
        self.model_prefix = os.path.join(model_dir, "sp_bangla")
        
        # Train SentencePiece model
        # Key parameters for Bangla:
        # - character_coverage: 0.9995 for non-Latin scripts
        # - split_by_whitespace: True (Bangla uses spaces)
        # - add_dummy_prefix: False (Bangla doesn't need space prefix)
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=0.9995,  # High coverage for Bangla script
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            split_by_whitespace=True,
            add_dummy_prefix=False,  # Don't add leading space
            normalization_rule_name='identity',  # Preserve Bangla text as-is
            max_sentence_length=16384,  # Safety net for long sentences
            num_threads=os.cpu_count() or 4,
            train_extremely_large_corpus=False,
        )
        
        # Clean up temp file
        os.unlink(train_file)
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{self.model_prefix}.model")
        
        print(f"  ✓ Trained tokenizer with {self.sp.get_piece_size():,} tokens")
        print(f"  Model saved to: {self.model_prefix}.model")
        
        return self
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        processed = self._preprocess_text(text)
        return self.sp.encode(processed)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        # Filter out special tokens (pad=0, unk=1, bos=2, eos=3)
        filtered = [i for i in ids if i >= 4]
        return self.sp.decode(filtered)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to subword pieces"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        processed = self._preprocess_text(text)
        return self.sp.encode_as_pieces(processed)
    
    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.sp.get_piece_size()
    
    @property
    def pad_id(self) -> int:
        return 0
    
    @property
    def unk_id(self) -> int:
        return 1
    
    def id_to_piece(self, id: int) -> str:
        """Convert ID to token piece"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.sp.id_to_piece(id)
    
    def save(self, path: str):
        """Save tokenizer (copies model and vocab files)"""
        if self.model_prefix is None:
            raise RuntimeError("Tokenizer not trained")
        
        import shutil
        base_path = Path(path).with_suffix('')
        shutil.copy(f"{self.model_prefix}.model", f"{base_path}.model")
        shutil.copy(f"{self.model_prefix}.vocab", f"{base_path}.vocab")
        print(f"  Tokenizer saved to {base_path}.model and {base_path}.vocab")
    
    @classmethod
    def load(cls, path: str) -> 'BanglaSentencePieceTokenizer':
        """Load tokenizer from path"""
        base_path = Path(path).with_suffix('')
        model_path = f"{base_path}.model"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        tokenizer = cls()
        tokenizer.sp = spm.SentencePieceProcessor()
        tokenizer.sp.load(model_path)
        tokenizer.model_prefix = str(base_path)
        tokenizer.vocab_size = tokenizer.sp.get_piece_size()
        
        return tokenizer


# =============================================================================
# Dataset
# =============================================================================

class NextWordDataset(Dataset):
    """PyTorch Dataset for next word prediction"""
    
    def __init__(self, sequences: List[Tuple[List[int], int]], context_len: int):
        self.sequences = sequences
        self.context_len = context_len
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx, tgt = self.sequences[idx]
        if len(ctx) < self.context_len:
            ctx = [0] * (self.context_len - len(ctx)) + ctx
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def make_sequences(tokens: List[int], context_len: int) -> List[Tuple[List[int], int]]:
    """Create (context, target) pairs from token sequence"""
    seqs = []
    for i in range(context_len, len(tokens)):
        seqs.append((tokens[i - context_len:i], tokens[i]))
    return seqs


# =============================================================================
# GRU Model (replacing LSTM)
# =============================================================================

class NextWordGRU(nn.Module):
    """
    GRU-based next word prediction model

    GRU advantages over LSTM:
    - Fewer parameters (2 gates vs 3 gates) -> faster training
    - Often similar performance to LSTM
    - Better gradient flow for shorter sequences
    - More memory efficient
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 384,
        hid_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.3,
        weight_tying: bool = True,
    ):
        super().__init__()

        self.weight_tying = weight_tying
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )

        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hid_dim)

        if weight_tying:
            # Project hid_dim -> emb_dim, then reuse embedding weight for output
            if hid_dim != emb_dim:
                self.proj = nn.Linear(hid_dim, emb_dim, bias=False)
            else:
                self.proj = nn.Identity()
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.proj = None
            self.fc = nn.Linear(hid_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        nn.init.normal_(self.emb.weight, mean=0, std=0.1)
        with torch.no_grad():
            self.emb.weight[0].fill_(0)

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        if self.weight_tying:
            if hasattr(self.proj, 'weight'):
                nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.output_bias)
        else:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        embedded = self.drop(self.emb(x))
        output, _ = self.gru(embedded)
        last_hidden = self.drop(self.layer_norm(output[:, -1, :]))

        if self.weight_tying:
            projected = self.proj(last_hidden)
            return F.linear(projected, self.emb.weight, self.output_bias)
        else:
            return self.fc(last_hidden)
    
    def predict_topk(
        self, 
        x: torch.Tensor, 
        k: int = 10, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self(x) / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            return torch.topk(probs, k, dim=-1)


# =============================================================================
# Data Loading
# =============================================================================

def load_bangla_data(max_samples: int = 50000) -> List[str]:
    """Load Bangla text from HuggingFace datasets"""
    all_texts = []
    bangla_pattern = re.compile(r'[\u0980-\u09FF]+')
    
    # Dataset 1: Bengali Wikipedia
    print("\n" + "=" * 60)
    print("Loading: wikimedia/wikipedia (Bengali)")
    print("=" * 60)
    try:
        ds = load_dataset('wikimedia/wikipedia', '20231101.bn', split='train', streaming=True)
        print("  ✓ Loaded (streaming mode)")
        
        texts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            if 'text' in item and item['text']:
                texts.append(str(item['text']))
            if (i + 1) % 10000 == 0:
                print(f"     Loaded {i + 1:,} articles...")
        
        all_texts.extend(texts)
        print(f"  ✓ Added {len(texts):,} Wikipedia articles")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Dataset 3: Bengali Reviews
    print("\n" + "=" * 60)
    print("Loading: shawon95/Bengali-Fake-Review-Dataset")
    print("=" * 60)
    try:
        ds = load_dataset('shawon95/Bengali-Fake-Review-Dataset', split='train')
        print(f"  ✓ Loaded {len(ds):,} samples")
        print(f"  Columns: {ds.column_names}")

        for col in ds.column_names:
            sample = ds[0][col]
            if isinstance(sample, str) and bangla_pattern.findall(sample):
                texts = [str(t) for t in ds[col] if t]
                all_texts.extend(texts)
                print(f"  ✓ Added {len(texts):,} reviews from '{col}'")
                break
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Dataset 4: Bengali newspaper articles
    print("\n" + "=" * 60)
    print("Loading: zabir-nabil/bangla_newspaper_dataset")
    print("=" * 60)
    try:
        ds = load_dataset('zabir-nabil/bangla_newspaper_dataset', split='train', streaming=True)
        print("  ✓ Loaded (streaming mode)")
        texts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            content = item.get('content') or item.get('text') or ''
            if content and bangla_pattern.findall(str(content)):
                texts.append(str(content))
            if (i + 1) % 10000 == 0:
                print(f"     Loaded {i + 1:,} articles...")
        all_texts.extend(texts)
        print(f"  ✓ Added {len(texts):,} newspaper articles")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"TOTAL TEXTS LOADED: {len(all_texts):,}")
    print("=" * 60)
    
    return all_texts


# =============================================================================
# Training Functions (MPS-optimized)
# =============================================================================

def train_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int = 1,
    epoch: int = 0,
    total_epochs: int = 0,
    show_progress: bool = True
) -> float:
    """
    Train one epoch with MPS optimizations and progress bar
    
    MPS best practices applied:
    - Gradient accumulation for effective larger batch sizes
    - Periodic MPS synchronization
    - Gradient clipping for stability
    - Progress bar with minimal overhead (tqdm updates ~10 times/sec)
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    # Create progress bar
    if show_progress:
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=False,  # Don't leave progress bar after completion
            dynamic_ncols=True,  # Adapt to terminal width
            mininterval=0.1,  # Update at most 10 times per second (minimal overhead)
        )
    else:
        pbar = loader
    
    running_loss = 0.0
    
    for batch_idx, (ctx, tgt) in enumerate(pbar):
        ctx, tgt = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

        logits = model(ctx)
        loss = criterion(logits, tgt)

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()
        
        # Update weights after accumulation steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        running_loss += batch_loss
        
        # Update progress bar with running average loss
        if show_progress and (batch_idx + 1) % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'}, refresh=False)
        
        # Periodic sync for MPS (helps with memory management)
        if device.type == "mps" and (batch_idx + 1) % 100 == 0:
            torch.mps.synchronize()
    
    # Close progress bar
    if show_progress:
        pbar.close()
    
    # Handle any remaining gradients
    if len(loader) % grad_accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True
) -> Tuple[float, float, float]:
    """Evaluate model with top-1 and top-5 accuracy"""
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0
    
    if show_progress:
        pbar = tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True, mininterval=0.1)
    else:
        pbar = loader
    
    for ctx, tgt in pbar:
        ctx, tgt = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

        out = model(ctx)
        total_loss += criterion(out, tgt).item()

        # Top-1 accuracy
        correct += (out.argmax(-1) == tgt).sum().item()

        # Top-5 accuracy
        _, top5 = out.topk(5, dim=-1)
        correct_top5 += (top5 == tgt.unsqueeze(1)).any(-1).sum().item()

        total += tgt.size(0)
    
    if show_progress:
        pbar.close()
    
    # Sync before returning
    sync_mps(device)
    
    return total_loss / len(loader), correct / total, correct_top5 / total


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    best_loss: float,
    history: dict,
    args,
    is_best: bool = False,
):
    """Save full training checkpoint for resume capability."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'history': history,
        'args': vars(args),
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
        },
    }

    # Save per-epoch checkpoint (keep all)
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, path)

    # Save 'latest' for easy resume
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    # Save best model separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)

    print(f"         └─ Checkpoint saved: epoch {epoch}" +
          (" [best]" if is_best else ""))


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, float, dict]:
    """Load checkpoint and restore all training state. Returns (epoch, best_loss, history)."""
    # If path is a directory, load checkpoint_latest.pt
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, 'checkpoints', 'checkpoint_latest.pt')

    print(f"Resuming from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # Restore RNG states for reproducibility
    rng = ckpt.get('rng_state', {})
    if 'python' in rng:
        random.setstate(rng['python'])
    if 'numpy' in rng:
        np.random.set_state(rng['numpy'])
    if 'torch' in rng:
        torch.random.set_rng_state(rng['torch'])

    return ckpt['epoch'], ckpt['best_loss'], ckpt['history']


# =============================================================================
# Predictor Class
# =============================================================================

class NextWordPredictor:
    """Easy-to-use prediction interface"""
    
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: BanglaSentencePieceTokenizer, 
        device: torch.device, 
        context_len: int
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_len = context_len
        self.model.eval()
    
    def predict(
        self, 
        text: str, 
        top_k: int = 10, 
        temperature: float = 1.0
    ) -> List[Tuple[str, float]]:
        """Predict next token candidates"""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) == 0:
            return []
        
        # Prepare context
        if len(tokens) > self.context_len:
            tokens = tokens[-self.context_len:]
        elif len(tokens) < self.context_len:
            tokens = [0] * (self.context_len - len(tokens)) + tokens
        
        x = torch.tensor([tokens], device=self.device)
        probs, indices = self.model.predict_topk(x, top_k, temperature)
        
        results = []
        for idx, prob in zip(indices[0].tolist(), probs[0].tolist()):
            # Skip special tokens (pad=0, unk=1, bos=2, eos=3)
            if idx >= 4:
                piece = self.tokenizer.id_to_piece(idx)
                # Clean up SentencePiece's underscore prefix
                piece = piece.replace('▁', '')
                if piece:
                    results.append((piece, prob))
        
        return results
    
    def demo(self, examples: List[str] = None):
        """Run demo predictions"""
        if examples is None:
            examples = [
                "আমি বাংলায়",
                "বাংলাদেশের রাজধানী",
                "আজকে আমি",
                "সে স্কুলে",
                "এটি একটি",
            ]
        
        print("\n" + "=" * 60)
        print("  বাংলা Next Word Prediction Demo (GRU + SentencePiece)")
        print("=" * 60)
        
        for text in examples:
            print(f"\n📝 Input: {text}")
            
            # Show tokenization
            tokens = self.tokenizer.tokenize(text)
            print(f"   Tokens: {tokens}")
            
            preds = self.predict(text, top_k=5)
            
            if preds:
                print("   Predictions:")
                for i, (word, prob) in enumerate(preds, 1):
                    bar = '█' * int(prob * 25)
                    print(f"   {i}. {word:12} {prob*100:5.1f}% {bar}")
            else:
                print("   No predictions available")
    
    def interactive(self):
        """Interactive prediction mode"""
        print("\n" + "=" * 60)
        print("  Interactive Mode - Type Bangla text to get predictions")
        print("  Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            try:
                text = input("\n📝 Enter text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not text:
                    continue
                
                # Show tokenization
                tokens = self.tokenizer.tokenize(text)
                print(f"   Tokens: {tokens}")
                
                preds = self.predict(text, top_k=10)
                
                if preds:
                    print("\n🎯 Predictions:")
                    for i, (word, prob) in enumerate(preds, 1):
                        print(f"   {i:2d}. {word} ({prob*100:.1f}%)")
                else:
                    print("   No Bangla words detected")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break


# =============================================================================
# Main Training Function
# =============================================================================

def main(args):
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # MPS-specific seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    
    # Device setup
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # =================================================================
    # Load Data & Tokenizer (with resume support)
    # =================================================================
    sequences_path = os.path.join(args.output_dir, 'sequences.pt')
    tokenizer_path = os.path.join(args.output_dir, 'sp_bangla')

    if args.resume and os.path.exists(sequences_path) and os.path.exists(f"{tokenizer_path}.model"):
        # Fast resume: load cached tokenizer + sequences
        print("\n" + "=" * 70)
        print(" RESUMING: Loading cached tokenizer and sequences")
        print("=" * 70)

        tokenizer = BanglaSentencePieceTokenizer.load(tokenizer_path)
        vocab_size = tokenizer.vocab_size_actual
        print(f"  ✓ Loaded tokenizer ({vocab_size:,} tokens)")

        saved = torch.load(sequences_path, weights_only=False)
        train_seqs = saved['train']
        val_seqs = saved['val']
        print(f"  ✓ Loaded {len(train_seqs):,} train + {len(val_seqs):,} val sequences")
    else:
        # Full pipeline: load data, train tokenizer, create sequences
        print("\n" + "=" * 70)
        print(" STEP 1: Loading Data")
        print("=" * 70)

        texts = load_bangla_data(max_samples=args.max_samples)

        if len(texts) == 0:
            print("ERROR: No texts loaded!")
            return

        # =================================================================
        # Train SentencePiece Tokenizer
        # =================================================================
        print("\n" + "=" * 70)
        print(" STEP 2: Training SentencePiece Tokenizer")
        print("=" * 70)

        tokenizer = BanglaSentencePieceTokenizer(
            vocab_size=args.vocab_size,
            model_type=args.sp_model_type,
        )
        tokenizer.train(texts, model_dir=args.output_dir)

        # Show sample tokenization
        print("\nSample tokenization:")
        sample_texts = ["আমি বাংলায় গান গাই", "বাংলাদেশের রাজধানী ঢাকা"]
        for text in sample_texts:
            pieces = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            print(f"  '{text}'")
            print(f"    → Pieces: {pieces}")
            print(f"    → IDs: {ids}")

        vocab_size = tokenizer.vocab_size_actual
        print(f"\n✓ Vocabulary size: {vocab_size:,}")

        # =================================================================
        # Create Sequences
        # =================================================================
        print("\n" + "=" * 70)
        print(" STEP 3: Creating Training Sequences")
        print("=" * 70)

        all_seqs = []
        for i, text in enumerate(texts):
            # Encode per-sentence to avoid cross-topic context leaking
            sentences = tokenizer._split_into_sentences(text)
            for sent in sentences:
                tokens = tokenizer.encode(sent)
                if len(tokens) > args.context_len:
                    seqs = make_sequences(tokens, args.context_len)
                    all_seqs.extend(seqs)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} texts, sequences: {len(all_seqs):,}")

        print(f"\n✓ Total sequences: {len(all_seqs):,}")

        if len(all_seqs) == 0:
            print("ERROR: No sequences created!")
            return

        # Split data
        random.shuffle(all_seqs)
        val_size = int(len(all_seqs) * args.val_split)
        train_seqs = all_seqs[val_size:]
        val_seqs = all_seqs[:val_size]

        # Cache sequences for fast resume
        print(f"  Caching sequences to {sequences_path}...")
        torch.save({'train': train_seqs, 'val': val_seqs}, sequences_path)
    
    # Create data loaders
    # MPS best practices:
    # - num_workers=0: MPS doesn't benefit from multiprocessing data loading
    # - pin_memory=False: Not applicable for MPS
    # - persistent_workers=False: Default, no benefit for num_workers=0
    train_loader = DataLoader(
        NextWordDataset(train_seqs, args.context_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True  # Helps with batch normalization consistency
    )
    val_loader = DataLoader(
        NextWordDataset(val_seqs, args.context_len),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"  Train: {len(train_seqs):,} sequences, {len(train_loader):,} batches")
    print(f"  Val: {len(val_seqs):,} sequences, {len(val_loader):,} batches")
    
    # =================================================================
    # Create Model
    # =================================================================
    print("\n" + "=" * 70)
    print(" STEP 4: Creating GRU Model")
    print("=" * 70)
    
    model = NextWordGRU(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        weight_tying=args.weight_tying,
    )

    model = model.to(device=device, dtype=torch.float32)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Architecture: GRU (replaces LSTM)")
    print(f"  Tokenizer: SentencePiece ({args.sp_model_type})")
    print(f"  Vocabulary: {vocab_size:,}")
    print(f"  Parameters: {n_params:,} (trainable: {n_trainable:,})")
    print(f"  Device: {device}")
    print(f"  Dtype: float32")
    
    # Sync after model creation
    sync_mps(device)
    
    # =================================================================
    # Training Setup
    # =================================================================
    print("\n" + "=" * 70)
    print(" STEP 5: Training")
    print("=" * 70)
    
    # AdamW with weight decay (better than plain Adam)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Cosine annealing with linear warmup
    warmup_epochs = max(1, args.epochs // 10)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=args.label_smoothing
    )

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top5': []}
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    start_epoch = 1

    # Gradient accumulation for effective larger batch size
    grad_accum_steps = args.grad_accum_steps
    effective_batch = args.batch_size * grad_accum_steps

    # Resume from checkpoint if requested
    if args.resume:
        start_epoch, best_loss, history = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1
        print(f"  Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.4f}")

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {grad_accum_steps} steps")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Weight tying: {args.weight_tying}")
    print(f"  LR schedule: warmup({warmup_epochs}) + cosine({args.epochs - warmup_epochs})")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, grad_accum_steps,
            epoch=epoch, total_epochs=args.epochs, show_progress=True,
        )

        # Evaluate
        val_loss, val_acc, val_top5 = evaluate(
            model, val_loader, criterion, device, show_progress=True
        )

        # Step the scheduler (epoch-based, not metric-based)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5'].append(val_top5)

        elapsed = time.time() - t0

        is_best = val_loss < best_loss
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Top1: {val_acc*100:.1f}% | Top5: {val_top5*100:.1f}% | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # Save best model
        if is_best:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print("         └─ ✓ Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                # Save final checkpoint before stopping
                save_checkpoint(
                    checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, best_loss=best_loss,
                    history=history, args=args, is_best=False,
                )
                break

        # Save checkpoint every epoch
        save_checkpoint(
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
            epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, best_loss=best_loss,
            history=history, args=args, is_best=is_best,
        )

        # Periodic cache clearing for MPS
        if device.type == "mps" and epoch % 5 == 0:
            empty_mps_cache(device)

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"\n✓ Training complete! Best val loss: {best_loss:.4f}")
    
    # =================================================================
    # Save Model
    # =================================================================
    print("\n" + "=" * 70)
    print(" STEP 6: Saving Model")
    print("=" * 70)
    
    # Save model (move to CPU for compatibility)
    model_path = os.path.join(args.output_dir, 'model.pt')
    model_cpu = model.cpu()
    torch.save({
        'model': model_cpu.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'context_len': args.context_len,
            'emb_dim': args.emb_dim,
            'hid_dim': args.hid_dim,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'weight_tying': args.weight_tying,
            'label_smoothing': args.label_smoothing,
            'model_type': 'gru',
            'tokenizer_type': 'sentencepiece',
            'sp_model_type': args.sp_model_type,
        },
        'history': history
    }, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # Move model back to device for inference
    model = model.to(device)
    
    # Tokenizer is already saved during training
    print(f"  ✓ Tokenizer saved to {args.output_dir}/sp_bangla.model")
    
    # =================================================================
    # Demo
    # =================================================================
    print("\n" + "=" * 70)
    print(" STEP 7: Demo")
    print("=" * 70)
    
    predictor = NextWordPredictor(model, tokenizer, device, args.context_len)
    predictor.demo()
    
    # Interactive mode if requested
    if args.interactive:
        predictor.interactive()
    
    print("\n" + "=" * 70)
    print(" DONE!")
    print("=" * 70)
    print(f"\nModel saved to: {args.output_dir}/")
    print(f"  - model.pt (GRU weights)")
    print(f"  - sp_bangla.model (SentencePiece tokenizer)")
    print(f"  - sp_bangla.vocab (Vocabulary)")
    print(f"\nTo load and use:")
    print(f"  python train_bangla_gru_sp.py --load {args.output_dir} --interactive")


def load_and_predict(args):
    """Load saved model and run predictions"""
    device = get_device()
    
    # Load tokenizer
    tokenizer_path = os.path.join(args.load, 'sp_bangla')
    tokenizer = BanglaSentencePieceTokenizer.load(tokenizer_path)
    print(f"✓ Loaded SentencePiece tokenizer ({tokenizer.vocab_size_actual:,} tokens)")
    
    # Load model
    model_path = os.path.join(args.load, 'model.pt')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Verify it's a GRU model
    model_type = config.get('model_type', 'gru')
    print(f"  Model type: {model_type.upper()}")
    
    model = NextWordGRU(
        vocab_size=config['vocab_size'],
        emb_dim=config['emb_dim'],
        hid_dim=config['hid_dim'],
        n_layers=config['n_layers'],
        dropout=config.get('dropout', 0.3),
        weight_tying=config.get('weight_tying', False),
    ).to(device)
    
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"✓ Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Create predictor
    predictor = NextWordPredictor(model, tokenizer, device, config['context_len'])
    
    if args.text:
        # Single prediction
        print(f"\n📝 Input: {args.text}")
        tokens = tokenizer.tokenize(args.text)
        print(f"   Tokens: {tokens}")
        preds = predictor.predict(args.text, top_k=10)
        print("\n🎯 Predictions:")
        for i, (word, prob) in enumerate(preds, 1):
            print(f"   {i:2d}. {word} ({prob*100:.1f}%)")
    else:
        # Demo and interactive
        predictor.demo()
        if args.interactive:
            predictor.interactive()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Bangla Next Word Prediction Model (GRU + SentencePiece, Mac M1 MPS)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument('--load', type=str, default=None,
                        help='Load saved model from directory (skip training)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint directory')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to predict next word for')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive prediction mode')

    # Data
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Max samples per dataset')
    parser.add_argument('--context_len', type=int, default=16,
                        help='Context length (previous tokens)')

    # Tokenizer (SentencePiece)
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='SentencePiece vocabulary size (8000-32000 recommended)')
    parser.add_argument('--sp_model_type', type=str, default='bpe',
                        choices=['bpe', 'unigram'],
                        help='SentencePiece model type')

    # Model (GRU)
    parser.add_argument('--emb_dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--hid_dim', type=int, default=512,
                        help='GRU hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--weight_tying', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Tie embedding and output weights (reduces model size)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0.0 to disable)')

    # Training
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (increase for effective larger batch)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=4,
                        help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')

    # Other
    parser.add_argument('--output_dir', type=str, default='bangla_gru_sp',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("  বাংলা Next Word Prediction - GRU + SentencePiece (Mac M1 MPS)")
    print("=" * 70)
    
    if args.load:
        load_and_predict(args)
    else:
        main(args)
