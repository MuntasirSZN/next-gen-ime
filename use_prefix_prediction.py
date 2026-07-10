#!/usr/bin/env python3
"""
Example: Using prefix-based prediction with your trained Bangla GRU model

This script shows how to:
1. Load your trained model
2. Use prefix filtering to get candidates starting with specific characters
"""

import torch
from prefix_predictor import PrefixAwarePredictor

# Assuming you have these imports from your original script
# from train_bangla_gru_sp import NextWordGRU, BanglaSentencePieceTokenizer


def load_model(model_dir: str):
    """Load trained model and tokenizer"""
    import os
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Load tokenizer (you need to import BanglaSentencePieceTokenizer)
    from train_bangla_gru_sp import BanglaSentencePieceTokenizer, NextWordGRU
    
    tokenizer_path = os.path.join(model_dir, 'sp_bangla')
    tokenizer = BanglaSentencePieceTokenizer.load(tokenizer_path)
    print(f"✓ Loaded tokenizer ({tokenizer.vocab_size_actual:,} tokens)")
    
    # Load model
    model_path = os.path.join(model_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    model = NextWordGRU(
        vocab_size=config['vocab_size'],
        emb_dim=config['emb_dim'],
        hid_dim=config['hid_dim'],
        n_layers=config['n_layers'],
        dropout=config.get('dropout', 0.3),
        weight_tying=config.get('weight_tying', False),
    ).to(device)
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"✓ Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    return model, tokenizer, device, config['context_len']


def main():
    # === Example 1: Load model and use prefix filtering ===
    print("=" * 70)
    print(" Example: Prefix-Based Prediction")
    print("=" * 70)
    
    # Load your trained model
    MODEL_DIR = "bangla_gru_sp"  # Change this to your model directory
    
    try:
        model, tokenizer, device, context_len = load_model(MODEL_DIR)
    except Exception as e:
        print(f"\n⚠ Error loading model: {e}")
        print(f"\nMake sure you have:")
        print(f"  1. Trained the model first: python train_bangla_gru_sp.py")
        print(f"  2. Set the correct MODEL_DIR path")
        print(f"  3. Both train_bangla_gru_sp.py and prefix_predictor.py in the same directory")
        return
    
    # Create predictor with prefix support
    predictor = PrefixAwarePredictor(model, tokenizer, device, context_len)
    
    # === Example 2: Get predictions with specific prefix ===
    print("\n" + "=" * 70)
    print(" Example: Predictions starting with 'কা'")
    print("=" * 70)
    
    text = "আমি বাংলায়"
    prefix = "কা"
    
    print(f"\n📝 Input: {text}")
    print(f"🔍 Prefix filter: '{prefix}'")
    
    predictions = predictor.predict_with_prefix(text, prefix=prefix, top_k=10)
    
    print("\n🎯 Predictions:")
    for i, (word, prob) in enumerate(predictions, 1):
        print(f"   {i:2d}. {word} ({prob*100:.1f}%)")
    
    # === Example 3: Compare with and without prefix ===
    print("\n" + "=" * 70)
    print(" Comparison: With vs Without Prefix")
    print("=" * 70)
    
    test_cases = [
        ("আমি বাংলায়", "গ"),
        ("বাংলাদেশের রাজধানী", "ঢা"),
        ("আজকে আমি", "কা"),
    ]
    
    for text, prefix in test_cases:
        print(f"\n📝 Input: {text}")
        
        # Without prefix
        no_prefix = predictor.predict_with_prefix(text, prefix="", top_k=5)
        print(f"   Without filter:")
        for i, (word, prob) in enumerate(no_prefix, 1):
            print(f"     {i}. {word} ({prob*100:.1f}%)")
        
        # With prefix
        with_prefix = predictor.predict_with_prefix(text, prefix=prefix, top_k=5)
        print(f"   With prefix '{prefix}':")
        if with_prefix:
            for i, (word, prob) in enumerate(with_prefix, 1):
                print(f"     {i}. {word} ({prob*100:.1f}%)")
        else:
            print(f"     No words starting with '{prefix}' found")
    
    # === Example 4: Interactive mode ===
    print("\n" + "=" * 70)
    print(" Interactive Mode Available")
    print("=" * 70)
    print("\nYou can now use interactive mode:")
    print("  predictor.interactive_with_prefix()")
    print("\nOr run programmatically:")
    print("  predictions = predictor.predict_with_prefix('আমি', prefix='ক', top_k=5)")
    
    # Uncomment the next line to start interactive mode
    # predictor.interactive_with_prefix()


# === Example 5: Using the advanced method ===
def advanced_example():
    """
    The advanced method filters at the logit level,
    which can be more efficient for large vocabularies
    """
    MODEL_DIR = "bangla_gru_sp"
    model, tokenizer, device, context_len = load_model(MODEL_DIR)
    predictor = PrefixAwarePredictor(model, tokenizer, device, context_len)
    
    text = "আমি বাংলায়"
    prefix = "কা"
    
    # Use advanced method (masks logits before softmax)
    predictions = predictor.predict_with_prefix_advanced(text, prefix=prefix, top_k=10)
    
    print(f"\nAdvanced method predictions (prefix='{prefix}'):")
    for i, (word, prob) in enumerate(predictions, 1):
        print(f"   {i:2d}. {word} ({prob*100:.1f}%)")


if __name__ == "__main__":
    main()
    
    print("\n" + "=" * 70)
    print(" Quick Reference")
    print("=" * 70)
    print("""
    # Basic usage:
    predictions = predictor.predict_with_prefix(
        text="আমি বাংলায়", 
        prefix="কা",      # Only words starting with 'কা'
        top_k=10          # Return top 10 matches
    )
    
    # Interactive mode:
    predictor.interactive_with_prefix()
    # Then type: আমি বাংলায় | কা
    
    # Advanced (logit-level filtering):
    predictions = predictor.predict_with_prefix_advanced(
        text="আমি বাংলায়",
        prefix="কা"
    )
    """)
