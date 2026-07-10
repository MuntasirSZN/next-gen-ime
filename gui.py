#!/usr/bin/env python3
"""
GUI for Bangla Next Word Prediction using NextWordPredictor and PrefixAwarePredictor.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk


def load_predictors(model_dir: str):
    """Load model and create both predictors."""
    import torch
    from train_bangla_gru_sp import BanglaSentencePieceTokenizer, NextWordGRU, NextWordPredictor
    from prefix_predictor import PrefixAwarePredictor

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = BanglaSentencePieceTokenizer.load(os.path.join(model_dir, "sp_bangla"))

    checkpoint = torch.load(
        os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]

    model = NextWordGRU(
        vocab_size=config["vocab_size"],
        emb_dim=config["emb_dim"],
        hid_dim=config["hid_dim"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.3),
        weight_tying=config.get("weight_tying", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    context_len = config["context_len"]
    next_word = NextWordPredictor(model, tokenizer, device, context_len)
    prefix_aware = PrefixAwarePredictor(model, tokenizer, device, context_len)

    return next_word, prefix_aware, device


class PredictorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bangla Next Word Predictor")
        self.root.geometry("700x600")
        self.root.minsize(500, 400)

        self.next_word_predictor = None
        self.prefix_predictor = None

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        # Status bar
        self.status_var = tk.StringVar(value="Loading model...")
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(side="bottom", fill="x")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Input
        ttk.Label(main, text="Input Text:", font=("", 13, "bold")).pack(anchor="w")
        self.text_entry = ttk.Entry(main, font=("", 14))
        self.text_entry.pack(fill="x", pady=(2, 8))
        self.text_entry.bind("<KeyRelease>", lambda e: self._on_input_change())

        # Prefix
        prefix_frame = ttk.Frame(main)
        prefix_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(prefix_frame, text="Prefix Filter:", font=("", 13, "bold")).pack(side="left")
        self.prefix_entry = ttk.Entry(prefix_frame, font=("", 14), width=20)
        self.prefix_entry.pack(side="left", padx=(8, 0))
        self.prefix_entry.bind("<KeyRelease>", lambda e: self._on_input_change())

        # Options
        opts = ttk.Frame(main)
        opts.pack(fill="x", pady=(0, 8))

        ttk.Label(opts, text="Predictor:").pack(side="left")
        self.predictor_var = tk.StringVar(value="prefix_advanced")
        for val, label in [
            ("next_word", "NextWordPredictor"),
            ("prefix_filter", "PrefixAware (filter)"),
            ("prefix_advanced", "PrefixAware (logit mask)"),
        ]:
            ttk.Radiobutton(opts, text=label, variable=self.predictor_var, value=val,
                            command=self._on_input_change).pack(side="left", padx=4)

        # Top-K / Temperature
        params = ttk.Frame(main)
        params.pack(fill="x", pady=(0, 8))

        ttk.Label(params, text="Top-K:").pack(side="left")
        self.topk_var = tk.IntVar(value=10)
        ttk.Spinbox(params, from_=1, to=50, textvariable=self.topk_var, width=5,
                     command=self._on_input_change).pack(side="left", padx=(2, 12))

        ttk.Label(params, text="Temperature:").pack(side="left")
        self.temp_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(params, from_=0.1, to=3.0, increment=0.1, textvariable=self.temp_var,
                     width=6, command=self._on_input_change).pack(side="left", padx=(2, 0))

        # Predict button
        self.predict_btn = ttk.Button(main, text="Predict", command=self._on_input_change)
        self.predict_btn.pack(pady=(0, 8))

        # Results
        ttk.Label(main, text="Predictions:", font=("", 13, "bold")).pack(anchor="w")

        columns = ("rank", "word", "probability")
        self.tree = ttk.Treeview(main, columns=columns, show="headings", height=15)
        self.tree.heading("rank", text="#")
        self.tree.heading("word", text="Word")
        self.tree.heading("probability", text="Probability")
        self.tree.column("rank", width=40, anchor="center")
        self.tree.column("word", width=250, anchor="w")
        self.tree.column("probability", width=300, anchor="w")

        scrollbar = ttk.Scrollbar(main, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Allow clicking a prediction to append it to input
        self.tree.bind("<Double-1>", self._on_result_double_click)

    def _load_model_async(self):
        def load():
            try:
                model_dir = os.path.join(os.path.dirname(__file__), "bangla_gru_sp")
                nw, pa, device = load_predictors(model_dir)
                self.next_word_predictor = nw
                self.prefix_predictor = pa
                self.root.after(0, lambda: self.status_var.set(f"Ready (device: {device})"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error loading model: {e}"))

        threading.Thread(target=load, daemon=True).start()

    def _on_input_change(self):
        if not self.next_word_predictor:
            return

        text = self.text_entry.get().strip()
        if not text:
            self._clear_results()
            return

        prefix = self.prefix_entry.get().strip()
        mode = self.predictor_var.get()
        top_k = self.topk_var.get()
        temperature = self.temp_var.get()

        def run():
            try:
                if mode == "next_word":
                    preds = self.next_word_predictor.predict(text, top_k=top_k, temperature=temperature)
                elif mode == "prefix_filter":
                    preds = self.prefix_predictor.predict_with_prefix(
                        text, prefix=prefix, top_k=top_k, temperature=temperature
                    )
                else:
                    preds = self.prefix_predictor.predict_with_prefix_advanced(
                        text, prefix=prefix, top_k=top_k, temperature=temperature
                    )
                self.root.after(0, lambda: self._show_results(preds))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Prediction error: {e}"))

        threading.Thread(target=run, daemon=True).start()

    def _show_results(self, predictions):
        self._clear_results()
        for i, (word, prob) in enumerate(predictions, 1):
            bar = "\u2588" * int(prob * 40)
            self.tree.insert("", "end", values=(i, word, f"{prob*100:.1f}%  {bar}"))
        self.status_var.set(f"{len(predictions)} predictions")

    def _clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _on_result_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        word = self.tree.item(sel[0])["values"][1]
        current = self.text_entry.get()
        separator = " " if current and not current.endswith(" ") else ""
        self.text_entry.delete(0, "end")
        self.text_entry.insert(0, current + separator + str(word))
        self.prefix_entry.delete(0, "end")
        self._on_input_change()


def main():
    root = tk.Tk()
    PredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
