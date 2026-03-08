#!/usr/bin/env python3
"""
PITOMADOM RTL Root Transformer — Training Script
Hebrew root-level bidirectional transformer with gematria features.

Architecture (~5M params):
  - Letter embeddings -> Root encoder (3 letters -> 1 root vector)
  - RTL positional encoding (right-to-left)
  - N transformer blocks (bidirectional attention + FFN)
  - Gematria sinusoidal features
  - Dissonance-gated distance bias (learnable)
  - Output: predict masked root letters (3 x 22-way classification)

Training: Masked Root Modeling (like BERT, but at root level)
  - Mask 15% of roots in sequence
  - Predict each masked root's 3 letters independently

Usage:
  python3 train_rtl.py --data hebrew.txt [--steps 50000] [--dim 256] [--layers 4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, sys, time, json, re, math, struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse


# ============================================================================
# HEBREW ALPHABET & GEMATRIA
# ============================================================================

# Standard gematria values
HE_GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8,
    'ט': 9, 'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60,
    'ע': 70, 'פ': 80, 'צ': 90, 'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    # Final forms — same gematria as regular
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

# Map final forms to regular
FINAL_TO_REGULAR = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}

# 22 standard Hebrew letters (no finals — we normalize)
HE_LETTERS = list('אבגדהוזחטיכלמנסעפצקרשת')
NUM_LETTERS = len(HE_LETTERS)  # 22

# Letter to index (0-21 = letters, 22 = PAD, 23 = MASK, 24 = UNK)
LETTER2IDX = {letter: i for i, letter in enumerate(HE_LETTERS)}
PAD_LETTER = NUM_LETTERS      # 22
MASK_LETTER = NUM_LETTERS + 1  # 23
UNK_LETTER = NUM_LETTERS + 2   # 24
LETTER_VOCAB_SIZE = NUM_LETTERS + 3  # 25

# Special root tokens
PAD_ROOT = (PAD_LETTER, PAD_LETTER, PAD_LETTER)
MASK_ROOT = (MASK_LETTER, MASK_LETTER, MASK_LETTER)

# Common prefixes and suffixes to strip for root extraction
PREFIXES = ['והת', 'הת', 'ומ', 'וה', 'של', 'וב', 'וכ', 'ול', 'ומ', 'וש',
            'ה', 'ב', 'כ', 'ל', 'מ', 'ש', 'ו', 'נ', 'י', 'ת', 'א']
SUFFIXES = ['ותיהם', 'ותיהן', 'ותיך', 'ותינו', 'יהם', 'יהן', 'ותי',
            'ים', 'ות', 'ית', 'ני', 'כם', 'כן', 'הם', 'הן',
            'ה', 'ת', 'י', 'ך', 'ם', 'ן']

# Sort by length descending so longer affixes match first
PREFIXES.sort(key=len, reverse=True)
SUFFIXES.sort(key=len, reverse=True)


def normalize_letter(ch: str) -> str:
    """Normalize final forms to regular."""
    return FINAL_TO_REGULAR.get(ch, ch)


def letter_to_idx(ch: str) -> int:
    """Convert Hebrew letter to index."""
    ch = normalize_letter(ch)
    return LETTER2IDX.get(ch, UNK_LETTER)


def is_hebrew(ch: str) -> bool:
    """Check if character is Hebrew."""
    return ch in HE_GEMATRIA


def extract_consonants(word: str) -> List[str]:
    """Extract and normalize Hebrew consonants from a word."""
    return [normalize_letter(ch) for ch in word if is_hebrew(ch)]


def extract_root(word: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract approximate 3-letter root from Hebrew word.
    Returns tuple of 3 letter indices, or None if can't extract.

    Heuristic:
    1. Strip known prefixes and suffixes
    2. Normalize final forms
    3. Take first 3 consonants as root
    """
    consonants = extract_consonants(word)
    if len(consonants) < 2:
        return None

    # Try stripping prefixes
    text = ''.join(consonants)
    for prefix in PREFIXES:
        prefix_norm = ''.join(normalize_letter(c) for c in prefix)
        if text.startswith(prefix_norm) and len(text) - len(prefix_norm) >= 2:
            text = text[len(prefix_norm):]
            break

    # Try stripping suffixes
    for suffix in SUFFIXES:
        suffix_norm = ''.join(normalize_letter(c) for c in suffix)
        if text.endswith(suffix_norm) and len(text) - len(suffix_norm) >= 2:
            text = text[:-len(suffix_norm)]
            break

    # Convert back to list
    letters = [ch for ch in text if ch in LETTER2IDX]

    if len(letters) < 2:
        return None

    # Pad or truncate to 3 letters
    if len(letters) == 2:
        # Two-letter root — duplicate last (common in Hebrew: e.g., סב→סבב)
        letters.append(letters[-1])
    elif len(letters) > 3:
        letters = letters[:3]

    return (LETTER2IDX[letters[0]], LETTER2IDX[letters[1]], LETTER2IDX[letters[2]])


def root_gematria(root: Tuple[int, int, int]) -> int:
    """Compute gematria value of a root (by letter indices)."""
    total = 0
    for idx in root:
        if 0 <= idx < NUM_LETTERS:
            total += HE_GEMATRIA[HE_LETTERS[idx]]
    return total


def extract_hebrew_words(text: str) -> List[str]:
    """Extract Hebrew words from text."""
    words = []
    current = []
    for ch in text:
        if is_hebrew(ch):
            current.append(ch)
        else:
            if current:
                words.append(''.join(current))
                current = []
    if current:
        words.append(''.join(current))
    return words


# ============================================================================
# DATASET
# ============================================================================

class HebrewRootDataset(Dataset):
    """
    Dataset of Hebrew root sequences for Masked Root Modeling.

    Each sample is a sequence of roots extracted from Hebrew text.
    15% of roots are masked for prediction.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 64,
        max_samples: int = 0,  # 0 = all
        mask_prob: float = 0.15,
    ):
        self.seq_len = seq_len
        self.mask_prob = mask_prob

        print(f"Loading Hebrew corpus from {data_path}...")
        t0 = time.time()

        # Extract all roots from corpus
        all_roots = []
        lines_read = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = extract_hebrew_words(line)
                for word in words:
                    root = extract_root(word)
                    if root is not None:
                        all_roots.append(root)
                lines_read += 1
                if lines_read % 100000 == 0:
                    print(f"  {lines_read} lines, {len(all_roots)} roots...")
                if max_samples > 0 and len(all_roots) >= max_samples * seq_len:
                    break

        print(f"  Extracted {len(all_roots)} roots from {lines_read} lines in {time.time()-t0:.1f}s")

        # Build root vocabulary
        root_counts = {}
        for r in all_roots:
            root_counts[r] = root_counts.get(r, 0) + 1
        self.root_vocab = sorted(root_counts.keys(), key=lambda r: -root_counts[r])
        print(f"  Unique roots: {len(self.root_vocab)}")
        print(f"  Top 20 roots: {[(HE_LETTERS[r[0]]+HE_LETTERS[r[1]]+HE_LETTERS[r[2]], root_counts[r]) for r in self.root_vocab[:20]]}")

        # Pack into sequences
        self.sequences = []
        for i in range(0, len(all_roots) - seq_len, seq_len):
            seq = all_roots[i:i+seq_len]
            self.sequences.append(seq)

        if max_samples > 0:
            self.sequences = self.sequences[:max_samples]

        print(f"  Packed into {len(self.sequences)} sequences of length {seq_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Convert to tensors: (seq_len, 3) for the 3 letters of each root
        input_roots = torch.zeros(self.seq_len, 3, dtype=torch.long)
        target_roots = torch.zeros(self.seq_len, 3, dtype=torch.long)
        mask_positions = torch.zeros(self.seq_len, dtype=torch.bool)
        gematria_vals = torch.zeros(self.seq_len, dtype=torch.float)

        for i, root in enumerate(seq):
            c1, c2, c3 = root
            target_roots[i] = torch.tensor([c1, c2, c3])
            gematria_vals[i] = root_gematria(root) / 500.0  # Normalize

            # Mask with probability
            if torch.rand(1).item() < self.mask_prob:
                mask_positions[i] = True
                # 80% replace with MASK, 10% random, 10% keep
                r = torch.rand(1).item()
                if r < 0.8:
                    input_roots[i] = torch.tensor([MASK_LETTER, MASK_LETTER, MASK_LETTER])
                elif r < 0.9:
                    # Random root
                    input_roots[i] = torch.randint(0, NUM_LETTERS, (3,))
                else:
                    input_roots[i] = torch.tensor([c1, c2, c3])
            else:
                input_roots[i] = torch.tensor([c1, c2, c3])

        return {
            'input_roots': input_roots,        # (seq_len, 3)
            'target_roots': target_roots,      # (seq_len, 3)
            'mask_positions': mask_positions,   # (seq_len,)
            'gematria': gematria_vals,          # (seq_len,)
        }


# ============================================================================
# MODEL
# ============================================================================

class GematriaSinusoidal(nn.Module):
    """Sinusoidal encoding of gematria values (like positional encoding but for numbers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Precompute frequency bands
        freqs = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        self.register_buffer('freqs', freqs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: (batch, seq_len) gematria values (normalized)
        Returns:
            (batch, seq_len, dim) sinusoidal encoding
        """
        # Scale values to useful range
        v = values.unsqueeze(-1) * 500.0  # Un-normalize
        angles = v * self.freqs  # (batch, seq_len, dim//2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class RTLPositionalEncoding(nn.Module):
    """RTL positional encoding — position 0 = rightmost (present in Hebrew)."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Reverse for RTL: position 0 = rightmost token
        pe = pe.flip(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns (seq_len, dim) positional encoding."""
        return self.pe[-seq_len:]


class RootEncoder(nn.Module):
    """Encode a root (3 letter indices) into a dense vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.letter_embed = nn.Embedding(LETTER_VOCAB_SIZE, dim)
        # Project 3 concatenated letter embeddings to root embedding
        self.proj = nn.Linear(3 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, roots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roots: (batch, seq_len, 3) letter indices
        Returns:
            (batch, seq_len, dim) root embeddings
        """
        b, s, _ = roots.shape
        # Embed each letter
        letters = self.letter_embed(roots)  # (b, s, 3, dim)
        # Concatenate and project
        concat = letters.reshape(b, s, -1)  # (b, s, 3*dim)
        return self.norm(self.proj(concat))  # (b, s, dim)


class DissonanceBias(nn.Module):
    """
    Learnable distance bias modulated by dissonance level.
    High dissonance -> less distance penalty -> allow far attention.
    """

    def __init__(self, num_heads: int, max_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        # Learnable base distance penalty per head
        self.distance_scale = nn.Parameter(torch.randn(num_heads) * 0.1)
        # Learnable dissonance sensitivity per head
        self.dissonance_sensitivity = nn.Parameter(torch.ones(num_heads) * 0.5)

    def forward(self, seq_len: int, dissonance: float = 0.5) -> torch.Tensor:
        """
        Returns attention bias of shape (num_heads, seq_len, seq_len).
        """
        positions = torch.arange(seq_len, device=self.distance_scale.device, dtype=torch.float)
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # (s, s)

        # Per-head distance penalty, modulated by dissonance
        # High dissonance -> penalty reduced -> allow far jumps
        penalty = self.distance_scale * (1.0 - dissonance * self.dissonance_sensitivity)
        # penalty: (num_heads,) -> (num_heads, 1, 1)
        bias = -penalty.abs().unsqueeze(-1).unsqueeze(-1) * dist.unsqueeze(0)

        return bias  # (num_heads, seq_len, seq_len)


class RTLTransformerBlock(nn.Module):
    """Transformer block with RTL attention and dissonance gating."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Multi-head self-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Dissonance-gated distance bias
        self.dissonance_bias = DissonanceBias(num_heads)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dissonance: float = 0.5) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            dissonance: 0.0-1.0
        Returns:
            (batch, seq_len, dim)
        """
        b, s, d = x.shape

        # Pre-norm attention
        normed = self.ln1(x)
        Q = self.q_proj(normed).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(normed).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(normed).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores + dissonance bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.dissonance_bias(s, dissonance)  # (num_heads, s, s)
        scores = scores + bias.unsqueeze(0)  # broadcast batch

        # Bidirectional: no causal mask
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        out = self.o_proj(out)

        # Residual
        x = x + out

        # Pre-norm FFN
        x = x + self.ff(self.ln2(x))

        return x


class HebrewRootTransformer(nn.Module):
    """
    PITOMADOM RTL Root Transformer.

    Root-level bidirectional transformer for Hebrew with:
    - CCC root encoding (3 letters -> embedding)
    - RTL positional encoding
    - Gematria sinusoidal features
    - Dissonance-gated attention
    - Masked root prediction head
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Root encoding
        self.root_encoder = RootEncoder(dim)

        # Positional encoding (RTL)
        self.pos_encoding = RTLPositionalEncoding(dim, max_len)

        # Gematria features
        self.gematria_enc = GematriaSinusoidal(dim)

        # Combine root + position + gematria
        self.input_proj = nn.Linear(dim * 2, dim)  # root_embed + gematria
        self.input_norm = nn.LayerNorm(dim)
        self.input_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            RTLTransformerBlock(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output: predict 3 letters independently
        self.output_norm = nn.LayerNorm(dim)
        self.head_c1 = nn.Linear(dim, NUM_LETTERS)  # First consonant
        self.head_c2 = nn.Linear(dim, NUM_LETTERS)  # Second consonant
        self.head_c3 = nn.Linear(dim, NUM_LETTERS)  # Third consonant

    def forward(
        self,
        input_roots: torch.Tensor,
        gematria: torch.Tensor,
        dissonance: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_roots: (batch, seq_len, 3) letter indices
            gematria: (batch, seq_len) normalized gematria values
            dissonance: 0.0-1.0

        Returns:
            (logits_c1, logits_c2, logits_c3) each (batch, seq_len, NUM_LETTERS)
        """
        b, s, _ = input_roots.shape

        # Encode roots
        root_emb = self.root_encoder(input_roots)  # (b, s, dim)

        # Gematria features
        gem_emb = self.gematria_enc(gematria)  # (b, s, dim)

        # Combine
        combined = torch.cat([root_emb, gem_emb], dim=-1)  # (b, s, 2*dim)
        x = self.input_proj(combined)  # (b, s, dim)

        # Add RTL positional encoding
        pos = self.pos_encoding(s)  # (s, dim)
        x = x + pos.unsqueeze(0)

        x = self.input_norm(x)
        x = self.input_drop(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, dissonance=dissonance)

        # Output heads
        x = self.output_norm(x)
        logits_c1 = self.head_c1(x)
        logits_c2 = self.head_c2(x)
        logits_c3 = self.head_c3(x)

        return logits_c1, logits_c2, logits_c3

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TRAINING
# ============================================================================

def compute_loss(
    model: HebrewRootTransformer,
    batch: dict,
    device: torch.device,
    dissonance: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Compute masked root prediction loss."""
    input_roots = batch['input_roots'].to(device)
    target_roots = batch['target_roots'].to(device)
    mask_pos = batch['mask_positions'].to(device)
    gematria = batch['gematria'].to(device)

    logits_c1, logits_c2, logits_c3 = model(input_roots, gematria, dissonance)

    # Only compute loss on masked positions
    if mask_pos.sum() == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss, {'loss': 0.0, 'acc': 0.0}

    # Gather masked positions
    # mask_pos is (batch, seq_len), logits are (batch, seq_len, NUM_LETTERS)
    # target_roots is (batch, seq_len, 3)
    masked_c1 = logits_c1[mask_pos]  # (num_masked, NUM_LETTERS)
    masked_c2 = logits_c2[mask_pos]
    masked_c3 = logits_c3[mask_pos]

    masked_targets = target_roots[mask_pos]  # (num_masked, 3)
    target_c1 = masked_targets[:, 0]  # (num_masked,)
    target_c2 = masked_targets[:, 1]
    target_c3 = masked_targets[:, 2]

    loss_c1 = F.cross_entropy(masked_c1, target_c1)
    loss_c2 = F.cross_entropy(masked_c2, target_c2)
    loss_c3 = F.cross_entropy(masked_c3, target_c3)

    loss = (loss_c1 + loss_c2 + loss_c3) / 3.0

    # Accuracy
    with torch.no_grad():
        acc_c1 = (masked_c1.argmax(-1) == target_c1).float().mean()
        acc_c2 = (masked_c2.argmax(-1) == target_c2).float().mean()
        acc_c3 = (masked_c3.argmax(-1) == target_c3).float().mean()
        acc = (acc_c1 + acc_c2 + acc_c3) / 3.0

    return loss, {
        'loss': loss.item(),
        'loss_c1': loss_c1.item(),
        'loss_c2': loss_c2.item(),
        'loss_c3': loss_c3.item(),
        'acc': acc.item(),
        'acc_c1': acc_c1.item(),
        'acc_c2': acc_c2.item(),
        'acc_c3': acc_c3.item(),
    }


def save_checkpoint(model, optimizer, step, loss, path):
    """Save training checkpoint."""
    torch.save({
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def export_numpy_weights(model: HebrewRootTransformer, path: str):
    """Export model weights as numpy arrays for pitomadom inference."""
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    np.savez(path, **weights)
    print(f"Exported {len(weights)} weight tensors to {path}")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    dataset = HebrewRootDataset(
        data_path=args.data,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        mask_prob=args.mask_prob,
    )

    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = HebrewRootTransformer(
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=args.dim * 4,
        max_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    num_params = model.count_params()
    print(f"\nModel: HebrewRootTransformer")
    print(f"  dim={args.dim}, heads={args.heads}, layers={args.layers}, ff_dim={args.dim*4}")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps: {args.steps}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.wd,
    )

    # Cosine LR schedule with warmup
    def lr_schedule(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    print(f"\n{'='*70}")
    print(f"  PITOMADOM — Training RTL Root Transformer")
    print(f"{'='*70}\n")

    model.train()
    step = 0
    running_loss = 0.0
    running_acc = 0.0
    t0 = time.time()
    log_interval = 100
    save_interval = 5000

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'training.log')
    log_file = open(log_path, 'w')

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            # Random dissonance for augmentation
            dissonance = np.random.uniform(0.1, 0.9)

            loss, metrics = compute_loss(model, batch, device, dissonance)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            running_loss += metrics['loss']
            running_acc += metrics['acc']
            step += 1

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_acc = running_acc / log_interval
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]['lr']
                steps_per_sec = step / elapsed

                msg = (f"step {step}/{args.steps} | loss {avg_loss:.4f} | "
                       f"acc {avg_acc:.3f} | lr {lr:.2e} | "
                       f"{steps_per_sec:.1f} steps/s | {elapsed:.0f}s")
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()

                running_loss = 0.0
                running_acc = 0.0

            if step % save_interval == 0:
                ckpt_path = os.path.join(args.save_dir, f'pitomadom_step{step}.pt')
                save_checkpoint(model, optimizer, step, metrics['loss'], ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.save_dir, 'pitomadom_final.pt')
    save_checkpoint(model, optimizer, step, metrics['loss'], final_path)

    # Export numpy weights for pitomadom inference
    npz_path = os.path.join(args.save_dir, 'pitomadom_rtl_weights.npz')
    export_numpy_weights(model, npz_path)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Steps: {step}, Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Final loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.3f}")
    print(f"  Checkpoint: {final_path}")
    print(f"  Numpy weights: {npz_path}")
    print(f"{'='*70}")

    log_file.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PITOMADOM RTL Root Transformer Training')

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to Hebrew text file')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length (roots)')
    parser.add_argument('--max_samples', type=int, default=0, help='Max training samples (0=all)')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Mask probability')

    # Model
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup', type=int, default=500, help='Warmup steps')
    parser.add_argument('--steps', type=int, default=50000, help='Total training steps')

    # Output
    parser.add_argument('--save_dir', type=str, default='./pitomadom_weights', help='Save directory')

    args = parser.parse_args()
    train(args)
